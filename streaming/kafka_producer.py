"""
Kafka Producer for Batch+Streaming Pipeline
=============================================
Sends posts from Kaggle CSV data to Kafka topic.
Uses pre-computed centroids for fast topic assignment (no ML inference).

This is ALTERNATIVE to demo_ready/producer.py:
- demo_ready: Sends raw posts, consumer does full ML clustering
- streaming/: Sends posts, consumer uses pre-computed centroids (faster)
"""

import json
import time
import sys
import os
import random
from datetime import datetime
from kafka import KafkaProducer

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIG ---
KAFKA_TOPIC = 'posts_stream_v1'  # Different topic from demo_ready
BOOTSTRAP_SERVERS = ['localhost:29092']
MAX_POSTS = 500  # Limit for demo speed

def create_producer():
    """Connect to Kafka broker"""
    print(f"🔄 Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
        print(f"✅ Connected to Kafka!")
        return producer
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return None

def load_kaggle_data():
    """Load posts from demo-ready/data CSVs or demo_state folder"""
    try:
        from src.utils.demo_state import load_demo_state
        import pandas as pd
        import glob
        
        # 1. Try demo-ready/data CSV files (New Source for E2E Simulation)
        demo_ready_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo-ready", "data"
        )
        
        if os.path.exists(demo_ready_dir):
            csv_files = glob.glob(os.path.join(demo_ready_dir, "*.csv"))
            if csv_files:
                all_posts = []
                print(f"📂 Loading data from demo-ready/data: {len(csv_files)} files found")
                
                for f in csv_files:
                    try:
                        df = pd.read_csv(f, encoding='utf-8')
                        fname = os.path.basename(f).lower()
                        
                        # Identify mapping based on file type
                        is_fb = 'fb' in fname or 'facebook' in fname
                        
                        for _, row in df.iterrows():
                            # Extract text
                            if is_fb:
                                content = str(row.get('summaried_text', row.get('text', '')))
                                source = 'Facebook'
                            else:
                                content = str(row.get('summary', row.get('content', '')))
                                source = 'News'
                                
                            if len(content) > 10:
                                all_posts.append({
                                    'content': content[:1000],
                                    'source': source,
                                    'time': str(row.get('published_at', row.get('timestamp', row.get('time', datetime.now().isoformat())))),
                                    'final_topic': str(row.get('final_topic', 'Unknown'))
                                })
                    except Exception as e:
                        print(f"⚠️ Warning: Failed to load {f}: {e}")
                
                if all_posts:
                    print(f"✅ Loaded {len(all_posts)} posts from demo-ready CSVs")
                    # Return as DataFrame for consistency
                    return pd.DataFrame(all_posts)

        # 2. Try latest demo state folder (Pre-computed parquet)
        demo_states_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo_states"
        )
        
        if os.path.exists(demo_states_dir):
            folders = sorted([
                f for f in os.listdir(demo_states_dir)
                if os.path.isdir(os.path.join(demo_states_dir, f))
            ], reverse=True)
            
            for folder in folders:
                folder_path = os.path.join(demo_states_dir, folder)
                try:
                    state = load_demo_state(folder_path)
                    df_results = state.get('df_results')
                    if df_results is not None and len(df_results) > 0:
                        print(f"📂 Loaded {len(df_results)} posts from {folder}")
                        return df_results
                except Exception as e:
                    print(f"⚠️ Failed to load {folder}: {e}")
        
        # 3. Last Fallback: Generic data directory
        print("📂 Last fallback: Scanning generic data/ directory...")
        csv_files = glob.glob("data/**/*.csv", recursive=True)
        all_posts = []
        
        for f in csv_files:
            try:
                df = pd.read_csv(f, encoding='utf-8')
                for _, row in df.iterrows():
                    content = str(row.get('content', row.get('title', '')))
                    if len(content) > 20:
                        all_posts.append({
                            'content': content[:500],
                            'source': 'News' if 'news' in f.lower() else 'Facebook',
                            'time': str(row.get('published_at', datetime.now().isoformat()))
                        })
            except Exception:
                pass
        
        print(f"📊 Loaded {len(all_posts)} posts from generic CSV files")
        return all_posts[:MAX_POSTS]
        
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return []

def run_producer():
    """Send posts to Kafka topic"""
    producer = create_producer()
    if not producer:
        return {"status": "error", "message": "Could not connect to Kafka"}
    
    # Load data
    posts = load_kaggle_data()
    if posts is None or (hasattr(posts, 'empty') and posts.empty) or (isinstance(posts, list) and not posts):
        return {"status": "error", "message": "No data found"}
    
    # Handle DataFrame vs list
    if hasattr(posts, 'to_dict'):
        records = posts.to_dict('records')
    else:
        records = posts
    
    # Limit for demo
    records = records[:MAX_POSTS]
    random.shuffle(records)
    
    print(f"🚀 Streaming {len(records)} posts to Kafka topic '{KAFKA_TOPIC}'...")
    
    sent = 0
    try:
        for i, row in enumerate(records):
            # Format message
            message = {
                "content": str(row.get('content', row.get('title', '')))[:500],
                "source": str(row.get('source', 'Unknown')),
                "time": str(row.get('time', row.get('published_at', datetime.now().isoformat()))),
                "final_topic": str(row.get('final_topic', 'Unknown'))  # Pre-computed topic
            }
            
            producer.send(KAFKA_TOPIC, value=message)
            sent += 1
            
            if i % 50 == 0:
                sys.stdout.write(f"\r📤 Sent {i+1}/{len(records)}...")
                sys.stdout.flush()
            
            # Realistic streaming delay
            time.sleep(random.uniform(0.01, 0.03))
        
        producer.flush()
        print(f"\n✅ Producer Complete: {sent} posts sent to '{KAFKA_TOPIC}'")
        
        return {"status": "success", "sent": sent, "topic": KAFKA_TOPIC}
        
    except KeyboardInterrupt:
        print("\n🛑 Producer stopped by user")
        return {"status": "interrupted", "sent": sent}
    finally:
        producer.close()

if __name__ == "__main__":
    result = run_producer()
    print(f"\n📊 Result: {result}")
