"""
Airflow DAG: Demo Streaming Pipeline (All-in-One)

Single DAG with PARALLEL Producer & Consumer:
[load_data] → [streaming_simulation] → [write_to_dashboard]

The streaming_simulation task runs Producer & Consumer in parallel threads.
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DB_URL = "postgresql://user:password@localhost:5432/trend_db"
default_args = {'owner': 'demo', 'retries': 0}


def find_demo_folder():
    import glob
    candidates = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
    return candidates[0] if candidates else os.path.join(PROJECT_ROOT, "demo/demo_data")


def load_data(**context):
    """Task 1: Load data from demo-ready CSVs (preferred), config override, or pre-computed state."""
    import pandas as pd
    import glob
    
    # 0. Check for explicit overrides from DAG run config
    dag_run = context.get('dag_run')
    custom_folder = dag_run.conf.get('demo_folder') if dag_run else None
    
    if custom_folder:
        print(f"🔧 Config override: Using demo folder {custom_folder}")
        if not os.path.exists(custom_folder):
            raise FileNotFoundError(f"Custom demo folder not found: {custom_folder}")
        
        # Load directly from this folder (skip CSV check)
        from src.utils.demo_state import load_demo_state
        state = load_demo_state(custom_folder)
        df = state.get('df_results')
        
        # Fallback for folder with just results.parquet
        if df is None:
             parquet_path = os.path.join(custom_folder, "results.parquet")
             if os.path.exists(parquet_path):
                 df = pd.read_parquet(parquet_path)
        
        if df is None or len(df) == 0:
             raise ValueError(f"No results found in {custom_folder}")

        temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
        os.makedirs(os.path.dirname(temp_file), exist_ok=True)
        df.to_parquet(temp_file)
        
        print(f"✅ Loaded {len(df)} posts from {custom_folder}")
        return {"posts": len(df), "folder": custom_folder, "source": "custom-override"}

    # 1. Try demo-ready/data CSV files (Primary source for E2E simulation)
    demo_ready_dir = os.path.join(PROJECT_ROOT, "demo-ready", "data")
    if os.path.exists(demo_ready_dir):
        csv_files = glob.glob(os.path.join(demo_ready_dir, "*.csv"))
        if csv_files:
            all_posts = []
            print(f"📂 Loading from CSVs in: {demo_ready_dir}")
            
            for f in csv_files:
                try:
                    df_csv = pd.read_csv(f, encoding='utf-8')
                    fname = os.path.basename(f).lower()
                    is_fb = 'fb' in fname or 'facebook' in fname
                    
                    for _, row in df_csv.iterrows():
                        if is_fb:
                            text = str(row.get('summaried_text', row.get('text', '')))
                            source = 'Facebook'
                        else:
                            text = str(row.get('summary', row.get('content', '')))
                            source = 'News'
                            
                        # Extract metrics safely (default to 0)
                        def get_metric(val):
                            try: return int(float(val)) if pd.notnull(val) else 0
                            except: return 0

                        likes = get_metric(row.get('likes', 0))
                        comments = get_metric(row.get('comments', 0))
                        shares = get_metric(row.get('shares', 0))
                            
                        if len(text) > 10:
                            all_posts.append({
                                'content': text,
                                'source': source,
                                'time': str(row.get('published_at', row.get('timestamp', row.get('time', datetime.now().isoformat())))),
                                'final_topic': str(row.get('final_topic', 'Unknown')),
                                'likes': likes,
                                'comments': comments,
                                'shares': shares,
                                'topic_type': str(row.get('topic_type', 'Trending' if row.get('final_topic') else 'Discovery')),
                                'score': float(row.get('score', 0.8)),
                                'category': str(row.get('category', 'T7'))
                            })
                except Exception as e:
                    print(f"⚠️ Warning loading {f}: {e}")
            
            if all_posts:
                df = pd.DataFrame(all_posts)
                temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
                os.makedirs(os.path.dirname(temp_file), exist_ok=True)
                df.to_parquet(temp_file)
                print(f"✅ Loaded {len(df)} posts from demo-ready CSVs")
                return {"posts": len(df), "source": "demo-ready-csv"}

    # 2. Fallback to old demo folder logic
    from src.utils.demo_state import load_demo_state
    demo_folder = find_demo_folder()
    print(f"📂 Fallback: Loading from {demo_folder}")
    
    state = load_demo_state(demo_folder)
    df = state.get('df_results')
    
    if df is None or len(df) == 0:
        raise ValueError("No results found in fallback either!")
    
    temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    df.to_parquet(temp_file)
    
    print(f"✅ Loaded {len(df)} posts from state fallback")
    return {"posts": len(df), "folder": demo_folder, "source": "demo-state"}


def produce_to_kafka(**context):
    """Task 2: Stream loaded data to Kafka for the external consumer."""
    import pandas as pd
    import json
    import time
    from kafka import KafkaProducer
    from datetime import datetime
    
    # Config
    KAFKA_TOPIC = 'posts_stream_v1'
    BOOTSTRAP_SERVERS = ['localhost:29092']
    
    temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
    if not os.path.exists(temp_file):
        print(f"⚠️ No data file found at {temp_file}")
        return
        
    df = pd.read_parquet(temp_file)
    total = len(df)
    
    print(f"\n{'='*60}")
    print(f"🚀 KAFKA PRODUCER: Streaming {total} posts to '{KAFKA_TOPIC}'")
    print(f"{'='*60}\n")
    
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )
    except Exception as e:
        print(f"❌ Failed to connect to Kafka at {BOOTSTRAP_SERVERS}: {e}")
        raise e
        
    sent_count = 0
    
    # Stream posts
    for i, row in df.iterrows():
        message = {
            "content": str(row.get('content', ''))[:1000],
            "source": str(row.get('source', 'Unknown')),
            "time": str(row.get('time', datetime.now().isoformat())),
            "final_topic": str(row.get('final_topic', 'Unknown')),
            "topic_type": str(row.get('topic_type', 'Discovery')),
            "score": float(row.get('score', 0.5)),
            "category": str(row.get('category', 'T7'))
        }
        
        producer.send(KAFKA_TOPIC, value=message)
        sent_count += 1
        
        # Log progress
        if sent_count % 50 == 0:
            print(f"📤 Sent {sent_count}/{total} posts...")
            
        # Add a small delay to simulate real-time stream
        time.sleep(0.05) 
        
    producer.flush()
    producer.close()
    
    print(f"\n✅ Finished streaming {sent_count} posts to Kafka.")
    print(f"   External 'kafka_consumer.py' should be processing these now.")
    print(f"{'='*60}\n")

# === SINGLE DAG ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='demo_streaming_pipeline',
        default_args=default_args,
        description='Demo Producer: Streams custom data to Kafka',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['demo', 'kafka', 'producer'],
    ) as dag:
        
        t1 = PythonOperator(task_id='load_data', python_callable=load_data)
        t2 = PythonOperator(task_id='produce_to_kafka', python_callable=produce_to_kafka)
        
        t1 >> t2

# === STANDALONE TEST ===
if __name__ == "__main__":
    print("🚀 Running Demo Producer (Standalone)...\n")
    
    # Simulate Airflow context with config
    class MockDagRun:
        conf = {"demo_folder": "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project/demo/demo_data_taxonomy_trained"}
    
    load_data(dag_run=MockDagRun())
    produce_to_kafka()
