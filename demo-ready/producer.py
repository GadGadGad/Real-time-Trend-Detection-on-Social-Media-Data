import json
import csv
import time
import os
import sys
import glob
import random
import pickle
import numpy as np
from datetime import datetime
from kafka import KafkaProducer

# --- CONFIG ---
KAFKA_TOPIC = 'posts-stream'
BOOTSTRAP_SERVERS = ['localhost:29092']
DATA_DIR = 'data'
CACHE_FILE = 'embeddings_cache.pkl'

def create_producer():
    print(f"🔄 Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    try:
        producer = KafkaProducer(
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_serializer=lambda x: json.dumps(x).encode('utf-8')
        )
        print(f"✅ Connected to Kafka!")
        return producer
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return None

def clean_text(text):
    import re
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_from_cache():
    """Load pre-computed embeddings from cache"""
    if not os.path.exists(CACHE_FILE):
        print(f"⚠️ Cache not found: {CACHE_FILE}")
        print("👉 Run 'python precompute_embeddings.py' first!")
        return None, None
    
    print(f"📂 Loading from cache: {CACHE_FILE}...")
    with open(CACHE_FILE, 'rb') as f:
        cache = pickle.load(f)
    
    if isinstance(cache, dict):
        posts = cache['posts']
        embeddings = cache['embeddings']
        print(f"✅ Loaded {len(posts)} posts with embeddings (Created: {cache.get('created_at', 'Unknown')})")
    elif isinstance(cache, tuple):
        # Handle format: (posts, embeddings)
        posts, embeddings = cache
        print(f"✅ Loaded {len(posts)} posts from import.")
    else:
        print("❌ Unknown cache format")
        return None, None
    return posts, embeddings

def load_data_raw(data_dir):
    """Load posts without embeddings (fallback)"""
    posts = []
    print(f"📂 Scanning data in: {data_dir}...")
    
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    content = f"{row.get('title', '')} {row.get('content', '')}"
                    if len(content) < 20: continue
                    
                    fname_lower = os.path.basename(f).lower()
                    if 'fb' in fname_lower or 'facebook' in fname_lower:
                        source = 'Facebook'
                    elif 'news' in fname_lower:
                        source = "News"
                    else:
                        source = "News"

                    posts.append({
                        "source": source,
                        "content": clean_text(content),
                        "time": row.get('published_at', datetime.now().isoformat())
                    })
        except Exception: pass
        
    print(f"📊 Loaded {len(posts)} posts (no embeddings).")
    random.shuffle(posts)
    return posts

def run_replay():
    producer = create_producer()
    if not producer: return

    # Try to load from cache first
    posts, embeddings = load_from_cache()
    
    use_cache = posts is not None
    if not use_cache:
        posts = load_data_raw(DATA_DIR)
        embeddings = None
    
    if not posts:
        print("⚠️ No data found!")
        return

    # Shuffle for demo randomness
    indices = list(range(len(posts)))
    random.shuffle(indices)

    print(f"🚀 Starting Stream Replay ({len(posts)} items, cache={'YES' if use_cache else 'NO'})...")
    try:
        for i, idx in enumerate(indices):
            post = posts[idx].copy()
            
            # Include embedding if cached
            if use_cache and embeddings is not None:
                post['embedding'] = embeddings[idx].tolist()
            
            producer.send(KAFKA_TOPIC, value=post)
            
            if i % 20 == 0:
                sys.stdout.write(f"\r📤 Sent {i}/{len(posts)}...")
                sys.stdout.flush()
            
            # Faster replay for demo (0.01s - 0.05s)
            time.sleep(random.uniform(0.01, 0.05))
            
        producer.flush()
        print("\n✅ Replay Complete.")
    except KeyboardInterrupt:
        print("\n🛑 Stopped.")
    finally:
        producer.close()

if __name__ == "__main__":
    run_replay()
