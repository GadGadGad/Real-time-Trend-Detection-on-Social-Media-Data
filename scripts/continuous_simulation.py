import time
import random
import sys
import os
from datetime import datetime

# Add project root
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

from streaming.kafka_producer import create_producer, load_kaggle_data

def run_continuous_simulation():
    producer = create_producer()
    if not producer:
        print("❌ Could not connect to Kafka")
        return

    print("📂 Loading data for simulation...")
    posts = load_kaggle_data()
    if posts is None or (hasattr(posts, 'empty') and posts.empty) or (isinstance(posts, list) and not posts):
        print("❌ No data found")
        return

    if hasattr(posts, 'to_dict'):
        records = posts.to_dict('records')
    else:
        records = posts

    print(f"🚀 Continuous Simulation Started (1 post every 3 seconds)...")
    print(f"   Topic: batch-stream")
    print(f"   Press Ctrl+C to stop")

    try:
        while True:
            row = random.choice(records)
            
            # Format message
            message = {
                "content": str(row.get('content', row.get('title', '')))[:500],
                "source": str(row.get('source', 'Unknown')),
                "time": datetime.now().isoformat(), # Use NOW for flow
                "final_topic": str(row.get('final_topic', 'Unknown'))
            }
            
            producer.send('batch-stream', value=message)
            print(f"📤 Sent: {message['content'][:50]}... | {message['final_topic']}")
            
            time.sleep(3)
            
    except KeyboardInterrupt:
        print("\n🛑 Simulation stopped")
    finally:
        producer.close()

if __name__ == "__main__":
    run_continuous_simulation()
