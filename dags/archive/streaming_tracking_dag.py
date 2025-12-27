"""
Airflow DAG: Pipeline 2 - Streaming Tracking (Online)

Purpose: Track new posts against existing clusters in real-time
Schedule: Continuous / triggered by Pipeline 1
Input: Uses cluster centroids from Pipeline 1 (batch_clustering_dag)

This runs AFTER Pipeline 1 to track trends.
"""

import os
import sys
import time
import json
import threading
import queue
from datetime import datetime, timedelta

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.sensors.external_task import ExternalTaskSensor
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

DB_URL = "postgresql://user:password@localhost:5432/trend_db"

default_args = {
    'owner': 'streaming_pipeline',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


def find_demo_folder():
    """Find the latest demo folder created by Pipeline 1."""
    import glob
    candidates = [
        os.path.join(PROJECT_ROOT, "demo/demo_data_batch"),  # From Pipeline 1
        os.path.join(PROJECT_ROOT, "demo/demo_data_refinedtrends"),  # Pre-computed
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    # Fallback to any demo folder
    others = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
    return others[0] if others else os.path.join(PROJECT_ROOT, "demo/demo_data")


def load_cluster_state(**context):
    """Task 1: Load clusters from Pipeline 1 output."""
    from src.utils.demo_state import load_demo_state
    
    demo_folder = find_demo_folder()
    print(f"📂 Loading clusters from: {demo_folder}")
    
    state = load_demo_state(demo_folder)
    df = state.get('df_results')
    
    if df is None or len(df) == 0:
        raise ValueError("No cluster data found! Run Pipeline 1 first.")
    
    # Get unique clusters
    if 'cluster' in df.columns:
        n_clusters = df['cluster'].nunique()
    elif 'final_topic' in df.columns:
        n_clusters = df['final_topic'].nunique()
    else:
        n_clusters = 0
    
    # Save to temp file
    temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
    os.makedirs(os.path.dirname(temp_file), exist_ok=True)
    df.to_parquet(temp_file)
    
    print(f"✅ Loaded {len(df)} posts, {n_clusters} clusters")
    context['ti'].xcom_push(key='demo_folder', value=demo_folder)
    context['ti'].xcom_push(key='n_clusters', value=n_clusters)
    
    return {"posts": len(df), "clusters": n_clusters, "folder": demo_folder}


def streaming_simulation(**context):
    """Task 2: Kafka Producer → Spark Consumer (parallel simulation)."""
    import pandas as pd
    
    temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
    df = pd.read_parquet(temp_file)
    total = len(df)
    batch_size = 500
    num_batches = (total + batch_size - 1) // batch_size
    
    # Shared Kafka queue
    kafka_queue = queue.Queue()
    processed_trends = []
    lock = threading.Lock()
    
    print(f"\n{'='*60}")
    print("🚀 STREAMING TRACKING (Pipeline 2)")
    print(f"{'='*60}")
    print(f"📊 Input: {total} posts from {num_batches} batches")
    print(f"{'='*60}\n")
    
    def producer_thread():
        """Kafka Producer: publishes batches."""
        for i in range(0, total, batch_size):
            batch = df.iloc[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            print(f"📤 [PRODUCER] Batch {batch_num}/{num_batches} → topic 'social_posts' ({len(batch)} posts)")
            kafka_queue.put((batch_num, batch))
            time.sleep(3)  # 3s between batches
        
        kafka_queue.put((None, None))  # End signal
        print(f"\n📤 [PRODUCER] ✅ Finished publishing {total} posts")
    
    def consumer_thread():
        """Spark Consumer: assigns posts to nearest cluster."""
        batch_trends = {}
        
        while True:
            try:
                batch_num, batch = kafka_queue.get(timeout=15)
                
                if batch_num is None:
                    break
                
                time.sleep(1)  # Simulate processing
                
                # Assign each post to its cluster (from pre-computed data)
                for _, row in batch.iterrows():
                    topic = row.get('final_topic', 'Unknown')
                    if topic not in batch_trends:
                        batch_trends[topic] = {
                            'count': 0,
                            'topic_type': row.get('topic_type', 'Discovery'),
                            'score_sum': 0,
                            'category': row.get('category', 'T7'),
                        }
                    batch_trends[topic]['count'] += 1
                    batch_trends[topic]['score_sum'] += row.get('score', 0.5)
                
                trending = sum(1 for t in batch_trends.values() if t['topic_type'] == 'Trending')
                print(f"⚡ [CONSUMER] Batch {batch_num}: {len(batch_trends)} trends ({trending} trending)")
                
            except queue.Empty:
                break
        
        # Save aggregated trends
        with lock:
            for topic, data in batch_trends.items():
                processed_trends.append({
                    'trend_name': topic,
                    'post_count': data['count'],
                    'topic_type': data['topic_type'],
                    'trend_score': data['score_sum'] / data['count'] if data['count'] > 0 else 0,
                    'category': data['category'],
                })
        
        print(f"\n⚡ [CONSUMER] ✅ Finished → {len(batch_trends)} unique trends")
    
    # Start threads in parallel
    producer = threading.Thread(target=producer_thread, name="KafkaProducer")
    consumer = threading.Thread(target=consumer_thread, name="SparkConsumer")
    
    producer.start()
    time.sleep(0.5)  # Small delay
    consumer.start()
    
    producer.join()
    consumer.join()
    
    # Save trends
    trends_df = pd.DataFrame(processed_trends)
    trends_file = os.path.join(PROJECT_ROOT, "output/.trends_temp.parquet")
    trends_df.to_parquet(trends_file)
    
    print(f"\n{'='*60}")
    print(f"✅ Streaming Complete: {total} posts → {len(processed_trends)} trends")
    print(f"{'='*60}\n")
    
    context['ti'].xcom_push(key='n_trends', value=len(processed_trends))
    return {"posts": total, "trends": len(processed_trends)}


def write_to_dashboard(**context):
    """Task 3: Write trends to PostgreSQL for Streamlit."""
    import pandas as pd
    from sqlalchemy import create_engine, text
    
    temp_file = os.path.join(PROJECT_ROOT, "output/.streaming_temp.parquet")
    trends_file = os.path.join(PROJECT_ROOT, "output/.trends_temp.parquet")
    
    df = pd.read_parquet(temp_file)
    trends = pd.read_parquet(trends_file)
    
    print(f"📊 WRITING TO DASHBOARD")
    print(f"   Trends: {len(trends)}")
    
    # Create records
    records = []
    for _, trend in trends.iterrows():
        trend_posts = df[df['final_topic'] == trend['trend_name']].head(10)
        posts_json = json.dumps([{
            'content': str(row.get('content', ''))[:500],
            'source': row.get('source', 'Social Media'),
            'time': str(row.get('time', datetime.now())),
            'similarity': float(row.get('score', 0.5)),
        } for _, row in trend_posts.iterrows()])
        
        records.append({
            'trend_name': str(trend['trend_name'])[:200] if trend['trend_name'] else 'Unknown',
            'trend_score': float(trend['trend_score']) if trend['trend_score'] else 0.0,
            'post_count': int(trend['post_count']),
            'topic_type': trend['topic_type'] or 'Discovery',
            'category': trend['category'] or 'T7',
            'representative_posts': posts_json,
            'score_n': 0.5,
            'score_f': 0.5,
            'sentiment': 'Neutral',
            'summary': f'Tracked from streaming pipeline at {datetime.now().strftime("%H:%M:%S")}',
            'created_at': datetime.now(),
        })
    
    # Write to DB
    try:
        engine = create_engine(DB_URL)
        
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE detected_trends"))
        
        records_df = pd.DataFrame(records)
        records_df.to_sql('detected_trends', engine, if_exists='append', index=False)
        
        print(f"   ✅ Written {len(records)} trends to PostgreSQL")
    except Exception as e:
        print(f"   ⚠️ DB write failed: {e}")
        print(f"   Saving to JSON instead...")
        
        # Fallback: save to JSON
        output_file = os.path.join(PROJECT_ROOT, "output/streaming_trends.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(records, f, ensure_ascii=False, indent=2, default=str)
        print(f"   ✅ Saved to {output_file}")
    
    print(f"\n{'='*60}")
    print("🎉 STREAMING PIPELINE COMPLETE!")
    print(f"{'='*60}")
    print("📊 Dashboard: http://localhost:8502")
    print(f"{'='*60}\n")
    
    return {"written": len(records)}


# === DAG DEFINITION ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='streaming_tracking_pipeline',
        default_args=default_args,
        description='Pipeline 2: Streaming trend tracking (uses clusters from Pipeline 1)',
        schedule=None,  # Triggered manually or by Pipeline 1
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['streaming', 'tracking', 'online', 'pipeline-2', 'kafka', 'spark'],
    ) as dag:
        
        # Optional: Wait for Pipeline 1 to complete
        # wait_for_batch = ExternalTaskSensor(
        #     task_id='wait_for_batch_pipeline',
        #     external_dag_id='batch_clustering_pipeline',
        #     external_task_id='save_demo_state',
        #     timeout=600,
        #     allowed_states=['success'],
        # )
        
        t1 = PythonOperator(
            task_id='load_cluster_state',
            python_callable=load_cluster_state,
        )
        
        t2 = PythonOperator(
            task_id='streaming_simulation',
            python_callable=streaming_simulation,
        )
        
        t3 = PythonOperator(
            task_id='write_to_dashboard',
            python_callable=write_to_dashboard,
        )
        
        # wait_for_batch >> t1 >> t2 >> t3
        t1 >> t2 >> t3


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("🚀 Running Streaming Tracking Pipeline...\n")
    print("=" * 60)
    print("⚠️ Requires clusters from Pipeline 1 (batch_clustering_dag)")
    print("=" * 60 + "\n")
    
    class MockTI:
        def __init__(self): self.data = {}
        def xcom_push(self, key, value): self.data[key] = value
        def xcom_pull(self, key, task_ids=None): return self.data.get(key)
    
    ti = MockTI()
    
    print("📥 TASK 1: Load Cluster State")
    load_cluster_state(ti=ti)
    
    print("\n🚀 TASK 2: Streaming Simulation")
    streaming_simulation(ti=ti)
    
    print("\n📊 TASK 3: Write to Dashboard")
    write_to_dashboard(ti=ti)
    
    print("\n" + "=" * 60)
    print("✅ Streaming Tracking Pipeline Complete!")
    print("=" * 60)
