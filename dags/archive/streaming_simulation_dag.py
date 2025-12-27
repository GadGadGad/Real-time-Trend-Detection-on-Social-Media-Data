"""
Airflow DAG: Pseudo-Streaming Demo

Simulates real-time event detection by processing posts one batch at a time.
Each batch simulates a "window" of incoming posts.

This creates a visual effect in Airflow UI showing progress over multiple tasks.
"""

import os
import sys
import time
from datetime import datetime, timedelta

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# === CONFIG ===
DEMO_FOLDER = os.path.join(PROJECT_ROOT, "demo/demo_data")
BATCH_SIZE = 100  # Posts per batch
NUM_BATCHES = 10  # Total batches to process
DELAY_BETWEEN_BATCHES = 2  # Seconds

default_args = {
    'owner': 'streaming_demo',
    'retries': 0,
}


def load_demo_state_task(**context):
    """Task 1: Load demo state from Kaggle export."""
    import glob
    
    # Find demo folder
    demo_folders = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
    if not demo_folders:
        raise FileNotFoundError("No demo folders found!")
    
    demo_folder = demo_folders[0]
    
    from src.utils.demo_state import load_demo_state
    state = load_demo_state(demo_folder)
    
    # Store metadata
    num_posts = len(state.get('df_results', []))
    context['ti'].xcom_push(key='demo_folder', value=demo_folder)
    context['ti'].xcom_push(key='num_posts', value=num_posts)
    
    print(f"✅ Loaded {num_posts} posts from {demo_folder}")
    return {"posts": num_posts, "folder": demo_folder}


def simulate_batch(**context):
    """Generic batch processing task."""
    batch_num = context['params']['batch_num']
    
    from src.utils.demo_state import load_demo_state, attach_new_post
    from sentence_transformers import SentenceTransformer
    import numpy as np
    
    ti = context['ti']
    demo_folder = ti.xcom_pull(key='demo_folder', task_ids='load_state')
    
    # Load state
    state = load_demo_state(demo_folder)
    df = state.get('df_results')
    centroids = state.get('centroids', {})
    trend_embeddings = state.get('trend_embeddings')
    cluster_mapping = state.get('cluster_mapping', {})
    metadata = state.get('metadata', {})
    
    # Get batch of posts
    start_idx = batch_num * BATCH_SIZE
    end_idx = start_idx + BATCH_SIZE
    batch_posts = df.iloc[start_idx:end_idx].to_dict('records')
    
    if not batch_posts:
        print(f"Batch {batch_num}: No more posts")
        return {"batch": batch_num, "processed": 0}
    
    # Load model
    model_name = metadata.get('model_name', 'dangvantuan/vietnamese-document-embedding')
    embedder = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Get trend keys
    trends = state.get('trends', {})
    trend_keys = list(trends.keys()) if isinstance(trends, dict) and trends else \
                 [f"Trend_{i}" for i in range(trend_embeddings.shape[0])]
    
    # Process batch
    results = []
    trending = 0
    discoveries = 0
    
    for post in batch_posts:
        result = attach_new_post(
            new_post=post,
            centroids=centroids,
            trend_embeddings=trend_embeddings,
            trend_keys=trend_keys,
            embedder=embedder,
            threshold=0.5,
            cluster_mapping=cluster_mapping,
        )
        results.append(result)
        
        if result.get('topic_type') == 'Trending':
            trending += 1
        else:
            discoveries += 1
    
    # Simulate processing delay
    time.sleep(DELAY_BETWEEN_BATCHES)
    
    print(f"✅ Batch {batch_num}: Processed {len(batch_posts)} posts | Trending: {trending} | Discoveries: {discoveries}")
    
    # Push batch stats
    ti.xcom_push(key=f'batch_{batch_num}_trending', value=trending)
    ti.xcom_push(key=f'batch_{batch_num}_discoveries', value=discoveries)
    
    return {"batch": batch_num, "processed": len(batch_posts), "trending": trending, "discoveries": discoveries}


def summarize_streaming(**context):
    """Final task: Summarize all batches."""
    import json
    
    ti = context['ti']
    
    total_trending = 0
    total_discoveries = 0
    
    for i in range(NUM_BATCHES):
        t = ti.xcom_pull(key=f'batch_{i}_trending', task_ids=f'batch_{i}') or 0
        d = ti.xcom_pull(key=f'batch_{i}_discoveries', task_ids=f'batch_{i}') or 0
        total_trending += t
        total_discoveries += d
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_batches": NUM_BATCHES,
        "batch_size": BATCH_SIZE,
        "total_trending": total_trending,
        "total_discoveries": total_discoveries,
        "total_processed": total_trending + total_discoveries,
    }
    
    # Save summary
    summary_path = os.path.join(PROJECT_ROOT, "output/streaming_summary.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*50)
    print("🎉 STREAMING SIMULATION COMPLETE")
    print("="*50)
    print(f"Batches: {NUM_BATCHES}")
    print(f"Total Trending: {total_trending}")
    print(f"Total Discoveries: {total_discoveries}")
    print(f"Summary: {summary_path}")
    print("="*50)
    
    return summary


# === DAG DEFINITION ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='streaming_simulation',
        default_args=default_args,
        description='Simulate real-time streaming with multiple batches',
        schedule=None,  # Manual trigger only
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['streaming', 'demo', 'simulation'],
    ) as dag:
        
        # Task 1: Load state
        load_task = PythonOperator(
            task_id='load_state',
            python_callable=load_demo_state_task,
        )
        
        # Create batch tasks dynamically
        batch_tasks = []
        for i in range(NUM_BATCHES):
            batch_task = PythonOperator(
                task_id=f'batch_{i}',
                python_callable=simulate_batch,
                params={'batch_num': i},
            )
            batch_tasks.append(batch_task)
        
        # Final summary task
        summary_task = PythonOperator(
            task_id='summarize',
            python_callable=summarize_streaming,
        )
        
        # Chain: load -> batch_0 -> batch_1 -> ... -> summary
        load_task >> batch_tasks[0]
        for i in range(len(batch_tasks) - 1):
            batch_tasks[i] >> batch_tasks[i + 1]
        batch_tasks[-1] >> summary_task


# === STANDALONE TEST ===
if __name__ == "__main__":
    class MockTI:
        def __init__(self):
            self.data = {}
        def xcom_push(self, key, value):
            self.data[key] = value
        def xcom_pull(self, key, task_ids=None):
            return self.data.get(key)
    
    ti = MockTI()
    print("🚀 Testing Streaming Simulation DAG...\n")
    
    load_demo_state_task(ti=ti)
    
    for i in range(3):  # Test 3 batches
        simulate_batch(ti=ti, params={'batch_num': i})
    
    summarize_streaming(ti=ti)
