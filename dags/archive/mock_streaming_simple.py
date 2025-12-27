"""
Airflow DAG: Mock Streaming Demo (Simple)

Simple 3-task flow showing Kafka → Spark → Dashboard.
Uses pre-computed Kaggle results.
"""

import os
import sys
import time
import json
from datetime import datetime

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {'owner': 'demo', 'retries': 0}


def find_demo_folder():
    import glob
    candidates = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
    return candidates[0] if candidates else os.path.join(PROJECT_ROOT, "demo/demo_data")


def kafka_producer(**context):
    """Simulate Kafka Producer - load and publish all posts."""
    from src.utils.demo_state import load_demo_state
    
    demo_folder = find_demo_folder()
    print(f"📤 KAFKA PRODUCER")
    print(f"   Loading from: {demo_folder}")
    
    state = load_demo_state(demo_folder)
    df = state.get('df_results')
    
    # Save posts for consumer
    posts_file = os.path.join(PROJECT_ROOT, "output/.kafka_posts.json")
    os.makedirs(os.path.dirname(posts_file), exist_ok=True)
    
    posts = df.to_dict('records')
    with open(posts_file, 'w') as f:
        json.dump(posts[:500], f)  # Limit to 500 for demo speed
    
    print(f"   ✅ Published {min(500, len(posts))} posts to Kafka topic")
    time.sleep(2)  # Simulate publishing time
    return {"published": min(500, len(posts))}


def spark_consumer(**context):
    """Simulate Spark Consumer - process posts."""
    print(f"⚡ SPARK CONSUMER")
    
    posts_file = os.path.join(PROJECT_ROOT, "output/.kafka_posts.json")
    with open(posts_file, 'r') as f:
        posts = json.load(f)
    
    print(f"   Processing {len(posts)} posts...")
    
    # Aggregate results
    trending = sum(1 for p in posts if p.get('topic_type') == 'Trending')
    discoveries = len(posts) - trending
    
    categories = {}
    topics = set()
    for p in posts:
        cat = p.get('category', 'T7')
        categories[cat] = categories.get(cat, 0) + 1
        topics.add(p.get('final_topic', 'Unknown'))
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "total": len(posts),
        "trending": trending,
        "discoveries": discoveries,
        "unique_topics": len(topics),
        "categories": categories,
    }
    
    results_file = os.path.join(PROJECT_ROOT, "output/streaming_results.json")
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"   ✅ Trending: {trending}, Discoveries: {discoveries}")
    print(f"   ✅ Unique topics: {len(topics)}")
    time.sleep(2)  # Simulate processing
    return results


def dashboard_output(**context):
    """Output to Dashboard."""
    print(f"📊 DASHBOARD UPDATE")
    
    results_file = os.path.join(PROJECT_ROOT, "output/streaming_results.json")
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    print("\n" + "="*50)
    print("🎉 STREAMING DEMO COMPLETE")
    print("="*50)
    print(f"Total Events:    {results['total']}")
    print(f"Trending:        {results['trending']}")
    print(f"Discoveries:     {results['discoveries']}")
    print(f"Unique Topics:   {results['unique_topics']}")
    print("-"*50)
    print("Categories:")
    for cat, count in sorted(results['categories'].items()):
        print(f"  {cat}: {count}")
    print("="*50)
    print(f"\nResults saved to: {results_file}")
    return results


# === DAG ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='mock_streaming_simple',
        default_args=default_args,
        description='Simple Kafka → Spark → Dashboard demo',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['streaming', 'demo'],
    ) as dag:
        
        t1 = PythonOperator(task_id='kafka_producer', python_callable=kafka_producer)
        t2 = PythonOperator(task_id='spark_consumer', python_callable=spark_consumer)
        t3 = PythonOperator(task_id='dashboard', python_callable=dashboard_output)
        
        t1 >> t2 >> t3


if __name__ == "__main__":
    print("🚀 Testing Simple Mock Streaming DAG...\n")
    kafka_producer()
    spark_consumer()
    dashboard_output()
