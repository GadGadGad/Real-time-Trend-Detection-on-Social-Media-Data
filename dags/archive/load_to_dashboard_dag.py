"""
Airflow DAG: Mock Streaming → Dashboard

Loads pre-computed Kaggle results and writes to PostgreSQL database
for the demo-ready/dashboard.py to display.

Flow:
[load_kaggle_data] → [write_to_postgres] → [done]

Dashboard will auto-refresh and show the data!
"""

import os
import sys
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

# Database config (same as demo-ready/dashboard.py)
DB_URL = "postgresql://user:password@localhost:5432/trend_db"

default_args = {'owner': 'demo', 'retries': 0}


def find_demo_folder():
    import glob
    candidates = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
    return candidates[0] if candidates else os.path.join(PROJECT_ROOT, "demo/demo_data")


def load_and_write_to_db(**context):
    """Load Kaggle results and write to PostgreSQL for Dashboard."""
    import pandas as pd
    from sqlalchemy import create_engine, text
    from src.utils.demo_state import load_demo_state
    
    demo_folder = find_demo_folder()
    print(f"📂 Loading from: {demo_folder}")
    
    # Load demo state
    state = load_demo_state(demo_folder)
    df = state.get('df_results')
    
    if df is None or len(df) == 0:
        raise ValueError("No results found!")
    
    print(f"✅ Loaded {len(df)} posts")
    
    # Convert to dashboard format
    # Group by final_topic (trend name)
    trends = df.groupby('final_topic').agg({
        'content': 'count',
        'topic_type': 'first', 
        'score': 'mean',
        'category': 'first',
    }).reset_index()
    
    trends.columns = ['trend_name', 'post_count', 'topic_type', 'trend_score', 'category']
    cluster_col = 'final_topic'  # For post lookup later
    
    # Create records for DB
    records = []
    for _, trend in trends.iterrows():
        # Get representative posts for this trend
        trend_posts = df[df['final_topic'] == trend['trend_name']].head(10)
        posts_json = json.dumps([{
            'content': str(row.get('content', ''))[:500],
            'source': row.get('source', 'Unknown'),
            'time': str(row.get('time', datetime.now())),
            'similarity': float(row.get('score', 0.5)),
        } for _, row in trend_posts.iterrows()])
        
        record = {
            'trend_name': trend['trend_name'][:200] if trend['trend_name'] else 'Unknown',
            'trend_score': float(trend['trend_score']) if trend['trend_score'] else 0.0,
            'post_count': int(trend['post_count']),
            'topic_type': trend['topic_type'] or 'Discovery',
            'category': trend['category'] or 'T7',
            'representative_posts': posts_json,
            'score_n': 0.5,  # News score placeholder
            'score_f': 0.5,  # Social score placeholder
            'sentiment': 'Neutral',
            'summary': 'Loaded from Kaggle pre-computed results',
            'created_at': datetime.now(),
        }
        records.append(record)
    
    # Write to DB
    engine = create_engine(DB_URL)
    
    # Clear old data
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    
    # Insert new data
    records_df = pd.DataFrame(records)
    records_df.to_sql('detected_trends', engine, if_exists='append', index=False)
    
    print(f"✅ Written {len(records)} trends to database")
    print(f"📊 Open Dashboard: streamlit run demo-ready/dashboard.py")
    
    return {"trends": len(records), "posts": len(df)}


def notify_complete(**context):
    """Print completion message."""
    print("\n" + "="*60)
    print("🎉 DATA LOADED TO DASHBOARD!")
    print("="*60)
    print("👉 Open Dashboard: streamlit run demo-ready/dashboard.py")
    print("👉 Or visit: http://localhost:8501")
    print("="*60)


# === DAG ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='load_to_dashboard',
        default_args=default_args,
        description='Load Kaggle results to Dashboard',
        schedule=None,
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['dashboard', 'demo'],
    ) as dag:
        
        t1 = PythonOperator(
            task_id='load_and_write',
            python_callable=load_and_write_to_db,
        )
        
        t2 = PythonOperator(
            task_id='complete',
            python_callable=notify_complete,
        )
        
        t1 >> t2


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("🚀 Loading Kaggle data to Dashboard DB...\n")
    load_and_write_to_db()
    notify_complete()
