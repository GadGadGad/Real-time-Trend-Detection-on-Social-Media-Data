"""
Demo Mode DAG: Quick demonstration using pre-computed results.

This DAG skips the expensive ML pipeline and directly loads results
that were pre-computed on Kaggle or another environment.

Usage:
    1. Export results from Kaggle as CSV
    2. Place in output/kaggle_results.csv
    3. Run this DAG for quick demo
"""

import os
import sys
import json
from datetime import datetime, timedelta

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.models import Variable
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

# === CONFIGURATION ===
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
PRECOMPUTED_RESULTS = os.path.join(PROJECT_ROOT, "output/kaggle_results.csv")
DEMO_STATE_PATH = os.path.join(PROJECT_ROOT, "demo-ready/demo_state.pkl")

sys.path.insert(0, PROJECT_ROOT)

default_args = {
    'owner': 'demo',
    'retries': 0,
}


def load_precomputed(**context):
    """Load pre-computed results from Kaggle export using demo_state format."""
    import pandas as pd
    import glob
    
    # Look for demo_state folders (created by save_demo_state)
    demo_folders = [
        os.path.join(PROJECT_ROOT, "demo"),
        os.path.join(PROJECT_ROOT, "demo_data"),
        os.path.join(PROJECT_ROOT, "output/demo_data"),
    ]
    
    # Also check for any demo_* folders
    demo_folders.extend(glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*")))
    demo_folders.extend(glob.glob(os.path.join(PROJECT_ROOT, "demo_*")))
    
    for folder in demo_folders:
        results_path = os.path.join(folder, 'results.parquet')
        if os.path.exists(results_path):
            # Load using demo_state format
            sys.path.insert(0, PROJECT_ROOT)
            from src.utils.demo_state import load_demo_state
            
            state = load_demo_state(folder)
            df = state.get('df_results')
            
            if df is not None and len(df) > 0:
                print(f"✅ Loaded {len(df)} results from {folder}")
                context['ti'].xcom_push(key='results_df', value=df.to_json())
                context['ti'].xcom_push(key='source_folder', value=folder)
                context['ti'].xcom_push(key='state', value={
                    'trends': state.get('trends', {}),
                    'cluster_mapping': state.get('cluster_mapping', {}),
                    'metadata': state.get('metadata', {}),
                })
                return {"count": len(df), "source": folder}
    
    # Fallback: try CSV files
    csv_patterns = [
        os.path.join(PROJECT_ROOT, "output/kaggle_results.csv"),
        os.path.join(PROJECT_ROOT, "output/results_*.csv"),
        os.path.join(PROJECT_ROOT, "results.csv"),
    ]
    
    for pattern in csv_patterns:
        files = glob.glob(pattern) if '*' in pattern else ([pattern] if os.path.exists(pattern) else [])
        if files:
            df = pd.read_csv(files[0])
            print(f"✅ Loaded {len(df)} results from {files[0]}")
            context['ti'].xcom_push(key='results_df', value=df.to_json())
            context['ti'].xcom_push(key='source_folder', value=os.path.dirname(files[0]))
            return {"count": len(df), "source": files[0]}
    
    raise FileNotFoundError("No pre-computed results found! Run save_demo_state() on Kaggle first.")


def generate_dashboard_data(**context):
    """Generate dashboard-ready statistics from results."""
    import pandas as pd
    
    ti = context['ti']
    df_json = ti.xcom_pull(key='results_df', task_ids='load_precomputed')
    df = pd.read_json(df_json)
    
    stats = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(df),
        "unique_topics": df['final_topic'].nunique() if 'final_topic' in df.columns else 0,
        "trending_count": len(df[df.get('topic_type', '') == 'Trending']) if 'topic_type' in df.columns else 0,
        "discovery_count": len(df[df.get('topic_type', '') == 'Discovery']) if 'topic_type' in df.columns else 0,
        "category_distribution": df['category'].value_counts().to_dict() if 'category' in df.columns else {},
        "sentiment_distribution": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
        "top_topics": df['final_topic'].value_counts().head(10).to_dict() if 'final_topic' in df.columns else {},
    }
    
    # Save stats
    stats_path = os.path.join(PROJECT_ROOT, "output/demo_stats.json")
    os.makedirs(os.path.dirname(stats_path), exist_ok=True)
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    print(f"📊 Dashboard stats saved to {stats_path}")
    print(f"   Topics: {stats['unique_topics']}, Trending: {stats['trending_count']}, Discovery: {stats['discovery_count']}")
    
    return stats


def print_summary(**context):
    """Print demo summary."""
    ti = context['ti']
    source = ti.xcom_pull(key='source_file', task_ids='load_precomputed')
    
    print("\n" + "="*50)
    print("🎉 DEMO MODE COMPLETE")
    print("="*50)
    print(f"Source: {source}")
    print(f"Stats: output/demo_stats.json")
    print("\n✅ Ready for dashboard demonstration!")
    print("="*50)


# === DAG DEFINITION ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='demo_mode_quick',
        default_args=default_args,
        description='Quick demo using pre-computed Kaggle results',
        schedule=None,  # Manual trigger only
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['demo', 'quick'],
    ) as dag:
        
        t1 = PythonOperator(
            task_id='load_precomputed',
            python_callable=load_precomputed,
        )
        
        t2 = PythonOperator(
            task_id='generate_dashboard_data',
            python_callable=generate_dashboard_data,
        )
        
        t3 = PythonOperator(
            task_id='print_summary',
            python_callable=print_summary,
        )
        
        t1 >> t2 >> t3


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
    print("🚀 Testing Demo Mode DAG...")
    load_precomputed(ti=ti)
    generate_dashboard_data(ti=ti)
    print_summary(ti=ti)
