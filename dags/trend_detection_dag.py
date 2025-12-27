"""
Apache Airflow DAG for Trend Detection Pipeline

This DAG runs the complete trend detection workflow:
1. Load posts and trends from CSV/JSON files
2. Run the hybrid matching pipeline (find_matches_hybrid)
3. Save results to output folder

Schedule: Daily at midnight (adjustable)

Setup:
    1. Install Airflow: pip install apache-airflow
    2. Copy this file to ~/airflow/dags/
    3. Set AIRFLOW_HOME and PROJECT_ROOT in the config below
    4. Start Airflow: airflow webserver & airflow scheduler
"""

import os
import sys
import json
import glob
from datetime import datetime, timedelta

# Airflow imports (optional for standalone testing)
try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    from airflow.models import Variable
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False
    print("⚠️ Airflow not installed. Running in standalone mode.")
    # Mock classes for standalone testing
    class DAG:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
    class PythonOperator:
        def __init__(self, *args, **kwargs):
            pass
    class BashOperator:
        def __init__(self, *args, **kwargs):
            pass

# === CONFIGURATION ===
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Pipeline parameters (matching notebook config)
PIPELINE_CONFIG = {
    "model_name": "dangvantuan/vietnamese-document-embedding",
    "threshold": 0.65,
    "coherence_threshold": 0.6,
    "semantic_floor": 0.55,
    "min_cluster_size": 3,
    "use_llm": True,
    "llm_provider": "gemini",
    "rerank": False,
    "use_cache": True,
    "taxonomy_method": "trained",
    "sentiment_method": "trained",
    "taxonomy_model_path": "models/taxonomy-classifier-vietnamese-v1",
    "sentiment_model_path": "models/sentiment-classifier-vietnamese-v1",
}

# Add project to path
sys.path.insert(0, PROJECT_ROOT)

# === DAG DEFAULT ARGS ===
default_args = {
    'owner': 'trend_detection',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

# === TASK FUNCTIONS ===

def load_data(**context):
    """
    Task 1: Load posts and trends from CSV/JSON files.
    Pushes file paths to XCom for next task.
    """
    import glob
    
    # Find post files (JSON)
    post_patterns = [
        os.path.join(DATA_DIR, "posts/*.json"),
        os.path.join(DATA_DIR, "crawlers/**/*.json"),
        os.path.join(PROJECT_ROOT, "crawlers/**/*.json"),
    ]
    post_files = []
    for pattern in post_patterns:
        post_files.extend(glob.glob(pattern, recursive=True))
    
    # Find trend files (CSV or JSON)
    trend_patterns = [
        os.path.join(DATA_DIR, "trends/*.csv"),
        os.path.join(DATA_DIR, "trends/*.json"),
        os.path.join(PROJECT_ROOT, "crawlers/google_trends/**/*.csv"),
        os.path.join(PROJECT_ROOT, "crawlers/new_data/trendings/*.csv"),  # Actual location
        os.path.join(PROJECT_ROOT, "crawlers/**/trending*.csv"),
    ]
    trend_files = []
    for pattern in trend_patterns:
        trend_files.extend(glob.glob(pattern, recursive=True))
    
    print(f"Found {len(post_files)} post files")
    print(f"Found {len(trend_files)} trend files")
    
    if not post_files:
        raise ValueError("No post files found!")
    if not trend_files:
        raise ValueError("No trend files found!")
    
    # Push to XCom
    context['ti'].xcom_push(key='post_files', value=post_files)
    context['ti'].xcom_push(key='trend_files', value=trend_files)
    
    return {"posts": len(post_files), "trends": len(trend_files)}


def run_pipeline(**context):
    """
    Task 2: Run the main trend detection pipeline.
    Uses find_matches_hybrid from main_pipeline.py
    """
    import pandas as pd
    import os
    import sys
    
    # Ensure project is in path
    sys.path.insert(0, PROJECT_ROOT)
    
    from src.pipeline.main_pipeline import find_matches_hybrid, load_json, load_trends
    
    # Get file paths from previous task
    ti = context['ti']
    post_files = ti.xcom_pull(key='post_files', task_ids='load_data')
    trend_files = ti.xcom_pull(key='trend_files', task_ids='load_data')
    
    # Load posts
    print("Loading posts...")
    posts = []
    for f in post_files:
        try:
            loaded = load_json(f)
            posts.extend(loaded)
        except Exception as e:
            print(f"Warning: Failed to load {f}: {e}")
    
    print(f"Loaded {len(posts)} posts")
    
    # Load trends
    print("Loading trends...")
    trends = load_trends(trend_files)
    print(f"Loaded {len(trends)} trends")
    
    # Get GEMINI API key from environment or Airflow Variables
    gemini_api_key = os.environ.get("GEMINI_API_KEY", "")
    if AIRFLOW_AVAILABLE and not gemini_api_key:
        try:
            gemini_api_key = Variable.get("GEMINI_API_KEY")
        except:
            pass
    
    # Run pipeline
    print("Running find_matches_hybrid...")
    matches, components = find_matches_hybrid(
        posts=posts,
        trends=trends,
        model_name=PIPELINE_CONFIG["model_name"],
        threshold=PIPELINE_CONFIG["threshold"],
        coherence_threshold=PIPELINE_CONFIG["coherence_threshold"],
        semantic_floor=PIPELINE_CONFIG["semantic_floor"],
        min_cluster_size=PIPELINE_CONFIG["min_cluster_size"],
        use_llm=PIPELINE_CONFIG["use_llm"],
        llm_provider=PIPELINE_CONFIG["llm_provider"],
        gemini_api_key=gemini_api_key,
        rerank=PIPELINE_CONFIG["rerank"],
        use_cache=PIPELINE_CONFIG["use_cache"],
        taxonomy_method=PIPELINE_CONFIG["taxonomy_method"],
        sentiment_method=PIPELINE_CONFIG["sentiment_method"],
        taxonomy_model_path=PIPELINE_CONFIG["taxonomy_model_path"],
        sentiment_model_path=PIPELINE_CONFIG["sentiment_model_path"],
        return_components=True,
    )
    
    print(f"Pipeline completed. Generated {len(matches)} results.")
    
    # Save results
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(OUTPUT_DIR, f"results_{timestamp}.csv")
    
    df_results = pd.DataFrame(matches)
    df_results.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
    # Push output path to XCom
    ti.xcom_push(key='output_path', value=output_path)
    ti.xcom_push(key='match_count', value=len(matches))
    
    return {"matches": len(matches), "output": output_path}


def generate_report(**context):
    """
    Task 3: Generate summary report of detected trends.
    """
    import pandas as pd
    
    ti = context['ti']
    output_path = ti.xcom_pull(key='output_path', task_ids='run_pipeline')
    
    if not output_path or not os.path.exists(output_path):
        print("No output file found, skipping report generation")
        return
    
    df = pd.read_csv(output_path)
    
    # Generate summary
    report = {
        "timestamp": datetime.now().isoformat(),
        "total_results": len(df),
        "trending_topics": df[df['topic_type'] == 'Trending']['final_topic'].nunique() if 'topic_type' in df.columns else 0,
        "discoveries": df[df['topic_type'] == 'Discovery']['final_topic'].nunique() if 'topic_type' in df.columns else 0,
        "top_categories": df['category'].value_counts().head(5).to_dict() if 'category' in df.columns else {},
        "sentiment_distribution": df['sentiment'].value_counts().to_dict() if 'sentiment' in df.columns else {},
    }
    
    # Save report
    report_path = output_path.replace('.csv', '_report.json')
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    print(f"Report saved to {report_path}")
    print(f"Summary: {report}")
    
    return report


# === DAG DEFINITION (only when Airflow is available) ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='trend_detection_pipeline',
        default_args=default_args,
        description='Daily trend detection from social media posts',
        schedule_interval='0 0 * * *',  # Run at midnight daily
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['trend-detection', 'nlp', 'ml'],
    ) as dag:
        
        # Task 1: Load data
        task_load = PythonOperator(
            task_id='load_data',
            python_callable=load_data,
            provide_context=True,
        )
        
        # Task 2: Run ML pipeline
        task_pipeline = PythonOperator(
            task_id='run_pipeline',
            python_callable=run_pipeline,
            provide_context=True,
            # Allocate more resources for ML task
            executor_config={
                "KubernetesExecutor": {
                    "request_memory": "8Gi",
                    "request_cpu": "2",
                }
            }
        )
        
        # Task 3: Generate report
        task_report = PythonOperator(
            task_id='generate_report',
            python_callable=generate_report,
            provide_context=True,
        )
        
        # Task 4: Cleanup old cache (optional)
        task_cleanup = BashOperator(
            task_id='cleanup_old_cache',
            bash_command=f'find {PROJECT_ROOT}/embeddings_cache -mtime +7 -delete 2>/dev/null || true',
        )
        
        # Define task order
        task_load >> task_pipeline >> task_report >> task_cleanup


# === MANUAL TRIGGER (for testing) ===
if __name__ == "__main__":
    # Test run without Airflow
    print("Testing DAG locally...")
    
    # Simulate XCom with dict
    class MockContext:
        def __init__(self):
            self.data = {}
        def xcom_push(self, key, value):
            self.data[key] = value
        def xcom_pull(self, key, task_ids=None):
            return self.data.get(key)
    
    class MockTI:
        def __init__(self):
            self.ctx = MockContext()
        def xcom_push(self, key, value):
            self.ctx.xcom_push(key, value)
        def xcom_pull(self, key, task_ids=None):
            return self.ctx.xcom_pull(key)
    
    ti = MockTI()
    
    # Run tasks
    print("\n=== Task 1: Load Data ===")
    load_data(ti=ti)
    
    print("\n=== Task 2: Run Pipeline ===")  
    run_pipeline(ti=ti)
    
    print("\n=== Task 3: Generate Report ===")
    generate_report(ti=ti)
    
    print("\n✅ Local test completed!")
