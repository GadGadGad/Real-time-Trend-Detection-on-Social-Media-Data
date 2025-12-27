from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.utils.trigger_rule import TriggerRule
from datetime import datetime, timedelta
import sys
import os

# Configuration
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
PYTHON_BIN = "python"  # Ensure this points to the correct env if needed

sys.path.insert(0, PROJECT_ROOT)

default_args = {
    'owner': 'trend_detection',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 0,
}

def decide_mode(**context):
    """
    Decide which task(s) to run based on 'mode' in dag_run.conf.
    Options: 'demo', 'live', 'hybrid' (default: 'demo')
    """
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', 'demo').lower()
    
    print(f"🔀 Branching Decision: Mode = {mode}")
    
    if mode == 'live':
        return ['trigger_live_ingest']
    elif mode == 'hybrid':
        return ['trigger_demo_ingest', 'trigger_live_ingest']
    else:
        # Default to demo
        return ['trigger_demo_ingest']

def run_demo_ingest(**context):
    """Refers to the logic in demo_streaming_pipeline.py"""
    # Import locally to avoid top-level import errors if missing deps
    from dags.demo_streaming_pipeline import load_data, produce_to_kafka
    
    print("🎬 Starting Demo Data Ingestion...")
    load_data(**context)
    produce_to_kafka(**context)
    print("✅ Demo Data Ingestion Complete.")

with DAG(
    dag_id='unified_pipeline',
    default_args=default_args,
    description='Unified Pipeline: Run Demo, Live, or Hybrid Data Ingestion',
    schedule=None, # Trigger manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['unified', 'demo', 'live', 'hybrid'],
) as dag:

    # 1. Branching Task
    branch_task = BranchPythonOperator(
        task_id='check_mode',
        python_callable=decide_mode
    )

    # 2. Demo Task (Wrapped Python Function)
    demo_task = PythonOperator(
        task_id='trigger_demo_ingest',
        python_callable=run_demo_ingest
    )

    # 3. Live Task (Bash Script Wrapper)
    # Using BashOperator is safer for long-running crawler processes and env isolation
    live_task = BashOperator(
        task_id='trigger_live_ingest',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_producer_live.py --categories thoi-su kinh-doanh --pages 1',
    )

    # 4. Join Task (Optional, just to have a clean end state)
    join_task = PythonOperator(
        task_id='pipeline_finished',
        python_callable=lambda: print("🎉 Pipeline Execution Finished"),
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS # Runs if at least one branch succeeds
    )

    # Graph
    branch_task >> [demo_task, live_task]
    demo_task >> join_task
    live_task >> join_task

if __name__ == "__main__":
    print("🚀 Testing content availability...")
    # Basic check
    from dags.demo_streaming_pipeline import load_data
    print("✅ Imports successful.")
