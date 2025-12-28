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

def decide_engine(**context):
    """
    Decide whether to use standard 'python' or 'spark' engine for processing.
    (default: 'python')
    """
    conf = context.get('dag_run').conf or {}
    engine = conf.get('engine', 'python').lower()
    
    print(f"⚙️  Engine Decision: {engine}")
    
    if engine == 'spark':
        return 'run_spark_processing'
    else:
        return 'run_consumer_processing'

with DAG(
    dag_id='unified_pipeline',
    default_args=default_args,
    description='Unified Pipeline: Run Demo, Live, or Hybrid Data Ingestion',
    schedule=None, # Trigger manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['unified', 'demo', 'live', 'hybrid', 'spark'],
) as dag:

    # 0. Seeding Task (New Robustness Step)
    seed_task = BashOperator(
        task_id='seed_initial_trends',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} scripts/seed_trends.py',
    )

    # 1. Branching Task for Mode
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
    live_task = BashOperator(
        task_id='trigger_live_ingest',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_producer_live.py --categories thoi-su kinh-doanh --pages 1',
    )

    # 3.5. Branching Task for Engine
    engine_branch = BranchPythonOperator(
        task_id='check_engine',
        python_callable=decide_engine,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # 4a. Standard Python Consumer Task
    consumer_task = BashOperator(
        task_id='run_consumer_processing',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_consumer.py --timeout 30',
    )

    # 4b. New Spark Consumer Task
    spark_task = BashOperator(
        task_id='run_spark_processing',
        # In production, this would be a long-running job, here we use finite execution sim or spark-submit
        bash_command=f'cd "{PROJECT_ROOT}" && spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming/spark_consumer.py',
    )

    # 5. Intelligence Task (Refine Trends)
    intelligence_task = BashOperator(
        task_id='run_ai_analysis',
        bash_command=f'cd "{PROJECT_ROOT}" && source .env && {PYTHON_BIN} demo-ready/intelligence_worker.py --max-cycles 5',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # 6. Verification Task
    verify_task = BashOperator(
        task_id='verify_pipeline_health',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} scripts/verify_pipeline.py',
        trigger_rule=TriggerRule.ALL_SUCCESS
    )

    # Graph
    seed_task >> branch_task >> [demo_task, live_task]
    [demo_task, live_task] >> engine_branch >> [consumer_task, spark_task]
    [consumer_task, spark_task] >> intelligence_task >> verify_task

if __name__ == "__main__":
    print("🚀 Testing content availability...")
    # Basic check
    from dags.demo_streaming_pipeline import load_data
    print("✅ Imports successful.")
