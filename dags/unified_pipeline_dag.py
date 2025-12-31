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
    Options: 'fast-demo' (instant), 'demo' (kafka), 'live', 'hybrid'
    Default: 'fast-demo' for instant demo experience
    """
    conf = context.get('dag_run').conf or {}
    mode = conf.get('mode', 'demo').lower()
    
    print(f"🔀 Branching Decision: Mode = {mode}")
    
    if mode == 'fast-demo':
        return ['load_precomputed_data']
    
    # For streaming modes, we start multiple tasks in parallel
    tasks_to_run = []
    if mode in ['demo', 'hybrid']:
        tasks_to_run.append('trigger_demo_ingest')
    if mode in ['live', 'hybrid']:
        tasks_to_run.append('trigger_live_ingest')
    
    # Engine and AI should also start in parallel with ingestion
    tasks_to_run.extend(['check_engine', 'run_ai_analysis'])
    return tasks_to_run


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
    description='Unified Pipeline: Run Fast-Demo, Demo, Live, or Hybrid Data Ingestion',
    schedule=None, # Trigger manually
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['unified', 'fast-demo', 'demo', 'live', 'hybrid', 'spark'],
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

    # 2a. Fast Demo Task - Load pre-computed data instantly (RECOMMENDED for demos)
    fast_demo_task = BashOperator(
        task_id='load_precomputed_data',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/load_demo_data.py',
    )

    # 2b. Demo Task (Kafka-based, slower but shows real streaming)
    # 2b. Demo Task (Kafka-based, slower but shows real streaming)
    demo_task = BashOperator(
        task_id='trigger_demo_ingest',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/producer_simple.py',
    )

    # 3. Live Task (Bash Script Wrapper)
    live_task = BashOperator(
        task_id='trigger_live_ingest',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_producer_live.py --categories thoi-su kinh-doanh --pages 1',
    )

    # 3.5. Branching Task for Engine (only for Kafka-based modes)
    engine_branch = BranchPythonOperator(
        task_id='check_engine',
        python_callable=decide_engine,
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # 4a. Standard Python Consumer Task
    consumer_task = BashOperator(
        task_id='run_consumer_processing',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_consumer.py --timeout 300',
    )

    # 4b. New Spark Consumer Task
    spark_task = BashOperator(
        task_id='run_spark_processing',
        bash_command=f'cd "{PROJECT_ROOT}" && spark-submit --packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0 streaming/spark_consumer.py',
    )

    # 5. Intelligence Task (Refine Trends) - Skip for fast-demo since AI is pre-computed
    intelligence_task = BashOperator(
        task_id='run_ai_analysis',
        bash_command=f'cd "{PROJECT_ROOT}" && source .env && {PYTHON_BIN} streaming/intelligence_worker.py --max-cycles 5',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # 6. Verification Task
    verify_task = BashOperator(
        task_id='verify_pipeline_health',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} scripts/verify_pipeline.py',
        trigger_rule=TriggerRule.NONE_FAILED_MIN_ONE_SUCCESS
    )

    # Graph - Two paths:
    # Path 1 (Fast Demo): seed -> check_mode -> load_precomputed_data -> verify
    # Path 2 (Kafka modes): 
    #   - seed -> check_mode -> [demo, live, engine_branch, intelligence] IN PARALLEL
    #   - engine_branch -> [consumer, spark]
    
    seed_task >> branch_task >> [fast_demo_task, demo_task, live_task, engine_branch, intelligence_task]
    fast_demo_task >> verify_task

    # Connect engine branch to its targets
    engine_branch >> [consumer_task, spark_task]
    
    # Verification runs after everything (theoretically - for streams it might never run)
    [consumer_task, spark_task, intelligence_task] >> verify_task
