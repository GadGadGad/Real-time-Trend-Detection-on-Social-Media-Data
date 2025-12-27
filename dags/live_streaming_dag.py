from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta
import os

# Configuration
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
PYTHON_BIN = "python" # Or specific path if needed, e.g., /home/gad/miniforge3/envs/SE363Final/bin/python

default_args = {
    'owner': 'trend_detection',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    dag_id='live_data_ingestion',
    default_args=default_args,
    description='Fetches live news from VnExpress and streams to Kafka',
    schedule_interval='*/30 * * * *', # Every 30 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['live', 'ingestion', 'kafka'],
) as dag:

    # Task: Run Live Producer
    # This invokes the script we just created.
    run_ingestion = BashOperator(
        task_id='run_live_crawl_and_produce',
        bash_command=f'cd "{PROJECT_ROOT}" && {PYTHON_BIN} streaming/kafka_producer_live.py --categories thoi-su kinh-doanh --pages 1',
    )

    run_ingestion
