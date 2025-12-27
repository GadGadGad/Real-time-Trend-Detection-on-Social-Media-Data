"""
Batch+Streaming Pipeline DAG (with Kafka)
==========================================
Runs the full pipeline:
1. Start Kafka (check/ensure)
2. Kafka Producer (send posts)
3. Kafka Consumer (assign to trends)
4. Dashboard Ready

This uses PRE-COMPUTED centroids for fast topic assignment.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import subprocess
import os
import sys
import time

# Project root
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"

default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=1),
}

dag = DAG(
    'batch_streaming_kafka',
    default_args=default_args,
    description='Batch+Streaming Pipeline with REAL Kafka (pre-computed centroids)',
    schedule='*/2 * * * *',  # Every 2 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['streaming', 'kafka', 'batch'],
)


def check_kafka(**context):
    """Check if Kafka is running, provide instructions if not"""
    import socket
    
    kafka_host = 'localhost'
    kafka_port = 29092
    
    print("🔍 Checking Kafka connection...")
    
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(5)
        result = sock.connect_ex((kafka_host, kafka_port))
        sock.close()
        
        if result == 0:
            print(f"✅ Kafka is running at {kafka_host}:{kafka_port}")
            return {"kafka_ready": True}
        else:
            print(f"❌ Kafka is NOT running!")
            print(f"")
            print(f"📋 To start Kafka, run:")
            print(f"   cd {PROJECT_ROOT}/demo-ready")
            print(f"   docker-compose up -d")
            print(f"")
            raise Exception("Kafka is not running. Start it first with docker-compose.")
            
    except Exception as e:
        raise Exception(f"Kafka check failed: {e}")


def run_producer(**context):
    """Run Kafka producer to send posts"""
    sys.path.insert(0, PROJECT_ROOT)
    
    # Import and run producer
    from streaming.kafka_producer import run_producer
    
    print("📤 Starting Kafka Producer...")
    result = run_producer()
    
    if result.get('status') == 'success':
        print(f"✅ Producer sent {result.get('sent', 0)} posts")
        return result
    else:
        raise Exception(f"Producer failed: {result}")


def run_consumer(**context):
    """Run Kafka consumer to process posts"""
    sys.path.insert(0, PROJECT_ROOT)
    
    # Import and run consumer
    from streaming.kafka_consumer import run_consumer
    
    print("📥 Starting Kafka Consumer...")
    result = run_consumer()
    
    if result.get('status') in ['success', 'interrupted']:
        print(f"✅ Consumer processed {result.get('processed', 0)} posts")
        return result
    else:
        raise Exception(f"Consumer failed: {result}")


def dashboard_ready(**context):
    """Final task - notify that dashboard is ready"""
    ti = context['ti']
    
    # Get results from previous tasks
    producer_result = ti.xcom_pull(task_ids='kafka_producer') or {}
    consumer_result = ti.xcom_pull(task_ids='kafka_consumer') or {}
    
    print("=" * 60)
    print("🎉 BATCH+STREAMING PIPELINE COMPLETE!")
    print("=" * 60)
    print(f"")
    print(f"📊 Summary:")
    print(f"   • Posts sent to Kafka: {producer_result.get('sent', 'N/A')}")
    print(f"   • Posts processed: {consumer_result.get('processed', 'N/A')}")
    print(f"   • Trends updated: {consumer_result.get('trends', 'N/A')}")
    print(f"")
    print(f"🖥️  To view in Dashboard:")
    print(f"   cd {PROJECT_ROOT}/demo-ready")
    print(f"   streamlit run dashboard.py")
    print(f"")
    print("=" * 60)
    
    return {
        "status": "complete",
        "posts_sent": producer_result.get('sent', 0),
        "posts_processed": consumer_result.get('processed', 0),
        "trends_updated": consumer_result.get('trends', 0)
    }


# Define tasks
check_kafka_task = PythonOperator(
    task_id='check_kafka',
    python_callable=check_kafka,
    dag=dag,
)

producer_task = PythonOperator(
    task_id='kafka_producer',
    python_callable=run_producer,
    dag=dag,
)

consumer_task = PythonOperator(
    task_id='kafka_consumer',
    python_callable=run_consumer,
    dag=dag,
)

dashboard_task = PythonOperator(
    task_id='dashboard_ready',
    python_callable=dashboard_ready,
    dag=dag,
)

# Task dependencies
check_kafka_task >> producer_task >> consumer_task >> dashboard_task
