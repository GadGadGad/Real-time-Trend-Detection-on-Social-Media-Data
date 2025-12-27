"""
Airflow DAG: Demo-Ready Pipeline

Orchestrates the actual demo-ready/ components:
1. Start Kafka (docker-compose)
2. Run producer.py (send posts to Kafka)
3. Run consumer.py (process from Kafka, write to DB)
4. Dashboard ready notification

This uses the REAL Kafka infrastructure from demo-ready/
"""

import os
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
DEMO_READY_DIR = os.path.join(PROJECT_ROOT, "demo-ready")

sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, DEMO_READY_DIR)

try:
    from airflow import DAG
    from airflow.operators.python import PythonOperator
    from airflow.operators.bash import BashOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    AIRFLOW_AVAILABLE = False

default_args = {
    'owner': 'demo_ready',
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}


def check_kafka(**context):
    """Task 1: Ensure Kafka is running."""
    import subprocess
    
    print("🔍 Checking Kafka status...")
    
    # Check if docker-compose services are up
    result = subprocess.run(
        ["docker", "ps", "--filter", "name=kafka", "--format", "{{.Names}}"],
        capture_output=True, text=True
    )
    
    if "kafka" in result.stdout:
        print("✅ Kafka is running")
        return {"kafka": "running"}
    
    print("⚠️ Kafka not running, starting...")
    subprocess.run(
        ["docker-compose", "up", "-d"],
        cwd=DEMO_READY_DIR
    )
    
    # Wait for Kafka to be ready
    import time
    time.sleep(10)
    
    print("✅ Kafka started")
    return {"kafka": "started"}


def run_producer(**context):
    """Task 2: Run producer.py to send posts to Kafka."""
    import subprocess
    import time
    
    print("📤 KAFKA PRODUCER")
    print("=" * 50)
    
    # Run producer with timeout (it should complete after sending all posts)
    result = subprocess.run(
        [sys.executable, "producer.py"],
        cwd=DEMO_READY_DIR,
        capture_output=True,
        text=True,
        timeout=300  # 5 minutes max
    )
    
    print(result.stdout)
    if result.stderr:
        print(f"Stderr: {result.stderr}")
    
    print("=" * 50)
    print("✅ Producer finished")
    
    return {"status": "completed"}


def run_consumer(**context):
    """Task 3: Run consumer.py to process posts from Kafka."""
    import subprocess
    import time
    
    print("⚡ KAFKA CONSUMER")
    print("=" * 50)
    
    # Run consumer with timeout
    # Consumer will process all messages and exit
    result = subprocess.run(
        [sys.executable, "consumer.py"],
        cwd=DEMO_READY_DIR,
        capture_output=True,
        text=True,
        timeout=600  # 10 minutes max
    )
    
    print(result.stdout[-5000:] if len(result.stdout) > 5000 else result.stdout)
    if result.stderr:
        print(f"Stderr (last 1000 chars): {result.stderr[-1000:]}")
    
    print("=" * 50)
    print("✅ Consumer finished")
    
    return {"status": "completed"}


def notify_dashboard(**context):
    """Task 4: Notify that dashboard is ready."""
    print("\n" + "=" * 60)
    print("🎉 DEMO-READY PIPELINE COMPLETE!")
    print("=" * 60)
    print("")
    print("📊 To view Dashboard:")
    print(f"   cd {DEMO_READY_DIR}")
    print("   streamlit run dashboard.py --server.port 8502")
    print("")
    print("🌐 Dashboard URL: http://localhost:8502")
    print("=" * 60)
    
    return {"dashboard": "ready"}


# === DAG DEFINITION ===
if AIRFLOW_AVAILABLE:
    with DAG(
        dag_id='demo_ready_pipeline',
        default_args=default_args,
        description='Orchestrates demo-ready/ Kafka Producer → Consumer → Dashboard',
        schedule=None,  # Manual trigger
        start_date=datetime(2024, 1, 1),
        catchup=False,
        tags=['demo-ready', 'kafka', 'real-pipeline'],
    ) as dag:
        
        # Task 1: Check/Start Kafka
        t1 = PythonOperator(
            task_id='check_kafka',
            python_callable=check_kafka,
        )
        
        # Task 2: Run Producer
        t2 = PythonOperator(
            task_id='run_producer',
            python_callable=run_producer,
            execution_timeout=timedelta(minutes=5),
        )
        
        # Task 3: Run Consumer  
        t3 = PythonOperator(
            task_id='run_consumer',
            python_callable=run_consumer,
            execution_timeout=timedelta(minutes=10),
        )
        
        # Task 4: Notify
        t4 = PythonOperator(
            task_id='notify_dashboard',
            python_callable=notify_dashboard,
        )
        
        t1 >> t2 >> t3 >> t4


# === STANDALONE TEST ===
if __name__ == "__main__":
    print("🚀 Running Demo-Ready Pipeline Locally...\n")
    
    check_kafka()
    run_producer()
    run_consumer()
    notify_dashboard()
