
import sys
import os
import unittest.mock
from datetime import datetime

# Add project root
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
sys.path.insert(0, PROJECT_ROOT)

# Mock Airflow Context
class MockDagRun:
    def __init__(self, mode='hybrid'):
        self.conf = {'mode': mode}

def simulate_airflow():
    print("🚀 Simulating AIRFLOW DAG Execution (Hybrid Mode)...")
    print("---------------------------------------------------")
    
    try:
        from dags.unified_pipeline_dag import decide_mode, run_demo_ingest
        from dags.demo_streaming_pipeline import load_data, produce_to_kafka
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return

    # 1. Simulate Task: check_mode
    print("\n[Task: check_mode]")
    context = {'dag_run': MockDagRun(mode='hybrid')}
    decision = decide_mode(**context)
    print(f"   -> Result: Next tasks = {decision}")
    
    # 2. Simulate Branch Execution
    if 'trigger_live_ingest' in decision:
        print("\n[Task: trigger_live_ingest]")
        print("   -> (Starting BashOperator logic for Live Crawler...)")
        import subprocess
        # Start Live in background logic
        cmd = f"python streaming/kafka_producer_live.py --categories thoi-su kinh-doanh --pages 1"
        print(f"   -> Running: {cmd}")
        # We won't block here for the simulation, just show intent or run briefly
        # subprocess.Popen(cmd, shell=True, cwd=PROJECT_ROOT)
        print("   -> ✅ Live Crawler Triggered (Simulated)")

    if 'trigger_demo_ingest' in decision:
        print("\n[Task: trigger_demo_ingest]")
        print("   -> Starting PythonOperator logic...")
        # Start Demo logic
        try:
           load_data(**context)
           produce_to_kafka(**context)
           print("   -> ✅ Demo Ingestion Complete")
        except Exception as e:
           print(f"   -> ❌ Demo Task Failed: {e}")

    print("\n---------------------------------------------------")
    print("🎉 Pipeline Simulation Finished.")

if __name__ == "__main__":
    simulate_airflow()
