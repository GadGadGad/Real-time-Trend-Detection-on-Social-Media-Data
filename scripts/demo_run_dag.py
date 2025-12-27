import os
import sys
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the functions from the DAG
from dags.intelligence_llm_dag import check_api_key, fetch_unanalyzed_trends, analyze_with_llm, log_completion

class MockTaskInstance:
    def __init__(self):
        self.xcom = {}
    
    def xcom_push(self, key, value):
        self.xcom[key] = value
    
    def xcom_pull(self, key=None, task_ids=None):
        if key:
            return self.xcom.get(key)
        # Default to whatever matches task_ids if key is None (simplified for this mock)
        if task_ids == 'fetch_unanalyzed_trends':
            return self.xcom.get('trends_to_analyze')
        return self.xcom.get(task_ids)

def run_demo():
    print("🚀 STARTING DAG LOGIC DEMO RUN")
    print("-" * 40)
    
    context = {'ti': MockTaskInstance()}
    
    try:
        # Step 1: Check API Key
        print("\nStep 1: Checking API Key...")
        check_api_key(**context)
        
        # Step 2: Fetch trends
        print("\nStep 2: Fetching Trends from DB...")
        count = fetch_unanalyzed_trends(**context)
        if count == 0:
            print("📭 No trends to process. Exiting demo.")
            return

        # Step 3: Analyze with LLM
        print("\nStep 3: Analyzing with LLM (Phase 3-5)...")
        # This will calculate re-embeddings and perform deduplication
        stats = analyze_with_llm(**context)
        context['ti'].xcom_push('analyze_with_llm', stats)
        
        # Step 4: Log completion
        print("\nStep 4: Final Summary")
        log_completion(**context)
        
    except Exception as e:
        print(f"\n❌ Demo Failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_demo()
