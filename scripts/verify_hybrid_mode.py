import subprocess
import time
import os
import signal
import sys
from sqlalchemy import create_engine, text

# Config
PROJECT_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"
DB_URL = "postgresql://user:password@localhost:5432/trend_db"

def run_background(cmd, log_file):
    with open(log_file, "w") as f:
        return subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=f, stderr=f, shell=True, preexec_fn=os.setsid)

def check_db_integrity():
    engine = create_engine(DB_URL)
    with engine.connect() as conn:
        # Check for Live Data
        live_count = conn.execute(text("SELECT COUNT(*) FROM detected_trends WHERE source LIKE '%VnExpress Live%' OR summary='Waiting for analysis...'")).scalar()
        
        # Check for Demo Data (using the Demo Source names)
        demo_count = conn.execute(text("SELECT COUNT(*) FROM detected_trends WHERE source IN ('Facebook', 'News') AND summary != 'Waiting for analysis...'")).scalar()
        
        # Check total
        total = conn.execute(text("SELECT COUNT(*) FROM detected_trends")).scalar()
        
        return live_count, demo_count, total

def main():
    print("🚀 STARTING HYBRID PIPELINE TEST")
    print("================================")
    
    # 1. Kill invalid processes
    print("🧹 Cleaning up old processes...")
    subprocess.run("pkill -f demo_streaming_pipeline.py", shell=True)
    subprocess.run("pkill -f kafka_producer_live.py", shell=True)
    
    # 2. Make sure DB is cleanish (optional, but checking flow)
    # subprocess.run("python scripts/clear_db.py", shell=True, cwd=PROJECT_ROOT)
    
    # 3. Start Live Producer
    print("🟢 Starting LIVE Producer (VnExpress)...")
    live_proc = run_background(
        "python streaming/kafka_producer_live.py --categories thoi-su --pages 1",
        "logs/test_live.log"
    )
    
    # 4. Start Demo Producer
    print("🔵 Starting DEMO Producer (Historical)...")
    demo_proc = run_background(
        "python dags/demo_streaming_pipeline.py", 
        "logs/test_demo.log"
    )
    
    print("⏳ Waiting 30s for data to flow...")
    try:
        for i in range(30):
            time.sleep(1)
            sys.stdout.write(f"\rTime: {i+1}/30s")
            sys.stdout.flush()
    except KeyboardInterrupt:
        pass
        
    print("\n\n📊 CHECKING RESULTS...")
    
    # Inspect DB
    # Note: kafka_consumer.py is assumed to be running!
    try:
        engine = create_engine(DB_URL)
        with engine.connect() as conn:
            # We check raw counts. 
            # Note: Demo Producer sends 'source'='Facebook'/'News'. Live sends 'source'='VnExpress Live'.
            res = conn.execute(text("SELECT source, COUNT(*) FROM detected_trends GROUP BY source")).fetchall()
            
            print("\nDatabase Content by Source:")
            found_live = False
            found_demo = False
            
            for row in res:
                s = row[0]
                c = row[1]
                print(f" - {s}: {c}")
                if "VnExpress Live" in str(s): found_live = True
                if "Facebook" in str(s) or "News" in str(s): found_demo = True
            
            if found_live and found_demo:
                print("\n✅ SUCCESS: Found both LIVE and DEMO data trends!")
            elif found_live:
                print("\n⚠️ PARTIAL: Only found LIVE data.")
            elif found_demo:
                print("\n⚠️ PARTIAL: Only found DEMO data.")
            else:
                print("\n❌ FAILURE: No data found.")

    except Exception as e:
        print(f"❌ Error checking DB: {e}")

    # Cleanup
    print("\n🛑 Stopping Test Producers...")
    os.killpg(os.getpgid(live_proc.pid), signal.SIGTERM)
    os.killpg(os.getpgid(demo_proc.pid), signal.SIGTERM)

if __name__ == "__main__":
    main()
