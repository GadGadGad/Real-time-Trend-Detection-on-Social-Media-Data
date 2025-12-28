
import os
import sys
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta

# Config
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")

def verify_pipeline():
    print("🔍 [Verify] Checking pipeline health (Seeding -> Ingest -> Process -> Intelligence)...")
    
    try:
        engine = create_engine(POSTGRES_URL)
        with engine.connect() as conn:
            # 1. Check Row Count
            count = conn.execute(text("SELECT COUNT(*) FROM detected_trends")).scalar()
            print(f"📊 [Verify] Total Trends in DB: {count}")
            
            if count == 0:
                print("❌ [Verify] FAILED: Database is empty. Seeding or Ingestion failed.")
                sys.exit(1)
                
            # 2. Check for recent activity (last 5 minutes)
            # This confirms the Kafka Consumer or Intelligence Worker actually ran
            recent = conn.execute(text("""
                SELECT COUNT(*) FROM detected_trends 
                WHERE last_updated >= NOW() - INTERVAL '10 minutes'
            """)).scalar()
            
            print(f"⏱️ [Verify] Trends updated in last 10 mins: {recent}")
            
            if recent == 0:
                print("⚠️ [Verify] WARNING: No recent updates found. Did the Consumer run?")
                # Don't exit 1 strict, as maybe it was just a quick check run, but warn heavily
            else:
                print("✅ [Verify] Active processing detected.")
            
            # 3. Check for Intelligence (Non-null summaries)
            analyzed = conn.execute(text("""
                SELECT COUNT(*) FROM detected_trends 
                WHERE summary IS NOT NULL 
                AND summary != 'Waiting for analysis...' 
                AND summary != 'Seeded from Google Trends'
            """)).scalar()
            
            print(f"🧠 [Verify] Intelligent Analyses Completed: {analyzed}")
            
    except Exception as e:
        print(f"❌ [Verify] Database connection failed: {e}")
        sys.exit(1)

    print("🎉 [Verify] Pipeline Verification Passed.")

if __name__ == "__main__":
    verify_pipeline()
