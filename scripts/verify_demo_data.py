import os
import sys
import time
from sqlalchemy import create_engine, text

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")

def verify_data():
    print("🔍 Connect to DB...")
    engine = create_engine(POSTGRES_URL)
    
    with engine.connect() as conn:
        # Check total count
        result = conn.execute(text("SELECT COUNT(*) FROM detected_trends"))
        count = result.scalar()
        print(f"📊 Total Trends in DB: {count}")
        
        if count == 0:
            print("❌ DB is empty! Seeding failed or pipeline broken.")
            return

        # Check for seeded trends
        res_seeded = conn.execute(text("SELECT COUNT(*) FROM detected_trends WHERE topic_type='Seeded'"))
        seeded_count = res_seeded.scalar()
        print(f"🌱 Seeded Trends: {seeded_count}")
        
        # Check for real-time detected trends (from Kafka)
        # Assuming non-seeded trends have a different topic_type or topic_type != 'Seeded'
        # Adjust logic if needed based on your pipeline
        res_real = conn.execute(text("SELECT COUNT(*) FROM detected_trends WHERE topic_type != 'Seeded'"))
        real_count = res_real.scalar()
        print(f"⚡ Real-time/Detected Trends: {real_count}")
        
        if seeded_count > 0:
            print("✅ Seeding Successful.")
        else:
            print("⚠️ Seeding might have failed.")
            
        if real_count > 0:
            print("✅ Real-time pipeline is processing data!")
        else:
            print("⏳ Real-time pipeline hasn't produced new trends yet (might be warming up or grouping).")

        # Show top 5 trends
        print("\n🏆 Top 5 Recent Trends:")
        trends = conn.execute(text("SELECT id, trend_name, trend_score, topic_type, post_count FROM detected_trends ORDER BY last_updated DESC LIMIT 5")).fetchall()
        for t in trends:
            print(f"   - [{t[0]}] {t[1]} (Score: {t[2]}, Type: {t[3]}, Posts: {t[4]})")

if __name__ == "__main__":
    verify_data()
