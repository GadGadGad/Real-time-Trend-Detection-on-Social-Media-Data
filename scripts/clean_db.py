
import os
import psycopg2
from urllib.parse import urlparse

# Get DB URL from env or use default
DB_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/trend_db')

def wipe_trends():
    try:
        conn = psycopg2.connect(DB_URL)
        cur = conn.cursor()
        
        print("🧹 Cleaning database...")
        
        # Option 1: Delete specific timestamp-like trends
        # cur.execute("DELETE FROM detected_trends WHERE trend_name LIKE '2025-%';")
        
        # Option 2: WIPE ALL to ensure a clean demo slate (Recommended for this specific user issue)
        cur.execute("TRUNCATE TABLE detected_trends CASCADE;")
        # cur.execute("TRUNCATE TABLE processed_posts CASCADE;")
        
        conn.commit()
        print("✅ Database wiped clean! Ready for fresh seeding.")
        
        cur.close()
        conn.close()
    except Exception as e:
        print(f"❌ Error cleaning DB: {e}")

if __name__ == "__main__":
    wipe_trends()
