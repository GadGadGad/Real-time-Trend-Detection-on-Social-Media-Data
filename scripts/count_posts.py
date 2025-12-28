
import os
import psycopg2

DB_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/trend_db')

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT COUNT(*) FROM processed_posts;")
    count = cur.fetchone()[0]
    print(f"📊 Processed Posts Count: {count}")
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
