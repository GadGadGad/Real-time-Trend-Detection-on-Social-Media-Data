
import os
import psycopg2

DB_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/trend_db')

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT trend_name, keywords FROM detected_trends LIMIT 3;")
    rows = cur.fetchall()
    print("📊 Sample Keywords Format:")
    for row in rows:
        print(f"   🔹 {row[0]}: keywords='{row[1]}' (type={type(row[1]).__name__})")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
