
import os
import psycopg2

DB_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/trend_db')

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT trend_name, post_count, interactions FROM detected_trends ORDER BY post_count DESC LIMIT 5;")
    rows = cur.fetchall()
    print("📊 Top 5 Trends by Processing Activity:")
    for row in rows:
        print(f"   🔹 {row[0]}: Posts={row[1]}, Interactions={row[2]}")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
