
import os
import psycopg2

DB_URL = os.getenv('POSTGRES_URL', 'postgresql://user:password@localhost:5432/trend_db')

try:
    conn = psycopg2.connect(DB_URL)
    cur = conn.cursor()
    cur.execute("SELECT trend_name, trend_score, volume FROM detected_trends ORDER BY trend_score DESC LIMIT 5;")
    rows = cur.fetchall()
    print("📊 Top 5 Seeded Trends by Score:")
    for row in rows:
        print(f"   🔹 {row[0]}: Score={row[1]}, Vol={row[2]}")
        
    cur.close()
    conn.close()
except Exception as e:
    print(f"❌ Error: {e}")
