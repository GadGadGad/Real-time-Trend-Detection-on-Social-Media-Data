from sqlalchemy import create_engine, text
import json

POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
engine = create_engine(POSTGRES_URL)

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT id, trend_name, summary, category, post_count, last_updated 
        FROM detected_trends 
        ORDER BY last_updated DESC 
        LIMIT 10
    """)).fetchall()
    
    print(f"{'ID':<5} | {'Trend Name':<50} | {'Summary':<20} | {'Updated'}")
    print("-" * 100)
    for row in result:
        summ = (row[2][:17] + "...") if row[2] else "N/A"
        print(f"{row[0]:<5} | {row[1][:50]:<50} | {summ:<20} | {row[5]}")
