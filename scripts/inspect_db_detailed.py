from sqlalchemy import create_engine, text
import json

POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def inspect_db_detailed():
    engine = create_engine(POSTGRES_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, trend_name, summary, post_count, representative_posts FROM detected_trends LIMIT 3;")).fetchall()
        for row in result:
            reps = row[4]
            print(f"ID: {row[0]}")
            print(f"Name: {row[1]}")
            print(f"Summary: {row[2]}")
            print(f"Count: {row[3]}")
            print(f"Reps Type: {type(reps)}")
            print(f"Reps Preview: {str(reps)[:200]}")
            if isinstance(reps, str):
                try:
                    parsed = json.loads(reps)
                    print(f"Parsed Reps Count: {len(parsed)}")
                except:
                    print("Failed to parse Reps JSON")
            print("-" * 20)

if __name__ == "__main__":
    inspect_db_detailed()
