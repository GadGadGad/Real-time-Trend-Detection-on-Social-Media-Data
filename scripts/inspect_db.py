from sqlalchemy import create_engine, text
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def inspect_db():
    engine = create_engine(POSTGRES_URL)
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, trend_name, post_count, summary, category FROM detected_trends LIMIT 5;")).fetchall()
        for row in result:
            print(f"ID: {row[0]}, Name: {row[1]}, Count: {row[2]}, Summary: {row[3]}, Cat: {row[4]}")

if __name__ == "__main__":
    inspect_db()
