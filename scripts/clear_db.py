from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def clear_db():
    engine = create_engine(POSTGRES_URL)
    with engine.connect() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends;"))
        conn.commit()
    print("✅ Database cleared: detected_trends table truncated.")

if __name__ == "__main__":
    clear_db()
