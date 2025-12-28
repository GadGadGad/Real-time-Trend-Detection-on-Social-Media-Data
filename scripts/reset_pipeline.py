from sqlalchemy import create_engine, text
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")

def reset():
    print("🗑️ Clearing Database for Fresh Demo...")
    try:
        engine = create_engine(POSTGRES_URL)
        with engine.begin() as conn:
            conn.execute(text("TRUNCATE TABLE detected_trends RESTART IDENTITY"))
        print("✅ Tables truncated. Ready for streaming.")
    except Exception as e:
        print(f"❌ Error resetting DB: {e}")

if __name__ == "__main__":
    reset()
