import os
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

load_dotenv()
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
engine = create_engine(POSTGRES_URL)

def verify():
    with engine.connect() as conn:
        result = conn.execute(text("SELECT id, trend_name, summary, advice_state, advice_business FROM detected_trends WHERE id = 173")).fetchone()
        if result:
            print(f"ID: {result[0]}")
            print(f"Name: {result[1]}")
            print(f"Summary: {result[2][:100]}...")
            print(f"Advice State: {result[3][:100]}...")
            print(f"Advice Business: {result[4][:100]}...")
        else:
            print("Trend 173 not found or not yet updated.")

if __name__ == "__main__":
    verify()
