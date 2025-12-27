import os
import json
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Load database URL
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def check_db():
    try:
        engine = create_engine(POSTGRES_URL)
        with engine.connect() as conn:
            # Check for unanalyzed trends
            result = conn.execute(text("SELECT id, trend_name, topic_type, category FROM detected_trends WHERE category = 'Unclassified' OR topic_type = 'Discovery' LIMIT 5;")).fetchall()
            
            if not result:
                print("📭 No unanalyzed trends found. Populating some dummy data for the demo...")
                # Insert a dummy "Discovery" trend to test the DAG logic
                conn.execute(text("""
                    INSERT INTO detected_trends (trend_name, topic_type, category, post_count, interactions, representative_posts, last_updated)
                    VALUES (:name, :type, :cat, :pc, :inter, :reps, NOW())
                """), {
                    "name": "New: Phim tài liệu về nghệ sĩ Việt Nam gây sốt",
                    "type": "Discovery",
                    "cat": "Unclassified",
                    "pc": 10,
                    "inter": 1500,
                    "reps": json.dumps([
                        {"content": "Phim tài liệu này hay quá, xem mà khóc luôn.", "source": "Facebook"},
                        {"content": "Mọi người đã xem phim về nghệ sĩ này chưa? Đang trend nè.", "source": "TikTok"}
                    ], ensure_ascii=False)
                })
                conn.commit()
                print("✅ Dummy 'Discovery' trend added.")
            else:
                print(f"✅ Found {len(result)} trends to analyze.")
                for row in result:
                    print(f" - [{row[0]}] {row[1]} ({row[2]})")
    except Exception as e:
        print(f"❌ DB Error: {e}")

if __name__ == "__main__":
    check_db()
