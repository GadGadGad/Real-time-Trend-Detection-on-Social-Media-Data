import json
from sqlalchemy import create_engine, text

POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"

def test_collision():
    engine = create_engine(POSTGRES_URL)
    with engine.begin() as conn:
        # Get the name of trend 830 (now refined)
        row = conn.execute(text("SELECT trend_name FROM detected_trends WHERE id = 830")).fetchone()
        refined_name = row[0]
        print(f"🎯 Target name for collision: '{refined_name}'")
        
        # Insert a NEW discovery trend that LLM will "refine" to this same name
        # We'll mock the LLM by just having this name in our test_db_state or similar
        # Actually, the quickest way is to just insert a trend with the SAME name
        # and see if the DAG merges it on next run.
        # Wait, the DAG logic merges IF current_name == refined_name AND id != target_id.
        
        conn.execute(text("""
            INSERT INTO detected_trends (trend_name, topic_type, category, post_count, interactions, representative_posts, summary, last_updated)
            VALUES (:name, :type, :cat, :pc, :inter, :reps, NULL, NOW())
        """), {
            "name": "Collision Test Trend",
            "type": "Discovery",
            "cat": "Unclassified",
            "pc": 10,
            "inter": 500,
            "reps": json.dumps([{"content": "Same event as 830!", "source": "News"}], ensure_ascii=False)
        })
        print("✅ Inserted Collision Test Trend (summary=NULL).")

if __name__ == "__main__":
    test_collision()
