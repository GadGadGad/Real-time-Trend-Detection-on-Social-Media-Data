"""
Kafka Consumer for Batch+Streaming Pipeline
=============================================
Consumes posts from Kafka and assigns to pre-computed trends.
NO ML INFERENCE - uses pre-computed centroids for fast lookup.

This is ALTERNATIVE to demo_ready/consumer.py:
- demo_ready: Does full ML (embedding + UMAP + HDBSCAN) - SLOWER
- streaming/: Uses pre-computed centroids (lookup only) - FASTER
"""

import json
import os
import sys
import time
import math
import numpy as np
from datetime import datetime
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIG ---
KAFKA_TOPIC = 'batch-stream'  # Same topic as producer
BOOTSTRAP_SERVERS = ['localhost:29092']
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
BATCH_SIZE = 50  # Process in batches for efficiency
SIMILARITY_THRESHOLD = 0.5  # Minimum similarity to assign to a trend

# Database engine
db_engine = create_engine(POSTGRES_URL)

def init_db():
    """Ensure detected_trends table exists"""
    with db_engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS detected_trends (
                id SERIAL PRIMARY KEY,
                trend_name TEXT,
                trend_score FLOAT,
                volume INT,
                representative_posts TEXT,
                topic_type TEXT,
                post_count INT,
                score_n FLOAT,
                score_f FLOAT,
                created_at TIMESTAMP,
                last_updated TIMESTAMP,
                summary TEXT,
                category TEXT,
                sentiment TEXT,
                advice_state TEXT,
                advice_business TEXT,
                keywords TEXT,
                embedding TEXT,
                google_vol INT DEFAULT 0,
                interactions INT DEFAULT 0
            );
        """))
        conn.commit()
    print("✅ Database initialized")

def load_centroids():
    """Load pre-computed centroids from demo_state"""
    try:
        from src.utils.demo_state import load_demo_state
        
        demo_states_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "demo_states"
        )
        
        if os.path.exists(demo_states_dir):
            folders = sorted([
                f for f in os.listdir(demo_states_dir)
                if os.path.isdir(os.path.join(demo_states_dir, f))
            ], reverse=True)
            
            for folder in folders:
                folder_path = os.path.join(demo_states_dir, folder)
                try:
                    state = load_demo_state(folder_path)
                    if state:
                        print(f"📂 Loaded state from {folder}")
                        return {
                            'cluster_centroids': state.get('centroids') or state.get('cluster_centroids'),
                            'trend_embeddings': state.get('trend_embeddings'),
                            'trend_keys': state.get('trend_keys', []) or (state.get('df_results')['final_topic'].unique().tolist() if 'df_results' in state else [])
                        }
                except Exception as e:
                    print(f"⚠️ Failed to load {folder}: {e}")
        
        print("⚠️ No pre-computed centroids found")
        return None
        
    except Exception as e:
        print(f"❌ Error loading centroids: {e}")
        return None

def assign_topic(content, final_topic_from_producer, centroids_state):
    """
    Assign post to a trend using pre-computed topic label.
    Falls back to centroid matching if label is generic.
    """
    # First: Use pre-computed topic from producer if specific
    if final_topic_from_producer and final_topic_from_producer != 'Unknown':
        return final_topic_from_producer, 0.95  # High confidence for pre-computed
    
    # Fallback: DO NOT hash to existing keys (prevents pollution)
    # If we don't have a specific topic, treat it as Unclassified
    return "Unclassified", 0.0
    
    return "Unclassified", 0.0

def calculate_realtime_score(g_vol, interactions, post_count):
    """Aligns with src.pipeline.trend_scoring heuristics"""
    MAX_VOL = 5010000
    g_score = (math.log10(g_vol + 1) / math.log10(MAX_VOL + 1)) * 100 if g_vol > 0 else 0
    g_score = min(100, g_score)

    MAX_INTERACTIONS = 20000
    f_score = (math.log10(interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
    f_score = min(100, f_score)

    MAX_ARTICLES = 20
    n_score = (math.log10(post_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if post_count > 0 else 0
    n_score = min(100, n_score)

    active_sources = 0
    if g_score > 10: active_sources += 1
    if f_score > 10: active_sources += 1
    if n_score > 10: active_sources += 1
    
    synergy_mult = 1.0
    if active_sources == 3: synergy_mult = 1.2
    elif active_sources == 2: synergy_mult = 1.1

    base_score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
    final_score = min(100, base_score * synergy_mult)
    return round(final_score, 1), round(g_score, 1), round(f_score, 1), round(n_score, 1)

def update_trend_in_db(trend_name, post, similarity):
    """Update or create trend in database"""
    with db_engine.begin() as conn:
        # Check if trend exists
        result = conn.execute(
            text("SELECT id, post_count, representative_posts, google_vol, interactions FROM detected_trends WHERE trend_name = :name"),
            {"name": trend_name}
        ).fetchone()
        
        if result:
            # Update existing trend
            trend_id, post_count, reps_json, g_vol, interactions = result
            post_count = (post_count or 0) + 1
            interactions = (interactions or 0) + 10  # Heuristic: +10 per post
            g_vol = g_vol or 0
            
            # Recalculate Score
            new_score, g_s, f_s, n_s = calculate_realtime_score(g_vol, interactions, post_count)
            
            # Parse existing reps
            try:
                reps = json.loads(reps_json) if reps_json else []
            except:
                reps = []
            
            # Add new post
            new_rep = {
                "content": post['content'][:300],
                "source": post['source'],
                "time": post['time'],
                "similarity": float(similarity)
            }
            reps = [new_rep] + reps[:999]  # Keep latest 1000 to match demo-ready

            # Update
            
            conn.execute(text("""
                UPDATE detected_trends
                SET post_count = :count,
                    volume = :vol,
                    interactions = :inter,
                    trend_score = :score,
                    score_n = :sn,
                    score_f = :sf,
                    representative_posts = :reps,
                    last_updated = :now
                WHERE id = :id
            """), {
                "count": post_count,
                "vol": g_vol + post_count, # Total Volume proxy
                "inter": interactions,
                "score": new_score,
                "sn": n_s,
                "sf": f_s,
                "reps": json.dumps(reps, ensure_ascii=False),
                "now": datetime.now(),
                "id": trend_id
            })
        else:
            # Insert new trend
            rep_posts = [{
                "content": post['content'][:300],
                "source": post['source'],
                "time": post['time'],
                "similarity": float(similarity)
            }]
            
            conn.execute(text("""
                INSERT INTO detected_trends (
                    trend_name, trend_score, volume, post_count,
                    representative_posts, topic_type, score_n, score_f,
                    created_at, last_updated, summary, category, sentiment
                ) VALUES (
                    :name, :score, 1, 1, :reps, :type, :sn, :sf,
                    :now, :now, :summary, :cat, :sent
                )
            """), {
                "name": trend_name,
                "score": 50.0,  # Initial score
                "reps": json.dumps(rep_posts, ensure_ascii=False),
                "type": "Social" if post['source'] in ['Facebook', 'Twitter'] else "News",
                "sn": 50.0,
                "sf": 50.0,
                "now": datetime.now(),
                "summary": "Waiting for analysis...",
                "cat": "Unclassified",
                "sent": "Neutral"
            })

def run_consumer():
    """Consume from Kafka and update database"""
    print(f"🔄 Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='latest',  # Start from latest to see fresh flow
            # no timeout for continuous flow
        )
        print(f"✅ Connected to Kafka topic '{KAFKA_TOPIC}'!")
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return {"status": "error", "message": str(e)}
    
    # Initialize
    init_db()
    centroids_state = load_centroids()
    
    print(f"🚀 Consumer Started - Waiting for messages...")
    
    processed = 0
    trends_updated = set()
    
    try:
        for message in consumer:
            post = message.value
            
            # Assign to trend
            trend_name, similarity = assign_topic(
                post.get('content', ''),
                post.get('final_topic', ''),
                centroids_state
            )
            
            # Update database
            update_trend_in_db(trend_name, post, similarity)
            
            processed += 1
            trends_updated.add(trend_name)
            
            if processed % 50 == 0:
                print(f"📊 Processed: {processed} posts | Trends: {len(trends_updated)}")
        
        print(f"\n✅ Consumer Complete: {processed} posts processed, {len(trends_updated)} trends updated")
        return {
            "status": "success",
            "processed": processed,
            "trends": len(trends_updated)
        }
        
    except KeyboardInterrupt:
        print("\n🛑 Consumer stopped by user")
        return {"status": "interrupted", "processed": processed}
    finally:
        consumer.close()

if __name__ == "__main__":
    result = run_consumer()
    print(f"\n📊 Result: {result}")
