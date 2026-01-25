"""
Kafka Consumer for Batch+Streaming Pipeline
=============================================
Consumes posts from Kafka and assigns to trends using embedding similarity.
Ported from demo-ready/consumer.py for accurate matching and trend discovery.
"""

import json
import os
import sys
import time
import math
import numpy as np
import argparse
from datetime import datetime
from kafka import KafkaConsumer
from sqlalchemy import create_engine, text
from sklearn.metrics.pairwise import cosine_similarity

# Add project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# --- CONFIG ---
KAFKA_TOPIC = 'posts_stream_v1'
BOOTSTRAP_SERVERS = ['localhost:29092']
POSTGRES_URL = "postgresql://user:password@localhost:5432/trend_db"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"
THRESHOLD = 0.65  # Minimum similarity to match existing trend
MIN_CLUSTER_SIZE = 3  # For HDBSCAN discovery
CLUSTER_EPSILON = 0.5
COHERENCE_THRESHOLD = 0.75
BATCH_SIZE = 50  # Process in micro-batches

# Database engine
db_engine = create_engine(POSTGRES_URL)

# --- LAZY MODEL LOADING ---
_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer
        device = "cpu"
        print(f"📥 Loading Embedding Model '{MODEL_NAME}'...")
        _embedding_model = SentenceTransformer(
            MODEL_NAME,
            device=device,
            trust_remote_code=True
        )
        print("✅ Embedding Model Loaded.")
    return _embedding_model

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

def load_active_trends():
    """Load all trends with embeddings from DB"""
    trends = []
    try:
        with db_engine.connect() as conn:
            result = conn.execute(text(
                "SELECT id, trend_name, embedding, post_count, trend_score, keywords, google_vol, interactions, representative_posts FROM detected_trends"
            ))
            for row in result:
                # Parse embedding (may be null for seeded trends without embedding)
                emb = None
                if row[2]:
                    try:
                        emb = np.array(json.loads(row[2]))
                        # Validate: must be proper vector, not NaN
                        if emb.ndim == 1 and len(emb) > 0 and not np.isnan(emb).any():
                            norm = np.linalg.norm(emb)
                            if norm > 0: emb = emb / norm
                        else:
                            emb = None  # Invalid embedding
                    except:
                        emb = None
                
                # Parse keywords (comma-separated string)
                kws = []
                if row[5]:
                    kws = [k.strip() for k in row[5].split(",") if k.strip()]
                
                trends.append({
                    "id": row[0],
                    "name": row[1],
                    "embedding": emb,
                    "post_count": row[3] or 0,
                    "trend_score": row[4] or 0,
                    "keywords": kws,
                    "google_vol": row[6] or 0,
                    "interactions": row[7] or 0,
                    "representative_posts": row[8]
                })
    except Exception as e:
        print(f"⚠️ Error loading trends: {e}")
    return trends

def calculate_realtime_score(g_vol, interactions, post_count):
    """Aligns with src.pipeline.trend_scoring heuristics"""
    MAX_VOL = 20000000
    g_score = (math.log10(g_vol + 1) / math.log10(MAX_VOL + 1)) * 100 if g_vol > 0 else 0
    g_score = min(100, g_score)

    MAX_INTERACTIONS = 100000
    f_score = (math.log10(interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
    f_score = min(100, f_score)

    MAX_ARTICLES = 10
    n_score = (math.log10(post_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if post_count > 0 else 0
    n_score = min(100, n_score)

    active_sources = sum([g_score > 10, f_score > 10, n_score > 10])
    synergy_mult = 1.2 if active_sources == 3 else (1.1 if active_sources == 2 else 1.0)

    base_score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
    final_score = min(100, base_score * synergy_mult)
    return round(final_score, 1), 0.0, 0.0

def process_batch(posts_batch, active_trends, model):
    """Process a batch of posts - prioritize pre-assigned topics from producer"""
    if not posts_batch:
        return 0, 0
    
    matches = 0
    new_trends = 0
    unmatched_posts = []
    unmatched_indices = []
    
    # Build trend lookup by name
    trend_by_name = {t['name'].lower(): t for t in active_trends}
    
    # PHASE 1: Use pre-assigned final_topic from producer (if available)
    for i, post in enumerate(posts_batch):
        final_topic = post.get('final_topic', '')
        
        # Skip if no topic or it's a noise/unassigned marker
        if not final_topic or final_topic in ('Unknown', 'Unassigned', 'N/A'):
            unmatched_posts.append(post)
            unmatched_indices.append(i)
            continue
        
        # Handle "New:" prefix topics - strip and check
        topic_key = final_topic.lower()
        if topic_key.startswith('new:'):
            topic_key = topic_key[4:].strip()
        
        # Match to existing trend
        trend = trend_by_name.get(topic_key)
        if trend:
            _update_trend_in_db(trend, post, 0.95)  # High similarity - pre-assigned
            matches += 1
        else:
            # Create new trend from pre-assigned topic
            _create_trend_from_topic(final_topic, post, active_trends)
            new_trends += 1
    
    # PHASE 2: Embedding matching for truly unmatched posts (no pre-assigned topic)
    if unmatched_posts and len(unmatched_posts) >= MIN_CLUSTER_SIZE:
        contents = [p['content'] for p in unmatched_posts]
        sources = [p['source'] for p in unmatched_posts]
        times = [p['time'] for p in unmatched_posts]
        
        post_embeddings = model.encode(contents, show_progress_bar=False)
        norms = np.linalg.norm(post_embeddings, axis=1, keepdims=True)
        post_embeddings = post_embeddings / np.where(norms > 0, norms, 1.0)
        
        # 2. Build trend embeddings matrix
        trends_with_emb = [t for t in active_trends if t['embedding'] is not None]
        
        cluster_unmatched_idx = []
        
        if trends_with_emb:
            trend_embeddings = np.array([t['embedding'] for t in trends_with_emb])
            sims = cosine_similarity(post_embeddings, trend_embeddings)
            
            for i in range(len(unmatched_posts)):
                best_idx = np.argmax(sims[i])
                best_sim = sims[i][best_idx]
                
                if best_sim >= THRESHOLD:
                    trend = trends_with_emb[best_idx]
                    _update_trend_in_db(trend, unmatched_posts[i], float(best_sim))
                    matches += 1
                else:
                    cluster_unmatched_idx.append(i)
        else:
            cluster_unmatched_idx = list(range(len(unmatched_posts)))
    
    # 3. Cluster truly unmatched posts for discovery (after embedding matching)
    if 'cluster_unmatched_idx' in dir() and len(cluster_unmatched_idx) >= MIN_CLUSTER_SIZE:
        try:
            import hdbscan
            # Use the embeddings that were computed in Phase 2 for unmatched posts
            unmatched_embs = post_embeddings[cluster_unmatched_idx]
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=MIN_CLUSTER_SIZE, 
                metric='euclidean', 
                cluster_selection_epsilon=CLUSTER_EPSILON
            )
            labels = clusterer.fit_predict(unmatched_embs)
            unique_labels = set(labels) - {-1}
            
            for label in unique_labels:
                member_local_idx = [j for j, l in enumerate(labels) if l == label]
                if len(member_local_idx) < MIN_CLUSTER_SIZE:
                    continue
                
                # Compute centroid from local embeddings
                cluster_local_embs = unmatched_embs[member_local_idx]
                centroid = np.mean(cluster_local_embs, axis=0).reshape(1, -1)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0: centroid = centroid / c_norm
                
                # Coherence check
                dists = cosine_similarity(cluster_local_embs, centroid).flatten()
                valid_mask = dists >= THRESHOLD
                filtered_local_idx = [member_local_idx[k] for k, valid in enumerate(valid_mask) if valid]
                
                if len(filtered_local_idx) < MIN_CLUSTER_SIZE:
                    continue
                
                # Recalculate centroid for filtered cluster
                filtered_embs = unmatched_embs[filtered_local_idx]
                centroid = np.mean(filtered_embs, axis=0).reshape(1, -1)
                c_norm = np.linalg.norm(centroid)
                if c_norm > 0: centroid = centroid / c_norm
                
                dists = cosine_similarity(filtered_embs, centroid).flatten()
                mean_coherence = np.mean(dists)
                
                if mean_coherence < COHERENCE_THRESHOLD:
                    continue
                
                # Create new trend - use original indices to get content
                # Map back from local idx to outer unmatched_posts
                best_rep_local = np.argmax(dists)
                outer_idx = cluster_unmatched_idx[filtered_local_idx[best_rep_local]]
                sample_content = contents[outer_idx]
                new_trend_name = "New: " + (sample_content[:40] + "..." if len(sample_content) > 40 else sample_content)
                
                # Build representative posts from unmatched_posts
                rep_posts = []
                for k, local_idx in enumerate(filtered_local_idx[:100]):
                    outer_i = cluster_unmatched_idx[local_idx]
                    rep_posts.append({
                        "content": contents[outer_i],
                        "source": sources[outer_i],
                        "time": times[outer_i],
                        "similarity": float(dists[k] if k < len(dists) else 0.8)
                    })
                
                # Insert new trend
                _insert_new_trend(new_trend_name, rep_posts, centroid.flatten().tolist(), [p['source'] for p in rep_posts])
                new_trends += 1
                
        except ImportError:
            pass
    
    return matches, new_trends

def _update_trend_in_db(trend, post, similarity):
    """Update existing trend with new post"""
    with db_engine.begin() as conn:
        new_post_count = trend['post_count'] + 1
        new_interactions = trend['interactions'] + 10
        new_score, _, _ = calculate_realtime_score(
            trend['google_vol'], new_interactions, new_post_count
        )
        
        # Parse existing reps
        try:
            reps = json.loads(trend['representative_posts']) if trend['representative_posts'] else []
        except:
            reps = []
        
        new_rep = {
            "content": post['content'][:300],
            "source": post['source'],
            "time": post['time'],
            "similarity": similarity
        }
        reps = [new_rep] + reps[:999]
        
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
            "count": new_post_count,
            "vol": trend['google_vol'] + new_post_count,
            "inter": new_interactions,
            "score": new_score,
            "sn": 0.0,
            "sf": 0.0,
            "reps": json.dumps(reps, ensure_ascii=False),
            "now": datetime.now(),
            "id": trend['id']
        })


def _create_trend_from_topic(topic_name, post, active_trends):
    """Create a new trend from a pre-assigned topic name"""
    rep_posts = [{
        "content": post['content'][:300],
        "source": post['source'],
        "time": post['time'],
        "similarity": 0.95
    }]
    sources = [post['source']]
    _insert_new_trend(topic_name, rep_posts, None, sources)

def _insert_new_trend(name, rep_posts, centroid, sources):
    """Insert a newly discovered trend"""
    with db_engine.begin() as conn:
        # Check if exists
        exists = conn.execute(text("SELECT id FROM detected_trends WHERE trend_name = :name"), {"name": name}).fetchone()
        if exists:
            return
        
        is_news = any(s.lower() in ['vnexpress', 'tuoitre', 'thanhnien', 'news'] for s in sources)
        topic_type = 'News' if is_news else 'Social'
        
        init_inter = len(rep_posts) * 10
        proper_score, _, _ = calculate_realtime_score(0, init_inter, len(rep_posts))
        
        conn.execute(text("""
            INSERT INTO detected_trends (
                trend_name, trend_score, volume, post_count,
                representative_posts, topic_type, score_n, score_f,
                created_at, last_updated, embedding,
                summary, category, sentiment, google_vol, interactions, keywords
            )
            VALUES (:name, :score, :vol, :vol, :rep, :type, :sn, :sf, :now, :now, :emb, :summ, :cat, :sent, 0, :inter, :kws)
        """), {
            "name": name,
            "score": proper_score,
            "vol": len(rep_posts),
            "rep": json.dumps(rep_posts, ensure_ascii=False),
            "type": topic_type,
            "sn": 0.0,
            "sf": 0.0,
            "now": datetime.now(),
            "emb": json.dumps(centroid),
            "summ": "Waiting for analysis...",
            "cat": "Unclassified",
            "sent": "Neutral",
            "inter": init_inter,
            "kws": json.dumps([], ensure_ascii=False)
        })
        print(f"   🌟 DISCOVERED: {name} ({len(rep_posts)} posts)")

def run_consumer(max_messages=None, timeout=None):
    """Consume from Kafka and process using embedding matching"""
    print(f"🔄 Connecting to Kafka at {BOOTSTRAP_SERVERS}...")
    
    try:
        consumer = KafkaConsumer(
            KAFKA_TOPIC,
            bootstrap_servers=BOOTSTRAP_SERVERS,
            value_deserializer=lambda x: json.loads(x.decode('utf-8')),
            auto_offset_reset='earliest',
            consumer_timeout_ms=timeout * 1000 if timeout else float('inf')
        )
        print(f"✅ Connected to Kafka topic '{KAFKA_TOPIC}'!")
    except Exception as e:
        print(f"❌ Failed to connect to Kafka: {e}")
        return {"status": "error", "message": str(e)}
    
    # Initialize
    init_db()
    model = get_embedding_model()
    
    print(f"🚀 Consumer Started - Waiting for messages...")
    
    processed = 0
    total_matches = 0
    total_new = 0
    posts_buffer = []
    
    try:
        for message in consumer:
            post = message.value
            posts_buffer.append(post)
            processed += 1
            
            # Process in batches
            if len(posts_buffer) >= BATCH_SIZE:
                active_trends = load_active_trends()
                matches, new_trends = process_batch(posts_buffer, active_trends, model)
                total_matches += matches
                total_new += new_trends
                print(f"📊 Processed: {processed} | Matched: {total_matches} | New Trends: {total_new}")
                posts_buffer = []
            
            if max_messages and processed >= max_messages:
                break
        
        # Process remaining
        if posts_buffer:
            active_trends = load_active_trends()
            matches, new_trends = process_batch(posts_buffer, active_trends, model)
            total_matches += matches
            total_new += new_trends
        
        print(f"\n✅ Consumer Complete: {processed} posts, {total_matches} matched, {total_new} new trends")
        return {"status": "success", "processed": processed, "matched": total_matches, "new_trends": total_new}
        
    except KeyboardInterrupt:
        print("\n🛑 Consumer stopped by user")
        return {"status": "interrupted", "processed": processed}
    finally:
        consumer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kafka Consumer for Trends (Embedding Mode)")
    parser.add_argument("--max-messages", type=int, help="Stop after processing N messages", default=None)
    parser.add_argument("--timeout", type=int, help="Stop if no message received for N seconds", default=None)
    args = parser.parse_args()

    result = run_consumer(max_messages=args.max_messages, timeout=args.timeout)
    print(f"\n📊 Result: {result}")
