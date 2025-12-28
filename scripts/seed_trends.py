
import os
import sys
import json
import glob
import numpy as np
from datetime import datetime
from sqlalchemy import create_engine, text

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from src.pipeline.main_pipeline import load_trends, refine_trends_preprocessing

# Config
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
DEMO_STATE_DIR = os.path.join(PROJECT_ROOT, "demo", "demo_data_taxonomy_trained")
TRENDS_DIR = os.path.join(PROJECT_ROOT, "data", "trendings")
if not os.path.exists(TRENDS_DIR):
    TRENDS_DIR = os.path.join(PROJECT_ROOT, "streaming", "data")


import math

def load_precomputed_embeddings():
    """Load pre-computed trend embeddings from demo state"""
    try:
        emb_path = os.path.join(DEMO_STATE_DIR, "trend_embeddings.npy")
        trends_path = os.path.join(DEMO_STATE_DIR, "trends.json")
        
        if os.path.exists(emb_path) and os.path.exists(trends_path):
            embeddings = np.load(emb_path)
            with open(trends_path, 'r', encoding='utf-8') as f:
                trends_dict = json.load(f)
            
            # Map trend name to embedding
            trend_names = list(trends_dict.keys())
            emb_map = {}
            for i, name in enumerate(trend_names):
                if i < len(embeddings):
                    # Normalize
                    emb = embeddings[i]
                    norm = np.linalg.norm(emb)
                    if norm > 0:
                        emb = emb / norm
                    emb_map[name.lower()] = emb.tolist()
            print(f"📦 [Seeder] Loaded {len(emb_map)} pre-computed embeddings from demo state.")
            return emb_map
    except Exception as e:
        print(f"⚠️ [Seeder] Failed to load pre-computed embeddings: {e}")
    return {}

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
    return round(final_score, 1), round(g_score, 1), round(f_score, 1), round(n_score, 1)

def seed_database():
    print(f"🌱 [Seeder] Starting trend seeding...")
    
    # 1. Database Setup
    engine = create_engine(POSTGRES_URL)
    
    # Ensure table exists (idempotent)
    with engine.connect() as conn:
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

    # 2. Load Trends
    # Look for trends.json FIRST as requested by user
    json_path = os.path.join(PROJECT_ROOT, "crawlers", "trendings", "trends.json")
    if os.path.exists(json_path):
        print(f"📄 [Seeder] Found primary JSON source: {json_path}")
        trend_files = [json_path]
    else:
        # Fallback to CSVs
        trend_files = glob.glob(os.path.join(TRENDS_DIR, "*.csv"))
        if not trend_files:
            trend_files = glob.glob(os.path.join(PROJECT_ROOT, "input", "se363-final-dataset", "trendings", "*.csv"))
    
    if not trend_files:
        print(f"⚠️ [Seeder] No trend files found (checked JSON and CSVs). Skipping seeding.")
        return

    print(f"📦 [Seeder] using sources: {trend_files}")
    
    # Load and Refine
    # Note: We disable expensive LLM refinement for seeding to be fast, relying on the robust Regex/Heuristics we added.
    raw_trends = load_trends(trend_files)
    refined_trends = refine_trends_preprocessing(
        raw_trends, 
        llm_provider="gemini", # Unused if refine=False/Use_LLM=False? 
        gemini_api_key=None, 
        llm_model_path=None, 
        debug_llm=False,
        use_llm=False # KEY: Fast seeding, only regex filtering
    )
    
    print(f"✨ [Seeder] Loaded {len(raw_trends)} raw -> {len(refined_trends)} refined trends.")
    
    # 3. Load Pre-computed Embeddings
    precomputed_emb_map = load_precomputed_embeddings()
    
    # 4. Insert into DB
    with engine.begin() as conn:
        inserted = 0
        skipped_no_emb = 0
        for name, data in refined_trends.items():
            # Check existence
            exists = conn.execute(text("SELECT 1 FROM detected_trends WHERE trend_name = :name"), {"name": name}).fetchone()
            if exists:
                continue
            
            # FIX: Ensure name is readable. If it looks like a hash (len > 30, no spaces), use first keyword
            display_name = name
            if len(name) > 30 and ' ' not in name:
                kws = data.get('keywords', [])
                if kws: display_name = kws[0]

            # FIX: Generate a "System" representative post so Dashboard doesn't show "No Content"
            kws_list = data.get('keywords', [])[:5]
            kws_str = ", ".join(kws_list)
            synth_post = {
                "source": "Google Trends (System)",
                "content": f"Chủ đề được khởi tạo từ dữ liệu xu hướng tìm kiếm. Từ khóa liên quan: {kws_str}. Đang chờ dữ liệu thực tế từ mạng xã hội...",
                "time": datetime.now().isoformat(),
                "similarity": 1.0
            }
            reps_json = json.dumps([synth_post], ensure_ascii=False)

            # Calculate Initial Score
            init_vol = min(int(data.get('volume', 1000)), 2000000000)
            init_score, g_s, f_s, n_s = calculate_realtime_score(init_vol, 0, 0)
            
            # Lookup Pre-computed Embedding (fall back to None)
            emb = precomputed_emb_map.get(display_name.lower())
            emb_json = json.dumps(emb) if emb else None

            # Insert 'Seeded' trend with embedding
            conn.execute(text("""
                INSERT INTO detected_trends (
                    trend_name, trend_score, volume, post_count,
                    representative_posts, topic_type, score_n, score_f,
                    created_at, last_updated, summary, category, sentiment, google_vol, keywords, interactions, embedding
                ) VALUES (
                    :name, :score, :vol, 0, :reps, 'Seeded', :sn, :sf,
                    :now, :now, 'Seeded from Google Trends (Waiting for new posts)', 'Seeded Topic', 'Neutral', :vol, :kws, 0, :emb
                )
            """), {
                "name": display_name,
                "score": init_score,
                "vol": init_vol,
                "reps": reps_json,
                "sn": n_s,
                "sf": f_s,
                "now": datetime.now(),
                "kws": kws_str,
                "emb": emb_json
            })
            inserted += 1
            if inserted % 10 == 0:
                print(f"   📊 Seeded {inserted} trends...")
            
    print(f"✅ [Seeder] Successfully seeded {inserted} new trends to DB.")

if __name__ == "__main__":
    seed_database()
