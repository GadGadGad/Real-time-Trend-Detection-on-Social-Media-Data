
import os
import sys
import time
import json
import logging
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.core.llm.llm_refiner import LLMRefiner
except ImportError:
    print("❌ Could not import LLMRefiner. Make sure you are in the correct directory.")
    sys.exit(1)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("IntelligenceWorker")

import math
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

# Database Setup
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
db_engine = create_engine(POSTGRES_URL)

def get_unanalyzed_trends():
    """Fetch trends that need analysis: 
    1. Discovery trends with placeholder summary
    2. Seeded trends that received enough REAL matches (post_count increased)
    """
    query = text("""
        SELECT id, trend_name, representative_posts, post_count 
        FROM detected_trends 
        WHERE (
            -- Discovery trends awaiting analysis
            (summary IS NULL OR summary = 'Waiting for analysis...')
            -- OR Seeded trends that got real matches (need T1-T7 classification)
            OR (category = 'Seeded Topic' AND post_count >= 5)
            OR (category LIKE '%Seeded%' AND post_count >= 5)
        )
        AND post_count >= 3
        ORDER BY post_count DESC
        LIMIT 5
    """)
    with db_engine.connect() as conn:
        result = conn.execute(query).fetchall()
        return result

def update_trend_analysis(trend_id, analysis_result):
    """Update trend with LLM analysis results. Implements Merge-on-Rename logic."""
    new_name = analysis_result[0]
    
    with db_engine.begin() as conn:
        # 1. Check if name already exists (Deduplication)
        existing = conn.execute(
            text("SELECT id, volume, post_count, interactions, representative_posts, google_vol FROM detected_trends WHERE trend_name = :name AND id != :id"),
            {"name": new_name, "id": trend_id}
        ).fetchone()
        
        if existing:
            e_id, e_vol, e_pc, e_inter, e_reps_raw, e_gvol = existing
            logger.info(f"🔄 Merging discovery {trend_id} into existing trend {e_id} ('{new_name}')")
            
            # Fetch current discovery data
            current = conn.execute(
                text("SELECT volume, post_count, interactions, representative_posts, google_vol FROM detected_trends WHERE id = :id"),
                {"id": trend_id}
            ).fetchone()
            
            if current:
                c_vol, c_pc, c_inter, c_reps_raw, c_gvol = current
                
                # Sum stats
                merged_pc = (e_pc or 0) + (c_pc or 0)
                merged_inter = (e_inter or 0) + (c_inter or 0)
                merged_gvol = max((e_gvol or 0), (c_gvol or 0)) # Google vol is usually same for same name
                
                # Merge Rep Posts (Keep unique ones)
                try: e_reps = json.loads(e_reps_raw) if e_reps_raw else []
                except: e_reps = []
                try: c_reps = json.loads(c_reps_raw) if c_reps_raw else []
                except: c_reps = []
                
                # Combine and deduplicate by content
                all_reps = e_reps + c_reps
                seen_content = set()
                merged_reps = []
                for r in all_reps:
                    txt = r.get('content', '')[:100]
                    if txt not in seen_content:
                        merged_reps.append(r)
                        seen_content.add(txt)
                
                merged_reps = merged_reps[:1000] # Limit reps
                
                # Recalculate score
                new_score, g_s, f_s, n_s = calculate_realtime_score(merged_gvol, merged_inter, merged_pc)
                
                # Update Existing
                conn.execute(text("""
                    UPDATE detected_trends SET 
                        volume = :vol, post_count = :pc, interactions = :inter,
                        representative_posts = :reps, trend_score = :score,
                        score_n = :sn, score_f = :sf,
                        summary = :summ, category = :cat, sentiment = :sent,
                        advice_state = :as, advice_business = :ab, topic_type = :type,
                        last_updated = :now
                    WHERE id = :e_id
                """), {
                    "vol": merged_gvol + merged_pc, "pc": merged_pc, "inter": merged_inter,
                    "reps": json.dumps(merged_reps, ensure_ascii=False), "score": new_score,
                    "sn": n_s, "sf": f_s, "summ": analysis_result[4], "cat": analysis_result[1],
                    "sent": analysis_result[5], "as": analysis_result[6].get('advice_state', 'N/A'),
                    "ab": analysis_result[6].get('advice_business', 'N/A'), "type": analysis_result[3],
                    "now": datetime.now(), "e_id": e_id
                })
                
                # Delete Discovery Row
                conn.execute(text("DELETE FROM detected_trends WHERE id = :id"), {"id": trend_id})
                logger.info(f"✅ Merged and deleted discovery {trend_id}")
                return
                
        # 2. Sequential Update (Normal path)
        conn.execute(text("""
            UPDATE detected_trends 
            SET 
                trend_name = :new_name,
                summary = :summary,
                category = :category,
                sentiment = :sentiment,
                advice_state = :advice_state,
                advice_business = :advice_business,
                topic_type = :topic_type,
                last_updated = :now
            WHERE id = :id
        """), {
            "id": trend_id,
            "new_name": analysis_result[0],
            "category": analysis_result[1],
            "topic_type": analysis_result[3],
            "summary": analysis_result[4],
            "sentiment": analysis_result[5],
            "advice_state": analysis_result[6].get('advice_state', 'N/A'),
            "advice_business": analysis_result[6].get('advice_business', 'N/A'),
            "now": datetime.now()
        })
        logger.info(f"✅ Updated Trend {trend_id}: {analysis_result[0]}")

def main():
    load_dotenv()
    
    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠️ GEMINI_API_KEY not found. Worker will fail to initialize LLM.")
    
    logger.info("🧠 Initializing Intelligence Worker (Slow Path)...")
    try:
        # Initialize Refiner (High Capacity mode for demo speed)
        refiner = LLMRefiner(provider="gemini", api_key=api_key, debug=True)
        if not refiner.enabled:
            logger.error("❌ Refiner disabled (missing key or library). Exiting.")
            return
    except Exception as e:
        logger.error(f"❌ Failed to init Refiner: {e}")
        return

    logger.info("🚀 Worker Started. Polling for unanalyzed trends...")
    
    while True:
        try:
            trends = get_unanalyzed_trends()
            
            if not trends:
                logger.debug("💤 No suitable trends found. Sleeping...")
                time.sleep(5)
                continue
                
            logger.info(f"🔍 Found {len(trends)} trends to analyze.")
            
            for trend in trends:
                t_id, t_name, t_posts_json, t_vol = trend
                
                try:
                    posts = json.loads(t_posts_json) if t_posts_json else []
                    if not posts:
                        continue
                        
                    logger.info(f"⚡ Analyzing Trend {t_id}: {t_name} ({len(posts)} posts)...")
                    
                    # Call LLM
                    analysis_result = refiner.refine_cluster(
                        cluster_name=t_name,
                        posts=posts,
                        original_category="Unclassified",
                        topic_type="Discovery"
                    )
                    
                    # Update DB
                    update_trend_analysis(t_id, analysis_result)
                    
                except Exception as e:
                    logger.error(f"⚠️ Error analyzing trend {t_id}: {e}")
                    
            time.sleep(2) # Short rest between batches
            
        except KeyboardInterrupt:
            logger.info("🛑 Worker stopped by user.")
            break
        except Exception as e:
            logger.error(f"🔥 Critical Worker Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    main()
