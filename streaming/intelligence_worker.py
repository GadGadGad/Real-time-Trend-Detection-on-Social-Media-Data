
import os
import sys
import time
import json
import logging
import argparse # Added for finite execution
from datetime import datetime
from sqlalchemy import create_engine, text
from dotenv import load_dotenv

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.core.llm.llm_refiner import LLMRefiner
    from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
    from src.core.analysis.sentiment import batch_analyze_sentiment, get_analyzer
except ImportError:
    print("❌ Could not import core modules. Make sure you are in the correct directory.")
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

    MAX_ARTICLES = 10
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
# Database Setup
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
db_engine = create_engine(POSTGRES_URL)

MODEL_NAME = "keepitreal/vietnamese-sbert"

def get_unanalyzed_trends():
    """Fetch trends that need analysis."""
    query = text("""
        SELECT id, trend_name, representative_posts, post_count 
        FROM detected_trends 
        WHERE (
            -- Discovery trends awaiting analysis
            (summary IS NULL OR summary = 'Waiting for analysis...' OR summary = 'Auto-detected trend')
            -- OR Seeded trends that got real matches (need T1-T7 classification)
            OR (category = 'Seeded Topic' AND post_count >= 5)
            OR (category LIKE '%Seeded%' AND post_count >= 5)
        )
        AND post_count >= 3
        ORDER BY post_count DESC
        LIMIT 15
    """)
    with db_engine.connect() as conn:
        result = conn.execute(query).fetchall()
        return result

def update_trend_analysis(trend_id, analysis_result, model=None):
    """Update trend with LLM analysis results + Re-embedding."""
    new_name = analysis_result[0]
    import numpy as np
    
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
                merged_gvol = max((e_gvol or 0), (c_gvol or 0)) 
                
                # Merge Rep Posts
                try: e_reps = json.loads(e_reps_raw) if e_reps_raw else []
                except: e_reps = []
                try: c_reps = json.loads(c_reps_raw) if c_reps_raw else []
                except: c_reps = []
                
                all_reps = (e_reps + c_reps)[:1000]
                
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
                        reasoning = :reasoning,
                        last_updated = :now
                    WHERE id = :e_id
                """), {
                    "vol": merged_gvol + merged_pc, "pc": merged_pc, "inter": merged_inter,
                    "reps": json.dumps(all_reps, ensure_ascii=False), "score": new_score,
                    "sn": n_s, "sf": f_s, "summ": analysis_result[4], "cat": analysis_result[1],
                    "sent": analysis_result[5], "as": analysis_result[6].get('advice_state', 'N/A'),
                    "ab": analysis_result[6].get('advice_business', 'N/A'), "type": analysis_result[3],
                    "reasoning": analysis_result[2],
                    "now": datetime.now(), "e_id": e_id
                })
                
                # Delete Discovery Row
                conn.execute(text("DELETE FROM detected_trends WHERE id = :id"), {"id": trend_id})
                logger.info(f"✅ Merged and deleted discovery {trend_id}")
                return
                
        # 2. Sequential Update (Normal path) + Re-embedding
        new_emb_json = None
        if model:
            # Re-embed if name is valid
            try:
                emb = model.encode(new_name)
                norm = np.linalg.norm(emb)
                if norm > 0: emb = emb / norm
                new_emb_json = json.dumps(emb.tolist())
                logger.info(f"🧠 Re-embedded new name: '{new_name}'")
            except Exception as e:
                logger.error(f"⚠️ Embedding failed: {e}")

        update_query = """
            UPDATE detected_trends 
            SET 
                trend_name = :new_name,
                summary = :summary,
                category = :category,
                sentiment = :sentiment,
                advice_state = :advice_state,
                advice_business = :advice_business,
                topic_type = :topic_type,
                reasoning = :reasoning,
                last_updated = :now
        """
        params = {
            "id": trend_id,
            "new_name": analysis_result[0],
            "category": analysis_result[1],
            "topic_type": analysis_result[3],
            "summary": analysis_result[4],
            "sentiment": analysis_result[5],
            "advice_state": analysis_result[6].get('advice_state', 'N/A'),
            "advice_business": analysis_result[6].get('advice_business', 'N/A'),
            "reasoning": analysis_result[2],
            "now": datetime.now()
        }
        
        if new_emb_json:
            update_query += ", embedding = :emb "
            params["emb"] = new_emb_json
            
        update_query += " WHERE id = :id "
        
        conn.execute(text(update_query), params)
        logger.info(f"✅ Updated Trend {trend_id}: {analysis_result[0]}")

def main(max_cycles=None):
    load_dotenv()
    
    # Check API Key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        logger.warning("⚠️ GEMINI_API_KEY not found. Worker will fail to initialize LLM.")
    
    logger.info("🧠 Initializing Intelligence Worker (Slow Path)...")
    
    # Load Embedding Model
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        logger.info("📥 Loading Embedding Model (this may take a moment)...")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        logger.error(f"⚠️ Failed to load embedding model: {e}")

    # --- CLASSIFIERS SETUP (Local Mode vs LLM Mode) ---
    use_llm_only = os.getenv("USE_LLM_ONLY", "false").lower() == "true"
    tax_clf = None
    if not use_llm_only:
        try:
            logger.info("🧠 Loading Local Classifiers (Taxonomy & Sentiment)...")
            tax_clf = TaxonomyClassifier(embedding_model=model) # Uses model from above
            get_analyzer() # Pre-load sentiment
        except Exception as e:
            logger.error(f"⚠️ Failed to load local classifiers: {e}")
            tax_clf = None
    else:
        logger.info("🤖 Running in LLM-ONLY Mode (Skipping local BERT classifiers)")

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
    
    cycles = 0
    while True:
        try:
            trends = get_unanalyzed_trends()
            
            if not trends:
                logger.debug("💤 No suitable trends found. Sleeping...")
                time.sleep(60)
                # Still count as a cycle? Maybe only count active cycles? 
                # Let's count every loop to be safe if user wants strictly limited time.
                cycles += 1
                if max_cycles and cycles >= max_cycles:
                    logger.info(f"🛑 Reached limit of {max_cycles} cycles.")
                    break
                continue
                
            logger.info(f"🔍 Found {len(trends)} trends to analyze.")
            
            for trend in trends:
                t_id, t_name, t_posts_json, t_vol = trend
                
                try:
                    posts = json.loads(t_posts_json) if t_posts_json else []
                    if not posts:
                        continue
                        
                    logger.info(f"⚡ Analyzing Trend {t_id}: {t_name} ({len(posts)} posts)...")
                    
                    # --- PRE-FILTER: Cheaper Heuristics before Expensive LLM ---
                    # Define inline instead of importing (main_pipeline uses local vars)
                    import re
                    import unicodedata
                    _garbage_list = {'cf', 'fo4', 'sou', 'scl', 'hit', 'vic', 'xoi', 'vck'}
                    def _normalize_text(t): return unicodedata.normalize('NFC', t.lower().strip())
                    def _mostly_numeric(t): return sum(c.isdigit() for c in t) / max(len(t), 1) > 0.6
                    def _too_many_symbols(t): return sum(not c.isalnum() and not c.isspace() for c in t) / max(len(t), 1) > 0.4
                    def _bad_vs_pattern(t): return bool(re.match(r'^[a-z]{2,4}\s+vs\s+[a-z]{2,4}$', t, re.I))
                    
                    norm_name = _normalize_text(t_name)
                    is_garbage = False
                    
                    if norm_name in _garbage_list: is_garbage = True
                    elif _mostly_numeric(norm_name): is_garbage = True
                    elif _too_many_symbols(t_name): is_garbage = True
                    elif _bad_vs_pattern(norm_name): is_garbage = True
                    
                    if is_garbage:
                         logger.warning(f"🗑️ Skipping Garbage Trend: '{t_name}'")
                         # Mark as ignored in DB so we don't pick it up again
                         with db_engine.begin() as conn:
                             conn.execute(text("UPDATE detected_trends SET summary = 'Ignored: Garbage Filter', category = 'Ignored' WHERE id = :id"), {"id": t_id})
                         continue
                    # -----------------------------------------------------------

                    
                    # --- LOCAL CLASSIFICATION (Optionally Override LLM) ---
                    local_cat = "Unclassified"
                    local_sent = "Neutral"
                    
                    if not use_llm_only and tax_clf:
                        try:
                            # Taxonomy on Name
                            local_cat, _ = tax_clf.classify(t_name)
                            
                            # Sentiment on Posts (Batch)
                            if posts:
                                contents = [p.get('content', '') for p in posts[:10]]
                                sents = batch_analyze_sentiment(contents)
                                from collections import Counter
                                if sents:
                                    local_sent = Counter(sents).most_common(1)[0][0]
                        except Exception as e:
                            logger.error(f"⚠️ Local classification failed: {e}")

                    # Call LLM
                    # We pass the local category as a hint/constraint
                    analysis_result = refiner.refine_cluster(
                        cluster_name=t_name,
                        posts=posts,
                        original_category=local_cat if local_cat != "Unclassified" else "Unclassified",
                        topic_type="Discovery"
                    )
                    
                    # Override if Local Models used (User Requirement)
                    if not use_llm_only and tax_clf:
                        # Unpack tuple to list
                        res_list = list(analysis_result)
                        # 1: Category, 5: Sentiment
                        if local_cat != "Unclassified": res_list[1] = local_cat
                        res_list[5] = local_sent
                        analysis_result = tuple(res_list)
                    
                    # Update DB (with model for embeddings)
                    update_trend_analysis(t_id, analysis_result, model=model)
                    
                except Exception as e:
                    logger.error(f"⚠️ Error analyzing trend {t_id}: {e}")
                    
            cycles += 1
            if max_cycles and cycles >= max_cycles:
                logger.info(f"🛑 Reached limit of {max_cycles} cycles.")
                break

            time.sleep(60) # Rest between batches
            
        except KeyboardInterrupt:
            logger.info("🛑 Worker stopped by user.")
            break
        except Exception as e:
            logger.error(f"🔥 Critical Worker Error: {e}")
            time.sleep(5)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI Worker for Trend Analysis")
    parser.add_argument("--max-cycles", type=int, help="Stop after N polling cycles", default=None)
    args = parser.parse_args()
    
    main(max_cycles=args.max_cycles)
