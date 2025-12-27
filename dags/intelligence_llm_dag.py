"""
Intelligence LLM DAG
====================
Airflow-scheduled version of intelligence_worker.py

This DAG runs every 5 minutes to:
1. Query PostgreSQL for unanalyzed trends
2. Call Gemini LLM API to analyze each trend
3. Update database with summary, category, sentiment, advice

Architecture:
  consumer.py (Fast Path, continuous) → writes raw trends to DB
  ↓
  [PostgreSQL - detected_trends table]
  ↓
  This DAG (Slow Path, scheduled) → enriches with LLM analysis
  ↓
  dashboard.py (reads enriched data)
"""

import os
import sys
import json
import logging
from datetime import datetime, timedelta
import numpy as np

from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.models import Variable

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
POSTGRES_URL = os.getenv(
    "POSTGRES_URL", 
    "postgresql://user:password@localhost:5432/trend_db"
)

# Number of trends to process per DAG run (to avoid timeout)
MAX_TRENDS_PER_RUN = 15

# Default embedding model
MODEL_NAME = "keepitreal/vietnamese-sbert"

# ---------------------------------------------------------------------------
# Task Functions
# ---------------------------------------------------------------------------

def check_api_key(**context):
    """Check if Gemini API key is available."""
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        # Try Airflow Variable
        try:
            api_key = Variable.get("GEMINI_API_KEY", default_var=None)
        except:
            pass
    
    if not api_key:
        raise ValueError("❌ GEMINI_API_KEY not found. Set as environment variable or Airflow Variable.")
    
    print(f"✅ GEMINI_API_KEY found (length: {len(api_key)})")
    return True


def fetch_unanalyzed_trends(**context):
    """Fetch trends that need LLM analysis from PostgreSQL."""
    from sqlalchemy import create_engine, text
    
    engine = create_engine(POSTGRES_URL)
    
    query = text("""
        SELECT id, trend_name, representative_posts, post_count 
        FROM detected_trends 
        WHERE (
            -- Discovery trends awaiting analysis
            (summary IS NULL OR summary = 'Waiting for analysis...')
            -- OR Seeded trends that got real matches
            OR (category = 'Seeded Topic' AND post_count >= 5)
            OR (category LIKE '%Seeded%' AND post_count >= 5)
        )
        AND post_count >= 3
        ORDER BY post_count DESC
        LIMIT :limit
    """)
    
    with engine.connect() as conn:
        result = conn.execute(query, {"limit": MAX_TRENDS_PER_RUN}).fetchall()
        
    trends = []
    for row in result:
        trends.append({
            "id": row[0],
            "trend_name": row[1],
            "representative_posts": row[2],
            "post_count": row[3]
        })
    
    print(f"📊 Found {len(trends)} trends to analyze")
    for t in trends:
        print(f"  - ID {t['id']}: {t['trend_name'][:50]}... ({t['post_count']} posts)")
    
    # Push to XCom for next task
    context['ti'].xcom_push(key='trends_to_analyze', value=trends)
    return len(trends)


def analyze_with_llm(**context):
    """Analyze trends using Gemini LLM."""
    from dotenv import load_dotenv
    from sqlalchemy import create_engine, text
    import math
    
    load_dotenv()
    
    # Get trends from previous task
    trends = context['ti'].xcom_pull(key='trends_to_analyze', task_ids='fetch_unanalyzed_trends')
    
    if not trends:
        print("💤 No trends to analyze. Skipping.")
        return {"analyzed": 0, "skipped": 0, "errors": 0}
    
    # Get API key
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = Variable.get("GEMINI_API_KEY", default_var=None)
        except:
            pass
    
    # Initialize LLM Refiner
    try:
        from src.core.llm.llm_refiner import LLMRefiner
        refiner = LLMRefiner(provider="gemini", api_key=api_key, debug=True)
        if not refiner.enabled:
            print("⚠️ LLM Refiner not enabled. Using mock analysis.")
            refiner = None
    except ImportError as e:
        print(f"⚠️ Could not import LLMRefiner: {e}. Using mock analysis.")
        refiner = None
    
    engine = create_engine(POSTGRES_URL)
    
    # Load embedding model for re-embedding refined names
    model = None
    try:
        from sentence_transformers import SentenceTransformer
        print(f"📥 Loading Embedding Model for refinement...")
        model = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        print(f"⚠️ Could not load embedding model in DAG: {e}")

    def calculate_realtime_score(g_vol, interactions, post_count):
        """Calculate trend score."""
        import math
        MAX_VOL = 5010000
        g_score = (math.log10(g_vol + 1) / math.log10(MAX_VOL + 1)) * 100 if g_vol > 0 else 0
        g_score = min(100, g_score)
        
        MAX_INTERACTIONS = 20000
        f_score = (math.log10(interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
        f_score = min(100, f_score)
        
        MAX_ARTICLES = 20
        n_score = (math.log10(post_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if post_count > 0 else 0
        n_score = min(100, n_score)
        
        active_sources = sum([1 for s in [g_score, f_score, n_score] if s > 10])
        synergy_mult = {3: 1.2, 2: 1.1}.get(active_sources, 1.0)
        
        base_score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
        final_score = min(100, base_score * synergy_mult)
        return round(final_score, 1), round(g_score, 1), round(f_score, 1), round(n_score, 1)
    
    stats = {"analyzed": 0, "merged": 0, "skipped": 0, "errors": 0}
    
    for trend in trends:
        t_id = trend['id']
        t_name = trend['trend_name']
        t_posts_json = trend['representative_posts']
        
        try:
            posts = json.loads(t_posts_json) if t_posts_json else []
            if not posts:
                print(f"⏭️ Skipping trend {t_id} (no posts)")
                stats['skipped'] += 1
                continue
            
            print(f"⚡ Analyzing Trend {t_id}: {t_name[:50]}... ({len(posts)} posts)")
            
            if refiner:
                # Real LLM analysis
                analysis_result = refiner.refine_cluster(
                    cluster_name=t_name,
                    posts=posts,
                    original_category="Unclassified",
                    topic_type="Discovery"
                )
            else:
                # Mock analysis (fallback)
                analysis_result = (t_name, "T1", "Reasoning", "Discovery", "Summary", "Neutral", {})
            
            new_name = analysis_result[0]
            
            # --- PHASE 4: DEDUPLICATION / MERGE-ON-RENAME ---
            with engine.begin() as conn:
                # Check if this refined name already exists as another trend
                existing = conn.execute(
                    text("SELECT id, post_count, interactions, representative_posts, google_vol FROM detected_trends WHERE trend_name = :name AND id != :id"),
                    {"name": new_name, "id": t_id}
                ).fetchone()
                
                if existing:
                    e_id, e_pc, e_inter, e_reps_raw, e_gvol = existing
                    print(f"🔄 Merging discovery {t_id} into existing trend {e_id} ('{new_name}')")
                    
                    # Merge data from current trend into existing one
                    current = conn.execute(
                        text("SELECT post_count, interactions, representative_posts, google_vol FROM detected_trends WHERE id = :id"),
                        {"id": t_id}
                    ).fetchone()
                    
                    if current:
                        c_pc, c_inter, c_reps_raw, c_gvol = current
                        merged_pc = (e_pc or 0) + (c_pc or 0)
                        merged_inter = (e_inter or 0) + (c_inter or 0)
                        merged_gvol = max((e_gvol or 0), (c_gvol or 0))
                        
                        # Merge representative posts
                        e_reps = json.loads(e_reps_raw) if e_reps_raw else []
                        c_reps = json.loads(c_reps_raw) if c_reps_raw else []
                        all_reps = (e_reps + c_reps)[:1000]
                        
                        score, gs, fs, ns = calculate_realtime_score(merged_gvol, merged_inter, merged_pc)
                        
                        conn.execute(text("""
                            UPDATE detected_trends SET 
                                post_count = :pc, interactions = :inter,
                                representative_posts = :reps, trend_score = :score,
                                score_n = :sn, score_f = :sf,
                                summary = :summ, category = :cat, sentiment = :sent,
                                advice_state = :as, advice_business = :ab,
                                last_updated = :now
                            WHERE id = :e_id
                        """), {
                            "pc": merged_pc, "inter": merged_inter, "reps": json.dumps(all_reps, ensure_ascii=False),
                            "score": score, "sn": ns, "sf": fs, "summ": analysis_result[4], 
                            "cat": analysis_result[1], "sent": analysis_result[5],
                            "as": analysis_result[6].get('advice_state', 'N/A'),
                            "ab": analysis_result[6].get('advice_business', 'N/A'),
                            "now": datetime.now(), "e_id": e_id
                        })
                        
                        # Delete the redundant trend
                        conn.execute(text("DELETE FROM detected_trends WHERE id = :id"), {"id": t_id})
                        stats['merged'] += 1
                        continue

                # --- REGULAR UPDATE + RE-EMBEDDING ---
                new_emb_json = None
                if model and new_name != t_name:
                    print(f"🧠 Re-embedding refined name: '{new_name}'")
                    new_emb = model.encode(new_name)
                    # Normalize
                    norm = np.linalg.norm(new_emb)
                    if norm > 0: new_emb = new_emb / norm
                    new_emb_json = json.dumps(new_emb.tolist())

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
                    "id": t_id, "new_name": new_name, "category": analysis_result[1],
                    "topic_type": analysis_result[3], "summary": analysis_result[4],
                    "sentiment": analysis_result[5],
                    "advice_state": analysis_result[6].get('advice_state', 'N/A'),
                    "advice_business": analysis_result[6].get('advice_business', 'N/A'),
                    "reasoning": analysis_result[2], "now": datetime.now()
                }
                
                if new_emb_json:
                    update_query += ", embedding = :emb "
                    params["emb"] = new_emb_json
                
                update_query += " WHERE id = :id "
                conn.execute(text(update_query), params)
            
            print(f"✅ Updated Trend {t_id}: {new_name[:50]}...")
            stats['analyzed'] += 1
            
        except Exception as e:
            print(f"❌ Error analyzing trend {t_id}: {e}")
            stats['errors'] += 1
    
    print(f"\n📊 Analysis Complete: {stats}")
    return stats


def log_completion(**context):
    """Log completion and stats."""
    stats = context['ti'].xcom_pull(task_ids='analyze_with_llm')
    trend_count = context['ti'].xcom_pull(task_ids='fetch_unanalyzed_trends')
    
    print("=" * 60)
    print("🧠 INTELLIGENCE LLM DAG COMPLETED")
    print("=" * 60)
    print(f"  Trends found: {trend_count}")
    if stats:
        print(f"  Analyzed: {stats.get('analyzed', 0)}")
        print(f"  Skipped:  {stats.get('skipped', 0)}")
        print(f"  Errors:   {stats.get('errors', 0)}")
    print("=" * 60)
    print("📺 View results in Streamlit dashboard:")
    print("   streamlit run demo-ready/dashboard.py")
    print("=" * 60)


# ---------------------------------------------------------------------------
# DAG Definition
# ---------------------------------------------------------------------------
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=2),
}

with DAG(
    dag_id='intelligence_llm',
    default_args=default_args,
    description='Scheduled LLM analysis for detected trends (Slow Path)',
    schedule='*/15 * * * *',  # Every 15 minutes
    start_date=datetime(2024, 1, 1),
    catchup=False,
    tags=['intelligence', 'llm', 'slow-path', 'gemini'],
    doc_md=__doc__,
) as dag:
    
    # Task 1: Check API Key
    check_key = PythonOperator(
        task_id='check_api_key',
        python_callable=check_api_key,
    )
    
    # Task 2: Fetch unanalyzed trends
    fetch_trends = PythonOperator(
        task_id='fetch_unanalyzed_trends',
        python_callable=fetch_unanalyzed_trends,
    )
    
    # Task 3: Analyze with LLM
    analyze = PythonOperator(
        task_id='analyze_with_llm',
        python_callable=analyze_with_llm,
    )
    
    # Task 4: Log completion
    log_done = PythonOperator(
        task_id='log_completion',
        python_callable=log_completion,
    )
    
    # Pipeline
    check_key >> fetch_trends >> analyze >> log_done
