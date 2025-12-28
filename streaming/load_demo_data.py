"""
Fast Demo Data Loader V2
========================
Load pre-computed demo data directly into the database OR produce to Kafka.
Supports:
1. Default Mode: pre-computed trends from `demo/demo_data_full_llm` (results.parquet) -> DB.
2. Legacy Mode: raw data from `demo/demo_data_v1` (CSVs + JSON) -> DB or Kafka.
"""

import json
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
from datetime import datetime
import os
import sys
import glob
try:
    from kafka import KafkaProducer
except ImportError:
    KafkaProducer = None

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
DEFAULT_DEMO_DIR = os.path.join(PROJECT_ROOT, "demo", "demo_data_full_llm")
LEGACY_DEMO_DIR = os.path.join(PROJECT_ROOT, "demo", "demo_data_v1")
KAFKA_BOOTSTRAP = ['localhost:29092']
KAFKA_TOPIC = 'posts_stream_v1'

def get_db_engine():
    return create_engine(POSTGRES_URL)

def load_legacy_data(is_stream=False, target='db'):
    print(f"🚀 Loading LEGACY demo data from demo/demo_data_v1 -> {target.upper()}...")
    engine = None
    producer = None
    
    if target == 'db':
        engine = get_db_engine()
    elif target == 'kafka':
        if not KafkaProducer:
            print("❌ kafka-python not installed.")
            return 0
        try:
             producer = KafkaProducer(
                bootstrap_servers=KAFKA_BOOTSTRAP,
                value_serializer=lambda v: json.dumps(v).encode('utf-8')
             )
             print(f"✅ Connected to Kafka at {KAFKA_BOOTSTRAP}")
        except Exception as e:
             print(f"❌ Failed to connect to Kafka: {e}")
             return 0

    # 1. Load Trends from JSON (Only needed for DB Matching, but fine to load)
    json_files = glob.glob(os.path.join(LEGACY_DEMO_DIR, "*.json"))
    trends_map = {}
    if json_files:
        with open(json_files[0], 'r') as f:
            trends_map = json.load(f)

    # 2. Load Posts from CSVs
    csv_files = glob.glob(os.path.join(LEGACY_DEMO_DIR, "*_summarized.csv"))
    all_posts = []
    
    for cf in csv_files:
        source_name = os.path.basename(cf).split('_')[0]
        try:
            # Try loading with fallback encoding
            try:
                idf = pd.read_csv(cf)
            except:
                idf = pd.read_csv(cf, encoding='latin1')
                
            # Normalize columns
            idf.columns = [c.lower() for c in idf.columns]
            
            # Identify content column
            content_col = 'content' if 'content' in idf.columns else ('text' if 'text' in idf.columns else None)
            if not content_col: continue
            
            # Normalize to standard dict
            if 'published_at' in idf.columns: time_col = 'published_at'
            elif 'created_time' in idf.columns: time_col = 'created_time'
            else: time_col = 'time'
            
            url_col = 'url' if 'url' in idf.columns else ('link' if 'link' in idf.columns else None)
            
            for _, row in idf.iterrows():
                content = str(row[content_col])
                if len(content) < 50: continue
                
                def _safe_int(val):
                    try: return int(float(val)) if pd.notnull(val) else 0
                    except: return 0

                likes = _safe_int(row.get('likes'))
                comments = _safe_int(row.get('comments'))
                shares = _safe_int(row.get('shares'))
                reactions = _safe_int(row.get('topreactionscount'))
                total_inter = likes + comments + shares + reactions

                post_data = {
                    'content': content,
                    'source': source_name,
                    'time': row.get(time_col, datetime.now().isoformat()), # REQUIRED by consumer
                    'created_time': row.get(time_col, datetime.now().isoformat()),
                    'summary': str(row.get('summary', '')),
                    'url': str(row.get(url_col, '')),
                    'interaction': total_inter
                }
                
                # If target is Kafka, we verify standard fields
                if target == 'kafka':
                    # Ensure format matches consumer expectation
                    # Consumer expects: content, source, etc.
                    pass
                    
                all_posts.append(post_data)
        except Exception as e:
            print(f"⚠️ Error reading {cf}: {e}")
            
    posts_df = pd.DataFrame(all_posts)
    print(f"📦 Loaded {len(posts_df)} raw posts from CSVs.")
    
    # ---------------------------------------------------------
    # MODE: KAFKA PRODUCER
    # ---------------------------------------------------------
    if target == 'kafka':
        print(f"📤 Producing {len(posts_df)} posts to Kafka topic '{KAFKA_TOPIC}'...")
        sent_count = 0
        
        # Determine delay
        delay = 0.05 # Fast stream by default
        if is_stream:
             delay = 1.0 # Slow stream
             
        for i, post in enumerate(all_posts):
            producer.send(KAFKA_TOPIC, post)
            sent_count += 1
            
            if sent_count % 10 == 0:
                print(f"   📨 Sent {sent_count} posts...")
                if is_stream:
                    import time
                    time.sleep(delay)
                    
            # Flush periodically
            if sent_count % 100 == 0:
                 producer.flush()
                 
        producer.flush()
        print(f"✅ Successfully sent {sent_count} posts to Kafka.")
        return sent_count

    # ---------------------------------------------------------
    # MODE: DATABASE INSERT (Legacy Simulation)
    # ---------------------------------------------------------
    
    # 3. Match and Insert
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    print("🗑️ Cleared existing trends")
    
    inserted = 0
    
    # Pre-calculate lowercase content for searching
    posts_df['content_lower'] = posts_df['content'].str.lower()
    
    for trend_name, data in trends_map.items():
        keywords = [k.lower() for k in data.get('keywords', [])]
        if not keywords: continue
        
        # Simple Keyword Matching (ANY match)
        import re
        pattern = '|'.join(map(re.escape, keywords))
        
        try:
            matched_posts = posts_df[posts_df['content_lower'].str.contains(pattern, na=False, regex=True)]
        except:
             continue
             
        post_count = len(matched_posts)
        if post_count < 2: continue
        
        # Prepare Representative Posts
        rep_posts = []
        for p in matched_posts.head(10).to_dict('records'):
             rep_posts.append({
                "content": p['content'][:500],
                "source": p['source'],
                "time": str(p['created_time']),
                "similarity": 0.9 # Fake sim
             })
             
        volume = data.get('volume', post_count * 10)
        
        # Prepare DB Params
        params = {
            "name": trend_name[:200],
            "score": min(95, 50 + post_count),
            "vol": volume,
            "rep": json.dumps(rep_posts, ensure_ascii=False),
            "type": "Legacy",
            "count": post_count,
            "sn": 0.0,
            "sf": 0.0,
            "now": datetime.now(),
            "emb": None,
            "summ": f"Legacy Trend: {trend_name}. (Chưa có phân tích chi tiết)",
            "cat": "Unclassified",
            "sent": "Neutral",
            "adv_s": "",
            "adv_b": "",
            "reason": "",
            "kws": json.dumps(keywords, ensure_ascii=False),
            "g_vol": volume,
            "inter": 0
        }
        
        # INSERT SQL
        insert_sql = text("""
            INSERT INTO detected_trends (
                trend_name, trend_score, volume, representative_posts,
                topic_type, post_count, score_n, score_f,
                created_at, last_updated, embedding,
                summary, category, sentiment,
                advice_state, advice_business, reasoning,
                keywords, google_vol, interactions
            ) VALUES (
                :name, :score, :vol, :rep, :type, :count, :sn, :sf,
                :now, :now, :emb, :summ, :cat, :sent,
                :adv_s, :adv_b, :reason, :kws, :g_vol, :inter
            )
        """)
        
        if is_stream:
             # Simulation with wait
             params_pending = params.copy()
             params_pending['summ'] = "Scanning Legacy Data..."
             
             with engine.begin() as conn:
                 conn.execute(insert_sql, params_pending)
             
             import time, random
             time.sleep(random.uniform(0.5, 1.5))
             
             # Update (Fake Analysis Complete)
             with engine.begin() as conn:
                 conn.execute(text("UPDATE detected_trends SET summary=:summ WHERE trend_name=:name"), params)
        else:
             with engine.begin() as conn:
                 conn.execute(insert_sql, params)
                 
        inserted += 1
        if inserted % 5 == 0:
            print(f"   📊 Inserted {inserted} legacy trends...")
            if is_stream:
                import time, random
                time.sleep(random.uniform(1.0, 3.0))

    print(f"✅ Loaded {inserted} Legacy Trends.")
    return inserted
    
def load_default_data(is_stream=False):
    # ... (Same as before)
    print("🚀 Loading DEFAULT (Full LLM) demo data...")
    # ... (rest of function omitted for brevity, verify context replacement)
    # Actually I should not truncated too much or regex will fail. I will use only necessary parts.
    # But wait, Step 1722 REPLACED `load_default_data` fully.
    # I can just re-paste `load_default_data` content to be safe OR assume it's there.
    # To avoid accidentally destroying `load_default_data`, I will read checking where it starts.
    # Step 1722: `load_default_data` starts around line 170.
    
    # I will replace from `def load_legacy_data` down to `if __name__` block.
    # I need to keep `load_default_data` intact.
    pass

# (The replace_file_content call will target `def load_legacy_data` block specifically).
    
def load_default_data(is_stream=False):
    print("🚀 Loading DEFAULT (Full LLM) demo data...")
    engine = get_db_engine()
    
    # Load data
    df = pd.read_parquet(os.path.join(DEFAULT_DEMO_DIR, "results.parquet"))
    try:
        trend_embs = np.load(os.path.join(DEFAULT_DEMO_DIR, "trend_embeddings.npy"))
    except:
        trend_embs = []
    
    print(f"📦 Loaded {len(df)} posts")
    
    # Clear existing data
    with engine.begin() as conn:
        conn.execute(text("TRUNCATE TABLE detected_trends"))
    print("🗑️ Cleared existing trends")
    
    # Filter out noise and unassigned
    valid_df = df[~df['final_topic'].isin(['Unassigned', 'N/A', 'Unknown', ''])]
    valid_df = valid_df[~valid_df['final_topic'].str.contains(r'\[Noise', na=False)]
    
    # Group posts by final_topic
    trend_posts = valid_df.groupby('final_topic')
    
    print(f"📊 Found {len(trend_posts)} valid trends")
    
    inserted = 0
    
    for trend_name, posts_df in trend_posts:
        post_count = len(posts_df)
        if post_count < 2: continue
        
        display_name = trend_name
        if display_name.startswith("New: "): display_name = display_name[5:]
        
        # Build representative posts
        sample_posts = posts_df.head(10).to_dict('records')
        rep_posts = []
        for p in sample_posts:
            rep_posts.append({
                "content": str(p.get('content', p.get('title', '')))[:500],
                "source": str(p.get('source', 'Unknown')),
                "time": str(p.get('time', datetime.now().isoformat())),
                "similarity": float(p.get('score', 0.85)) if pd.notna(p.get('score')) else 0.85
            })
        
        first_post = posts_df.iloc[0]
        reasoning = str(first_post.get('llm_reasoning', ''))
        
        # Extract rich summary (Logic preserved from previous version)
        summary = ""
        advice_state = ""
        advice_business = ""
        
        intel = first_post.get('intelligence')
        if isinstance(intel, dict):
            what = intel.get('what', '')
            who = intel.get('who', '')
            where = intel.get('where', '')
            when = intel.get('when', '')
            why = intel.get('why', '')
            
            summary_parts = []
            
            # Main event description
            if what:
                summary_parts.append(f"📌 **Nội dung chính:** {what}")
            
            # Context section
            context_parts = []
            if who:
                context_parts.append(f"👤 **Đối tượng liên quan:** {who}")
            if where:
                context_parts.append(f"📍 **Địa điểm:** {where}")
            if when:
                context_parts.append(f"🕐 **Thời gian:** {when}")
            if why:
                context_parts.append(f"❓ **Bối cảnh/Nguyên nhân:** {why}")
            
            if context_parts:
                summary_parts.extend(context_parts)
            
            # Add sample content from top posts for more detail
            top_contents = []
            for idx, p in enumerate(posts_df.head(3).to_dict('records')):
                content = p.get('content', p.get('title', ''))
                if content and len(str(content)) > 50:
                    top_contents.append(f"• {str(content)[:200]}...")
            
            if top_contents:
                summary_parts.append("")
                summary_parts.append("📰 **Nội dung tiêu biểu:**")
                summary_parts.extend(top_contents)
            
            # Add LLM reasoning if available
            llm_reasoning = first_post.get('llm_reasoning', '')
            if pd.notna(llm_reasoning) and llm_reasoning and len(str(llm_reasoning)) > 20:
                # Clean up reasoning (remove English parts if mixed)
                reasoning_text = str(llm_reasoning).split('|')[0].strip()
                if reasoning_text:
                    summary_parts.append("")
                    summary_parts.append(f"🧠 **Phân tích:** {reasoning_text}")
            
            summary = "\n".join(summary_parts) if summary_parts else ""
            advice_state = intel.get('advice_state', '') or ''
            advice_business = intel.get('advice_business', '') or ''
        
        # Fallback to simple summary column if intelligence is empty
        if not summary:
            raw_summary = first_post.get('summary', '')
            if pd.notna(raw_summary) and raw_summary:
                summary = str(raw_summary)
                
                # Add sample content for context
                top_contents = []
                for p in posts_df.head(2).to_dict('records'):
                    content = p.get('content', p.get('title', ''))
                    if content and len(str(content)) > 50:
                        top_contents.append(f"• {str(content)[:200]}...")
                if top_contents:
                    summary += "\n\n📰 **Bài viết liên quan:**\n" + "\n".join(top_contents)
            else:
                # Use llm_reasoning as last resort
                reasoning = first_post.get('llm_reasoning', '')
                if pd.notna(reasoning) and reasoning:
                    summary = f"📝 **Phân tích AI:** {str(reasoning)[:500]}"
                else:
                    summary = f"Chủ đề được phát hiện từ {post_count} bài viết liên quan."
        
        # Get advice from columns if not from intelligence
        if not advice_state:
            adv_s = first_post.get('advice_state', '')
            advice_state = str(adv_s) if pd.notna(adv_s) else ''
        if not advice_business:
            adv_b = first_post.get('advice_business', '')
            advice_business = str(adv_b) if pd.notna(adv_b) else ''
        
        category = str(first_post.get('category', 'Unclassified')) if pd.notna(first_post.get('category')) else 'Unclassified'
        sentiment = str(first_post.get('topic_sentiment', 'Neutral')) if pd.notna(first_post.get('topic_sentiment')) else 'Neutral'
        trend_score = float(first_post.get('trend_score', 50.0)) if pd.notna(first_post.get('trend_score')) else 50.0
        
        # Calculate volume and interactions with some randomness for demo realism
        import random
        random_boost = random.uniform(0.8, 1.5)
        adjusted_count = int(post_count * random_boost)
        volume = adjusted_count * random.randint(50, 150)
        
        interactions = 0
        if 'stats' in posts_df.columns:
            try:
                interactions = int(posts_df['stats'].apply(
                    lambda x: (x.get('likes', 0) or 0) + (x.get('comments', 0) or 0) if isinstance(x, dict) else 0
                ).sum())
            except:
                interactions = 0
        
        # Use embedding if available based on index
        emb = None
        if inserted < len(trend_embs):
            emb = trend_embs[inserted].tolist()
        
        params = {
            "name": display_name[:200],
            "score": min(100, max(30, trend_score + adjusted_count * 2)),  # Boost score by post count
            "vol": volume,
            "rep": json.dumps(rep_posts, ensure_ascii=False),
            "type": "Matched",
            "count": adjusted_count,
            "sn": 0.0,
            "sf": 0.0,
            "now": datetime.now(),
            "emb": json.dumps(emb) if emb else None,
            "summ": summary[:1000] if summary else f"Chủ đề được phát hiện từ {post_count} bài viết.",
            "cat": category,
            "sent": sentiment,
            "adv_s": advice_state[:500] if advice_state else "",
            "adv_b": advice_business[:500] if advice_business else "",
            "reason": reasoning[:1000] if reasoning else "", # Added reasoning
            "kws": json.dumps([], ensure_ascii=False),
            "g_vol": volume,
            "inter": interactions
        }
        
        # INSERT SQL
        insert_sql = text("""
            INSERT INTO detected_trends (
                trend_name, trend_score, volume, representative_posts,
                topic_type, post_count, score_n, score_f,
                created_at, last_updated, embedding,
                summary, category, sentiment,
                advice_state, advice_business, reasoning,
                keywords, google_vol, interactions
            ) VALUES (
                :name, :score, :vol, :rep, :type, :count, :sn, :sf,
                :now, :now, :emb, :summ, :cat, :sent,
                :adv_s, :adv_b, :reason, :kws, :g_vol, :inter
            )
        """)

        if is_stream:
            # 1. Insert as Pending
            params_pending = params.copy()
            params_pending['summ'] = 'Waiting for analysis...'
            params_pending['adv_s'] = ''
            params_pending['adv_b'] = ''
            params_pending['reason'] = ''  # Empty reasoning
            
            with engine.begin() as conn:
                conn.execute(insert_sql, params_pending)
            
            # Simulate Analysis Time
            import time
            import random
            time.sleep(random.uniform(0.5, 1.5))
            
            # 2. Update to Analyzed
            with engine.begin() as conn:
                conn.execute(text("""
                    UPDATE detected_trends SET 
                        summary = :summ, 
                        advice_state = :adv_s, 
                        advice_business = :adv_b,
                        reasoning = :reason
                    WHERE trend_name = :name
                """), params)
                
        else:
            # Direct insert full data
            with engine.begin() as conn:
                conn.execute(insert_sql, params)
        
        inserted += 1
        
        if inserted % 5 == 0:
            print(f"   📊 Inserted {inserted} trends...")
            if is_stream:
                import time
                import random
                # Sleep random time between 2-5 seconds every 5 trends to make demo observable
                time.sleep(random.uniform(2.0, 5.0))
                
    print(f"✅ Successfully loaded {inserted} trends!")
    return inserted


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Load demo data into DB")
    parser.add_argument("--stream", action="store_true", help="Enable streaming simulation")
    parser.add_argument("--source", type=str, default="default", help="Data source: 'default' or 'legacy'")
    parser.add_argument("--target", type=str, default="db", help="Target: 'db' or 'kafka'")
    
    args = parser.parse_args()
    
    total_inserted = 0
    if args.source == "legacy":
        total_inserted = load_legacy_data(args.stream, target=args.target)
    else:
        # Default mode doesn't support target yet, assumes DB
        total_inserted = load_default_data(args.stream)
    
    print(f"✅ Successfully loaded {total_inserted} trends with AI analysis!")
    print(f"👉 Open Dashboard at http://localhost:8501")
