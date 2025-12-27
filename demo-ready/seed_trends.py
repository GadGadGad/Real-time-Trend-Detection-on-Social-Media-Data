
import os
import sys
import json
import random
import time
from datetime import datetime
import numpy as np
from sqlalchemy import create_engine, text
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
console = Console()

# Add parent directory to path to import src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from src.pipeline.trend_scoring import calculate_unified_score
except ImportError:
    print("⚠️ Could not import calculate_unified_score. Using dummy scoring.")
    def calculate_unified_score(trend_data, cluster_posts):
        return 61.0, {"G": 50, "F": 50, "N": 50}

# Configuration
POSTGRES_URL = os.getenv("POSTGRES_URL", "postgresql://user:password@localhost:5432/trend_db")
MODEL_NAME = os.getenv("MODEL_NAME", "dangvantuan/vietnamese-document-embedding")

# Default Seeds (if none provided)
DEFAULT_SEEDS = [
    "Bão Yagi gây thiệt hại nặng nề tại miền Bắc",
    "Tour concert Blackpink tại Hà Nội",
    "Giá xăng dầu tiếp tục tăng mạnh",
    "Đội tuyển bóng đá Việt Nam chuẩn bị cho AFF Cup",
    "Vụ án Vạn Thịnh Phát và bà Trương Mỹ Lan"
]

def get_db_engine():
    return create_engine(POSTGRES_URL)

def init_db_for_seeding():
    """Ensure DB table exists with correct schema before seeding."""
    engine = get_db_engine()
    with engine.connect() as conn:
        conn.execute(text("DROP TABLE IF EXISTS detected_trends"))
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS detected_trends (
                id SERIAL PRIMARY KEY,
                trend_name TEXT,
                trend_score FLOAT,
                volume INT,
                representative_posts TEXT, -- JSON string of list of dicts
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
                keywords TEXT, -- JSON string of list of keywords
                embedding TEXT, -- JSON string of vector for matching
                google_vol INT DEFAULT 0,
                interactions INT DEFAULT 0
            );
        """))
        conn.commit()
    console.print("✅ Database initialized (Schema Checked).")

def seed_trends(seeds=None, json_data=None):
    # Ensure DB is ready with correct schema
    init_db_for_seeding()

    if not seeds and not json_data:
        seeds = DEFAULT_SEEDS
        
    count = 0
    if json_data:
        console.print(f"[bold cyan]🚀 Initializing Trend Seeder from JSON ({len(json_data)} topics)...[/bold cyan]")
        trend_names = list(json_data.keys())
    else:
        console.print(f"[bold cyan]🚀 Initializing Trend Seeder with {len(seeds)} topics...[/bold cyan]")
        trend_names = seeds
    
    # 1. Initialize Embedder
    console.print(f"   🧠 Loading Embedding Model ({MODEL_NAME})...")
    try:
        embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
    except Exception as e:
        console.print(f"[bold red]❌ Failed to load model '{MODEL_NAME}'![/bold red]")
        console.print(f"[yellow]Tip: Check your internet connection or run 'check_model_download.py' manually.[/yellow]")
        console.print(f"Error details: {e}")
        sys.exit(1)
    
    # 2. Compute Embeddings
    console.print("   📐 Computing Embeddings for seeds...")
    
    if json_data:
        # For JSON, we can use keywords to make a better embedding representation
        texts_to_embed = []
        for t in trend_names:
            keywords = json_data[t].get("keywords", [])
            # Combine name + keywords for rich semantic signal
            # [SMART QUERY LOGIC]
            # 1. Use trend name as anchor
            unique_signals = [t]
            # 2. Add distinctive entities/keywords
            for kw in keywords:
                 if len(unique_signals) >= 6: break # 1 anchor + 5 keywords
                 # Avoid redundancy
                 is_redundant = any(s.lower() in kw.lower() or kw.lower() in s.lower() for s in unique_signals)
                 if not is_redundant:
                     unique_signals.append(kw)
            text_rep = " ".join(unique_signals)
            texts_to_embed.append(text_rep)
        embeddings = embedder.encode(texts_to_embed)
    else:
        embeddings = embedder.encode(trend_names)
    
    # 3. Connect to DB
    engine = get_db_engine()
    
    # 4. Insert Seeds
    with engine.begin() as conn:
        for i, t_name in enumerate(trend_names):
            embedding_json = json.dumps(embeddings[i].tolist())
            
            # Extract Metadata
            if json_data:
                details = json_data[t_name]
                volume = details.get("volume", 100)
                keywords = details.get("keywords", [])
                time_str = details.get("time", datetime.now().isoformat())
                summary = f"Chủ đề hot: {t_name}. Keywords: {', '.join(keywords[:5])}"
                rep_posts = [{"content": t_name, "source": "System", "time": time_str, "similarity": 1.0}]
            else:
                volume = 1
                keywords = []
                time_str = datetime.now().isoformat()
                summary = f"Chủ đề được khởi tạo sẵn: {t_name}"
                rep_posts = [{"content": t_name, "source": "System", "time": time_str, "similarity": 1.0}]

            # Check overlap
            exists = conn.execute(text("SELECT id FROM detected_trends WHERE trend_name = :name"), {"name": t_name}).fetchone()
            if exists:
                console.print(f"   ⚠️ Trend '{t_name}' already exists. Skipping.")
                continue
            
            params = {
                "name": t_name,
                "score": 60.0,
                "vol": volume,
                "rep": json.dumps(rep_posts, ensure_ascii=False),
                "type": "Specific", 
                "sn": 70.0,
                "sf": 70.0,
                "now": datetime.now(),
                "emb": embedding_json,
                "summ": summary,
                "cat": "Seeded Topic",
                "sent": "Neutral",
                "kws": json.dumps(keywords, ensure_ascii=False),
                "g_vol": volume,
                "inter": random.randint(100, 500)
            }
            
            # Calculate more realistic unified score
            u_score, u_meta = calculate_unified_score({"volume": volume}, rep_posts)
            params["score"] = u_score
            params["sn"] = u_meta.get("N", 50.0)
            params["sf"] = u_meta.get("F", 50.0)
            
            conn.execute(text("""
                INSERT INTO detected_trends (
                    trend_name, trend_score, volume, post_count, 
                    representative_posts, topic_type, 
                    score_n, score_f, created_at, last_updated, embedding,
                    summary, category, sentiment, keywords,
                    google_vol, interactions
                )
                VALUES (:name, :score, :vol, 1, :rep, :type, :sn, :sf, :now, :now, :emb, :summ, :cat, :sent, :kws, :g_vol, :inter)
            """), params)
            count += 1
            console.print(f"   ✅ Seeded: [green]{t_name}[/green] (Vol: {volume})")
            
    console.print(f"[bold green]✨ Seeding Complete! Added {count} trends.[/bold green]")
    console.print("👉 The Consumer will now recognize these topics immediately.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode input")
    parser.add_argument("--file", "-f", type=str, help="Read seeds from a line-separated text file")
    parser.add_argument("--json", "-j", type=str, help="Read seeds from a JSON file (full metadata)")
    args = parser.parse_args()
    
    seeds = []
    json_data = None
    
    if args.json:
        try:
            with open(args.json, 'r', encoding='utf-8') as f:
                json_data = json.load(f)
        except Exception as e:
            console.print(f"[red]Error reading JSON: {e}[/red]")
            sys.exit(1)
            
    elif args.interactive:
        console.print("[yellow]Enter trend keywords (one per line). Empty line to finish:[/yellow]")
        while True:
            line = input("> ").strip()
            if not line: break
            seeds.append(line)
    elif args.file:
        with open(args.file, 'r', encoding='utf-8') as f:
            seeds = [l.strip() for l in f if l.strip()]
            
    seed_trends(seeds, json_data)
