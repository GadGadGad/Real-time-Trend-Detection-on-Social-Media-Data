"""
Pre-compute embeddings for all posts and save to cache for faster demo.
Run this ONCE before the demo, then consumer will load from cache.
"""
import os
import json
import glob
import csv
import pickle
import numpy as np
from datetime import datetime
from sentence_transformers import SentenceTransformer
from rich.console import Console
from rich.progress import Progress

console = Console()

# Config
DATA_DIR = 'data'
MODEL_NAME = os.getenv("MODEL_NAME", "dangvantuan/vietnamese-document-embedding")
CACHE_FILE = 'embeddings_cache.pkl'

def clean_text(text):
    import re
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_all_posts(data_dir):
    """Load all posts from data directory"""
    posts = []
    console.print(f"📂 Scanning data in: {data_dir}...")
    
    # Load CSVs
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    for f in csv_files:
        try:
            with open(f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    content = f"{row.get('title', '')} {row.get('content', '')}"
                    if len(content) < 20: continue
                    
                    fname_lower = os.path.basename(f).lower()
                    if 'fb' in fname_lower or 'facebook' in fname_lower:
                        source = 'Social'
                    elif 'news' in fname_lower:
                        source = 'News'
                    else:
                        source = 'Unknown'
                    
                    posts.append({
                        "content": clean_text(content),
                        "source": source,
                        "time": row.get('publish_date', row.get('created_time', datetime.now().isoformat())),
                        "stats": {
                            "likes": int(row.get('likes', 0) or 0),
                            "comments": int(row.get('comments', 0) or 0),
                            "shares": int(row.get('shares', 0) or 0)
                        }
                    })
        except Exception as e:
            console.print(f"[red]Error loading {f}: {e}[/red]")
    
    # Load JSONs
    json_files = glob.glob(os.path.join(data_dir, "**/*.json"), recursive=True)
    for f in json_files:
        if 'trend_refine' in f or 'embeddings_cache' in f:
            continue  # Skip trend seed files and cache
        try:
            with open(f, 'r', encoding='utf-8') as file:
                data = json.load(file)
                if isinstance(data, list):
                    for item in data:
                        content = item.get('content', item.get('text', ''))
                        if len(content) < 20: continue
                        posts.append({
                            "content": clean_text(content),
                            "source": item.get('source', 'Unknown'),
                            "time": item.get('time', item.get('published_at', datetime.now().isoformat())),
                            "stats": item.get('stats', {})
                        })
        except Exception as e:
            console.print(f"[red]Error loading {f}: {e}[/red]")
    
    console.print(f"[green]📊 Loaded {len(posts)} posts total.[/green]")
    return posts

def compute_embeddings(posts, model_name):
    """Compute embeddings for all posts"""
    console.print(f"🧠 Loading model: {model_name}...")
    model = SentenceTransformer(model_name, trust_remote_code=True)
    console.print("✅ Model loaded!")
    
    contents = [p['content'] for p in posts]
    
    console.print(f"📐 Computing embeddings for {len(contents)} posts...")
    with Progress() as progress:
        task = progress.add_task("Embedding...", total=len(contents))
        
        batch_size = 64
        all_embeddings = []
        
        for i in range(0, len(contents), batch_size):
            batch = contents[i:i+batch_size]
            embs = model.encode(batch, show_progress_bar=False)
            all_embeddings.extend(embs)
            progress.update(task, advance=len(batch))
    
    console.print(f"✅ Computed {len(all_embeddings)} embeddings!")
    return np.array(all_embeddings)

def save_cache(posts, embeddings, cache_file):
    """Save posts and embeddings to pickle cache"""
    cache = {
        "posts": posts,
        "embeddings": embeddings,
        "created_at": datetime.now().isoformat()
    }
    
    with open(cache_file, 'wb') as f:
        pickle.dump(cache, f)
    
    console.print(f"[bold green]💾 Cache saved to {cache_file} ({os.path.getsize(cache_file) / 1024 / 1024:.1f} MB)[/bold green]")

def main():
    console.print("[bold cyan]🚀 Pre-computing Embeddings for Demo[/bold cyan]")
    
    # Check if cache exists
    if os.path.exists(CACHE_FILE):
        console.print(f"[yellow]⚠️ Cache already exists: {CACHE_FILE}[/yellow]")
        response = input("Overwrite? (y/n): ")
        if response.lower() != 'y':
            console.print("Aborted.")
            return
    
    # Load posts
    posts = load_all_posts(DATA_DIR)
    if not posts:
        console.print("[red]No posts found![/red]")
        return
    
    # Compute embeddings
    embeddings = compute_embeddings(posts, MODEL_NAME)
    
    # Save cache
    save_cache(posts, embeddings, CACHE_FILE)
    
    console.print("\n[bold green]✨ Pre-computation complete![/bold green]")
    console.print("👉 Now run producer.py with --use-cache flag for fast demo.")

if __name__ == "__main__":
    main()
