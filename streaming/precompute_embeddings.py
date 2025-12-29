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
# Config
DATA_DIR = '../data/demo-ready_archieve' # Relative to streaming/
MODEL_NAME = os.getenv("MODEL_NAME", "dangvantuan/vietnamese-document-embedding")
CACHE_FILE = 'embeddings_cache.pkl'

def clean_text(text):
    import re
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_all_posts(data_dir):
    """Load all posts from data directory"""
    posts = []
    
    # Resolve absolute path for robustness
    if not os.path.isabs(data_dir):
        data_dir = os.path.abspath(os.path.join(os.getcwd(), data_dir))
        
    console.print(f"📂 Scanning data in: {data_dir}...")
    
    if not os.path.exists(data_dir):
        console.print(f"[red]Directory {data_dir} does not exist![/red]")
        return []

    # Load CSVs
    csv_files = glob.glob(os.path.join(data_dir, "**/*.csv"), recursive=True)
    csv_files.extend(glob.glob(os.path.join(data_dir, "*.csv"))) 
    csv_files = list(set(csv_files))
    
    for f in csv_files:
        try:
            filename = os.path.basename(f).lower()
            with open(f, 'r', encoding='utf-8') as file:
                # Basic check for header
                has_header = True
                try:
                    sample = file.read(4096)
                    file.seek(0)
                    has_header = csv.Sniffer().has_header(sample)
                except:
                    # Sniffer failed, likely complex csv with header
                    has_header = True 
                    file.seek(0)
                
                if 'fb_' in filename and not has_header:
                     # Force read as list for Facebook headless
                     reader = csv.reader(file)
                     for row in reader:
                         if not row: continue
                         # Assume last column is summary/content
                         content = row[-1]
                         if len(content) < 20: continue
                         posts.append({
                            "content": clean_text(content),
                            "source": "Facebook",
                            "time": datetime.now().isoformat(), # Mock time for headless
                            "stats": {"likes": 0, "comments": 0, "shares": 0}
                         })
                else:
                    # Standard DictReader
                    reader = csv.DictReader(file)
                    for row in reader:
                        # Try various summary keys
                        summary = row.get('summary') or row.get('refine_summary') or row.get('short_description')
                        content_raw = row.get('content') or row.get('title') or ""
                        
                        final_content = f"{row.get('title', '')} {summary or content_raw}".strip()
                        if len(final_content) < 20: 
                             # Fallback to direct content if title/summary not enough
                             final_content = content_raw

                        if len(final_content) < 20: continue
                        
                        source = 'News'
                        if 'fb' in filename or 'facebook' in filename: source = 'Social'

                        posts.append({
                            "content": clean_text(final_content),
                            "source": source,
                            "time": row.get('published_at') or row.get('time') or datetime.now().isoformat(),
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
    json_files.extend(glob.glob(os.path.join(data_dir, "*.json")))
    json_files = list(set(json_files))
    
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
