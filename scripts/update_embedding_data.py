import json
import os

nb_path = 'notebooks/analysis-playground_v2.ipynb'
if not os.path.exists(nb_path):
    print(f"Error: {nb_path} does not exist.")
    exit(1)

with open(nb_path, 'r') as f:
    nb = json.load(f)

# Logic to find Cell 17 (where 'posts' is defined)
target_idx = -1
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if "# ==========================================" in source and "def load_merged_posts" in source:
            print(f"Found target 'posts' loading cell at index {i} (Merged Version)")
            target_idx = i
            break
        # Identify by unique combination of load calls (Old version fallback)
        if "load_google_trends(" in source and "posts = load_social_data" in source:
            print(f"Found target 'posts' loading cell at index {i} (Old Version)")
            target_idx = i
            break

if target_idx == -1:
    print("Could not find the 'posts' loading cell.")
    exit(1)

# New source code to inject
new_code = """# ==========================================
# LOAD DATA (Refined Merged Sources Priority)
# ==========================================
import os
import glob
import pandas as pd
from src.utils.data_loader import load_social_data, load_news_data # Keep imports if needed or define helpers

DATA_DIR = '../summarized_data'
if not os.path.exists(DATA_DIR): DATA_DIR = 'summarized_data'

# Helper to load merged csv as posts
def load_merged_posts(path, source_type='News'):
    loaded = []
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            # content priority: refined_summary > summary > text
            content = row.get('refined_summary')
            if pd.isna(content) or content == "": content = row.get('summary')
            if pd.isna(content) or content == "": 
                # For news, fallback to title ONLY as requested by user.
                # Previously used title + \\n + content, which was too noisy.
                t_raw = row.get('title', '')
                content = str(t_raw) if t_raw else row.get('content', '')
            
            # Source
            src = row.get('source')
            if not src:
                if source_type == 'Facebook':
                    src = f"Face: {row.get('pageName', 'Unknown')}"
                else:
                    src = source_type.upper()
            
            # Stats
            stats = {'likes': row.get('likes', 0), 'comments': row.get('comments', 0), 'shares': row.get('shares', 0)}
            
            loaded.append({
                "source": src,
                "content": str(content),
                "title": row.get('title', ''),
                "url": row.get('url') or row.get('postUrl', ''),
                "time": row.get('time') or row.get('published_at', ''),
                "stats": stats
            })
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return loaded

# Load Trends
trend_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/trendings/*.csv")
trends = load_google_trends(trend_files)
print(f"Loaded {len(trends)} trends.")

# Load Posts (Refined > Raw)
posts = []

# 1. Facebook
fb_merged = os.path.join(DATA_DIR, 'facebook_merged.csv')
if os.path.exists(fb_merged):
    print(f"Loading merged Facebook data from {fb_merged}")
    posts.extend(load_merged_posts(fb_merged, 'Facebook'))
else:
    print("Merged Facebook data not found. Loading raw...")
    fb_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json")
    posts.extend(load_social_data(fb_files))

# 2. News
NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
news_merged_count = 0
for source in NEWS_SOURCES:
    n_path = os.path.join(DATA_DIR, f'{source}_merged.csv')
    if os.path.exists(n_path):
        posts.extend(load_merged_posts(n_path, source))
        news_merged_count += 1

if news_merged_count == 0:
    print("Merged News data not found. Loading raw...")
    news_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/news/**/*.csv", recursive=True)
    posts.extend(load_news_data(news_files))

if LIMIT_POSTS:
    posts = posts[:LIMIT_POSTS]
    
# Helper: Extract contents
post_contents = [p.get('content', '') for p in posts]
print(f"Loaded {len(posts)} posts for analysis.")
"""

# Replace
nb['cells'][target_idx]['source'] = [line + '\n' for line in new_code.splitlines()]

with open(nb_path, 'w') as f:
    json.dump(nb, f, indent=4)
    
print(f"Updated Cell {target_idx} in {nb_path}")
