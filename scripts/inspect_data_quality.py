import os
import sys
sys.path.append(os.getcwd())
import glob
import pandas as pd
from src.utils.data_loader import load_social_data, load_news_data

DATA_DIR = 'summarized_data'

# Mirroring Notebook Cell 17 logic for Merged Posts
def load_merged_posts(path, source_type='News'):
    loaded = []
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            # refined fallback logic
            content = row.get('refined_summary')
            if pd.isna(content) or content == "": content = row.get('summary')
            if pd.isna(content) or content == "": 
                t_raw = row.get('title', '')
                content = str(t_raw) if t_raw else row.get('content', '')
            
            src = row.get('source')
            if not src:
                src = f"Face: {row.get('pageName', 'Unknown')}" if source_type == 'Facebook' else source_type.upper()
            
            loaded.append({
                "source": src,
                "content": str(content),
                "title": row.get('title', ''),
                "url": row.get('url') or row.get('postUrl', ''),
                "time": row.get('time') or row.get('published_at', '')
            })
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return loaded

posts = []

# 1. Load Facebook Merged
fb_merged = os.path.join(DATA_DIR, 'facebook_merged.csv')
if os.path.exists(fb_merged):
    posts.extend(load_merged_posts(fb_merged, 'Facebook'))
else:
    fb_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json")
    posts.extend(load_social_data(fb_files))

# 2. Load News Merged
NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
for source in NEWS_SOURCES:
    n_path = os.path.join(DATA_DIR, f'{source}_merged.csv')
    if os.path.exists(n_path):
        posts.extend(load_merged_posts(n_path, source))
    else:
        # Fallback to raw not meant for this test as we know merged exists
        pass

# --- INSPECTION ---
print(f"Total Posts Loaded: {len(posts)}")
df = pd.DataFrame(posts)

# Filter Short/Empty
df['length'] = df['content'].astype(str).str.len()
short_df = df[df['length'] < 20].copy()

print(f"\n--- BAD DATA REPORT (Length < 20 chars) ---")
print(f"Count: {len(short_df)} / {len(df)} ({len(short_df)/len(df):.1%})")

if not short_df.empty:
    print("\nBreakdown by Source:")
    print(short_df['source'].value_counts())
    
    print("\n--- SAMPLES ---")
    for i, row in short_df.head(10).iterrows():
        print(f"[{row['source']}] Len={row['length']}")
        print(f"Title: {row['title']}")
        print(f"Content: '{row['content']}'")
        print("---")
else:
    print("No short posts found.")
