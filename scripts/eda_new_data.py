import pandas as pd
import numpy as np
import json
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns
from dateutil import parser
import re

# Setup
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

DATA_DIR = "crawlers/new_data"

def load_news_data(data_dir):
    print(f"📂 Loading News data from {data_dir}/news/...")
    files = glob.glob(os.path.join(data_dir, "news", "**", "*.csv"), recursive=True)
    dfs = []
    for f in files:
        try:
            # Extract source from folder name (e.g., .../news/nld/articles.csv -> nld)
            source = os.path.basename(os.path.dirname(f))
            df = pd.read_csv(f)
            df['source_type'] = 'News'
            df['publisher'] = source
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
    
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        # Parse Dates
        full_df['published_at'] = pd.to_datetime(full_df['published_at'], errors='coerce', utc=True)
        print(f"✅ Loaded {len(full_df)} news articles from {len(files)} files.")
        return full_df
    return pd.DataFrame()

def load_facebook_data(data_dir):
    print(f"📂 Loading Facebook data from {data_dir}/facebook/...")
    files = glob.glob(os.path.join(data_dir, "facebook", "*.json"))
    all_posts = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as json_file:
                data = json.load(json_file)
                # Ensure list
                if isinstance(data, dict): data = [data]
                
                for post in data:
                    post['source_file'] = os.path.basename(f)
                    all_posts.append(post)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if all_posts:
        df = pd.DataFrame(all_posts)
        df['source_type'] = 'Social'
        df['publisher'] = df['pageName']
        # Parse Dates (Handle ISO format)
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        print(f"✅ Loaded {len(df)} facebook posts from {len(files)} files.")
        return df
    return pd.DataFrame()

def parse_vietnamese_date(date_str):
    if pd.isna(date_str): return None
    # Format: "lúc 02:00:00 UTC+7 11 tháng 12, 2025"
    # Regex to extract day, month, year, time
    try:
        # Simple extraction
        date_str = str(date_str).lower()
        date_str = date_str.replace("lúc ", "").replace(" utc+7", "")
        # Replace "tháng" with /
        date_str = re.sub(r'\s+tháng\s+', '/', date_str)
        # Remove commas
        date_str = date_str.replace(",", "")
        return pd.to_datetime(date_str, dayfirst=True, errors='coerce')
    except:
        return None

def parse_volume(vol_str):
    if pd.isna(vol_str): return 0
    vol_str = str(vol_str).lower().strip()
    vol_str = vol_str.replace("n+", "").replace(".", "").replace(",", "").strip()
    try:
        val = int(vol_str)
        # If it was "200 N+", it means 200 * 1000? Or just 200,000?
        # Usually Google Trends "100 N+" means > 100,000 in Vietnamese (Ngàn)
        # But wait, "N" can be Ngàn (Thousand). 
        # Let's assume N = 1000.
        return val * 1000
    except:
        return 0

def load_trends_data(data_dir):
    print(f"📂 Loading Trends data from {data_dir}/trendings/...")
    files = glob.glob(os.path.join(data_dir, "trendings", "*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Rename columns if needed (Vietnamese headers)
            # "Xu hướng","Lượng tìm kiếm","Đã bắt đầu"
            df.rename(columns={
                "Xu hướng": "trend",
                "Lượng tìm kiếm": "volume_raw",
                "Đã bắt đầu": "start_time",
                "Đã kết thúc": "end_time"
            }, inplace=True)
            
            df['volume'] = df['volume_raw'].apply(parse_volume)
            df['start_time'] = df['start_time'].apply(parse_vietnamese_date)
            dfs.append(df)
        except Exception as e:
            print(f"⚠️ Error reading {f}: {e}")
            
    if dfs:
        full_df = pd.concat(dfs, ignore_index=True)
        print(f"✅ Loaded {len(full_df)} trends from {len(files)} files.")
        return full_df
    return pd.DataFrame()

def run_eda():
    print(f"🚀 Starting EDA on {DATA_DIR}...\n")
    
    # 1. Load Data
    df_news = load_news_data(DATA_DIR)
    df_fb = load_facebook_data(DATA_DIR)
    df_trends = load_trends_data(DATA_DIR)
    
    # 2. Unified Content DataFrame
    # Map columns: content/text -> text, published_at/time -> date
    news_subset = df_news[['publisher', 'published_at', 'content', 'title']].rename(
        columns={'published_at': 'date', 'content': 'text'}
    )
    news_subset['type'] = 'News'
    
    fb_subset = df_fb[['publisher', 'time', 'text']].rename(
        columns={'time': 'date'}
    )
    fb_subset['type'] = 'Facebook'
    fb_subset['title'] = fb_subset['text'].apply(lambda x: str(x)[:50] + "..." if pd.notna(x) else "")
    
    df_all = pd.concat([news_subset, fb_subset], ignore_index=True)
    df_all['length'] = df_all['text'].str.len().fillna(0)
    
    print("\n" + "="*40)
    print("📊 DATA SUMMARY")
    print("="*40)
    print(df_all.groupby('type')['text'].count())
    print("\nPublishers/Pages:")
    print(df_all.groupby(['type', 'publisher']).size())
    
    # 3. Visualizations
    fig = plt.figure(figsize=(15, 12))
    gs = fig.add_gridspec(3, 2)
    
    # Plot 1: Source Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    sns.countplot(data=df_all, x='type', ax=ax1)
    ax1.set_title("Total Posts by Source Type")
    
    # Plot 2: Content Length
    ax2 = fig.add_subplot(gs[0, 1])
    sns.histplot(data=df_all, x='length', hue='type', bins=50, log_scale=(False, True), ax=ax2)
    ax2.set_title("Content Length Distribution (Log Scale)")
    
    # Plot 3: Time Series
    ax3 = fig.add_subplot(gs[1, :])
    # Group by day
    df_all['day'] = df_all['date'].dt.date
    daily_counts = df_all.groupby(['day', 'type']).size().reset_index(name='count')
    sns.lineplot(data=daily_counts, x='day', y='count', hue='type', marker='o', ax=ax3)
    ax3.set_title("Daily Volume")
    plt.xticks(rotation=45)
    
    # Plot 4: Top Trends
    if not df_trends.empty:
        ax4 = fig.add_subplot(gs[2, 0])
        top_trends = df_trends.sort_values('volume', ascending=False).head(10)
        sns.barplot(data=top_trends, y='trend', x='volume', ax=ax4, palette='viridis')
        ax4.set_title("Top 10 Google Trends by Volume")
        
    # Plot 5: FB Interactions (if available)
    if not df_fb.empty and 'likes' in df_fb.columns:
        ax5 = fig.add_subplot(gs[2, 1])
        # Melt interactions
        try:
            df_fb['likes'] = pd.to_numeric(df_fb['likes'], errors='coerce').fillna(0)
            df_fb['shares'] = pd.to_numeric(df_fb['shares'], errors='coerce').fillna(0)
            df_fb['comments'] = pd.to_numeric(df_fb['comments'], errors='coerce').fillna(0)
            
            # Log transform for better viz
            fb_viz = df_fb.copy()
            fb_viz['log_likes'] = np.log1p(fb_viz['likes'])
            fb_viz['log_shares'] = np.log1p(fb_viz['shares'])
            fb_viz['log_comments'] = np.log1p(fb_viz['comments'])
            
            fb_melted = fb_viz.melt(value_vars=['log_likes', 'log_shares', 'log_comments'], 
                                   var_name='Interaction', value_name='Log Count')
            
            sns.boxplot(data=fb_melted, x='Interaction', y='Log Count', ax=ax5)
            ax5.set_title("Facebook Interactions (Log Scale)")
            ax5.set_xticklabels(['Likes', 'Shares', 'Comments'])
        except Exception as e:
            print(f"Skipping interaction plot: {e}")

    plt.tight_layout()
    plt.show() # NOTE: If in notebook, this shows. If script, might just close. 
    # For script, we usually save.
    
    out_file = "eda_summary.png"
    fig.savefig(out_file)
    print(f"\n✅ EDA Plots saved to {out_file}")

if __name__ == "__main__":
    run_eda()
