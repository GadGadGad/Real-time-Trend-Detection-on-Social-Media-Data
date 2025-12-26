import pandas as pd
import glob
import json
import random
import os
import re
from datetime import datetime
from rich.console import Console

console = Console()

# Configuration
OUTPUT_FILE = "ground_truth_annotation_sample.csv"
SAMPLE_SIZE = 300
SEARCH_ROOT = "/home/gad/My Study/Code Storages/University/HK7/SE363/Final Project"

def load_all_data():
    console.print(f"[dim]Loading data from {SEARCH_ROOT}...[/dim]")
    posts = []
    
    # 1. Load Facebook Data
    # 1a. Old Facebook folder
    fb_paths = [
        "crawlers/new_data/facebook/*.json"
    ]
    
    for pattern in fb_paths:
        full_pattern = os.path.join(SEARCH_ROOT, pattern)
        fb_files = glob.glob(full_pattern)
        console.print(f"Found {len(fb_files)} Facebook files in {pattern}.")
        
        for f in fb_files:
            try:
                with open(f, 'r') as file:
                    data = json.load(file)
                    if isinstance(data, list):
                        for item in data:
                            item['source_type'] = 'Facebook'
                            posts.append(item)
            except Exception as e:
                pass

    # 2. Load News Data
    news_paths = [
        "crawlers/old_data/news/**/*.csv",
        "crawlers/new_data/news/**/*.csv"
    ]
    
    for pattern in news_paths:
        full_pattern = os.path.join(SEARCH_ROOT, pattern)
        news_files = glob.glob(full_pattern, recursive=True)
        console.print(f"Found {len(news_files)} News files in {pattern}.")
        
        for f in news_files:
            try:
                df = pd.read_csv(f)
                records = df.to_dict('records')
                for item in records:
                    item['source_type'] = 'News'
                    posts.append(item)
            except Exception as e:
                pass

    console.print(f"[bold green]Total raw posts loaded: {len(posts)}[/bold green]")
    return posts

def clean_content(text):
    if not isinstance(text, str): return ""
    # Basic cleaning
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def sample_data(posts, n=300):
    # Filter out empty content
    valid_posts = [p for p in posts if isinstance(p.get('content'), str) and len(p.get('content', '')) > 20]
    
    if len(valid_posts) < n:
        console.print(f"[yellow]Warning: Only {len(valid_posts)} valid posts found. Sampling all.[/yellow]")
        return valid_posts
    
    return random.sample(valid_posts, n)

def main():
    posts = load_all_data()
    sampled = sample_data(posts, SAMPLE_SIZE)
    
    data_for_csv = []
    for i, p in enumerate(sampled):
        content = clean_content(p.get('content', ''))
        # Use title if content is short (common in news)
        if len(content) < 50 and 'title' in p and isinstance(p['title'], str):
             content = f"{p['title']} - {content}"
             
        row = {
            'id': i + 1,
            'source': p.get('source_type', 'Unknown'),
            'original_id': p.get('post_id') or p.get('url') or 'N/A',
            'published_date': p.get('created_time') or p.get('published_date') or 'N/A',
            'content_snippet': content[:500], # Trucate for readability in Excel
            'full_content': content, 
            'GROUND_TRUTH_EVENT_ID': '', # User fills this
            'NOTES': '' # User fills this
        }
        data_for_csv.append(row)
        
    df = pd.DataFrame(data_for_csv)
    
    # Save to CSV
    output_path = os.path.join(SEARCH_ROOT, "scripts", "eval", OUTPUT_FILE)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    console.print(f"[bold green]Successfully created annotation file at:[/bold green]")
    console.print(f"[blue]{output_path}[/blue]")
    console.print("\n[bold yellow]INSTRUCTIONS:[/bold yellow]")
    console.print("1. Open this CSV file in Excel or Google Sheets.")
    console.print("2. Read the 'content_snippet'.")
    console.print("3. Group similar events by assigning them the same ID in the [bold]GROUND_TRUTH_EVENT_ID[/bold] column.")
    console.print("   - Example: All posts about 'Gold Price' -> ID 1")
    console.print("   - All posts about 'Storm Yagi' -> ID 2")
    console.print("   - Unrelated/Noise posts -> ID -1")
    console.print("4. Save the file and run the [bold]calculate_metrics.py[/bold] script.")

if __name__ == "__main__":
    main()
