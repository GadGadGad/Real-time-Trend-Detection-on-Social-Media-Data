import json
import os
import csv
import re
from datetime import datetime

def clean_text(text):
    if not text: return ""
    return re.sub(r'\s+', ' ', str(text)).strip()

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified = []
            for item in data:
                text = item.get('text') or item.get('content') or ''
                time_str = item.get('time') or item.get('time_label') or ''
                # Prioritize structured timestamp if available
                p_timestamp = item.get('timestamp')
                published_at = p_timestamp if p_timestamp else time_str
                
                page_name = item.get('pageName') or item.get('page_name') or 'Unknown'
                
                clean = clean_text(text)
                if len(clean) < 20: continue # Filter noise
                
                unified.append({
                    "source": f"Face: {page_name}",
                    "content": clean,
                    "title": "",
                    "url": item.get('url') or item.get('postUrl') or '',
                    "stats": item.get('stats') or {'likes': item.get('likes', 0), 'comments': item.get('comments', 0), 'shares': item.get('shares', 0)},
                    "time": time_str,
                    "published_at": published_at,
                    "timestamp": p_timestamp
                })
            return unified
    except Exception as e:
        print(f"Error loading JSON {filepath}: {e}")
        return []

def load_social_data(files):
    all_data = []
    for f in files: all_data.extend(load_json(f))
    return all_data

def load_news_data(files):
    all_data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    content_str = f"{row.get('title', '')}\n{row.get('content', '')}"
                    if len(content_str.strip()) < 20: # Skip empty/short content
                        continue
                        
                    all_data.append({
                        "source": os.path.basename(os.path.dirname(f)).upper(),
                        "content": content_str,
                        "title": row.get('title', ''), "url": row.get('url', ''),
                        "stats": {'likes': 0, 'comments': 0, 'shares': 0},
                        "time": row.get('published_at', '')
                    })
        except: pass
    return all_data

def load_google_trends(csv_files):
    trends = {}
    for filepath in csv_files:
        # Extract timestamp from filename (e.g., 20251208-1452)
        file_time = None
        match = re.search(r'(\d{8}-\d{4})', os.path.basename(filepath))
        if match:
            try:
                file_time = datetime.strptime(match.group(1), "%Y%m%d-%H%M")
            except: pass
            
        # Support pre-refined JSON
        if filepath.endswith('.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # If loaded items don't have time, add file_time
                    for k, v in loaded.items():
                        if 'time' not in v and file_time:
                            v['time'] = file_time.isoformat()
                    trends.update(loaded)
                    continue
            except Exception as e:
                print(f"Error loading JSON {filepath}: {e}")
                continue
        
        # Original CSV parsing
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) 
                for row in reader:
                    if len(row) < 5: continue
                    main_trend = row[0].strip()
                    vol_str = row[1].strip()
                    # Parse volume
                    clean_vol = vol_str.upper().replace(',', '').replace('.', '')
                    multiplier = 1000 if 'N' in clean_vol or 'K' in clean_vol else (1000000 if 'M' in clean_vol or 'TR' in clean_vol else 1)
                    num_parts = re.findall(r'\d+', clean_vol)
                    volume = int(num_parts[0]) * multiplier if num_parts else 0
                    
                    keywords = [k.strip() for k in row[4].split(',') if k.strip()]
                    if main_trend not in keywords: keywords.insert(0, main_trend)
                    trends[main_trend] = {
                        "keywords": keywords, 
                        "volume": volume,
                        "time": file_time.isoformat() if file_time else None
                    }
        except Exception: pass
    return trends
