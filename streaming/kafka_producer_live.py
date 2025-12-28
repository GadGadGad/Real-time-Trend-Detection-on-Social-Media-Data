import sys
import os
import json
import time
import pandas as pd
from kafka import KafkaProducer
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

try:
    from src.scrapers.social.vnexpress_crawler import VnExpressCrawler
except ImportError:
    # Fallback if run from a different CWD
    sys.path.append(os.path.join(PROJECT_ROOT, "src", "scrapers", "social"))
    from vnexpress_crawler import VnExpressCrawler

# Kafka Config
KAFKA_BROKER = os.environ.get("KAFKA_BROKER_URL", "localhost:29092")
KAFKA_TOPIC = "posts_stream_v1"

def json_serializer(data):
    return json.dumps(data).encode("utf-8")

def run_live_producer(categories=['thoi-su', 'kinh-doanh'], pages=1):
    print(f"🚀 [Live Producer] Starting live crawl for {categories}...")
    
    # Setup temporary output for this run
    output_dir = Path(PROJECT_ROOT) / "data" / "live_crawl_temp"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize Crawler
    crawler = VnExpressCrawler(output_dir, use_cache=False) # Disable cache for live data
    
    # Run Crawler
    crawler.crawl(categories=categories, pages=pages, workers=5)
    
    # Read Resulting CSV
    csv_path = output_dir / "articles.csv"
    if not csv_path.exists():
        print("⚠️ [Live Producer] No articles.csv found. Crawler might have failed or found nothing.")
        return

    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"⚠️ [Live Producer] Failed to read CSV: {e}")
        return

    if df.empty:
        print("⚠️ [Live Producer] No new articles found.")
        return

    print(f"📦 [Live Producer] Found {len(df)} articles. Preparing to stream to Kafka...")

    # Connect to Kafka
    try:
        producer = KafkaProducer(
            bootstrap_servers=KAFKA_BROKER,
            value_serializer=json_serializer
        )
    except Exception as e:
        print(f"❌ [Live Producer] Failed to connect to Kafka at {KAFKA_BROKER}: {e}")
        return

    # Stream Posts
    success_count = 0
    for _, row in df.iterrows():
        # Map CSV columns to Pipeline format
        # Pipeline expects: {"content": "...", "source": "...", "timestamp": "...", "url": "..."}
        post = {
            "content": f"{row.get('title', '')} . {row.get('short_description', '')} . {row.get('content', '')}",
            "source": "VnExpress Live",
            "url": row.get('url', ''),
            "time": datetime.now().isoformat(), # Use current time for "live" feel
            "final_topic": row.get('category', 'General'), # Provide a suggestion
            "original_published_at": row.get('published_at', ''),
            "category": row.get('category', 'General')
        }
        
        try:
            producer.send(KAFKA_TOPIC, post)
            success_count += 1
        except Exception as e:
            print(f"⚠️ Failed to send post: {e}")

    producer.flush()
    print(f"✅ [Live Producer] Successfully sent {success_count} posts to topic '{KAFKA_TOPIC}'.")
    
    # Cleanup (Optional: remove temp file to keep it clean for next run)
    if csv_path.exists():
        os.remove(csv_path)

if __name__ == "__main__":
    # Allow arguments for categories
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--categories", nargs="+", default=['thoi-su', 'kinh-doanh'])
    parser.add_argument("--pages", type=int, default=1)
    args = parser.parse_args()
    
    run_live_producer(categories=args.categories, pages=args.pages)
