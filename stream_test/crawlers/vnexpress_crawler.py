import argparse
import json
import logging
import random
import time
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional
from datetime import datetime
from dateutil.relativedelta import relativedelta
from kafka import KafkaProducer
import rich.logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from utils import Cache, sha1, normalize_url, resolve_category_id

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[rich.logging.RichHandler(show_path=False)])
log = logging.getLogger(__name__)

config = {} # Giả lập hoặc load từ file nếu cần
BASE = "https://vnexpress.net/"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Khởi tạo Kafka
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

class VnExpressCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.cache = Cache(output_dir / ".cache", enabled=use_cache)
        self.seen_articles = set()

    def safe_get(self, url: str) -> Optional[str]:
        cached = self.cache.get(f"html:{url}")
        if cached: return cached["html"]
        r = self.session.get(url, headers=HEADERS, timeout=15)
        if r.status_code == 200:
            self.cache.set(f"html:{url}", {"html": r.text})
            return r.text
        return None

    def discover_articles(self, category_id: str, pages: int = 2, progress_context=None, task_id=None, from_ts=None, to_ts=None) -> List[dict]:
        articles_data = []
        for p in range(1, pages + 1):
            if from_ts and to_ts:
                url = f"{BASE}/category/day/cateid/{category_id}/fromdate/{from_ts}/todate/{to_ts}/allcate/{category_id}/page/{p}"
            else:
                url = f"{BASE}/{category_id}-p{p}"

            html = self.safe_get(url)
            if not html: continue

            soup = BeautifulSoup(html, "lxml")
            for article_block in soup.select("article.item-news"):
                link_el = article_block.select_one("h3.title-news a, .title-news a, a.article__link")
                if not link_el: continue
                href = link_el.get("href")
                if href:
                    if href.startswith("/"): href = BASE + href
                    url = normalize_url(href)
                    articles_data.append({
                        "url": url,
                        "short_description": "",
                        "category_source": category_id
                    })
            if progress_context: progress_context.update(task_id, advance=1)
            time.sleep(random.uniform(0.5, 1.0))
        return articles_data

    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        html = self.safe_get(url)
        if not html: return None
        soup = BeautifulSoup(html, "lxml")
        title = soup.select_one("h1.title_news_detail, h1.title-detail")
        published = soup.select_one(".date, span.date")
        content_el = soup.select_one(".fck_detail, .sidebar_1 .Normal")
        
        return {
            "source": "vnexpress",
            "url": url,
            "title": title.text.strip() if title else "",
            "short_description": short_description,
            "category": category_source,
            "published_at": published.text.strip() if published else "",
            "content": content_el.get_text("\n", strip=True) if content_el else ""
        }

    def save_article(self, article: dict):
        # LOGIC KAFKA Ở ĐÂY
        producer.send('news_data', value=article)
        self.seen_articles.add(article['url'])

    def crawl(self, categories: list[str], pages: int = 2, workers: int = 5):
        # Logic crawl giữ nguyên
        with Progress(TextColumn("{task.description}"), BarColumn(), TimeRemainingColumn(), console=console) as progress:
            all_data = []
            d_task = progress.add_task("Discovering...", total=len(categories)*pages)
            for cat in categories:
                all_data.extend(self.discover_articles(cat, pages, progress, d_task))
            
            unique = [a for a in all_data if a['url'] not in self.seen_articles]
            s_task = progress.add_task(f"Sending {len(unique)}...", total=len(unique))
            
            with concurrent.futures.ThreadPoolExecutor(workers) as exc:
                futures = [exc.submit(self.fetch_article, a['url'], a['short_description'], a['category_source']) for a in unique]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res: 
                        self.save_article(res)
                        progress.update(s_task, advance=1)
        producer.flush()