import json
import logging
import random
import time
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from kafka import KafkaProducer
import rich.logging
from rich.console import Console
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from utils import Cache, sha1, normalize_url

console = Console()
logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[rich.logging.RichHandler(show_path=False)])
log = logging.getLogger(__name__)

BASE = "https://vietnamnet.vn"
producer = KafkaProducer(
    bootstrap_servers=['localhost:9092'],
    value_serializer=lambda x: json.dumps(x).encode('utf-8')
)

class VietnamNetCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.cache = Cache(output_dir / ".cache_vietnamnet", enabled=use_cache)
        self.seen_articles = set()

    def safe_get(self, url):
        cached = self.cache.get(f"html:{url}")
        if cached: return cached["html"]
        r = self.session.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=15)
        if r.status_code == 200:
            self.cache.set(f"html:{url}", {"html": r.text})
            return r.text
        return None

    def discover_articles(self, category_slug, pages, progress_context=None, task_id=None):
        articles_data = []
        for p in range(1, pages + 1):
            url = f"{BASE}/{category_slug}" if p == 1 else f"{BASE}/{category_slug}-page{p}"
            html = self.safe_get(url)
            if not html: continue

            soup = BeautifulSoup(html, "lxml")
            for item in soup.select(".verticalPost, .horizontalPost"):
                link_el = item.select_one("a")
                if link_el and link_el.get("href"):
                    href = link_el.get("href")
                    if href.startswith("/"): href = BASE + href
                    articles_data.append({"url": normalize_url(href), "short_description": "", "category_source": category_slug})
            if progress_context: progress_context.update(task_id, advance=1)
            time.sleep(1)
        return articles_data

    def fetch_article(self, url, short_description, category_source):
        html = self.safe_get(url)
        if not html: return None
        soup = BeautifulSoup(html, "lxml")
        title = soup.select_one(".content-detail h1, .article-detail h1")
        published = soup.select_one(".bread-crumb-detail .date")
        content_el = soup.select_one("#maincontent, .content-detail")
        
        return {
            "source": "vietnamnet",
            "url": url,
            "title": title.text.strip() if title else "",
            "category": category_source,
            "published_at": published.text.strip() if published else "",
            "content": content_el.get_text("\n", strip=True) if content_el else ""
        }

    def save_article(self, article):
        producer.send('news_data', value=article)
        self.seen_articles.add(article['url'])

    def crawl(self, categories, pages=2, workers=5):
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