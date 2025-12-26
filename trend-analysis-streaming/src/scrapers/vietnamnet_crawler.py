"""
vietnamnet_csv_crawler.py (STEP 1: DISCOVERER - VietnamNet Edition)
- Crawls vietnamnet.vn categories.
- Pagination: /slug -> /slug-page2
- INPUTS: Category slugs (e.g., 'thoi-su', 'kinh-doanh')
- OUTPUTS: articles.csv
"""

import argparse
import csv
import json
import logging
import random
import time
import toml
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional

import rich.logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from src.utils.text_processing.utils import Cache, sha1, normalize_url

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[rich.logging.RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)]
)
log = logging.getLogger(__name__)

try:
    config = toml.load("config.toml")
    crawler_cfg = config.get("crawler", {})
    files_cfg = config.get("files", {})
except Exception:
    crawler_cfg = {}
    files_cfg = {}

BASE = "https://vietnamnet.vn"
MIN_SLEEP = crawler_cfg.get("min_sleep", 1.0)
MAX_SLEEP = crawler_cfg.get("max_sleep", 2.0)
HEADERS = {"User-Agent": crawler_cfg.get("user_agent", "Mozilla/5.0")}
DEFAULT_OUTPUT = files_cfg.get("default_output_dir", "data_vietnamnet")
MAX_WORKERS = crawler_cfg.get("max_workers", 10)

class VietnamNetCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.cache = Cache(output_dir / ".cache_vietnamnet", enabled=use_cache)
        self.article_csv = output_dir / "articles.csv"
        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()
        self.stats = {"saved": 0}
        self._load_seen()
        self._init_csvs()

    def _load_seen(self):
        if self.seen_file.exists():
            with open(self.seen_file, "r", encoding="utf-8") as f:
                self.seen_articles = {line.strip() for line in f}

    def _init_csvs(self):
        if not self.article_csv.exists():
            csv.writer(open(self.article_csv, "w", newline="", encoding="utf-8")).writerow(
                ["article_id", "url", "title", "short_description", "author", "category", "published_at", "content"]
            )

    def safe_get(self, url: str) -> Optional[str]:
        cached = self.cache.get(f"html:{url}")
        if cached: return cached["html"]
        try:
            r = self.session.get(url, headers=HEADERS, timeout=15)
            if r.status_code == 200:
                self.cache.set(f"html:{url}", {"html": r.text})
                return r.text
        except Exception: pass
        time.sleep(MIN_SLEEP)
        return None

    def discover_articles(self, category_slug: str, pages: int = 2, progress_context=None, task_id=None) -> List[dict]:
        articles_data = []
        category_slug = category_slug.strip("/")

        for p in range(1, pages + 1):

            if p == 1:
                url = f"{BASE}/{category_slug}"
            else:
                url = f"{BASE}/{category_slug}-page{p}"

            html = self.safe_get(url)
            if not html:
                if progress_context: progress_context.update(task_id, advance=1)
                continue

            soup = BeautifulSoup(html, "lxml")


            items = soup.select(".verticalPost, .horizontalPost, .feature-box, .vnn-title")

            for item in items:
                link_el = item.select_one("a")
                if not link_el: continue

                href = link_el.get("href")
                if not href: continue

                if href.startswith("/"): href = BASE + href

                # Filter out garbage
                if "vietnamnet.vn" not in href or "/video/" in href: continue

                url = normalize_url(href)
                desc = item.select_one(".summary, .sa-po")

                if url not in self.seen_articles:
                    articles_data.append({
                        "url": url,
                        "short_description": desc.text.strip() if desc else "",
                        "category_source": category_slug
                    })

            if progress_context: progress_context.update(task_id, advance=1)
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        return articles_data

    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        if url in self.seen_articles: return None
        html = self.safe_get(url)
        if not html: return None

        soup = BeautifulSoup(html, "lxml")

        title = soup.select_one(".content-detail h1, .article-detail h1")
        published = soup.select_one(".bread-crumb-detail .date, .article-relate .date, .bread-crumb-detail__time")
        content_el = soup.select_one("#maincontent, .maincontent, .content-detail")

        if content_el:
            # Clean generic embedded boxes
            for g in content_el.select(".inner-article, .related-box, .box-hightlight, .table-content"):
                g.decompose()


        bread = soup.select(".bread-crumb-detail ul li a")
        cat_text = bread[-1].text.strip() if bread else category_source


        author_text = ""
        if content_el:
            last_p = content_el.find_all('p')
            if last_p:

                 last = last_p[-1]
                 if last.find("strong") or last.find("b"):
                     author_text = last.text.strip()

        return {
            "article_id": sha1(url),
            "url": url,
            "title": title.text.strip() if title else "",
            "short_description": short_description,
            "author": author_text,
            "category": cat_text,
            "category": cat_text,
            "published_at": published.text.strip() if published else "",
            "content": content_el.get_text("\n", strip=True) if content_el else ""
        }

    def save_article(self, article: dict):
        try:
            with open(self.article_csv, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow([
                    article["article_id"], article["url"], article["title"],
                    article["short_description"], article["author"],
                    article["category"],
                    article["published_at"], article["content"]
                ])
            with open(self.seen_file, "a", encoding="utf-8") as f: f.write(f"{article['url']}\n")
            self.stats["saved"] += 1
            if hasattr(self, 'stream_callback'):
                # Chuyển đổi sang format mà Spark job mong đợi
                stream_data = {
                    "title": article["title"],
                    "content": article["content"],
                    "source": "VIETNAMNET", # Nhãn nguồn cho Dashboard
                    "url": article["url"],
                    "published_at": str(article["published_at"])
                }
                self.stream_callback(stream_data)
        except Exception: pass

    def crawl(self, categories: list[str], pages: int = 2, workers: int = MAX_WORKERS):
        log.info(f"Starting VietnamNet Crawler: {categories}")
        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(), console=console
        ) as progress:

            all_articles = []
            d_task = progress.add_task("Discovering...", total=len(categories)*pages)
            for cat in categories:
                all_articles.extend(self.discover_articles(cat, pages, progress, d_task))

            unique = [a for i, a in enumerate(all_articles) if a['url'] not in self.seen_articles and a['url'] not in [x['url'] for x in all_articles[:i]]]
            s_task = progress.add_task(f"Saving {len(unique)}", total=len(unique))

            with concurrent.futures.ThreadPoolExecutor(workers) as exc:
                futures = [exc.submit(self.fetch_article, a['url'], a['short_description'], a['category_source']) for a in unique]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res: self.save_article(res)
                    progress.update(s_task, advance=1)

        console.print(Panel(f"Done. Saved: {self.stats['saved']}", title="VietnamNet Summary", style="bold blue"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", nargs="+", required=True)
    parser.add_argument("-p", "--pages", type=int, default=2)
    parser.add_argument("-o", "--output", default="data_vietnamnet")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    VietnamNetCrawler(Path(args.output)).crawl(args.category, args.pages, args.workers)
