"""
nld_csv_crawler.py (STEP 1: DISCOVERER - Nguoi Lao Dong "Robust" Edition)
- Crawls nld.com.vn categories.
- Improved Selectors for better article detection.
- INPUTS: Category slugs (e.g., 'thoi-su', 'kinh-te')
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
from contextlib import nullcontext

import rich.logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn

from utils import Cache, sha1, normalize_url

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

BASE = "https://nld.com.vn"
# NLD cần sleep lâu hơn xíu để tránh bị chặn connection
MIN_SLEEP = crawler_cfg.get("min_sleep", 1.5)
MAX_SLEEP = crawler_cfg.get("max_sleep", 2.5)
RETRY_COUNT = crawler_cfg.get("retry_count", 3)
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Referer": "https://nld.com.vn/",
}
DEFAULT_OUTPUT = files_cfg.get("default_output_dir", "data_nld")
MAX_WORKERS = crawler_cfg.get("max_workers", 10)

class NLDCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.cache = Cache(output_dir / ".cache_nld", enabled=use_cache)
        self.start_time = time.time()
        self.stats = {"cache_hits": 0, "articles_saved": 0}
        self.article_csv = output_dir / "articles.csv"
        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()
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
        if cached:
            self.stats["cache_hits"] += 1
            return cached["html"]
        for i in range(RETRY_COUNT):
            try:
                r = self.session.get(url, headers=HEADERS, timeout=15)
                if r.status_code == 200:
                    self.cache.set(f"html:{url}", {"html": r.text})
                    return r.text
                elif r.status_code == 404:
                    log.warning(f"404 Not Found: {url}")
                    return None
            except Exception as e:
                log.warning(f"Error requesting {url}: {e}")
            time.sleep(MIN_SLEEP + i)
        return None

    def discover_articles(self, category_slug: str, pages: int = 2, progress_context=None, task_id=None) -> List[dict]:
        articles_data = []
        category_slug = category_slug.replace(".htm", "").strip("/")

        for p in range(1, pages + 1):
            if p == 1:
                url = f"{BASE}/{category_slug}.htm"
            else:
                url = f"{BASE}/{category_slug}/trang-{p}.htm"

            html = self.safe_get(url)
            if not html:
                if progress_context: progress_context.update(task_id, advance=1)
                continue

            soup = BeautifulSoup(html, "lxml")

            # --- CHIẾN THUẬT QUÉT RỘNG (ROBUST SELECTORS) ---
            # Thay vì tìm div cha, tìm thẳng thẻ a có chứa tiêu đề
            # NLD hay dùng: h2.title-news a, h3.title-news a, .news-item__title a
            link_elements = soup.select(
                "h2.title-news a, h3.title-news a, .news-item__title a, .box-category-item .box-category-link-title"
            )

            found_on_page = 0
            for link_el in link_elements:
                href = link_el.get("href")
                title = link_el.get("title") or link_el.text.strip()

                if not href: continue

                # Xử lý link tương đối/tuyệt đối
                if href.startswith("/"):
                    href = BASE + href
                elif not href.startswith("http"):
                    continue # Bỏ qua link javascript hoặc rác

                # Lọc rác (Video, Magazine, Quảng cáo)
                if "nld.com.vn" not in href or "/video" in href or "e-magazine" in href:
                    continue

                url = normalize_url(href)

                # Tìm đoạn mô tả ngắn (Sapo) - Cố gắng tìm thẻ p hoặc div sapo gần link đó nhất
                # Cách này hơi hacky: Tìm cha của link, rồi tìm sapo trong cha đó
                short_desc = ""
                parent = link_el.find_parent(["div", "li"], class_=lambda x: x and ("news" in x or "item" in x))
                if parent:
                    desc_el = parent.select_one(".sapo, .news-sapo, .box-category-sapo")
                    if desc_el:
                        short_desc = desc_el.text.strip()

                if url not in self.seen_articles:
                    articles_data.append({
                        "url": url,
                        "short_description": short_desc,
                        "category_source": category_slug
                    })
                    found_on_page += 1

            # Debug log nếu trang rỗng (giúp ông biết sai ở đâu)
            if found_on_page == 0:
                log.warning(f"⚠️ Không tìm thấy bài nào tại: {url} (Check lại Slug category xem đúng chưa?)")

            if progress_context: progress_context.update(task_id, advance=1)
            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        return articles_data

    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        if url in self.seen_articles: return None
        html = self.safe_get(url)
        if not html: return None

        soup = BeautifulSoup(html, "lxml")

        # 1. Title (Thường là h1)
        title = soup.select_one("h1.title-content, h1.title-detail")

        # 2. Date
        published = soup.select_one(".date-time, .ad-item-time")

        # 3. Content (Quan trọng nhất)
        # NLD nội dung nằm trong .content-news-detail hoặc .detail-content
        content_el = soup.select_one(".content-news-detail, .detail-content, .news-content")

        if content_el:
            # Xóa rác: Quảng cáo, Tin liên quan chèn giữa bài
            for garbage in content_el.select(".box-ads, .news-relate, .VCSortableInPreviewMode, .related-container"):
                garbage.decompose()



        # 5. Breadcrumb / Category
        category_text = category_source
        bread = soup.select(".breadcrumb-item a, .breadcrumb a")
        if bread:
            category_text = bread[-1].text.strip()

        # 6. Author
        author_text = ""
        # Tên tác giả thường ở cuối hoặc đầu
        author_el = soup.select_one(".author-info .name, .author")
        if author_el:
            author_text = author_el.text.strip()
        elif content_el:
             # Fallback: Tìm đoạn văn cuối cùng in đậm hoặc căn phải
             last_p = content_el.find_all('p')
             if last_p:
                 txt = last_p[-1].text.strip()
                 if len(txt) < 50: author_text = txt

        return {
            "article_id": sha1(url),
            "url": url,
            "title": title.text.strip() if title else "",
            "short_description": short_description,
            "author": author_text,
            "category": category_text,
            "category": category_text,
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
            self.stats["articles_saved"] += 1
        except Exception: pass

    def crawl(self, categories: list[str], pages: int = 2, workers: int = MAX_WORKERS):
        log.info(f"Starting NLD Crawler (Robust Mode): {categories}")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(), console=console
        ) as progress:

            # Phase 1: Discovery
            all_articles = []
            disc_task = progress.add_task("Discovering...", total=len(categories)*pages)
            for cat in categories:
                all_articles.extend(self.discover_articles(cat, pages, progress, disc_task))

            if not all_articles:
                log.warning("Không tìm thấy bài nào. Kiểm tra lại slug category hoặc IP có bị chặn không.")
                return

            # Phase 2: Fetch & Save
            unique = [a for i, a in enumerate(all_articles) if a['url'] not in self.seen_articles and a['url'] not in [x['url'] for x in all_articles[:i]]]

            if not unique:
                log.info("Tất cả bài viết đã được cào từ trước.")
                return

            save_task = progress.add_task(f"Saving {len(unique)} articles", total=len(unique))

            with concurrent.futures.ThreadPoolExecutor(workers) as exc:
                futures = [exc.submit(self.fetch_article, a['url'], a['short_description'], a['category_source']) for a in unique]
                for f in concurrent.futures.as_completed(futures):
                    res = f.result()
                    if res: self.save_article(res)
                    progress.update(save_task, advance=1)

        console.print(Panel(f"Done. Saved: {self.stats['articles_saved']}", title="NLD Summary", style="green"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", nargs="+", required=True)
    parser.add_argument("-p", "--pages", type=int, default=2)
    parser.add_argument("-o", "--output", default="data_nld")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS)
    args = parser.parse_args()

    NLDCrawler(Path(args.output)).crawl(args.category, args.pages, args.workers)
