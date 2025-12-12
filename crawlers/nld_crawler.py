"""
nld_csv_crawler.py (STEP 1: DISCOVERER - Nguoi Lao Dong "Robust" Edition)
- Crawls nld.com.vn categories.
- Uses Selenium to click the 'Xem thêm' (View more) button to load more content.
  (NLD doesn't expose traditional pagination URLs)
- INPUTS: Category slugs (e.g., 'thoi-su', 'kinh-te')
- OUTPUTS: articles.csv
"""

import argparse
import csv
import json
import logging
import os
import random
import time
import toml
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional
from contextlib import nullcontext

# Selenium imports for dynamic content loading
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By

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
    def __init__(self, output_dir: Path, use_cache=True, browser_type: str = "firefox"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.browser_type = browser_type
        self.driver = None  # Will be initialized when needed
        self.cache = Cache(output_dir / ".cache_nld", enabled=use_cache)
        self.start_time = time.time()
        self.stats = {"cache_hits": 0, "articles_saved": 0}
        self.article_csv = output_dir / "articles.csv"
        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()
        self._load_seen()
        self._init_csvs()

    def _create_browser(self):
        """Creates a Selenium browser driver for dynamic content loading."""
        if self.browser_type == "firefox":
            log.info("🦊 Initializing Firefox browser for 'Xem thêm' button clicks...")
            options = FirefoxOptions()
            options.add_argument("--headless")
            options.set_preference("general.useragent.override", HEADERS["User-Agent"])
            return webdriver.Firefox(options=options)
        elif self.browser_type in ["chrome", "chromium"]:
            log.info("🌐 Initializing Chrome/Chromium browser for 'Xem thêm' button clicks...")
            options = ChromeOptions()
            options.add_argument("--headless")
            options.add_argument("--no-sandbox")
            options.add_argument("--disable-dev-shm-usage")
            options.add_argument(f"--user-agent={HEADERS['User-Agent']}")
            if self.browser_type == "chromium":
                chromium_paths = ["/usr/bin/chromium", "/usr/bin/chromium-browser", "/snap/bin/chromium"]
                for path in chromium_paths:
                    if os.path.exists(path):
                        options.binary_location = path
                        break
            return webdriver.Chrome(options=options)
        else:
            raise ValueError(f"Unsupported browser: {self.browser_type}")

    def _close_browser(self):
        """Closes the browser driver if it's open."""
        if self.driver:
            try:
                self.driver.quit()
            except Exception:
                pass
            self.driver = None

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
        """
        Discovers articles from category pages using Selenium.
        NLD uses a 'Xem thêm' (View more) button to load more content.
        The 'pages' parameter is repurposed as the number of button click iterations.
        """
        articles_data = []
        if category_slug == "the-gioi":
            category_slug = "quoc-te"
        category_slug = category_slug.replace(".htm", "").strip("/")

        # Use base URL (page 1)
        url = f"{BASE}/{category_slug}.htm"
        
        log.info(f"Using Selenium with 'Xem thêm' button clicks on [cyan]{url}[/cyan]")
        
        # Initialize browser if not already done
        if not self.driver:
            self.driver = self._create_browser()
        
        try:
            self.driver.get(url)
            time.sleep(2)  # Wait for initial page load
            
            click_count = 0
            no_button_count = 0
            
            # Click "Xem thêm" button multiple times to load more content
            while click_count < pages:
                try:
                    # Scroll down a bit to ensure the button is visible
                    self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight - 500);")
                    time.sleep(0.5)
                    
                    # Find and click the "Xem thêm" button
                    # Using multiple selectors for robustness
                    view_more_btn = None
                    selectors = [
                        "a.view-more",
                        "a[class*='view-more']",
                        ".btn-viewmore",
                        "//a[contains(text(), 'Xem thêm')]",
                        "//button[contains(text(), 'Xem thêm')]",
                    ]
                    
                    for selector in selectors:
                        try:
                            if selector.startswith("//"):
                                view_more_btn = self.driver.find_element(By.XPATH, selector)
                            else:
                                view_more_btn = self.driver.find_element(By.CSS_SELECTOR, selector)
                            if view_more_btn and view_more_btn.is_displayed():
                                break
                        except:
                            continue
                    
                    if view_more_btn and view_more_btn.is_displayed():
                        # Click the button using JavaScript for reliability
                        self.driver.execute_script("arguments[0].click();", view_more_btn)
                        click_count += 1
                        no_button_count = 0
                        log.info(f"Clicked 'Xem thêm' button ({click_count}/{pages})")
                        time.sleep(random.uniform(1.5, 2.5))  # Wait for content to load
                        
                        if progress_context and task_id:
                            progress_context.update(task_id, advance=1)
                    else:
                        no_button_count += 1
                        if no_button_count >= 3:
                            log.info(f"'Xem thêm' button no longer available after {click_count} clicks")
                            break
                        time.sleep(1)
                        
                except Exception as e:
                    no_button_count += 1
                    if no_button_count >= 3:
                        log.info(f"Could not find 'Xem thêm' button after {click_count} clicks: {e}")
                        break
                    time.sleep(1)
            
            # Parse all loaded articles from the page source
            html = self.driver.page_source
            soup = BeautifulSoup(html, "lxml")

            link_elements = soup.select(
                "h2.title-news a, h3.title-news a, .news-item__title a, .box-category-item .box-category-link-title"
            )

            for link_el in link_elements:
                href = link_el.get("href")
                title = link_el.get("title") or link_el.text.strip()

                if not href: continue

                if href.startswith("/"):
                    href = BASE + href
                elif not href.startswith("http"):
                    continue


                if "nld.com.vn" not in href or "/video" in href or "e-magazine" in href:
                    continue

                article_url = normalize_url(href)

                short_desc = ""
                parent = link_el.find_parent(["div", "li"], class_=lambda x: x and ("news" in x or "item" in x))
                if parent:
                    desc_el = parent.select_one(".sapo, .news-sapo, .box-category-sapo")
                    if desc_el:
                        short_desc = desc_el.text.strip()

                if article_url not in self.seen_articles:
                    articles_data.append({
                        "url": article_url,
                        "short_description": short_desc,
                        "category_source": category_slug
                    })

            log.info(f"Found {len(articles_data)} articles after {click_count} 'Xem thêm' clicks")
            
        except Exception as e:
            log.error(f"Error during Selenium discovery: {e}")
        
        return articles_data

    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        if url in self.seen_articles: return None
        html = self.safe_get(url)
        if not html: return None

        soup = BeautifulSoup(html, "lxml")


        title = soup.select_one("h1.title-content, h1.title-detail")


        published = soup.select_one(".date-time, .ad-item-time")

        content_el = soup.select_one(".content-news-detail, .detail-content, .news-content")

        if content_el:

            for garbage in content_el.select(".box-ads, .news-relate, .VCSortableInPreviewMode, .related-container"):
                garbage.decompose()




        category_text = category_source
        bread = soup.select(".breadcrumb-item a, .breadcrumb a")
        if bread:
            category_text = bread[-1].text.strip()


        author_text = ""
        author_el = soup.select_one(".author-info .name, .author")
        if author_el:
            author_text = author_el.text.strip()
        elif content_el:
             # Fallback
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
        log.info(f"Starting NLD Crawler (Selenium Mode): {categories}")

        with Progress(
            TextColumn("[progress.description]{task.description}"),
            BarColumn(), "[progress.percentage]{task.percentage:>3.1f}%",
            TimeRemainingColumn(), console=console
        ) as progress:

            # Discovery
            all_articles = []
            disc_task = progress.add_task("Discovering...", total=len(categories)*pages)
            for cat in categories:
                all_articles.extend(self.discover_articles(cat, pages, progress, disc_task))

            if not all_articles:
                log.warning("Không tìm thấy bài nào. Kiểm tra lại slug category hoặc IP có bị chặn không.")
                return

            # Fetch & Save
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


        self._close_browser()
        
        console.print(Panel(f"Done. Saved: {self.stats['articles_saved']}", title="NLD Summary", style="green"))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--category", nargs="+", required=True)
    parser.add_argument("-p", "--pages", type=int, default=5, help="Number of 'Xem thêm' button clicks")
    parser.add_argument("-o", "--output", default="data_nld")
    parser.add_argument("-w", "--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("-b", "--browser", type=str, choices=["firefox", "chrome", "chromium"], default="firefox",
        help="Browser to use for scrolling (default: firefox)")
    args = parser.parse_args()

    NLDCrawler(Path(args.output), browser_type=args.browser).crawl(args.category, args.pages, args.workers)
