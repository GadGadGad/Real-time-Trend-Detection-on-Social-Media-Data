"""
tuoitre_csv_crawler.py (STEP 1: DISCOVERER - Tuoi Tre Edition)
- Crawls TuoiTre.vn categories to discover articles.
- Uses Selenium to click the 'Xem thêm' (View more) button to load more content.
  (TuoiTre doesn't expose traditional pagination URLs - accessing /trang-2.htm redirects to page 1)
- Uses Threading for concurrent article fetching.
- INPUTS: A list of categories (slugs) and number of 'Xem thêm' clicks.
- OUTPUTS: articles.csv (the "to-do list")
"""

import argparse
import csv
import json
import logging
import random
import time
import toml
import sys
import os
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional
from contextlib import nullcontext

# Selenium imports for infinite scroll handling
from selenium import webdriver
from selenium.webdriver.firefox.options import Options as FirefoxOptions
from selenium.webdriver.chrome.options import Options as ChromeOptions
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

import rich.logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
    log.info("Loaded settings from [bold cyan]config.toml[/bold cyan]")
except Exception:
    log.warning("config.toml not found. Using default values.")
    crawler_cfg = {}
    files_cfg = {}

BASE = "https://tuoitre.vn"
MIN_SLEEP = crawler_cfg.get("min_sleep", 1.0)
MAX_SLEEP = crawler_cfg.get("max_sleep", 2.0)
RETRY_COUNT = crawler_cfg.get("retry_count", 3)
HEADERS = {"User-Agent": crawler_cfg.get("user_agent", "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}
DEFAULT_OUTPUT = files_cfg.get("default_output_dir", "data_tuoitre")
MAX_WORKERS = crawler_cfg.get("max_workers", 10)


class TuoiTreCrawler:
    def __init__(self, output_dir: Path, use_cache=True, browser_type: str = "firefox"):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()
        self.browser_type = browser_type
        self.driver = None


        self.cache = Cache(output_dir / ".cache_tuoitre", enabled=use_cache)

        self.start_time = time.time()
        self.stats = {"cache_hits": 0, "articles_saved": 0}

        self.article_csv = output_dir / "articles.csv"
        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()

        self._load_seen()
        self._init_csvs()

    def _create_browser(self):
        """Creates a Selenium browser driver for infinite scroll handling."""
        if self.browser_type == "firefox":
            log.info("🦊 Initializing Firefox browser for scroll-based pagination...")
            options = FirefoxOptions()
            options.add_argument("--headless")  # Run in headless mode
            options.set_preference("general.useragent.override", HEADERS["User-Agent"])
            return webdriver.Firefox(options=options)
        elif self.browser_type in ["chrome", "chromium"]:
            log.info("🌐 Initializing Chrome/Chromium browser for scroll-based pagination...")
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
        """Loads previously scraped URLs to enable resuming."""
        if self.seen_file.exists():
            log.info("Loading previously seen URLs for resumability...")
            with open(self.seen_file, "r", encoding="utf-8") as f:
                self.seen_articles = {line.strip() for line in f}
            log.info(f"Loaded {len(self.seen_articles)} seen URLs. Will skip these.")

    def _init_csvs(self):
        """Initializes CSV files with headers if they don't exist."""
        def init(path, header):
            if not path.exists():
                csv.writer(open(path, "w", newline="", encoding="utf-8")).writerow(header)
                log.info(f"Created new file: [bold]{path.name}[/bold]")

        init(self.article_csv, ["article_id", "url", "title", "short_description", "author", "category", "published_at", "content"])

    def safe_get(self, url: str) -> Optional[str]:
        """Cached and retrying HTTP GET request."""
        cached = self.cache.get(f"html:{url}")
        if cached and "html" in cached:
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
                else:
                    log.warning(f"Failed to get {url} (Status: {r.status_code}). Retrying...")
            except Exception as e:
                log.warning(f"Error getting {url}: {e}. Retrying...")
            time.sleep(MIN_SLEEP + (i * MIN_SLEEP))

        log.error(f"Failed to get {url} after {RETRY_COUNT} retries.")
        return None

    def discover_articles(self, category_slug: str, pages: int = 2, progress_context=None, task_id=None) -> List[dict]:
        """
        Discovers articles from category pages using Selenium.
        TuoiTre uses a 'Xem thêm' (View more) button to load more content.
        The 'pages' parameter is repurposed as the number of button click iterations.
        """
        articles_data = []
        seen_urls_in_session = set()

        # Clean slug input (e.g., remove .htm if user added it)
        category_slug = category_slug.replace(".htm", "").strip("/")
        
        # Use base URL (page 1) - TuoiTre redirects all trang-X URLs to page 1 anyway
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
                    view_more_btn = None
                    selectors = [
                        "a.view-more",
                        "a[class*='view-more']",
                        "//a[contains(text(), 'Xem thêm')]",
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
                        # Click the button
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
            
            # Parse all loaded articles
            html = self.driver.page_source
            soup = BeautifulSoup(html, "lxml")
            

            items = soup.select(".box-category-item, .list-news-content .news-item, .box-news-layout")
            
            for item in items:
                link_el = item.select_one("a.box-category-link-title, a.focus-link, a.box-name-link")
                desc_el = item.select_one(".box-content-brief, .sapo")

                if not link_el:
                    continue

                href = link_el.get("href")
                if href:
                    if href.startswith("/"):
                        href = BASE + href

                    # Skip non-article links (e.g., videos, podcasts if structure differs)
                    if "tuoitre.vn/video" in href or "tuoitre.vn/podcast" in href:
                        continue

                    article_url = normalize_url(href)
                    if article_url not in seen_urls_in_session and article_url not in self.seen_articles:
                        seen_urls_in_session.add(article_url)
                        description = desc_el.text.strip() if desc_el else ""
                        articles_data.append({
                            "url": article_url,
                            "short_description": description,
                            "category_source": category_slug
                        })
            
            log.info(f"Found {len(articles_data)} articles after {click_count} 'Xem thêm' clicks")
            
        except Exception as e:
            log.error(f"Error during Selenium discovery: {e}")
        
        return articles_data


    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        """Fetches a single article's HTML metadata."""
        if url in self.seen_articles:
            return None

        html = self.safe_get(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")


        title = soup.select_one("h1.detail-title, h1.article-title")


        published = soup.select_one(".detail-time, .date-time")

        content_el = soup.select_one("#main-detail-body, .detail-content, .fck_detail")


        if content_el:
            for garbage in content_el.select(".VCSortableInPreviewMode, .relate-container, .box-hightlight"):
                garbage.decompose()




        category_text = category_source
        breadcrumb = soup.select("ul.bread-crumbs li a, .breadcrumbs a")
        if breadcrumb:

            category_text = breadcrumb[-1].text.strip()


        author_text = ""
        author_el = soup.select_one(".author-info, .author")
        if author_el:
            author_text = author_el.text.strip()


        if not author_text and content_el:
            last_p = content_el.select("p")
            if last_p:
                possible_author = last_p[-1].text.strip()
                if len(possible_author) < 50 and not possible_author.endswith("."):
                    author_text = possible_author

        article = {
            "article_id": sha1(url),
            "url": url,
            "title": title.text.strip() if title else "",
            "short_description": short_description,
            "author": author_text,
            "category": category_text,
            "category": category_text,
            "published_at": published.text.strip() if published else "",
            "content": content_el.get_text("\n", strip=True) if content_el else "",
        }
        return article

    def save_article(self, article: dict):
        """Saves one article to articles.csv."""
        try:
            writer = csv.writer(open(self.article_csv, "a", newline="", encoding="utf-8"))
            writer.writerow([
                article["article_id"], article["url"], article["title"],
                article["short_description"], article["author"],
                article["category"],
                article["published_at"], article["content"]
            ])
            with open(self.seen_file, "a", encoding="utf-8") as f:
                f.write(f"{article['url']}\n")
            self.stats["articles_saved"] += 1
            if hasattr(self, 'stream_callback'):
                # Chuyển đổi sang format mà Spark job mong đợi
                stream_data = {
                    "title": article["title"],
                    "content": article["content"],
                    "source": "TUOITRE", # Nhãn nguồn cho Dashboard
                    "url": article["url"],
                    "published_at": str(article["published_at"])
                }
                self.stream_callback(stream_data)
        except Exception as e:
            log.error(f"Failed to save article {article.get('url')}: {e}")

    def print_summary(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        summary = (
            f"Discovery Complete (Tuoi Tre) ✨\n\n"
            f"[bold green]New Articles Found:[/bold green] {self.stats['articles_saved']}\n"
            f"Total Time: {total_time:.2f} seconds\n"
            f"Cache Hits: {self.stats['cache_hits']}\n"
            f"Output: [italic]{self.article_csv.resolve()}[/italic]"
        )
        console.print(Panel(summary, title="Tuoi Tre Summary", border_style="bold blue", padding=(1, 2)))

    def crawl(self, categories: list[str], pages: int = 2, workers: int = MAX_WORKERS, no_progress: bool = False):
        """Main crawl orchestration function."""

        def fetch_and_save_article(article_info):
            try:
                article = self.fetch_article(
                    article_info["url"],
                    article_info["short_description"],
                    article_info["category_source"]
                )
                if article:
                    self.save_article(article)
                    return 1
            except Exception as e:
                log.error(f"Critical error crawling {article_info['url']}: {e}", exc_info=False)
            return 0

        progress_manager = (
            Progress(
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TimeRemainingColumn(),
                "•",
                TimeElapsedColumn(),
                console=console,
                transient=False,
            )
            if not no_progress else nullcontext()
        )
        if no_progress:
            log.info("Discovering articles (progress bar disabled)...")

        with progress_manager as progress:
            all_articles_data = []

            log.info("Running with Selenium-based infinite scroll pagination.")
            discover_task_id = progress.add_task(f"[cyan]Discovering...", total=len(categories) * pages) if not no_progress else None

            for category in categories:
                log.info(f"Discovering articles in [cyan]{category}[/cyan]...")
                articles_data = self.discover_articles(
                    category,
                    pages,
                    progress_context=progress,
                    task_id=discover_task_id
                )
                all_articles_data.extend(articles_data)
                log.info(f" > Found {len(articles_data)} items in [cyan]{category}[/cyan].")

            unique_new_articles = []
            seen_in_this_run = set()
            for article in all_articles_data:
                url = article["url"]
                if url not in self.seen_articles and url not in seen_in_this_run:
                    unique_new_articles.append(article)
                    seen_in_this_run.add(url)

            log.info(f"Total found: {len(all_articles_data)}. Unique new: [bold green]{len(unique_new_articles)}[/bold green]")

            if not unique_new_articles:
                log.warning("No new articles to save.")
                self.print_summary()
                return

            crawl_task_id = progress.add_task(f"[green]Saving {len(unique_new_articles)} articles", total=len(unique_new_articles)) if not no_progress else None
            if no_progress:
                log.info(f"Saving {len(unique_new_articles)} articles...")

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(fetch_and_save_article, unique_new_articles)
                if not no_progress:
                    for _ in results:
                        progress.update(crawl_task_id, advance=1)
                else:
                    for _ in results: pass


        self._close_browser()
        
        log.info("Crawl complete.")
        self.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Tuoi Tre Article Discoverer",
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--category", "-c",
        required=True,
        nargs="+",
        help="Categories slugs (e.g., 'the-gioi', 'cong-nghe')"
    )
    parser.add_argument("--pages", "-p", type=int, default=5, help="Number of scroll iterations per category (default: 5)")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help=f"Workers (default: {MAX_WORKERS})")
    parser.add_argument("--browser", "-b", type=str, choices=["firefox", "chrome", "chromium"], default="firefox",
        help="Browser to use for scrolling (default: firefox)")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    console.rule(f"[bold]Tuoi Tre Crawler[/bold]: [cyan]{', '.join(args.category)}[/cyan]")

    c = TuoiTreCrawler(Path(args.output), use_cache=(not args.no_cache), browser_type=args.browser)
    c.crawl(args.category, args.pages * 5 + 1, workers=args.workers)
