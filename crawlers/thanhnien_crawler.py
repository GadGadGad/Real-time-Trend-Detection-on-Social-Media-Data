"""
thanhnien_csv_crawler.py (STEP 1: DISCOVERER - Thanh Nien Edition)
- Crawls ThanhNien.vn categories.
- Pagination logic: /slug.htm (page 1) -> /slug/trang-2.htm (page 2+)
- INPUTS: Category slugs (e.g., 'cong-nghe-game', 'thoi-su')
- OUTPUTS: articles.csv
"""

import argparse
import csv
import json
import logging
import random
import time
import toml
import sys
from pathlib import Path
import requests
from bs4 import BeautifulSoup
import concurrent.futures
from typing import List, Optional
from contextlib import nullcontext

import rich.logging
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn

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
    log.info("Loaded settings from [bold cyan]config.toml[/bold cyan]")
except Exception:
    log.warning("config.toml not found. Using default values.")
    crawler_cfg = {}
    files_cfg = {}

BASE = "https://thanhnien.vn"
MIN_SLEEP = crawler_cfg.get("min_sleep", 1.0)
MAX_SLEEP = crawler_cfg.get("max_sleep", 2.0)
RETRY_COUNT = crawler_cfg.get("retry_count", 3)
HEADERS = {"User-Agent": crawler_cfg.get("user_agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")}
DEFAULT_OUTPUT = files_cfg.get("default_output_dir", "data_thanhnien")
MAX_WORKERS = crawler_cfg.get("max_workers", 10)


class ThanhNienCrawler:
    def __init__(self, output_dir: Path, use_cache=True):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session = requests.Session()


        self.cache = Cache(output_dir / ".cache_thanhnien", enabled=use_cache)

        self.start_time = time.time()
        self.stats = {"cache_hits": 0, "articles_saved": 0}

        self.article_csv = output_dir / "articles.csv"
        self.seen_file = self.output_dir / ".seen_articles.txt"
        self.seen_articles = set()

        self._load_seen()
        self._init_csvs()

    def _load_seen(self):
        if self.seen_file.exists():
            log.info("Loading previously seen URLs...")
            with open(self.seen_file, "r", encoding="utf-8") as f:
                self.seen_articles = {line.strip() for line in f}
            log.info(f"Loaded {len(self.seen_articles)} seen URLs.")

    def _init_csvs(self):
        def init(path, header):
            if not path.exists():
                csv.writer(open(path, "w", newline="", encoding="utf-8")).writerow(header)
                log.info(f"Created new file: [bold]{path.name}[/bold]")

        init(self.article_csv, ["article_id", "url", "title", "short_description", "author", "category", "published_at", "content"])

    def safe_get(self, url: str) -> Optional[str]:
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
                    log.warning(f"Failed {url} (Status: {r.status_code}). Retrying...")
            except Exception as e:
                log.warning(f"Error {url}: {e}. Retrying...")
            time.sleep(MIN_SLEEP + (i * MIN_SLEEP))

        log.error(f"Failed to get {url} after {RETRY_COUNT} retries.")
        return None

    def discover_articles(self, category_slug: str, pages: int = 2, progress_context=None, task_id=None) -> List[dict]:
        """Discovers articles using Thanh Nien pagination."""
        articles_data = []
        seen_urls_in_session = set()

        # Clean input
        category_slug = category_slug.replace(".htm", "").strip("/")

        for p in range(1, pages + 1):

            if p == 1:
                url = f"{BASE}/{category_slug}.htm"
            else:
                url = f"{BASE}/{category_slug}/trang-{p}.htm"

            html = self.safe_get(url)
            if not html:
                if progress_context and task_id:
                    progress_context.update(task_id, advance=1)
                continue

            soup = BeautifulSoup(html, "lxml")
            found_on_page = 0

            items = soup.select(".box-category-item, .story, .timeline-item")

            for item in items:
                link_el = item.select_one("a.box-category-link-title, a.story__title, a.title")
                desc_el = item.select_one(".box-category-sapo, .story__summary, .summary")

                if not link_el:
                    continue

                href = link_el.get("href")
                if href:
                    if href.startswith("/"):
                        href = BASE + href

            # Skip video, podcast, magazine
                    if "/video/" in href or "/podcast/" in href:
                        continue

                    url = normalize_url(href)
                    if url not in seen_urls_in_session and url not in self.seen_articles:
                        seen_urls_in_session.add(url)
                        description = desc_el.text.strip() if desc_el else ""
                        articles_data.append({
                            "url": url,
                            "short_description": description,
                            "category_source": category_slug
                        })
                        found_on_page += 1

            if progress_context and task_id:
                progress_context.update(task_id, advance=1)

            time.sleep(random.uniform(MIN_SLEEP, MAX_SLEEP))

        return articles_data

    def fetch_article(self, url: str, short_description: str, category_source: str) -> Optional[dict]:
        if url in self.seen_articles:
            return None

        html = self.safe_get(url)
        if not html:
            return None

        soup = BeautifulSoup(html, "lxml")


        title = soup.select_one(".detail-title, .main-title")


        published = soup.select_one(".detail-time, .meta-time")

        content_el = soup.select_one(".detail-content, #cnt-content, .content-detail")

        if content_el:

            for garbage in content_el.select(".more-news, .relate-news, .box-ads, .VCSortableInPreviewMode, .player-control"):
                garbage.decompose()




        category_text = category_source
        breadcrumb = soup.select(".breadcrumbs .breadcrumb-item a, .breadcrumb a")
        if breadcrumb:
            category_text = breadcrumb[-1].text.strip()


        author_text = ""
        author_el = soup.select_one(".detail-author, .author-info .name, .bottom-info .author")
        if author_el:
            author_text = author_el.text.strip()

        if not author_text and content_el:

            paragraphs = content_el.find_all('p')
            if paragraphs:
                last_p_text = paragraphs[-1].get_text().strip()
                if len(last_p_text) < 50:
                    author_text = last_p_text

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
        except Exception as e:
            log.error(f"Failed to save article {article.get('url')}: {e}")

    def print_summary(self):
        end_time = time.time()
        total_time = end_time - self.start_time
        summary = (
            f"Discovery Complete (Thanh Nien) ✨\n\n"
            f"[bold green]New Articles Found:[/bold green] {self.stats['articles_saved']}\n"
            f"Total Time: {total_time:.2f} seconds\n"
            f"Cache Hits: {self.stats['cache_hits']}\n"
            f"Output: [italic]{self.article_csv.resolve()}[/italic]"
        )
        console.print(Panel(summary, title="Thanh Nien Summary", border_style="bold blue", padding=(1, 2)))

    def crawl(self, categories: list[str], pages: int = 2, workers: int = MAX_WORKERS, no_progress: bool = False):
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

            discover_task_id = progress.add_task(f"[cyan]Discovering...", total=len(categories) * pages) if not no_progress else None

            for category in categories:
                log.info(f"Discovering in [cyan]{category}[/cyan]...")
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

            with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
                results = executor.map(fetch_and_save_article, unique_new_articles)
                if not no_progress:
                    for _ in results:
                        progress.update(crawl_task_id, advance=1)
                else:
                    for _ in results: pass

        log.info("Crawl complete.")
        self.print_summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Thanh Nien Article Discoverer")
    parser.add_argument("--category", "-c", required=True, nargs="+", help="Categories slugs (e.g., 'cong-nghe-game', 'thoi-su')")
    parser.add_argument("--pages", "-p", type=int, default=2, help="Number of pages per category")
    parser.add_argument("--output", "-o", default=DEFAULT_OUTPUT, help=f"Output directory (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--workers", "-w", type=int, default=MAX_WORKERS, help=f"Workers (default: {MAX_WORKERS})")
    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    console.rule(f"[bold]Thanh Nien Crawler[/bold]: [cyan]{', '.join(args.category)}[/cyan]")

    c = ThanhNienCrawler(Path(args.output), use_cache=(not args.no_cache))
    c.crawl(args.category, args.pages, workers=args.workers)
