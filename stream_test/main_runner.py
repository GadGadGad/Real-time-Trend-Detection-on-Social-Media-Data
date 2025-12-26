import argparse
import logging
import sys
import toml
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler

# Import crawler classes

sys.path.append(str(Path(__file__).parent / "crawlers"))

try:
    from crawlers.vnexpress_crawler import VnExpressCrawler
    from crawlers.nld_crawler import NLDCrawler
    from crawlers.thanhnien_crawler import ThanhNienCrawler
    from crawlers.tuoitre_crawler import TuoiTreCrawler
    from crawlers.vietnamnet_crawler import VietnamNetCrawler
except ImportError as e:
    print(f"Error importing crawlers: {e}")
    print("Make sure you are running this script from the project root.")
    sys.exit(1)


console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, show_path=False, markup=True)]
)
log = logging.getLogger("run_crawlers")

CRAWLER_MAP = {
    "vnexpress": VnExpressCrawler,
    "nld": NLDCrawler,
    "thanhnien": ThanhNienCrawler,
    "tuoitre": TuoiTreCrawler,
    "vietnamnet": VietnamNetCrawler
}

def load_config():
    try:
        return toml.load("config.toml")
    except Exception:
        log.warning("config.toml not found or invalid. Using defaults.")
        return {}

def main():
    parser = argparse.ArgumentParser(description="Unified Crawler Runner")
    
    parser.add_argument(
        "--crawlers", 
        nargs="+", 
        choices=list(CRAWLER_MAP.keys()),
        help="Specific crawlers to run (space-separated)"
    )
    parser.add_argument(
        "--all", 
        action="store_true", 
        help="Run all crawlers"
    )
    
    parser.add_argument(
        "--category", "-c", 
        nargs="+", 
        required=True, 
        help="Categories to crawl (e.g., 'thoi-su', 'kinh-doanh')"
    )
    
    parser.add_argument(
        "--pages", "-p", 
        type=int, 
        default=2, 
        help="Number of pages to crawl per category"
    )
    
    parser.add_argument(
        "--workers", "-w", 
        type=int, 
        default=None, 
        help="Number of worker threads (overrides config)"
    )

    parser.add_argument(
        "--output-base", "-o",
        default=None,
        help="Base output directory (overrides config)"
    )

    parser.add_argument("--no-cache", action="store_true", help="Disable caching")

    args = parser.parse_args()

    if not args.crawlers and not args.all:
        parser.error("You must specify --crawlers [names] or --all")

    config = load_config()
    crawler_cfg = config.get("crawler", {})
    files_cfg = config.get("files", {})


    workers = args.workers if args.workers else crawler_cfg.get("max_workers", 5)
    

    output_base = args.output_base if args.output_base else files_cfg.get("default_output_dir", "data")
    output_base_path = Path(output_base)

    crawlers_to_run = []
    if args.all:
        crawlers_to_run = list(CRAWLER_MAP.keys())
    else:
        crawlers_to_run = args.crawlers

    log.info(f"🚀 Starting run for: [bold cyan]{', '.join(crawlers_to_run)}[/bold cyan]")
    log.info(f"Categories: {args.category}")
    log.info(f"Pages: {args.pages} | Workers: {workers}")

    for name in crawlers_to_run:
        crawler_cls = CRAWLER_MAP[name]
        
        # Create specific output dir for this crawler
        crawler_output_dir = output_base_path / name
        
        log.info(f"\n[bold green]>>> Running {name} crawler...[/bold green]")
        try:
            crawler = crawler_cls(crawler_output_dir, use_cache=(not args.no_cache))
            
            crawler.crawl(
                categories=args.category,
                pages=args.pages,
                workers=workers
            )
        except Exception as e:
            log.error(f"Failed to run {name}: {e}")

    log.info("\n[bold green]✨ All requested crawlers finished.[/bold green]")

if __name__ == "__main__":
    main()
