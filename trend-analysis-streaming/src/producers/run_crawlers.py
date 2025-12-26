import argparse
import logging
import sys
import toml
import json
from pathlib import Path
from rich.console import Console
from rich.logging import RichHandler
from kafka import KafkaProducer

sys.path.append(str(Path(__file__).parent / "crawlers"))

try:
    from src.scrapers.vnexpress_crawler import VnExpressCrawler
    from src.scrapers.nld_crawler import NLDCrawler
    from src.scrapers.thanhnien_crawler import ThanhNienCrawler
    from src.scrapers.tuoitre_crawler import TuoiTreCrawler
    from src.scrapers.vietnamnet_crawler import VietnamNetCrawler
except ImportError as e:
    print(f"Error importing crawlers: {e}")
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
    if Path("config.toml").exists():
        return toml.load("config.toml")
    return {}

def main():
    parser = argparse.ArgumentParser(description="Unified Crawler Runner with Streaming")
    parser.add_argument("--crawlers", nargs="+", choices=list(CRAWLER_MAP.keys()))
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--category", "-c", nargs="+", required=True)
    parser.add_argument("--pages", "-p", type=int, default=2)
    parser.add_argument("--workers", "-w", type=int, default=None)
    parser.add_argument("--output-base", "-o", default=None)
    parser.add_argument("--no-cache", action="store_true")
    
    # Tính năng mở rộng
    parser.add_argument("--stream", action="store_true", help="Stream data directly to Kafka")
    parser.add_argument("--kafka-server", default="localhost:9092")
    parser.add_argument("--topic", default="raw_data")

    args = parser.parse_args()

    if not args.crawlers and not args.all:
        parser.error("You must specify --crawlers [names] or --all")

    config = load_config()
    crawler_cfg = config.get("crawler", {})
    files_cfg = config.get("files", {})

    workers = args.workers if args.workers else crawler_cfg.get("max_workers", 5)
    output_base = args.output_base if args.output_base else files_cfg.get("default_output_dir", "data")
    output_base_path = Path(output_base)

    # Khởi tạo Kafka Producer nếu có --stream
    producer = None
    if args.stream:
        log.info(f"📡 [bold yellow]Streaming mode enabled.[/bold yellow] Target: {args.kafka_server}")
        producer = KafkaProducer(
            bootstrap_servers=[args.kafka_server],
            value_serializer=lambda x: json.dumps(x, ensure_ascii=False).encode('utf-8')
        )

    crawlers_to_run = list(CRAWLER_MAP.keys()) if args.all else args.crawlers

    for name in crawlers_to_run:
        crawler_cls = CRAWLER_MAP[name]
        crawler_output_dir = output_base_path / name
        log.info(f"\n[bold green]>>> Running {name} crawler...[/bold green]")
        
        crawler = crawler_cls(crawler_output_dir, use_cache=(not args.no_cache))
        
        # Inject Kafka producer vào instance của crawler
        # Giả định các crawler class có thuộc tính stream_handler
        if producer:
            def kafka_callback(data):
                producer.send(args.topic, value=data)
            
            crawler.stream_callback = kafka_callback

        crawler.crawl(
            categories=args.category,
            pages=args.pages,
            workers=workers
        )

    if producer:
        producer.flush()
        producer.close()
    
    log.info("\n[bold green]✨ All tasks finished.[/bold green]")

if __name__ == "__main__":
    main()