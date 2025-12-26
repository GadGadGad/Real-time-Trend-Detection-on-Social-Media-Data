import json
import textwrap
from datetime import datetime
from time import sleep

from kafka import KafkaConsumer
from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich import box

# Cấu hình Kafka Consumer
TOPIC_NAME = 'news_data'
BOOTSTRAP_SERVERS = ['localhost:9092']

consumer = KafkaConsumer(
    TOPIC_NAME,
    bootstrap_servers=BOOTSTRAP_SERVERS,
    auto_offset_reset='latest',  # Chỉ nhận tin mới nhất (không hiện lại tin cũ)
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

console = Console()

def generate_table(articles_buffer):
    """Hàm tạo bảng từ danh sách bài viết"""
    table = Table(
        title="[bold cyan]STREAMING NEWS DASHBOARD[/bold cyan]",
        box=box.ROUNDED,
        show_lines=True,
        width=100
    )

    # Định nghĩa các cột
    table.add_column("Source", style="bold green", width=12)
    table.add_column("Category", style="magenta", width=12)
    table.add_column("Time", style="yellow", width=18)
    table.add_column("Title", style="white")

    # Thêm dữ liệu vào bảng (Đảo ngược để tin mới nhất lên đầu)
    for article in reversed(articles_buffer):
        # Cắt ngắn tiêu đề nếu quá dài
        title = textwrap.shorten(article.get('title', 'No Title'), width=50, placeholder="...")
        
        # Format thời gian crawl
        crawled_at = article.get('crawled_at', 0)
        time_str = datetime.fromtimestamp(crawled_at).strftime('%H:%M:%S')

        table.add_row(
            article.get('source', '').upper(),
            article.get('category', ''),
            time_str,
            title
        )
    return table

def run_dashboard():
    articles_buffer = []  # Lưu giữ 10 tin gần nhất để hiển thị
    
    # Chế độ Live của Rich giúp bảng tự động render lại mà không bị giật
    with Live(generate_table(articles_buffer), refresh_per_second=4, console=console) as live:
        
        # Lắng nghe Kafka liên tục
        for message in consumer:
            article = message.value
            
            # Thêm vào buffer
            articles_buffer.append(article)
            
            # Chỉ giữ lại 15 tin mới nhất để bảng không bị tràn màn hình
            if len(articles_buffer) > 15:
                articles_buffer.pop(0)
            
            # Cập nhật bảng
            live.update(generate_table(articles_buffer))

if __name__ == "__main__":
    try:
        console.print("[bold yellow]🚀 Đang kết nối tới Kafka... Đợi dữ liệu...[/bold yellow]")
        run_dashboard()
    except KeyboardInterrupt:
        console.print("\n[bold red]🛑 Đã dừng Dashboard.[/bold red]")