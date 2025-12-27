"""
Mock Streaming Demo: Full Architecture Simulation

Simulates the complete Kafka → Spark → Dashboard pipeline
using pre-computed Kaggle results. Shows visual progress without
actually running the ML pipeline.

Flow:
1. Kafka Producer: Simulates publishing posts to Kafka topic
2. Spark Consumer: Simulates processing (loads pre-computed results)
3. Dashboard: Displays real-time updates

Usage:
    python demo_streaming_full.py --demo-folder demo/demo_data
"""

import os
import sys
import time
import json
import argparse
import threading
from datetime import datetime
from queue import Queue

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn
from rich.layout import Layout

console = Console()

# Message queue (simulates Kafka)
kafka_queue = Queue()
results_queue = Queue()
STOP_SIGNAL = "__STOP__"


class MockKafkaProducer:
    """Simulates Kafka Producer - publishes posts from demo state."""
    
    def __init__(self, demo_folder, batch_size=10, delay=0.5):
        self.demo_folder = demo_folder
        self.batch_size = batch_size
        self.delay = delay
        self.running = False
        self.stats = {"produced": 0, "batches": 0}
        
    def load_posts(self):
        """Load posts from demo state."""
        from src.utils.demo_state import load_demo_state
        state = load_demo_state(self.demo_folder)
        df = state.get('df_results')
        return df.to_dict('records') if df is not None else []
    
    def run(self):
        """Run producer - publish posts to queue."""
        self.running = True
        posts = self.load_posts()
        
        console.print(f"[cyan]📤 Kafka Producer: {len(posts)} posts to publish[/cyan]")
        
        for i in range(0, len(posts), self.batch_size):
            if not self.running:
                break
                
            batch = posts[i:i+self.batch_size]
            kafka_queue.put(batch)
            self.stats["produced"] += len(batch)
            self.stats["batches"] += 1
            
            time.sleep(self.delay)
        
        kafka_queue.put(STOP_SIGNAL)
        console.print(f"[green]✅ Producer done: {self.stats['produced']} posts[/green]")
    
    def stop(self):
        self.running = False


class MockSparkConsumer:
    """Simulates Spark Streaming Consumer - processes batches."""
    
    def __init__(self, demo_folder):
        self.demo_folder = demo_folder
        self.running = False
        self.stats = {"processed": 0, "trending": 0, "discoveries": 0}
        self.precomputed_results = {}
        
    def load_precomputed(self):
        """Load pre-computed results mapping."""
        from src.utils.demo_state import load_demo_state
        state = load_demo_state(self.demo_folder)
        df = state.get('df_results')
        
        if df is not None:
            # Create mapping from content to result
            for _, row in df.iterrows():
                content = row.get('content', '')[:100]
                self.precomputed_results[content] = row.to_dict()
        
        return state.get('cluster_mapping', {})
    
    def process_batch(self, batch):
        """Process a batch - look up pre-computed results."""
        results = []
        
        for post in batch:
            content = post.get('content', '')[:100]
            
            # Look up pre-computed result
            if content in self.precomputed_results:
                result = self.precomputed_results[content]
            else:
                # Fallback for new posts
                result = {
                    'content': content,
                    'final_topic': 'Discovery',
                    'topic_type': 'Discovery',
                    'category': 'T7',
                    'sentiment': 'Neutral',
                    'score': 0.5,
                }
            
            results.append(result)
            
            if result.get('topic_type') == 'Trending':
                self.stats["trending"] += 1
            else:
                self.stats["discoveries"] += 1
        
        self.stats["processed"] += len(batch)
        return results
    
    def run(self):
        """Run consumer - process batches from queue."""
        self.running = True
        self.load_precomputed()
        
        console.print(f"[cyan]⚡ Spark Consumer: Ready to process[/cyan]")
        
        while self.running:
            batch = kafka_queue.get()
            
            if batch == STOP_SIGNAL:
                break
            
            results = self.process_batch(batch)
            results_queue.put(results)
            
            # Simulate processing time
            time.sleep(0.1)
        
        results_queue.put(STOP_SIGNAL)
        console.print(f"[green]✅ Consumer done: {self.stats['processed']} processed[/green]")
    
    def stop(self):
        self.running = False


class MockDashboard:
    """Simulates Dashboard - displays streaming results."""
    
    def __init__(self):
        self.running = False
        self.all_results = []
        self.stats = {
            "total": 0,
            "trending": 0,
            "discoveries": 0,
            "topics": set(),
            "categories": {},
        }
    
    def update_stats(self, results):
        """Update dashboard stats."""
        for r in results:
            self.stats["total"] += 1
            
            if r.get('topic_type') == 'Trending':
                self.stats["trending"] += 1
            else:
                self.stats["discoveries"] += 1
            
            topic = r.get('final_topic', 'Unknown')
            self.stats["topics"].add(topic)
            
            category = r.get('category', 'Unknown')
            self.stats["categories"][category] = self.stats["categories"].get(category, 0) + 1
    
    def create_display(self, latest_results):
        """Create rich display panel."""
        # Stats table
        stats_table = Table(show_header=False, box=None, padding=(0, 2))
        stats_table.add_column("Metric", style="dim")
        stats_table.add_column("Value", style="bold")
        
        stats_table.add_row("Total Processed", f"[cyan]{self.stats['total']}[/]")
        stats_table.add_row("Trending", f"[green]{self.stats['trending']}[/]")
        stats_table.add_row("Discoveries", f"[yellow]{self.stats['discoveries']}[/]")
        stats_table.add_row("Unique Topics", f"[magenta]{len(self.stats['topics'])}[/]")
        
        # Category distribution
        cat_str = " | ".join([f"{k}: {v}" for k, v in sorted(self.stats['categories'].items())])
        
        # Latest posts
        latest_table = Table(title="Latest Events", show_header=True, header_style="bold")
        latest_table.add_column("Topic", width=30)
        latest_table.add_column("Type", width=12)
        latest_table.add_column("Category", width=8)
        latest_table.add_column("Score", width=8)
        
        for r in latest_results[-5:]:
            topic = (r.get('final_topic', '')[:28] + "..") if len(r.get('final_topic', '')) > 30 else r.get('final_topic', '')
            topic_type = r.get('topic_type', 'Discovery')
            type_color = "green" if topic_type == "Trending" else "yellow"
            
            latest_table.add_row(
                topic,
                f"[{type_color}]{topic_type}[/]",
                r.get('category', 'T7'),
                f"{r.get('score', 0):.2f}",
            )
        
        return Panel(
            f"{stats_table}\n\nCategories: {cat_str}\n\n{latest_table}",
            title=f"🌊 Real-time Event Detection Dashboard - {datetime.now().strftime('%H:%M:%S')}",
            border_style="blue",
        )
    
    def run(self):
        """Run dashboard - display streaming results."""
        self.running = True
        
        console.print(f"[cyan]📊 Dashboard: Ready to display[/cyan]\n")
        
        with Live(self.create_display([]), refresh_per_second=4, console=console) as live:
            while self.running:
                try:
                    results = results_queue.get(timeout=1)
                except:
                    continue
                
                if results == STOP_SIGNAL:
                    break
                
                self.all_results.extend(results)
                self.update_stats(results)
                live.update(self.create_display(self.all_results))
        
        # Final summary
        console.print("\n" + "="*60)
        console.print("[bold green]🎉 STREAMING DEMO COMPLETE[/bold green]")
        console.print("="*60)
        console.print(f"Total Events: {self.stats['total']}")
        console.print(f"Trending: {self.stats['trending']}")
        console.print(f"Discoveries: {self.stats['discoveries']}")
        console.print(f"Unique Topics: {len(self.stats['topics'])}")
        console.print("="*60)
    
    def stop(self):
        self.running = False


def run_demo(demo_folder, batch_size=20, delay=0.3, limit=None):
    """Run the full mock streaming demo."""
    console.print("\n[bold cyan]" + "="*60 + "[/bold cyan]")
    console.print("[bold cyan]   KAFKA + SPARK + DASHBOARD STREAMING DEMO[/bold cyan]")
    console.print("[bold cyan]" + "="*60 + "[/bold cyan]\n")
    console.print(f"Demo folder: {demo_folder}")
    console.print(f"Batch size: {batch_size}, Delay: {delay}s")
    console.print("[dim]Using pre-computed Kaggle results (no ML inference)[/dim]\n")
    
    # Initialize components
    producer = MockKafkaProducer(demo_folder, batch_size=batch_size, delay=delay)
    consumer = MockSparkConsumer(demo_folder)
    dashboard = MockDashboard()
    
    # Start threads
    producer_thread = threading.Thread(target=producer.run, daemon=True)
    consumer_thread = threading.Thread(target=consumer.run, daemon=True)
    dashboard_thread = threading.Thread(target=dashboard.run, daemon=True)
    
    try:
        producer_thread.start()
        time.sleep(0.5)
        consumer_thread.start()
        time.sleep(0.5)
        dashboard_thread.start()
        
        # Wait for completion
        producer_thread.join()
        consumer_thread.join()
        dashboard_thread.join()
        
    except KeyboardInterrupt:
        console.print("\n[yellow]⏹ Demo stopped by user[/yellow]")
        producer.stop()
        consumer.stop()
        dashboard.stop()


def main():
    parser = argparse.ArgumentParser(description='Mock Streaming Demo')
    parser.add_argument('--demo-folder', default='demo/demo_data', help='Path to demo state folder')
    parser.add_argument('--batch-size', type=int, default=20, help='Posts per batch')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between batches')
    parser.add_argument('--limit', type=int, default=None, help='Limit posts to process')
    args = parser.parse_args()
    
    # Find demo folder
    demo_folder = args.demo_folder
    if not os.path.isabs(demo_folder):
        demo_folder = os.path.join(PROJECT_ROOT, demo_folder)
    
    if not os.path.exists(demo_folder):
        import glob
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
        if candidates:
            demo_folder = candidates[0]
        else:
            console.print("[red]❌ No demo folder found![/red]")
            return 1
    
    run_demo(
        demo_folder=demo_folder,
        batch_size=args.batch_size,
        delay=args.delay,
        limit=args.limit,
    )
    return 0


if __name__ == '__main__':
    exit(main())
