"""
Pseudo-Streaming Demo Simulator.

Simulates real-time event detection by feeding posts one by one
from pre-computed Kaggle results. Creates a "live" demo effect.

Usage:
    python streaming_simulator.py --demo-folder demo/demo_data --delay 0.5
"""

import os
import sys
import time
import argparse
import random
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.panel import Panel
from rich.layout import Layout
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def create_live_display(posts_processed, current_post, stats):
    """Create a rich live display for the demo."""
    layout = Layout()
    
    # Stats table
    stats_table = Table(title="📊 Live Stats", show_header=False, box=None)
    stats_table.add_row("Posts Processed", f"[bold cyan]{posts_processed}[/]")
    stats_table.add_row("Trending Topics", f"[bold green]{stats.get('trending', 0)}[/]")
    stats_table.add_row("Discoveries", f"[bold yellow]{stats.get('discoveries', 0)}[/]")
    stats_table.add_row("Active Clusters", f"[bold magenta]{stats.get('clusters', 0)}[/]")
    
    # Current post panel
    if current_post:
        content = current_post.get('content', '')[:200] + "..."
        source = current_post.get('source', 'Unknown')
        topic = current_post.get('final_topic', 'Processing...')
        score = current_post.get('score', 0)
        
        post_info = f"[dim]{source}[/]\n\n{content}\n\n[bold]→ {topic}[/] (score: {score:.2f})"
        post_panel = Panel(post_info, title="🔴 Live Post", border_style="red")
    else:
        post_panel = Panel("[dim]Waiting for posts...[/]", title="🔴 Live Post")
    
    return Panel(
        f"{stats_table}\n\n{post_panel}",
        title=f"🌊 Streaming Demo - {datetime.now().strftime('%H:%M:%S')}",
        border_style="blue"
    )


def simulate_streaming(demo_folder, delay=0.5, limit=None, shuffle=False):
    """
    Simulate real-time streaming using pre-computed demo state.
    
    Args:
        demo_folder: Path to demo state folder (from save_demo_state)
        delay: Seconds between each post
        limit: Max posts to process
        shuffle: Randomize post order for variety
    """
    from src.utils.demo_state import load_demo_state, attach_new_post
    from sentence_transformers import SentenceTransformer
    
    console.print(f"\n[bold cyan]🌊 Pseudo-Streaming Demo[/bold cyan]")
    console.print(f"Loading state from: {demo_folder}\n")
    
    # Load demo state
    state = load_demo_state(demo_folder)
    
    df_results = state.get('df_results')
    centroids = state.get('centroids', {})
    trend_embeddings = state.get('trend_embeddings')
    cluster_mapping = state.get('cluster_mapping', {})
    metadata = state.get('metadata', {})
    
    if df_results is None or len(df_results) == 0:
        console.print("[red]❌ No results found in demo state![/red]")
        return
    
    # Get trend keys - handle case where trends was loaded from list
    trends = state.get('trends', {})
    if isinstance(trends, dict) and trends:
        trend_keys = list(trends.keys())
    else:
        # Generate placeholder trend keys based on embedding count
        num_trends = trend_embeddings.shape[0] if trend_embeddings is not None else 0
        trend_keys = [f"Trend_{i}" for i in range(num_trends)]
        console.print(f"[yellow]⚠️ Generated {num_trends} placeholder trend keys[/yellow]")
    
    # Load embedding model
    model_name = metadata.get('model_name', 'dangvantuan/vietnamese-document-embedding')
    console.print(f"Loading model: {model_name}...")
    embedder = SentenceTransformer(model_name, trust_remote_code=True)
    
    # Prepare posts to simulate
    posts = df_results.to_dict('records')
    if shuffle:
        random.shuffle(posts)
    if limit:
        posts = posts[:limit]
    
    console.print(f"\n[bold green]▶ Starting stream with {len(posts)} posts[/bold green]")
    console.print(f"  Delay: {delay}s per post")
    console.print(f"  Press Ctrl+C to stop\n")
    
    # Stats tracking
    stats = {
        'trending': 0,
        'discoveries': 0,
        'clusters': set(),
        'topics': set(),
    }
    
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task("[cyan]Streaming...", total=len(posts))
            
            for i, post in enumerate(posts):
                # Simulate real-time attachment
                result = attach_new_post(
                    new_post=post,
                    centroids=centroids,
                    trend_embeddings=trend_embeddings,
                    trend_keys=trend_keys,
                    embedder=embedder,
                    threshold=0.5,
                    attach_threshold=0.6,
                    cluster_mapping=cluster_mapping,
                )
                
                # Update stats
                if result.get('topic_type') == 'Trending':
                    stats['trending'] += 1
                else:
                    stats['discoveries'] += 1
                
                if result.get('cluster_id', -1) != -1:
                    stats['clusters'].add(result['cluster_id'])
                
                stats['topics'].add(result.get('final_topic', 'Unknown'))
                
                # Display current post
                source = post.get('source', 'Unknown')[:20]
                topic = result.get('final_topic', 'Unknown')[:30]
                score = result.get('score', 0)
                category = result.get('category', '')
                
                progress.update(task, advance=1, description=f"[cyan]{source}[/] → [green]{topic}[/] ({score:.2f})")
                
                # Print detailed info periodically
                if (i + 1) % 10 == 0 or i < 5:
                    console.print(
                        f"  [{i+1:4d}] [dim]{source:20s}[/] → "
                        f"[{'green' if result.get('topic_type') == 'Trending' else 'yellow'}]{topic:30s}[/] "
                        f"[dim]({score:.2f})[/] {category}"
                    )
                
                time.sleep(delay)
        
        # Final summary
        console.print(f"\n[bold green]✅ Streaming Complete![/bold green]")
        console.print(f"   Processed: {len(posts)} posts")
        console.print(f"   Trending: {stats['trending']}")
        console.print(f"   Discoveries: {stats['discoveries']}")
        console.print(f"   Unique Topics: {len(stats['topics'])}")
        console.print(f"   Active Clusters: {len(stats['clusters'])}")
        
    except KeyboardInterrupt:
        console.print(f"\n[yellow]⏹ Stopped by user after {i+1} posts[/yellow]")


def main():
    parser = argparse.ArgumentParser(description='Pseudo-Streaming Demo Simulator')
    parser.add_argument('--demo-folder', default='demo/demo_data', help='Path to demo state folder')
    parser.add_argument('--delay', type=float, default=0.3, help='Delay between posts (seconds)')
    parser.add_argument('--limit', type=int, default=None, help='Max posts to process')
    parser.add_argument('--shuffle', action='store_true', help='Randomize post order')
    args = parser.parse_args()
    
    # Find demo folder
    demo_folder = args.demo_folder
    if not os.path.isabs(demo_folder):
        demo_folder = os.path.join(PROJECT_ROOT, demo_folder)
    
    # Auto-detect if not found
    if not os.path.exists(demo_folder):
        import glob
        candidates = glob.glob(os.path.join(PROJECT_ROOT, "demo/demo_*"))
        if candidates:
            demo_folder = candidates[0]
            console.print(f"[yellow]Auto-detected: {demo_folder}[/yellow]")
        else:
            console.print("[red]❌ No demo folder found. Run save_demo_state() first.[/red]")
            return 1
    
    simulate_streaming(
        demo_folder=demo_folder,
        delay=args.delay,
        limit=args.limit,
        shuffle=args.shuffle,
    )
    return 0


if __name__ == '__main__':
    exit(main())
