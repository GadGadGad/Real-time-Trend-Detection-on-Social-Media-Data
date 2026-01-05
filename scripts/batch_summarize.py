"""
Batch Summarization Script
==========================
Run summarization on all posts and cache results for later use.

Usage:
    # Summarize and save
    python scripts/batch_summarize.py --input data/fb_data.json --output data/fb_summaries.json
    
    # Summarize with custom model
    python scripts/batch_summarize.py --input data/fb_data.json --output data/fb_summaries.json --model vit5-base
    
    # Resume (skip already summarized)
    python scripts/batch_summarize.py --input data/fb_data.json --output data/fb_summaries.json --resume
"""

import json
import argparse
import os
import sys
import pandas as pd
from pathlib import Path
from rich.console import Console
from rich.progress import track

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.analysis.summarizer import Summarizer

console = Console()


def load_posts(input_path: str) -> list:
    """Load posts from JSON or CSV file. Supports various formats."""
    console.print(f"[cyan]📂 Loading data from {input_path}...[/cyan]")
    
    ext = Path(input_path).suffix.lower()
    
    # Handle CSV files (news articles)
    if ext == '.csv':
        df = pd.read_csv(input_path)
        posts = df.to_dict('records')
        console.print(f"[green]✅ Loaded {len(posts)} articles from CSV[/green]")
        return posts
    
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Handle different JSON structures
    if isinstance(data, list):
        posts = data
    elif isinstance(data, dict):
        # Try common keys
        for key in ['posts', 'data', 'articles', 'items']:
            if key in data:
                posts = data[key]
                break
        else:
            # If it's a dict of posts by ID
            if all(isinstance(v, dict) for v in data.values()):
                posts = list(data.values())
            else:
                posts = [data]
    else:
        raise ValueError(f"Unsupported data format: {type(data)}")
    
    console.print(f"[green]✅ Loaded {len(posts)} posts[/green]")
    return posts


def get_post_text(post: dict) -> str:
    """Extract text content from a post/article dict."""
    # Try common content field names (both social posts and news articles)
    for key in ['content', 'text', 'body', 'message', 'description', 'postText', 
                'article_text', 'full_text', 'article_content']:
        if key in post and post[key] and not pd.isna(post[key]):
            return str(post[key])
    
    # For news: combine title + description if no full text
    title = post.get('title', '')
    desc = post.get('description', '') or post.get('excerpt', '')
    if title and desc:
        return f"{title}. {desc}"
    
    return ""


def get_post_id(post: dict, index: int) -> str:
    """Get a unique ID for a post/article."""
    for key in ['id', 'postId', 'post_id', '_id', 'url', 'link']:
        if key in post and post[key] and not pd.isna(post[key]):
            return str(post[key])
    return str(index)


def load_existing_summaries(output_path: str) -> dict:
    """Load existing summaries for resume functionality."""
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            console.print(f"[yellow]📥 Loaded {len(data)} existing summaries for resume[/yellow]")
            return data
    return {}


def save_summaries(summaries: dict, output_path: str):
    """Save summaries to JSON file."""
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(summaries, f, ensure_ascii=False, indent=2)
    console.print(f"[green]💾 Saved {len(summaries)} summaries to {output_path}[/green]")


def batch_summarize(
    input_path: str,
    output_path: str,
    model_name: str = "vit5-large",
    max_length: int = 256,
    resume: bool = False,
    checkpoint_interval: int = 50
):
    """
    Summarize all posts and save results.
    
    Args:
        input_path: Path to input JSON with posts
        output_path: Path to save summaries JSON
        model_name: Summarizer model name
        max_length: Max summary length in tokens
        resume: If True, skip already summarized posts
        checkpoint_interval: Save progress every N posts
    """
    posts = load_posts(input_path)
    summaries = load_existing_summaries(output_path) if resume else {}
    
    to_summarize = []
    for i, post in enumerate(posts):
        post_id = get_post_id(post, i)
        if post_id not in summaries:
            text = get_post_text(post)
            if text and len(text) > 50:  # Skip very short posts
                to_summarize.append({
                    'id': post_id,
                    'text': text,
                    'index': i
                })
    
    if not to_summarize:
        console.print("[green]✅ All posts already summarized![/green]")
        return summaries
    
    console.print(f"[cyan]📝 Summarizing {len(to_summarize)} posts (skipped {len(posts) - len(to_summarize)})...[/cyan]")
    
    summarizer = Summarizer(model_name=model_name)
    summarizer.load_model()
    
    if not summarizer.enabled:
        console.print("[red]❌ Failed to load summarizer[/red]")
        return summaries
    
    batch_size = 8  # Safe GPU batch size
    
    for i in range(0, len(to_summarize), batch_size):
        batch = to_summarize[i:i + batch_size]
        batch_texts = [item['text'] for item in batch]
        
        try:
            batch_summaries = summarizer.summarize_batch(batch_texts, max_length=max_length)
            
            # Store results
            for item, summary in zip(batch, batch_summaries):
                summaries[item['id']] = {
                    'original_length': len(item['text']),
                    'summary': summary,
                    'summary_length': len(summary),
                    'index': item['index']
                }
            
            # Checkpoint
            if (i + batch_size) % checkpoint_interval == 0:
                save_summaries(summaries, output_path)
                console.print(f"[dim]💾 Checkpoint: {len(summaries)} summaries saved[/dim]")
                
        except Exception as e:
            console.print(f"[red]Error in batch {i}: {e}[/red]")
            save_summaries(summaries, output_path)
            raise
    
    save_summaries(summaries, output_path)
    summarizer.unload_model()
    
    total_original = sum(s['original_length'] for s in summaries.values())
    total_summary = sum(s['summary_length'] for s in summaries.values())
    compression = (1 - total_summary / total_original) * 100 if total_original else 0
    
    console.print(f"\n[bold green]✅ Summarization Complete![/bold green]")
    console.print(f"   Posts processed: {len(summaries)}")
    console.print(f"   Total compression: {compression:.1f}%")
    console.print(f"   Output: {output_path}")
    
    return summaries


def load_summaries_for_use(summary_path: str) -> dict:
    """
    Load pre-computed summaries for use in your pipeline.
    
    Returns a dict: {post_id: summary_text}
    
    Usage:
        summaries = load_summaries_for_use('data/fb_summaries.json')
        for post in posts:
            post_id = get_post_id(post)
            if post_id in summaries:
                post['summary'] = summaries[post_id]
    """
    with open(summary_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Return simplified dict: id -> summary text
    return {pid: info['summary'] for pid, info in data.items()}


def merge_summaries_into_posts(posts: list, summary_path: str) -> list:
    """
    Merge pre-computed summaries into posts list.
    Adds 'summary' field to each post that has a cached summary.
    
    Usage:
        posts = json.load(open('data/fb_data.json'))
        posts_with_summaries = merge_summaries_into_posts(posts, 'data/fb_summaries.json')
    """
    summaries = load_summaries_for_use(summary_path)
    
    for i, post in enumerate(posts):
        post_id = get_post_id(post, i)
        if post_id in summaries:
            post['summary'] = summaries[post_id]
    
    console.print(f"[green]✅ Merged {len(summaries)} summaries into posts[/green]")
    return posts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch summarize posts with caching")
    parser.add_argument("--input", "-i", required=True, help="Input JSON file with posts")
    parser.add_argument("--output", "-o", required=True, help="Output JSON file for summaries")
    parser.add_argument("--model", "-m", default="vit5-large", 
                        choices=["vit5-large", "vit5-base", "bartpho"],
                        help="Summarization model")
    parser.add_argument("--max-length", type=int, default=256, help="Max summary length")
    parser.add_argument("--resume", "-r", action="store_true", help="Resume from existing output")
    parser.add_argument("--checkpoint", type=int, default=50, help="Save every N posts")
    
    args = parser.parse_args()
    
    batch_summarize(
        input_path=args.input,
        output_path=args.output,
        model_name=args.model,
        max_length=args.max_length,
        resume=args.resume,
        checkpoint_interval=args.checkpoint
    )
