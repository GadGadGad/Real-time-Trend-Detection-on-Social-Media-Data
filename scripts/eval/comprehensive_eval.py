"""
Comprehensive Pipeline Evaluation Script.
Computes ALL available metrics for clustering/event detection quality.

Usage:
    python scripts/eval/comprehensive_eval.py --state demo/demo_finetuned_reranker
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from datetime import datetime
from rich.console import Console
from rich.table import Table
from sklearn.metrics import silhouette_score, calinski_harabasz_score

console = Console()

def load_demo_state(save_dir):
    """Load demo state files."""
    state = {}
    
    results_path = os.path.join(save_dir, 'results.parquet')
    if os.path.exists(results_path):
        state['df_results'] = pd.read_parquet(results_path)
    
    emb_path = os.path.join(save_dir, 'post_embeddings.npy')
    if os.path.exists(emb_path):
        state['post_embeddings'] = np.load(emb_path)
    
    labels_path = os.path.join(save_dir, 'cluster_labels.npy')
    if os.path.exists(labels_path):
        state['cluster_labels'] = np.load(labels_path)
    
    meta_path = os.path.join(save_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            state['metadata'] = json.load(f)
    
    return state

def compute_clustering_metrics(labels, embeddings):
    """Internal clustering quality metrics."""
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_embs = embeddings[valid_mask]
    
    metrics = {}
    n_clusters = len(set(valid_labels))
    metrics['num_clusters'] = n_clusters
    metrics['noise_ratio'] = (labels == -1).sum() / len(labels) * 100
    metrics['avg_cluster_size'] = len(valid_labels) / max(n_clusters, 1)
    
    if n_clusters > 1 and len(valid_embs) > n_clusters:
        try:
            metrics['silhouette'] = silhouette_score(valid_embs, valid_labels)
        except:
            metrics['silhouette'] = 0.0
        try:
            metrics['calinski_harabasz'] = calinski_harabasz_score(valid_embs, valid_labels)
        except:
            metrics['calinski_harabasz'] = 0.0
    else:
        metrics['silhouette'] = 0.0
        metrics['calinski_harabasz'] = 0.0
    
    return metrics

def compute_topic_diversity(df):
    """Measure diversity of detected topics."""
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    
    # Exclude noise/unassigned
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    
    unique_topics = valid_topics[group_col].nunique()
    total_posts = len(df)
    assigned_posts = len(valid_topics)
    
    return {
        'unique_topics': unique_topics,
        'assigned_posts': assigned_posts,
        'assignment_rate': assigned_posts / total_posts * 100 if total_posts > 0 else 0,
        'avg_posts_per_topic': assigned_posts / max(unique_topics, 1)
    }

def compute_source_diversity(df):
    """Measure source diversity within topics."""
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    source_col = 'source' if 'source' in df.columns else None
    
    if source_col is None:
        return {'avg_sources_per_topic': 0, 'multi_source_topics': 0}
    
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    
    sources_per_topic = valid_topics.groupby(group_col)[source_col].nunique()
    
    return {
        'avg_sources_per_topic': sources_per_topic.mean() if len(sources_per_topic) > 0 else 0,
        'max_sources_per_topic': sources_per_topic.max() if len(sources_per_topic) > 0 else 0,
        'multi_source_topics': (sources_per_topic > 1).sum(),
        'single_source_topics': (sources_per_topic == 1).sum()
    }

def compute_temporal_coherence(df):
    """Measure temporal coherence - posts in same topic should be close in time."""
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    time_col = 'time' if 'time' in df.columns else None
    
    if time_col is None:
        return {'avg_time_span_hours': 0, 'temporal_coherence': 0}
    
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    
    time_spans = []
    for topic, group in valid_topics.groupby(group_col):
        if len(group) < 2:
            continue
        try:
            times = pd.to_datetime(group[time_col])
            span_hours = (times.max() - times.min()).total_seconds() / 3600
            time_spans.append(span_hours)
        except:
            pass
    
    avg_span = np.mean(time_spans) if time_spans else 0
    # Temporal coherence: shorter span = higher coherence (inverse, normalized)
    # Score 1.0 for < 1 hour, 0.0 for > 168 hours (1 week)
    coherence = max(0, 1 - avg_span / 168)
    
    return {
        'avg_time_span_hours': avg_span,
        'max_time_span_hours': max(time_spans) if time_spans else 0,
        'temporal_coherence_score': coherence
    }

def compute_confidence_metrics(df):
    """Metrics about matching confidence."""
    if 'score' not in df.columns:
        return {}
    
    scores = df['score'].dropna()
    return {
        'avg_confidence': scores.mean(),
        'median_confidence': scores.median(),
        'high_confidence_rate': (scores > 0.7).sum() / len(scores) * 100 if len(scores) > 0 else 0,
        'low_confidence_rate': (scores < 0.3).sum() / len(scores) * 100 if len(scores) > 0 else 0
    }

def compute_sentiment_metrics(df):
    """Sentiment distribution metrics."""
    if 'sentiment' not in df.columns:
        return {}
    
    dist = df['sentiment'].value_counts()
    total = len(df)
    
    return {
        'positive_rate': dist.get('Positive', 0) / total * 100,
        'negative_rate': dist.get('Negative', 0) / total * 100,
        'neutral_rate': dist.get('Neutral', 0) / total * 100,
        'sentiment_polarity': (dist.get('Positive', 0) - dist.get('Negative', 0)) / total
    }

def compute_topic_type_metrics(df):
    """Topic type distribution."""
    if 'topic_type' not in df.columns:
        return {}
    
    dist = df['topic_type'].value_counts()
    total = len(df)
    
    return {
        'trending_rate': dist.get('Trending', 0) / total * 100,
        'discovery_rate': dist.get('Discovery', 0) / total * 100,
        'noise_rate': dist.get('Noise', 0) / total * 100
    }

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Pipeline Evaluation")
    parser.add_argument('--state', required=True, help='Path to demo state directory')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    args = parser.parse_args()
    
    console.print(f"[bold cyan]📊 Comprehensive Pipeline Evaluation[/bold cyan]")
    console.print(f"   State: {args.state}\n")
    
    # Load state
    state = load_demo_state(args.state)
    if 'df_results' not in state:
        console.print("[red]Error: Could not load results.[/red]")
        return
    
    df = state['df_results']
    all_metrics = {'state': args.state, 'evaluated_at': datetime.now().isoformat()}
    
    # 1. Clustering Quality
    if 'cluster_labels' in state and 'post_embeddings' in state:
        console.print("[dim]Computing clustering metrics...[/dim]")
        all_metrics['clustering'] = compute_clustering_metrics(
            state['cluster_labels'], state['post_embeddings']
        )
    
    # 2. Topic Diversity
    console.print("[dim]Computing topic diversity...[/dim]")
    all_metrics['diversity'] = compute_topic_diversity(df)
    
    # 3. Source Diversity
    console.print("[dim]Computing source diversity...[/dim]")
    all_metrics['source_diversity'] = compute_source_diversity(df)
    
    # 4. Temporal Coherence
    console.print("[dim]Computing temporal coherence...[/dim]")
    all_metrics['temporal'] = compute_temporal_coherence(df)
    
    # 5. Confidence Metrics
    console.print("[dim]Computing confidence metrics...[/dim]")
    all_metrics['confidence'] = compute_confidence_metrics(df)
    
    # 6. Sentiment Metrics
    console.print("[dim]Computing sentiment metrics...[/dim]")
    all_metrics['sentiment'] = compute_sentiment_metrics(df)
    
    # 7. Topic Type Distribution
    console.print("[dim]Computing topic type distribution...[/dim]")
    all_metrics['topic_types'] = compute_topic_type_metrics(df)
    
    # Display Results
    console.print("\n")
    
    # Table 1: Clustering
    if 'clustering' in all_metrics:
        t1 = Table(title="Clustering Quality")
        t1.add_column("Metric", style="bold")
        t1.add_column("Value", justify="right")
        for k, v in all_metrics['clustering'].items():
            t1.add_row(k, f"{v:.2f}")
        console.print(t1)
        console.print()
    
    # Table 2: Diversity & Coverage
    t2 = Table(title="Topic Diversity & Coverage")
    t2.add_column("Metric", style="bold")
    t2.add_column("Value", justify="right")
    for k, v in all_metrics['diversity'].items():
        t2.add_row(k, f"{v:.2f}")
    for k, v in all_metrics['source_diversity'].items():
        t2.add_row(k, f"{v:.2f}")
    console.print(t2)
    console.print()
    
    # Table 3: Temporal & Confidence
    t3 = Table(title="Temporal Coherence & Confidence")
    t3.add_column("Metric", style="bold")
    t3.add_column("Value", justify="right")
    for k, v in all_metrics['temporal'].items():
        t3.add_row(k, f"{v:.2f}")
    for k, v in all_metrics['confidence'].items():
        t3.add_row(k, f"{v:.2f}")
    console.print(t3)
    console.print()
    
    # Table 4: Distribution
    t4 = Table(title="Sentiment & Topic Type Distribution")
    t4.add_column("Metric", style="bold")
    t4.add_column("Value", justify="right")
    for k, v in all_metrics['sentiment'].items():
        t4.add_row(k, f"{v:.2f}")
    for k, v in all_metrics['topic_types'].items():
        t4.add_row(k, f"{v:.2f}")
    console.print(t4)
    
    # Save results
    output_path = args.output or os.path.join(args.state, 'comprehensive_eval.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_metrics, f, ensure_ascii=False, indent=2, default=str)
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")

if __name__ == "__main__":
    main()
