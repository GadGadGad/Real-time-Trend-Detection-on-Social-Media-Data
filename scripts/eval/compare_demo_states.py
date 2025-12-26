"""
Compare two pipeline demo states to measure performance differences.
Useful for comparing base models vs fine-tuned models.

Usage:
    python scripts/eval/compare_demo_states.py --state1 demo_state_base --state2 demo_state_tuned
"""
import argparse
import os
import json
import numpy as np
import pandas as pd
from collections import Counter
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    adjusted_rand_score, 
    normalized_mutual_info_score, 
    silhouette_score,
    calinski_harabasz_score
)

console = Console()

def load_demo_state(save_dir):
    """Load demo state files."""
    state = {}
    
    # Results
    results_path = os.path.join(save_dir, 'results.parquet')
    if os.path.exists(results_path):
        state['df_results'] = pd.read_parquet(results_path)
    else:
        # Try JSON fallback
        json_path = os.path.join(save_dir, 'results.json')
        if os.path.exists(json_path):
            state['df_results'] = pd.read_json(json_path)
    
    # Cluster Mapping
    mapping_path = os.path.join(save_dir, 'cluster_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            state['cluster_mapping'] = json.load(f)
    
    # Post Embeddings
    emb_path = os.path.join(save_dir, 'post_embeddings.npy')
    if os.path.exists(emb_path):
        state['post_embeddings'] = np.load(emb_path)
    
    # Cluster Labels
    labels_path = os.path.join(save_dir, 'cluster_labels.npy')
    if os.path.exists(labels_path):
        state['cluster_labels'] = np.load(labels_path)
    
    # Metadata
    meta_path = os.path.join(save_dir, 'metadata.json')
    if os.path.exists(meta_path):
        with open(meta_path, 'r', encoding='utf-8') as f:
            state['metadata'] = json.load(f)
    
    return state

def compute_clustering_metrics(labels, embeddings):
    """Compute internal clustering quality metrics."""
    # Filter out noise (-1) for some metrics
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_embs = embeddings[valid_mask]
    
    metrics = {}
    
    # Number of clusters
    n_clusters = len(set(valid_labels))
    metrics['num_clusters'] = n_clusters
    metrics['noise_ratio'] = (labels == -1).sum() / len(labels) * 100
    
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

def compute_topic_metrics(df):
    """Compute topic distribution and matching metrics."""
    metrics = {}
    
    if 'topic_type' in df.columns:
        topic_dist = df['topic_type'].value_counts()
        metrics['trending'] = topic_dist.get('Trending', 0)
        metrics['discovery'] = topic_dist.get('Discovery', 0)
        metrics['noise'] = topic_dist.get('Noise', 0)
        metrics['match_rate'] = df['is_matched'].mean() * 100 if 'is_matched' in df.columns else 0
    
    if 'score' in df.columns:
        metrics['avg_score'] = df['score'].mean()
        metrics['high_confidence'] = (df['score'] > 0.7).sum()
    
    if 'sentiment' in df.columns:
        sent_dist = df['sentiment'].value_counts()
        metrics['positive_pct'] = sent_dist.get('Positive', 0) / len(df) * 100
        metrics['negative_pct'] = sent_dist.get('Negative', 0) / len(df) * 100
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Compare two pipeline demo states")
    parser.add_argument('--state1', required=True, help='Path to first demo state (baseline)')
    parser.add_argument('--state2', required=True, help='Path to second demo state (tuned)')
    parser.add_argument('--name1', default='Baseline', help='Name for first state')
    parser.add_argument('--name2', default='Fine-tuned', help='Name for second state')
    args = parser.parse_args()
    
    console.print(f"[bold cyan]📊 Comparing Pipeline States[/bold cyan]")
    console.print(f"   State 1: {args.state1} ({args.name1})")
    console.print(f"   State 2: {args.state2} ({args.name2})\n")
    
    # Load states
    state1 = load_demo_state(args.state1)
    state2 = load_demo_state(args.state2)
    
    if 'df_results' not in state1 or 'df_results' not in state2:
        console.print("[red]Error: Could not load results from one or both states.[/red]")
        return
    
    # === Clustering Metrics ===
    table1 = Table(title="Clustering Quality")
    table1.add_column("Metric", style="bold")
    table1.add_column(args.name1, justify="right")
    table1.add_column(args.name2, justify="right", style="green")
    table1.add_column("Δ", justify="right")
    
    if 'cluster_labels' in state1 and 'post_embeddings' in state1:
        m1 = compute_clustering_metrics(state1['cluster_labels'], state1['post_embeddings'])
    else:
        m1 = {'num_clusters': 0, 'noise_ratio': 0, 'silhouette': 0, 'calinski_harabasz': 0}
    
    if 'cluster_labels' in state2 and 'post_embeddings' in state2:
        m2 = compute_clustering_metrics(state2['cluster_labels'], state2['post_embeddings'])
    else:
        m2 = {'num_clusters': 0, 'noise_ratio': 0, 'silhouette': 0, 'calinski_harabasz': 0}
    
    for key, label in [
        ('num_clusters', 'Num Clusters'),
        ('noise_ratio', 'Noise Ratio (%)'),
        ('silhouette', 'Silhouette Score'),
        ('calinski_harabasz', 'Calinski-Harabasz')
    ]:
        v1, v2 = m1.get(key, 0), m2.get(key, 0)
        diff = v2 - v1
        sign = '+' if diff >= 0 else ''
        table1.add_row(label, f"{v1:.2f}", f"{v2:.2f}", f"{sign}{diff:.2f}")
    
    console.print(table1)
    console.print()
    
    # === Topic Metrics ===
    table2 = Table(title="Topic Distribution & Matching")
    table2.add_column("Metric", style="bold")
    table2.add_column(args.name1, justify="right")
    table2.add_column(args.name2, justify="right", style="green")
    table2.add_column("Δ", justify="right")
    
    t1 = compute_topic_metrics(state1['df_results'])
    t2 = compute_topic_metrics(state2['df_results'])
    
    for key, label in [
        ('trending', 'Trending Topics'),
        ('discovery', 'Discovery Topics'),
        ('noise', 'Noise Posts'),
        ('match_rate', 'Match Rate (%)'),
        ('avg_score', 'Avg Confidence Score'),
        ('high_confidence', 'High Confidence (>0.7)'),
        ('positive_pct', 'Positive %'),
        ('negative_pct', 'Negative %')
    ]:
        v1, v2 = t1.get(key, 0), t2.get(key, 0)
        diff = v2 - v1
        sign = '+' if diff >= 0 else ''
        table2.add_row(label, f"{v1:.2f}", f"{v2:.2f}", f"{sign}{diff:.2f}")
    
    console.print(table2)
    console.print()
    
    # === Cross-state comparison (if same posts) ===
    if 'cluster_labels' in state1 and 'cluster_labels' in state2:
        if len(state1['cluster_labels']) == len(state2['cluster_labels']):
            ari = adjusted_rand_score(state1['cluster_labels'], state2['cluster_labels'])
            nmi = normalized_mutual_info_score(state1['cluster_labels'], state2['cluster_labels'])
            console.print(f"[bold]Cross-State Agreement:[/bold]")
            console.print(f"   ARI (cluster similarity): {ari:.4f}")
            console.print(f"   NMI (mutual information): {nmi:.4f}")
            console.print(f"   [dim](1.0 = identical clustering, 0.0 = no agreement)[/dim]")

if __name__ == "__main__":
    main()
