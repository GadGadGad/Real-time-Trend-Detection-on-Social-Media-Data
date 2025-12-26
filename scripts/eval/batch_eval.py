"""
Batch Evaluation Script - Test multiple demo state folders and compare results.

Usage:
    python scripts/eval/batch_eval.py demo/data_4 demo/demo_bge_reranker demo/demo_finetuned_reranker --ground-truth scripts/eval/ground_truth_sample.csv
"""
import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from rich.console import Console
from rich.table import Table
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)

warnings.filterwarnings('ignore')
console = Console()

# ==================== METRICS FUNCTIONS ====================

def load_demo_state(save_dir):
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
    return state

def load_ground_truth(gt_path):
    gt_df = pd.read_csv(gt_path)
    gt_df.columns = [c.strip().lower().replace('grouth_truth', 'ground_truth') for c in gt_df.columns]
    return gt_df

def compute_all_metrics(state, gt_df=None):
    """Compute all metrics for a single state."""
    metrics = {}
    df = state.get('df_results')
    if df is None:
        return metrics
    
    # Clustering metrics
    if 'cluster_labels' in state and 'post_embeddings' in state:
        labels = state['cluster_labels']
        embs = state['post_embeddings']
        valid_mask = labels != -1
        valid_labels = labels[valid_mask]
        valid_embs = embs[valid_mask]
        
        n_clusters = len(set(valid_labels))
        metrics['Clusters'] = n_clusters
        metrics['Noise%'] = (labels == -1).sum() / len(labels) * 100
        
        if n_clusters > 1 and len(valid_embs) > n_clusters:
            try:
                metrics['Silhouette'] = silhouette_score(valid_embs, valid_labels)
            except:
                metrics['Silhouette'] = 0.0
    
    # Topic diversity
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    metrics['Topics'] = valid_topics[group_col].nunique()
    metrics['Assigned%'] = len(valid_topics) / len(df) * 100
    
    # Source diversity
    if 'source' in df.columns:
        sources_per_topic = valid_topics.groupby(group_col)['source'].nunique()
        metrics['Src/Topic'] = sources_per_topic.mean() if len(sources_per_topic) > 0 else 0
    
    # Confidence
    if 'score' in df.columns:
        metrics['AvgConf'] = df['score'].mean()
    
    # P/R/F1 if ground truth provided
    if gt_df is not None:
        merged = match_gt(gt_df, df)
        if len(merged) > 0:
            true_labels = merged['ground_truth'].fillna(-1).astype(int).tolist()
            
            if 'cluster_id' in merged.columns:
                pred_labels = merged['cluster_id'].fillna(-1).astype(int).tolist()
            elif 'final_topic' in merged.columns:
                topics = merged['final_topic'].fillna('Noise')
                topic_to_id = {t: i for i, t in enumerate(topics.unique())}
                pred_labels = [topic_to_id[t] for t in topics]
            else:
                pred_labels = list(range(len(merged)))
            
            # BCubed F1
            bc_f1 = bcubed_f1(true_labels, pred_labels)
            metrics['BC-F1'] = bc_f1
            
            # Standard metrics
            metrics['ARI'] = adjusted_rand_score(true_labels, pred_labels)
            metrics['NMI'] = normalized_mutual_info_score(true_labels, pred_labels)
            metrics['Purity'] = cluster_purity(true_labels, pred_labels)
    
    return metrics

def match_gt(gt_df, pred_df):
    if 'original_index' in pred_df.columns:
        return gt_df.merge(pred_df, left_on='id', right_on='original_index', how='inner')
    elif 'id' in pred_df.columns:
        return gt_df.merge(pred_df, on='id', how='inner')
    else:
        min_len = min(len(gt_df), len(pred_df))
        return pd.concat([gt_df.head(min_len).reset_index(drop=True), 
                         pred_df.head(min_len).reset_index(drop=True)], axis=1)

def bcubed_f1(true_labels, pred_labels):
    n = len(true_labels)
    if n == 0:
        return 0
    
    true_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        true_clusters[t].add(i)
        pred_clusters[p].add(i)
    
    precision_sum = recall_sum = 0
    for i in range(n):
        t_label, p_label = true_labels[i], pred_labels[i]
        correct = len(true_clusters[t_label] & pred_clusters[p_label])
        precision_sum += correct / len(pred_clusters[p_label])
        recall_sum += correct / len(true_clusters[t_label])
    
    p, r = precision_sum / n, recall_sum / n
    return 2 * p * r / (p + r) if (p + r) > 0 else 0

def cluster_purity(true_labels, pred_labels):
    clusters = defaultdict(list)
    for t, p in zip(true_labels, pred_labels):
        clusters[p].append(t)
    total = sum(max(set(items), key=items.count) and items.count(max(set(items), key=items.count)) for items in clusters.values())
    return total / len(true_labels) if true_labels else 0

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="Batch Evaluation - Compare multiple demo states")
    parser.add_argument('states', nargs='+', help='Paths to demo state directories')
    parser.add_argument('--ground-truth', help='Path to ground truth CSV')
    parser.add_argument('--output', help='Output JSON path')
    args = parser.parse_args()
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"[bold cyan]   BATCH PIPELINE EVALUATION[/bold cyan]")
    console.print(f"[bold cyan]{'='*60}[/bold cyan]")
    console.print(f"   Evaluating {len(args.states)} states...")
    console.print()
    
    # Load ground truth
    gt_df = load_ground_truth(args.ground_truth) if args.ground_truth else None
    
    # Evaluate all states
    all_results = {}
    for state_path in args.states:
        name = os.path.basename(state_path.rstrip('/'))
        console.print(f"[dim]Evaluating {name}...[/dim]")
        
        state = load_demo_state(state_path)
        if 'df_results' not in state:
            console.print(f"[yellow]  Skipping {name}: no results found[/yellow]")
            continue
        
        metrics = compute_all_metrics(state, gt_df)
        all_results[name] = metrics
    
    # Build comparison table
    if not all_results:
        console.print("[red]No valid states found.[/red]")
        return
    
    # Get all metric names
    all_metrics = set()
    for m in all_results.values():
        all_metrics.update(m.keys())
    all_metrics = sorted(all_metrics)
    
    # Create table
    table = Table(title="Comparison Results")
    table.add_column("Metric", style="bold")
    for name in all_results.keys():
        table.add_column(name, justify="right")
    
    for metric in all_metrics:
        row = [metric]
        values = [all_results[name].get(metric, '') for name in all_results.keys()]
        
        # Highlight best value (green)
        numeric_vals = [(i, v) for i, v in enumerate(values) if isinstance(v, (int, float))]
        if numeric_vals:
            # For Noise%, lower is better; for others, higher is better
            if 'Noise' in metric:
                best_idx = min(numeric_vals, key=lambda x: x[1])[0]
            else:
                best_idx = max(numeric_vals, key=lambda x: x[1])[0]
            
            for i, v in enumerate(values):
                if isinstance(v, (int, float)):
                    if i == best_idx:
                        row.append(f"[bold green]{v:.4f}[/bold green]")
                    else:
                        row.append(f"{v:.4f}")
                else:
                    row.append(str(v))
        else:
            row.extend([str(v) for v in values])
        
        table.add_row(*row)
    
    console.print()
    console.print(table)
    
    # Save results
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(all_results, f, ensure_ascii=False, indent=2)
        console.print(f"\n[dim]Results saved to {args.output}[/dim]")
    
    console.print(f"\n[bold cyan]{'='*60}[/bold cyan]\n")

if __name__ == "__main__":
    main()
