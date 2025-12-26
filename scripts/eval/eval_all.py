"""
All-in-One Pipeline Evaluation Script.
Combines: Comprehensive metrics, P/R/F1, and optional LLM evaluation.

Usage:
    # Full evaluation (no LLM)
    python scripts/eval/eval_all.py --state demo/demo_finetuned_reranker --ground-truth scripts/eval/ground_truth_sample.csv

    # With LLM evaluation
    python scripts/eval/eval_all.py --state demo/demo_finetuned_reranker --ground-truth scripts/eval/ground_truth_sample.csv --llm --sample-size 10
"""
import argparse
import os
import json
import warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from itertools import combinations
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.progress import track
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score,
    adjusted_rand_score, normalized_mutual_info_score, v_measure_score
)
from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings('ignore')
console = Console()

# ==================== DATA LOADING ====================

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

def load_ground_truth(gt_path):
    """Load ground truth labels."""
    gt_df = pd.read_csv(gt_path)
    gt_df.columns = [c.strip().lower().replace('grouth_truth', 'ground_truth') for c in gt_df.columns]
    return gt_df

# ==================== CLUSTERING METRICS ====================

def compute_clustering_metrics(labels, embeddings):
    """Internal clustering quality metrics."""
    valid_mask = labels != -1
    valid_labels = labels[valid_mask]
    valid_embs = embeddings[valid_mask]
    
    metrics = {}
    n_clusters = len(set(valid_labels))
    metrics['Num Clusters'] = n_clusters
    metrics['Noise Ratio (%)'] = (labels == -1).sum() / len(labels) * 100
    metrics['Avg Cluster Size'] = len(valid_labels) / max(n_clusters, 1)
    
    if n_clusters > 1 and len(valid_embs) > n_clusters:
        try:
            metrics['Silhouette'] = silhouette_score(valid_embs, valid_labels)
        except:
            metrics['Silhouette'] = 0.0
        try:
            metrics['Calinski-Harabasz'] = calinski_harabasz_score(valid_embs, valid_labels)
        except:
            metrics['Calinski-Harabasz'] = 0.0
    
    return metrics

def compute_topic_diversity(df):
    """Measure diversity of detected topics."""
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    
    unique_topics = valid_topics[group_col].nunique()
    total_posts = len(df)
    assigned_posts = len(valid_topics)
    
    return {
        'Unique Topics': unique_topics,
        'Assigned Posts': assigned_posts,
        'Assignment Rate (%)': assigned_posts / total_posts * 100 if total_posts > 0 else 0
    }

def compute_source_diversity(df):
    """Measure source diversity within topics."""
    group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
    source_col = 'source' if 'source' in df.columns else None
    
    if source_col is None:
        return {}
    
    valid_topics = df[~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])]
    sources_per_topic = valid_topics.groupby(group_col)[source_col].nunique()
    
    return {
        'Avg Sources/Topic': sources_per_topic.mean() if len(sources_per_topic) > 0 else 0,
        'Multi-source Topics': (sources_per_topic > 1).sum()
    }

def compute_confidence_metrics(df):
    """Metrics about matching confidence."""
    if 'score' not in df.columns:
        return {}
    
    scores = df['score'].dropna()
    return {
        'Avg Confidence': scores.mean(),
        'High Confidence (>70%)': (scores > 0.7).sum() / len(scores) * 100 if len(scores) > 0 else 0
    }

def compute_sentiment_metrics(df):
    """Sentiment distribution metrics."""
    if 'sentiment' not in df.columns:
        return {}
    
    dist = df['sentiment'].value_counts()
    total = len(df)
    
    return {
        'Positive (%)': dist.get('Positive', 0) / total * 100,
        'Negative (%)': dist.get('Negative', 0) / total * 100,
        'Neutral (%)': dist.get('Neutral', 0) / total * 100
    }

# ==================== P/R/F1 METRICS ====================

def bcubed_precision_recall_f1(true_labels, pred_labels):
    """Compute BCubed Precision, Recall, F1."""
    n = len(true_labels)
    if n == 0:
        return 0, 0, 0
    
    true_clusters = defaultdict(set)
    pred_clusters = defaultdict(set)
    
    for i, (t, p) in enumerate(zip(true_labels, pred_labels)):
        true_clusters[t].add(i)
        pred_clusters[p].add(i)
    
    precision_sum = recall_sum = 0
    
    for i in range(n):
        t_label, p_label = true_labels[i], pred_labels[i]
        true_cluster = true_clusters[t_label]
        pred_cluster = pred_clusters[p_label]
        correct_in_pred = len(true_cluster & pred_cluster)
        
        precision_sum += correct_in_pred / len(pred_cluster)
        recall_sum += correct_in_pred / len(true_cluster)
    
    precision = precision_sum / n
    recall = recall_sum / n
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def pairwise_precision_recall_f1(true_labels, pred_labels):
    """Compute Pairwise Precision, Recall, F1."""
    n = len(true_labels)
    if n < 2:
        return 0, 0, 0
    
    tp = fp = fn = 0
    
    for i, j in combinations(range(n), 2):
        same_true = true_labels[i] == true_labels[j]
        same_pred = pred_labels[i] == pred_labels[j]
        
        if same_true and same_pred:
            tp += 1
        elif not same_true and same_pred:
            fp += 1
        elif same_true and not same_pred:
            fn += 1
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1

def cluster_purity(true_labels, pred_labels):
    """Compute cluster purity."""
    clusters = defaultdict(list)
    for t, p in zip(true_labels, pred_labels):
        clusters[p].append(t)
    
    total_correct = 0
    for cluster_items in clusters.values():
        most_common = max(set(cluster_items), key=cluster_items.count)
        total_correct += cluster_items.count(most_common)
    
    return total_correct / len(true_labels) if true_labels else 0

def match_predictions_to_gt(gt_df, pred_df):
    """Match predictions to ground truth."""
    if 'original_index' in pred_df.columns:
        return gt_df.merge(pred_df, left_on='id', right_on='original_index', how='inner')
    elif 'id' in pred_df.columns:
        return gt_df.merge(pred_df, on='id', how='inner')
    else:
        min_len = min(len(gt_df), len(pred_df))
        return pd.concat([gt_df.head(min_len).reset_index(drop=True), 
                         pred_df.head(min_len).reset_index(drop=True)], axis=1)

# ==================== LLM EVALUATION ====================

def evaluate_cluster_with_llm(cluster_name, posts, model):
    """Use LLM to evaluate cluster coherence."""
    posts_text = "\n".join([f"- {p[:300]}" for p in posts[:5]])
    
    prompt = f"""Đánh giá cluster tin tức này (1-5 điểm):

Cluster: "{cluster_name}"
Posts:
{posts_text}

Trả lời JSON: {{"coherence": X, "relevance": X, "distinctiveness": X}}"""
    
    try:
        response = model.generate_content(prompt)
        import re
        json_match = re.search(r'\{[^}]+\}', response.text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        console.print(f"[yellow]LLM error: {e}[/yellow]")
    
    return {"coherence": 0, "relevance": 0, "distinctiveness": 0}

# ==================== MAIN ====================

def main():
    parser = argparse.ArgumentParser(description="All-in-One Pipeline Evaluation")
    parser.add_argument('--state', required=True, help='Path to demo state directory')
    parser.add_argument('--ground-truth', help='Path to ground truth CSV (for P/R/F1)')
    parser.add_argument('--llm', action='store_true', help='Enable LLM evaluation')
    parser.add_argument('--sample-size', type=int, default=10, help='LLM sample size')
    parser.add_argument('--api-key', type=str, help='Gemini API key')
    parser.add_argument('--model', type=str, default='gemini-1.5-flash', help='LLM model')
    parser.add_argument('--output', type=str, help='Output JSON path')
    args = parser.parse_args()
    
    console.print(f"\n[bold cyan]{'='*50}[/bold cyan]")
    console.print(f"[bold cyan]   ALL-IN-ONE PIPELINE EVALUATION[/bold cyan]")
    console.print(f"[bold cyan]{'='*50}[/bold cyan]")
    console.print(f"   State: {args.state}")
    if args.ground_truth:
        console.print(f"   Ground Truth: {args.ground_truth}")
    console.print()
    
    # Load data
    state = load_demo_state(args.state)
    if 'df_results' not in state:
        console.print("[red]Error: Could not load results.[/red]")
        return
    
    df = state['df_results']
    all_results = {'state': args.state, 'evaluated_at': datetime.now().isoformat()}
    
    # ===== Section 1: Clustering Quality =====
    console.print("[bold]1. CLUSTERING QUALITY[/bold]")
    t1 = Table(show_header=False, box=None)
    t1.add_column("Metric", style="dim")
    t1.add_column("Value", justify="right")
    
    if 'cluster_labels' in state and 'post_embeddings' in state:
        clust_metrics = compute_clustering_metrics(state['cluster_labels'], state['post_embeddings'])
        all_results['clustering'] = clust_metrics
        for k, v in clust_metrics.items():
            t1.add_row(k, f"{v:.2f}")
    console.print(t1)
    console.print()
    
    # ===== Section 2: Topic Diversity =====
    console.print("[bold]2. TOPIC DIVERSITY & SOURCE[/bold]")
    t2 = Table(show_header=False, box=None)
    t2.add_column("Metric", style="dim")
    t2.add_column("Value", justify="right")
    
    div_metrics = {**compute_topic_diversity(df), **compute_source_diversity(df)}
    all_results['diversity'] = div_metrics
    for k, v in div_metrics.items():
        t2.add_row(k, f"{v:.2f}")
    console.print(t2)
    console.print()
    
    # ===== Section 3: Confidence & Sentiment =====
    console.print("[bold]3. CONFIDENCE & SENTIMENT[/bold]")
    t3 = Table(show_header=False, box=None)
    t3.add_column("Metric", style="dim")
    t3.add_column("Value", justify="right")
    
    conf_metrics = {**compute_confidence_metrics(df), **compute_sentiment_metrics(df)}
    all_results['distribution'] = conf_metrics
    for k, v in conf_metrics.items():
        t3.add_row(k, f"{v:.2f}")
    console.print(t3)
    console.print()
    
    # ===== Section 4: P/R/F1 (if ground truth provided) =====
    if args.ground_truth and os.path.exists(args.ground_truth):
        console.print("[bold]4. P/R/F1 METRICS (vs Ground Truth)[/bold]")
        
        gt_df = load_ground_truth(args.ground_truth)
        merged = match_predictions_to_gt(gt_df, df)
        
        true_labels = merged['ground_truth'].fillna(-1).astype(int).tolist()
        
        if 'cluster_id' in merged.columns:
            pred_labels = merged['cluster_id'].fillna(-1).astype(int).tolist()
        elif 'final_topic' in merged.columns:
            topics = merged['final_topic'].fillna('Noise')
            topic_to_id = {t: i for i, t in enumerate(topics.unique())}
            pred_labels = [topic_to_id[t] for t in topics]
        else:
            pred_labels = list(range(len(merged)))
        
        # BCubed
        bc_p, bc_r, bc_f1 = bcubed_precision_recall_f1(true_labels, pred_labels)
        
        # Pairwise
        pw_p, pw_r, pw_f1 = pairwise_precision_recall_f1(true_labels, pred_labels)
        
        # Standard metrics
        ari = adjusted_rand_score(true_labels, pred_labels)
        nmi = normalized_mutual_info_score(true_labels, pred_labels)
        vm = v_measure_score(true_labels, pred_labels)
        purity = cluster_purity(true_labels, pred_labels)
        
        t4 = Table(title="")
        t4.add_column("Metric", style="bold")
        t4.add_column("Precision", justify="right")
        t4.add_column("Recall", justify="right")
        t4.add_column("F1", justify="right", style="green")
        t4.add_row("BCubed", f"{bc_p:.4f}", f"{bc_r:.4f}", f"{bc_f1:.4f}")
        t4.add_row("Pairwise", f"{pw_p:.4f}", f"{pw_r:.4f}", f"{pw_f1:.4f}")
        console.print(t4)
        
        t5 = Table(show_header=False, box=None)
        t5.add_column("Metric", style="dim")
        t5.add_column("Value", justify="right")
        t5.add_row("ARI", f"{ari:.4f}")
        t5.add_row("NMI", f"{nmi:.4f}")
        t5.add_row("V-Measure", f"{vm:.4f}")
        t5.add_row("Purity", f"{purity:.4f}")
        console.print(t5)
        
        all_results['prf1'] = {
            'bcubed': {'precision': bc_p, 'recall': bc_r, 'f1': bc_f1},
            'pairwise': {'precision': pw_p, 'recall': pw_r, 'f1': pw_f1},
            'ari': ari, 'nmi': nmi, 'v_measure': vm, 'purity': purity
        }
        console.print()
    
    # ===== Section 5: LLM Evaluation (optional) =====
    if args.llm:
        api_key = args.api_key or os.getenv("GEMINI_API_KEY")
        if not api_key:
            console.print("[yellow]LLM eval skipped: No GEMINI_API_KEY[/yellow]")
        else:
            console.print("[bold]5. LLM CLUSTER EVALUATION[/bold]")
            
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(args.model)
            
            group_col = 'final_topic' if 'final_topic' in df.columns else 'cluster_id'
            valid_mask = ~df[group_col].isin(['Unassigned', 'Noise', '[Noise]', 'Discovery', -1])
            clusters = df[valid_mask][group_col].unique()
            
            import random
            sample = random.sample(list(clusters), min(args.sample_size, len(clusters)))
            
            llm_results = []
            total_c = total_r = total_d = 0
            
            for cid in track(sample, description="LLM Eval"):
                cluster_df = df[df[group_col] == cid]
                posts = cluster_df['post_content'].head(5).tolist() if 'post_content' in cluster_df.columns else []
                
                if posts:
                    result = evaluate_cluster_with_llm(str(cid), posts, model)
                    result['cluster'] = str(cid)[:30]
                    llm_results.append(result)
                    total_c += result.get('coherence', 0)
                    total_r += result.get('relevance', 0)
                    total_d += result.get('distinctiveness', 0)
            
            n = len(llm_results)
            if n > 0:
                console.print(f"   Avg Coherence:      {total_c/n:.2f}/5")
                console.print(f"   Avg Relevance:      {total_r/n:.2f}/5")
                console.print(f"   Avg Distinctiveness: {total_d/n:.2f}/5")
                console.print(f"   [bold]Overall LLM Score:  {(total_c+total_r+total_d)/(3*n):.2f}/5[/bold]")
                
                all_results['llm'] = {
                    'coherence': total_c/n, 'relevance': total_r/n, 
                    'distinctiveness': total_d/n, 'samples': llm_results
                }
    
    # Save results
    output_path = args.output or os.path.join(args.state, 'eval_all.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    console.print(f"\n[dim]Results saved to {output_path}[/dim]")
    console.print(f"[bold cyan]{'='*50}[/bold cyan]\n")

if __name__ == "__main__":
    main()
