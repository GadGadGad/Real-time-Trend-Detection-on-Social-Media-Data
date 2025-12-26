"""
LLM-as-a-Judge Evaluation for Pipeline Clusters.
Uses Gemini/LLM to evaluate cluster coherence and topic relevance.

Usage:
    python scripts/eval/llm_eval_clusters.py --state demo/demo_finetuned_reranker --sample-size 10
"""
import argparse
import os
import json
import random
import pandas as pd
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv

load_dotenv()
console = Console()

def load_demo_state(save_dir):
    """Load demo state files."""
    state = {}
    
    results_path = os.path.join(save_dir, 'results.parquet')
    if os.path.exists(results_path):
        state['df_results'] = pd.read_parquet(results_path)
    
    mapping_path = os.path.join(save_dir, 'cluster_mapping.json')
    if os.path.exists(mapping_path):
        with open(mapping_path, 'r', encoding='utf-8') as f:
            state['cluster_mapping'] = json.load(f)
    
    return state

def get_cluster_samples(df, cluster_id, n=5):
    """Get sample posts from a cluster."""
    cluster_posts = df[df['cluster_id'] == cluster_id]
    if len(cluster_posts) == 0:
        return []
    
    samples = cluster_posts.sample(min(n, len(cluster_posts)))
    return samples['post_content'].tolist()

def evaluate_cluster_with_llm(cluster_name, posts, model):
    """Use LLM to evaluate cluster coherence."""
    posts_text = "\n".join([f"- {p[:300]}" for p in posts[:5]])
    
    prompt = f"""Bạn là chuyên gia đánh giá chất lượng clustering tin tức.

Cluster Name: "{cluster_name}"

Sample Posts trong cluster:
{posts_text}

Đánh giá cluster này theo các tiêu chí sau (thang điểm 1-5):

1. **Coherence** (Các bài viết có cùng chủ đề không?): 1-5
2. **Topic Relevance** (Tên cluster có mô tả đúng nội dung không?): 1-5
3. **Distinctiveness** (Cluster có rõ ràng, không bị trộn lẫn topics không?): 1-5

Trả lời CHÍNH XÁC theo format JSON:
{{"coherence": X, "relevance": X, "distinctiveness": X, "reasoning": "..."}}
"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        
        # Extract JSON
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            return result
    except Exception as e:
        console.print(f"[yellow]LLM error: {e}[/yellow]")
    
    return {"coherence": 0, "relevance": 0, "distinctiveness": 0, "reasoning": "Error"}

def evaluate_topic_assignment(post_content, assigned_topic, model):
    """Use LLM to evaluate if a post was correctly assigned to a topic."""
    prompt = f"""Bạn là chuyên gia đánh giá việc gán bài viết vào chủ đề.

Bài viết: "{post_content[:400]}"

Được gán vào topic: "{assigned_topic}"

Đánh giá:
1. Bài viết này có THỰC SỰ liên quan đến topic "{assigned_topic}" không?
2. Đây có phải là topic PHÙ HỢP NHẤT cho bài viết này không?

Trả lời CHÍNH XÁC theo format JSON:
{{"is_relevant": true/false, "is_best_fit": true/false, "confidence": 1-5, "better_topic": "..." }}
"""
    
    try:
        response = model.generate_content(prompt)
        text = response.text
        
        import re
        json_match = re.search(r'\{[^}]+\}', text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group())
    except Exception as e:
        pass
    
    return {"is_relevant": False, "is_best_fit": False, "confidence": 0}

def main():
    parser = argparse.ArgumentParser(description="LLM-based evaluation of pipeline clusters")
    parser.add_argument('--state', required=True, help='Path to demo state directory')
    parser.add_argument('--sample-size', type=int, default=10, help='Number of clusters to evaluate')
    parser.add_argument('--eval-assignments', action='store_true', help='Also evaluate topic assignments')
    parser.add_argument('--api-key', type=str, help='Gemini API key (or set GEMINI_API_KEY env)')
    args = parser.parse_args()
    
    # Setup Gemini
    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]Error: GEMINI_API_KEY required. Set via --api-key or environment.[/red]")
        return
    
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")
    
    console.print(f"[bold cyan]🧠 LLM Evaluation of Pipeline Results[/bold cyan]")
    console.print(f"   State: {args.state}")
    console.print(f"   Sample Size: {args.sample_size}\n")
    
    # Load state
    state = load_demo_state(args.state)
    if 'df_results' not in state:
        console.print("[red]Error: Could not load results.[/red]")
        return
    
    df = state['df_results']
    cluster_mapping = state.get('cluster_mapping', {})
    
    # Get unique clusters (exclude noise)
    clusters = df[df['cluster_id'] != -1]['cluster_id'].unique()
    sample_clusters = random.sample(list(clusters), min(args.sample_size, len(clusters)))
    
    console.print(f"[dim]Evaluating {len(sample_clusters)} clusters...[/dim]\n")
    
    # Evaluate clusters
    results = []
    for cid in track(sample_clusters, description="Evaluating clusters"):
        cluster_name = df[df['cluster_id'] == cid]['final_topic'].iloc[0] if len(df[df['cluster_id'] == cid]) > 0 else f"Cluster {cid}"
        posts = get_cluster_samples(df, cid, n=5)
        
        if posts:
            eval_result = evaluate_cluster_with_llm(cluster_name, posts, model)
            eval_result['cluster_id'] = cid
            eval_result['cluster_name'] = cluster_name
            eval_result['num_posts'] = len(df[df['cluster_id'] == cid])
            results.append(eval_result)
    
    # Display results
    table = Table(title="LLM Cluster Quality Evaluation")
    table.add_column("Cluster", style="bold", max_width=40)
    table.add_column("#Posts", justify="right")
    table.add_column("Coherence", justify="right")
    table.add_column("Relevance", justify="right")
    table.add_column("Distinct", justify="right")
    
    total_coherence = 0
    total_relevance = 0
    total_distinct = 0
    
    for r in results:
        table.add_row(
            r.get('cluster_name', '')[:40],
            str(r.get('num_posts', 0)),
            f"{r.get('coherence', 0)}/5",
            f"{r.get('relevance', 0)}/5",
            f"{r.get('distinctiveness', 0)}/5"
        )
        total_coherence += r.get('coherence', 0)
        total_relevance += r.get('relevance', 0)
        total_distinct += r.get('distinctiveness', 0)
    
    console.print(table)
    
    # Summary
    n = len(results)
    if n > 0:
        console.print(f"\n[bold]📊 Average Scores:[/bold]")
        console.print(f"   Coherence:      {total_coherence/n:.2f}/5")
        console.print(f"   Relevance:      {total_relevance/n:.2f}/5")
        console.print(f"   Distinctiveness: {total_distinct/n:.2f}/5")
        console.print(f"   [bold]Overall:        {(total_coherence+total_relevance+total_distinct)/(3*n):.2f}/5[/bold]")
    
    # Save detailed results
    output_path = os.path.join(args.state, 'llm_eval_results.json')
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    console.print(f"\n[dim]Detailed results saved to {output_path}[/dim]")

if __name__ == "__main__":
    main()
