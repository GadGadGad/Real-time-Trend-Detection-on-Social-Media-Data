"""
LLM-as-a-Judge Evaluation for Pipeline Clusters
(PAIRWISE + LLM-INFERRED TOPIC CONSISTENCY)

Metrics:
- Pairwise win-rate (coherence comparison)
- LLM-inferred Topic Consistency (semantic purity)

Usage:
    python scripts/eval/llm_eval_clusters.py \
        --state demo/demo_finetuned_reranker \
        --sample-size 10 \
        --pairwise-runs 3 \
        --topic-sample 15
"""

import argparse
import os
import json
import random
import re
import pandas as pd
from collections import defaultdict
from rich.console import Console
from rich.table import Table
from rich.progress import track
from dotenv import load_dotenv

load_dotenv()
console = Console()

# -------------------------
# Data loading
# -------------------------
def load_demo_state(save_dir):
    state = {}
    results_path = os.path.join(save_dir, "results.parquet")
    mapping_path = os.path.join(save_dir, "cluster_mapping.json")

    if os.path.exists(results_path):
        state["df_results"] = pd.read_parquet(results_path)

    if os.path.exists(mapping_path):
        with open(mapping_path, "r", encoding="utf-8") as f:
            state["cluster_mapping"] = json.load(f)

    return state


def get_cluster_samples(df, group_col, cid, n=5):
    cluster_df = df[df[group_col] == cid]
    if len(cluster_df) == 0:
        return []
    return (
        cluster_df.sample(min(n, len(cluster_df)))["post_content"]
        .astype(str)
        .str.slice(0, 300)
        .tolist()
    )


# -------------------------
# Utils
# -------------------------
def safe_json_extract(text):
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except Exception:
        return None


# -------------------------
# Pairwise judging
# -------------------------
def pairwise_judge(posts_a, posts_b, model):
    prompt = f"""
Bạn là chuyên gia đánh giá chất lượng clustering tin tức.
Hai cluster dưới đây KHÔNG có tên, chỉ gồm các bài viết mẫu.

Cluster A:
{chr(10).join("- " + p for p in posts_a)}

Cluster B:
{chr(10).join("- " + p for p in posts_b)}

Câu hỏi:
1. Cluster nào COHERENT hơn (các bài cùng chủ đề hơn)?
2. Cluster nào RÕ RÀNG hơn, ít bị trộn topic hơn?

Trả lời CHÍNH XÁC theo JSON:
{{
  "better_cluster": "A" | "B" | "Tie",
  "confidence": 1-5,
  "reasoning": "..."
}}
"""
    response = model.generate_content(prompt)
    return safe_json_extract(response.text)


# -------------------------
# Topic Consistency
# -------------------------
def infer_cluster_topic(posts, model):
    prompt = f"""
Bạn là chuyên gia phân tích chủ đề tin tức.

Dưới đây là các bài viết thuộc CÙNG MỘT CLUSTER:
{chr(10).join("- " + p for p in posts)}

Nhiệm vụ:
1. Suy ra CHỦ ĐỀ CHUNG của cluster
2. Đặt một tên topic NGẮN GỌN (≤ 10 từ)
3. Mô tả topic trong 1 câu

Trả lời CHÍNH XÁC theo JSON:
{{
  "topic_name": "...",
  "topic_description": "..."
}}
"""
    res = model.generate_content(prompt)
    return safe_json_extract(res.text)


def check_post_topic(post, topic, model):
    prompt = f"""
Bạn là chuyên gia đánh giá nội dung tin tức.

Topic:
"{topic['topic_name']}"
Mô tả:
"{topic['topic_description']}"

Bài viết:
"{post}"

Câu hỏi:
Bài viết này có PHÙ HỢP với topic trên không?

Trả lời CHÍNH XÁC theo JSON:
{{
  "is_relevant": true | false,
  "confidence": 1-5
}}
"""
    res = model.generate_content(prompt)
    return safe_json_extract(res.text)


def compute_topic_consistency(df, group_col, cid, model, sample_n=15):
    cluster_df = df[df[group_col] == cid]
    if len(cluster_df) == 0:
        return None

    posts = (
        cluster_df.sample(min(sample_n, len(cluster_df)))["post_content"]
        .astype(str)
        .str.slice(0, 400)
        .tolist()
    )

    topic = infer_cluster_topic(posts[:5], model)
    if not topic:
        return None

    judgments = []
    for p in posts:
        r = check_post_topic(p, topic, model)
        if r:
            judgments.append(r)

    if not judgments:
        return None

    simple = sum(j["is_relevant"] for j in judgments) / len(judgments)
    weighted = (
        sum(j["confidence"] for j in judgments if j["is_relevant"])
        / sum(j["confidence"] for j in judgments)
    )

    return {
        "topic": topic,
        "simple_consistency": round(simple, 3),
        "weighted_consistency": round(weighted, 3),
        "n_checked": len(judgments),
    }


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser("LLM cluster evaluation")
    parser.add_argument("--state", required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--pairwise-runs", type=int, default=3)
    parser.add_argument("--topic-sample", type=int, default=15)
    parser.add_argument("--api-key", type=str)
    parser.add_argument("--model", type=str, default="gemma-3-27b-it")
    args = parser.parse_args()

    api_key = args.api_key or os.getenv("GEMINI_API_KEY")
    if not api_key:
        console.print("[red]GEMINI_API_KEY required[/red]")
        return

    import google.generativeai as genai

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(args.model)

    state = load_demo_state(args.state)
    if "df_results" not in state:
        console.print("[red]results.parquet not found[/red]")
        return

    df = state["df_results"]
    group_col = "cluster_id" if "cluster_id" in df.columns else "final_topic"
    valid_df = df[df[group_col] != -1] if group_col == "cluster_id" else df
    clusters = list(valid_df[group_col].unique())
    sampled = random.sample(clusters, min(args.sample_size, len(clusters)))

    # -------------------------
    # Pairwise
    # -------------------------
    win_stats = defaultdict(lambda: {"win": 0, "loss": 0, "tie": 0})

    for _ in track(range(args.pairwise_runs), description="Pairwise judging"):
        random.shuffle(sampled)
        for i in range(0, len(sampled) - 1, 2):
            a, b = sampled[i], sampled[i + 1]
            pa = get_cluster_samples(df, group_col, a)
            pb = get_cluster_samples(df, group_col, b)
            if not pa or not pb:
                continue

            r = pairwise_judge(pa, pb, model)
            if not r:
                continue

            if r["better_cluster"] == "A":
                win_stats[a]["win"] += 1
                win_stats[b]["loss"] += 1
            elif r["better_cluster"] == "B":
                win_stats[b]["win"] += 1
                win_stats[a]["loss"] += 1
            else:
                win_stats[a]["tie"] += 1
                win_stats[b]["tie"] += 1

    # -------------------------
    # Topic consistency
    # -------------------------
    topic_stats = {}
    for cid in track(sampled, description="Topic consistency"):
        res = compute_topic_consistency(
            df, group_col, cid, model, args.topic_sample
        )
        if res:
            topic_stats[cid] = res

    # -------------------------
    # Report
    # -------------------------
    table = Table(title="LLM Cluster Evaluation")
    table.add_column("Cluster")
    table.add_column("Win-rate")
    table.add_column("Topic")
    table.add_column("Consistency")

    for cid in sampled:
        s = win_stats[cid]
        total = s["win"] + s["loss"]
        winrate = s["win"] / total if total > 0 else 0.0

        topic = topic_stats.get(cid, {})
        table.add_row(
            str(cid),
            f"{winrate:.2f}",
            topic.get("topic", {}).get("topic_name", "-"),
            str(topic.get("weighted_consistency", "-")),
        )

    console.print(table)

    out = {
        "pairwise": win_stats,
        "topic_consistency": topic_stats,
    }

    out_path = os.path.join(args.state, "llm_cluster_eval.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    console.print(f"\n[dim]Saved results to {out_path}[/dim]")


if __name__ == "__main__":
    main()
