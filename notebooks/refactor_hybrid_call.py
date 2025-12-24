import json
import os

NB_PATH = "notebooks/analysis-playground.ipynb"

# The new cells to insert
NEW_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": ["### 🔬 Decomposed Hybrid Pipeline\n", "Instead of running the black-box `find_matches_hybrid`, we break it down into stages here for inspection."]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- STEP 1: Detailed Setup & Imports ---\n",
            "from src.pipeline.pipeline_stages import run_summarization_stage, run_sahc_clustering, calculate_match_scores\n",
            "from src.utils.text_processing.vectorizers import get_embeddings\n",
            "from src.pipeline.main_pipeline import extract_dynamic_anchors\n",
            "from src.core.analysis.clustering import extract_cluster_labels\n",
            "from src.pipeline.trend_scoring import calculate_unified_score\n",
            "from src.core.analysis.sentiment import batch_analyze_sentiment\n",
            "from sentence_transformers import SentenceTransformer, CrossEncoder\n",
            "from rich.console import Console\n",
            "from dateutil import parser\n",
            "import torch\n",
            "import numpy as np\n",
            "\n",
            "console = Console()\n",
            "\n",
            "# Setup Devices & Models\n",
            "embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'\n",
            "print(f\"🚀 Using Device: {embedding_device}\")\n",
            "\n",
            "embedder = SentenceTransformer(MODEL_NAME, device=embedding_device)\n",
            "\n",
            "reranker = None\n",
            "if RERANK:\n",
            "    try: \n",
            "        reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=embedding_device)\n",
            "        print(f\"✅ Reranker initialized: {CROSS_ENCODER_MODEL}\")\n",
            "    except Exception as e:\n",
            "        print(f\"⚠️ Failed to load reranker: {e}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- STEP 2: Preprocessing & Keywords ---\n",
            "print(\"📝 Preprocessing posts...\")\n",
            "post_contents = [p.get('content', '')[:500] for p in posts]\n",
            "\n",
            "if USE_KEYWORDS:\n",
            "    from src.core.extraction.keyword_extractor import KeywordExtractor\n",
            "    print(\"🔑 Extracting high-signal keywords...\")\n",
            "    kw_extractor = KeywordExtractor()\n",
            "    post_contents_enriched = kw_extractor.batch_extract(post_contents)\n",
            "else:\n",
            "    post_contents_enriched = post_contents\n",
            "\n",
            "# Summarization Stage\n",
            "print(\"Run Summarization Stage...\")\n",
            "post_contents_enriched = run_summarization_stage(post_contents_enriched, USE_LLM, summarize_all=False)"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- STEP 3: Generate Embeddings ---\n",
            "print(f\"🚀 Generating Embeddings ({EMBEDDING_METHOD})...\")\n",
            "post_embeddings = get_embeddings(\n",
            "    post_contents_enriched, \n",
            "    method=EMBEDDING_METHOD, \n",
            "    model_name=MODEL_NAME,\n",
            "    existing_model=embedder,\n",
            "    device=embedding_device,\n",
            "    cache_dir=\"embeddings_cache\" if USE_CACHE else None\n",
            ")\n",
            "print(f\"✅ Embeddings Shape: {post_embeddings.shape}\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- STEP 4: SAHC Clustering ---\n",
            "print(\"🧩 Running SAHC Clustering...\")\n",
            "cluster_labels = run_sahc_clustering(posts, post_embeddings, min_cluster_size=5)\n",
            "unique_labels = sorted([l for l in set(cluster_labels) if l != -1])\n",
            "print(f\"✅ Found {len(unique_labels)} clusters.\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# --- STEP 5: Matching Clusters to Trends ---\n",
            "print(\"🔗 Matching Clusters to Trends...\")\n",
            "\n",
            "# 5a. Embed Trends\n",
            "trend_keys = list(trends.keys())\n",
            "trend_queries = [\" \".join(trends[t]['keywords']) for t in trend_keys]\n",
            "if trend_queries:\n",
            "    trend_embeddings = get_embeddings(\n",
            "        trend_queries, \n",
            "        method=EMBEDDING_METHOD, \n",
            "        model_name=MODEL_NAME,\n",
            "        existing_model=embedder,\n",
            "        device=embedding_device,\n",
            "        cache_dir=\"embeddings_cache\" if USE_CACHE else None\n",
            "    )\n",
            "else:\n",
            "    trend_embeddings = []\n",
            "\n",
            "# 5b. Label Clusters\n",
            "anchors = extract_dynamic_anchors(posts, trends)\n",
            "cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=LABELING_METHOD, anchors=anchors)\n",
            "\n",
            "matches_hybrid = []\n",
            "\n",
            "print(\"😊 Analyzing sentiment (batch)...\")\n",
            "sentiments = batch_analyze_sentiment(post_contents)\n",
            "\n",
            "for label in unique_labels:\n",
            "    indices = [i for i, l in enumerate(cluster_labels) if l == label]\n",
            "    cluster_posts = [posts[i] for i in indices]\n",
            "    cluster_query = cluster_names.get(label, f\"Cluster {label}\")\n",
            "    \n",
            "    assigned_trend, topic_type, best_match_score = calculate_match_scores(\n",
            "        cluster_query, label, trend_embeddings, trend_keys, trend_queries, \n",
            "        embedder, reranker, RERANK, THRESHOLD\n",
            "    )\n",
            "    \n",
            "    # Calculate Scores (Simplified for notebook view)\n",
            "    trend_data = trends.get(assigned_trend, {'volume': 0})\n",
            "    t_time_str = trend_data.get('time')\n",
            "    t_time = parser.parse(t_time_str) if t_time_str else None\n",
            "    \n",
            "    unified_score, _ = calculate_unified_score(trend_data, cluster_posts, trend_time=t_time)\n",
            "    \n",
            "    for i, p in enumerate(cluster_posts):\n",
            "         # We need to map cluster_post index back to original index for sentiment, \n",
            "         # or just lookup sentiment for this post content/index if possible.\n",
            "         # Simply: indices[i] is the index in original 'posts' and 'sentiments'\n",
            "         original_idx = indices[i]\n",
            "         \n",
            "         matches_hybrid.append({\n",
            "            \"source\": p.get('source'), \"time\": p.get('time'), \"post_content\": p.get('content'),\n",
            "            \"trend\": assigned_trend, \"score\": float(best_match_score), \n",
            "            \"trend_score\": unified_score,\n",
            "            \"is_matched\": (topic_type == \"Trending\"),\n",
            "            \"final_topic\": assigned_trend,\n",
            "            \"cluster_id\": int(label),\n",
            "            \"topic_type\": topic_type,\n",
            "            \"category\": \"Unclassified\", # Skipping taxonomy for speed in playground\n",
            "            \"sentiment\": sentiments[original_idx]\n",
            "        })\n",
            "\n",
            "# Add unassigned posts?\n",
            "# For playground, usually we focus on what matched or clustered. \n",
            "# But original find_matches_hybrid saves unassigned as 'Noise' if save_all=True.\n",
            "unassigned_indices = [i for i, l in enumerate(cluster_labels) if l == -1]\n",
            "for idx in unassigned_indices:\n",
            "    matches_hybrid.append({\n",
            "        \"source\": posts[idx].get('source'), \"time\": posts[idx].get('time'), \"post_content\": posts[idx].get('content'),\n",
            "        \"trend\": \"Unassigned\", \"score\": 0.0, \"trend_score\": 0,\n",
            "        \"is_matched\": False, \"final_topic\": \"Unassigned\", \"topic_type\": \"Noise\",\n",
            "        \"category\": \"Noise\", \"sentiment\": sentiments[idx]\n",
            "    })\n",
            "\n",
            "print(f\"🎉 Final Matches Generated: {len(matches_hybrid)}\")"
        ]
    }
]

def main():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells_list = []
    replaced = False
    
    for cell in nb['cells']:
        # Identify the target cell by source content
        source_code = "".join(cell.get('source', []))
        if "matches_hybrid = find_matches_hybrid" in source_code:
            print("✅ Found target cell. Replacing...")
            new_cells_list.extend(NEW_CELLS)
            replaced = True
        else:
            new_cells_list.append(cell)
            
    if replaced:
        nb['cells'] = new_cells_list
        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("✅ Notebook updated successfully.")
    else:
        print("⚠️ Target cell not found in notebook.")

if __name__ == "__main__":
    main()
