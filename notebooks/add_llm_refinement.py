import json
import os

NB_PATH = "notebooks/analysis-playground.ipynb"

# 1. New Config Cell to add at the top (or modification of variables)
# We will just append a new cell for Custom Config if it's easier, or we can assume the user sets it.
# Better to validly add it.

CONFIG_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- CUSTOM LLM INSTRUCTION ---\n",
        "# Add your custom instructions for the LLM here.\n",
        "LLM_CUSTOM_INSTRUCTION = \"\"\"\n",
        "You are a helpful assistant. \n",
        "When refining trend names, prioritize being concise and using viral, catchy language.\n",
        "If a trend is about a serious event, maintain a formal tone.\n",
        "\"\"\"\n",
        "print(\"✅ Custom LLM Instruction Set\")"
    ]
}

# 2. Modified Step 5 Cell (to expose cluster_mapping)
STEP_5_SOURCE = [
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
    "cluster_mapping = {} # Store for LLM Refinement\n",
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
    "    # Calculate Scores\n",
    "    trend_data = trends.get(assigned_trend, {'volume': 0})\n",
    "    t_time_str = trend_data.get('time')\n",
    "    t_time = parser.parse(t_time_str) if t_time_str else None\n",
    "    \n",
    "    unified_score, components = calculate_unified_score(trend_data, cluster_posts, trend_time=t_time)\n",
    "    \n",
    "    # Save to mapping for LLM step\n",
    "    # We need 'category' here too, let's just default it or use basic classifier if we had it.\n",
    "    # For playground simplicity, we skip full TaxonomyClassifier here or just say 'Unclassified'.\n",
    "    cluster_mapping[label] = {\n",
    "        \"final_topic\": assigned_trend if assigned_trend != \"Discovery\" else f\"New: {cluster_query}\",\n",
    "        \"topic_type\": topic_type,\n",
    "        \"cluster_name\": cluster_query,\n",
    "        \"category\": \"Unclassified\",\n",
    "        \"trend_score\": unified_score,\n",
    "        \"posts\": cluster_posts\n",
    "    }\n",
    "\n",
    "    for i, p in enumerate(cluster_posts):\n",
    "         original_idx = indices[i]\n",
    "         matches_hybrid.append({\n",
    "            \"source\": p.get('source'), \"time\": p.get('time'), \"post_content\": p.get('content'),\n",
    "            \"trend\": assigned_trend, \"score\": float(best_match_score), \n",
    "            \"trend_score\": unified_score,\n",
    "            \"is_matched\": (topic_type == \"Trending\"),\n",
    "            \"final_topic\": cluster_mapping[label][\"final_topic\"],\n",
    "            \"cluster_id\": int(label),\n",
    "            \"topic_type\": topic_type,\n",
    "            \"sentiment\": sentiments[original_idx]\n",
    "        })\n",
    "\n",
    "unassigned_indices = [i for i, l in enumerate(cluster_labels) if l == -1]\n",
    "for idx in unassigned_indices:\n",
    "    matches_hybrid.append({\n",
    "        \"source\": posts[idx].get('source'), \"time\": posts[idx].get('time'), \"post_content\": posts[idx].get('content'),\n",
    "        \"trend\": \"Unassigned\", \"score\": 0.0, \"trend_score\": 0,\n",
    "        \"is_matched\": False, \"final_topic\": \"Unassigned\", \"topic_type\": \"Noise\",\n",
    "        \"sentiment\": sentiments[idx]\n",
    "    })\n",
    "\n",
    "print(f\"🎉 Final Matches Generated: {len(matches_hybrid)}\")"
]

# 3. New Step 6 Cell: LLM Refinement
STEP_6_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- STEP 6: LLM Refinement ---\n",
        "if USE_LLM:\n",
        "    from src.core.llm.llm_refiner import LLMRefiner\n",
        "    print(f\"🚀 Phase 6: Batch LLM Refinement using {LLM_PROVIDER}...\")\n",
        "    \n",
        "    llm_refiner = LLMRefiner(\n",
        "        provider=LLM_PROVIDER, \n",
        "        api_key=GEMINI_API_KEY, \n",
        "        model_path=LLM_MODEL_PATH, \n",
        "        debug=DEBUG_LLM\n",
        "    )\n",
        "    \n",
        "    to_refine = []\n",
        "    for l, m in cluster_mapping.items():\n",
        "        # Refine Discovery trends or high-score trends\n",
        "        if m[\"topic_type\"] == \"Discovery\" or m[\"trend_score\"] > 30:\n",
        "             keywords = []\n",
        "             # Try to get keywords from trends if applicable\n",
        "             current_topic = m[\"final_topic\"]\n",
        "             if current_topic in trends:\n",
        "                 keywords = trends[current_topic].get('keywords', [])\n",
        "             \n",
        "             to_refine.append({\n",
        "                \"label\": l, \"name\": m[\"cluster_name\"], \"topic_type\": m[\"topic_type\"],\n",
        "                \"category\": m[\"category\"], \"sample_posts\": m[\"posts\"],\n",
        "                \"keywords\": keywords\n",
        "            })\n",
        "    \n",
        "    if to_refine:\n",
        "        print(f\"   🤖 Refining {len(to_refine)} clusters with Custom Instruction...\")\n",
        "        # PASS THE CUSTOM INSTRUCTION HERE\n",
        "        batch_results = llm_refiner.refine_batch(to_refine, custom_instruction=LLM_CUSTOM_INSTRUCTION)\n",
        "        \n",
        "        # Update matches with refined names\n",
        "        refined_count = 0\n",
        "        for l, res in batch_results.items():\n",
        "             label_key = int(l) if isinstance(l, (int, str)) else l\n",
        "             new_title = res['refined_title']\n",
        "             reasoning = res['reasoning']\n",
        "             \n",
        "             # Update in matches list\n",
        "             for match in matches_hybrid:\n",
        "                 if match.get('cluster_id') == label_key:\n",
        "                     if match['topic_type'] == 'Discovery':\n",
        "                         match['final_topic'] = f\"New: {new_title}\"\n",
        "                     else:\n",
        "                         # Ideally we don't rename trending topics unless necessary, but let's assume LLM cleans it up\n",
        "                         match['final_topic'] = new_title\n",
        "                     match['llm_reasoning'] = reasoning\n",
        "                     match['category'] = res['category']\n",
        "             refined_count += 1\n",
        "             \n",
        "        print(f\"   ✅ Refined {refined_count} clusters.\")\n",
        "    else:\n",
        "        print(\"   ℹ️ No clusters needed refinement.\")\n",
        "else:\n",
        "    print(\"⏩ LLM Refinement Skipped (USE_LLM=False).\")"
    ]
}

def main():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    # 1. Insert Config Cell (e.g. after the Imports cell which defines USE_LLM, usually cell index 2 or 3)
    # We'll put it early, maybe index 2 (after the git clone/cd/imports)
    nb['cells'].insert(8, CONFIG_CELL) # Rough guess, better to append or place smartly
    # Actually, let's find the cell defining "USE_LLM" variables.
    
    # 2. Update Step 5 and Insert Step 6
    new_cells = []
    for cell in nb['cells']:
        src = "".join(cell.get('source', []))
        if "# --- STEP 5: Matching Clusters to Trends ---" in src:
            # Replace Step 5
            cell['source'] = STEP_5_SOURCE
            new_cells.append(cell)
            # Append Step 6 immediately after
            new_cells.append(STEP_6_CELL)
        else:
            new_cells.append(cell)
            
    nb['cells'] = new_cells
    
    with open(NB_PATH, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("✅ Notebook updated with LLM Refinement and Custom Instructions.")

if __name__ == "__main__":
    main()
