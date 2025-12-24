import json

nb_path = "notebooks/analysis-playground.ipynb"

with open(nb_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The new cell content
new_cell_source = [
    "# --- STEP 5.5: LLM REFINEMENT ---\n",
    "if LLM_PROVIDER != \"none\":\n",
    "    from src.core.llm.llm_refiner import LLMRefiner\n",
    "    print(f\"🚀 Refining Clusters with {LLM_PROVIDER}...\")\n",
    "    \n",
    "    # Initialize\n",
    "    llm = LLMRefiner(provider=LLM_PROVIDER, api_key=GEMINI_API_KEY, model_path=LLM_MODEL_PATH)\n",
    "    \n",
    "    # Prepare clusters for refinement\n",
    "    # Group by cluster_id\n",
    "    cluster_groups = {}\n",
    "    for m in matches_hybrid:\n",
    "        if m['topic_type'] == 'Noise': continue\n",
    "        cid = m.get('cluster_id')\n",
    "        if cid is None: continue\n",
    "        if cid not in cluster_groups:\n",
    "            cluster_groups[cid] = {\n",
    "                'label': cid,\n",
    "                'name': m['trend'], # Current tentative name\n",
    "                'sample_posts': [],\n",
    "                'topic_type': m['topic_type'],\n",
    "                'keywords': trends.get(m['trend'], {}).get('keywords', [])\n",
    "            }\n",
    "        # Add post if not duplicate\n",
    "        if len(cluster_groups[cid]['sample_posts']) < 5:\n",
    "            cluster_groups[cid]['sample_posts'].append({'content': m['post_content'], 'time': m['time']})\n",
    "\n",
    "    to_refine = list(cluster_groups.values())\n",
    "    \n",
    "    # Run Batch Refinement\n",
    "    if to_refine:\n",
    "        print(f\"   🤖 Batch Refining {len(to_refine)} clusters...\")\n",
    "        # Use the pipeline's logic for instruction if needed, or default\n",
    "        refined_results = llm.refine_batch(to_refine)\n",
    "        \n",
    "        # Update matches_hybrid\n",
    "        updated_count = 0\n",
    "        noise_count = 0\n",
    "        \n",
    "        # Create a map for fast lookup\n",
    "        refine_map = {}\n",
    "        for cid, res in refined_results.items():\n",
    "            refine_map[int(cid)] = res\n",
    "            \n",
    "        # Apply updates\n",
    "        new_matches = []\n",
    "        for m in matches_hybrid:\n",
    "            cid = m.get('cluster_id')\n",
    "            if cid is not None and int(cid) in refine_map:\n",
    "                res = refine_map[int(cid)]\n",
    "                \n",
    "                # Filter Noise logic (Sync with pipeline)\n",
    "                is_routine_c = (res['category'] == 'C' and m['trend_score'] < 90)\n",
    "                event_type = res.get('event_type', 'Specific')\n",
    "                \n",
    "                if event_type == 'Generic' or is_routine_c:\n",
    "                   if m['trend_score'] < 80 or is_routine_c:\n",
    "                       # Mark as Noise and SKIP adding to new list (Filter out)\n",
    "                       noise_count += 1\n",
    "                       continue\n",
    "                \n",
    "                # Update valid match\n",
    "                m['final_topic'] = res['refined_title']\n",
    "                m['category'] = res['category']\n",
    "                m['llm_reasoning'] = res['reasoning']\n",
    "                m['topic_type'] = 'Trending' if m['topic_type'] == 'Discovery' else m['topic_type']\n",
    "                updated_count += 1\n",
    "            else:\n",
    "                # Keep original if no refinement or was already Noise\n",
    "                pass \n",
    "            \n",
    "            new_matches.append(m)\n",
    "            \n",
    "        matches_hybrid = new_matches\n",
    "        print(f\"   ✅ Refined {updated_count} posts. Filtered out {noise_count} noise posts.\")\n"
]

# Find Step 5 index
insert_idx = -1
for i, cell in enumerate(nb['cells']):
    src = "".join(cell['source'])
    if "# --- STEP 5: Matching Clusters to Trends ---" in src:
        insert_idx = i + 1
        break

if insert_idx != -1:
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": new_cell_source
    }
    nb['cells'].insert(insert_idx, new_cell)
    
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=4, ensure_ascii=False)
    print(f"Successfully added Step 5.5 at cell index {insert_idx}")
else:
    print("Could not find Step 5 anchor.")
