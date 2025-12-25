import json

notebook_path = 'notebooks/analysis-playground-v1.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = "".join(cell['source'])
    
    # 1. Update Global Config to include USE_CACHE and CACHE_DIR
    if "MODEL_NAME =" in source and "EMBEDDING_METHOD =" in source:
        print("Updating Global Configuration with Caching toggles.")
        if "USE_CACHE =" not in source:
            # Inject after EMBEDDING_METHOD
            new_source = []
            for line in cell['source']:
                new_source.append(line)
                if "EMBEDDING_METHOD =" in line:
                    new_source.append("\n# --- CACHING CONFIG ---\n")
                    new_source.append("USE_CACHE = True\n")
                    new_source.append("CACHE_DIR = 'embeddings_cache'\n")
            cell['source'] = new_source

    # 2. Update Master Preprocessing to use get_embeddings for EVERYTHING
    if "# === MASTER PREPROCESSING (Optimized) ===" in source:
        print("Standardizing Master Preprocessing for full caching.")
        cell['source'] = [
            "# === MASTER PREPROCESSING (Optimized) ===\n",
            "print('✂️ Segmenting & Embedding (Global Source of Truth)...')\n",
            "\n",
            "# 1. Segmentation\n",
            "post_contents_seg = batch_segment_texts(post_contents_enriched)\n",
            "\n",
            "# 2. Trend Processing (Smart Query)\n",
            "trend_keys = list(trends.keys())\n",
            "trend_queries_raw = [create_smart_trend_query(k, trends[k]['keywords']) for k in trend_keys]\n",
            "trend_queries_seg = batch_segment_texts(trend_queries_raw)\n",
            "\n",
            "# 3. Embed (Using cache and trust_remote_code)\n",
            "post_embeddings = get_embeddings(\n",
            "    post_contents_seg, \n",
            "    model_name=MODEL_NAME, \n",
            "    trust_remote_code=True, \n",
            "    cache_dir=CACHE_DIR if USE_CACHE else None\n",
            ")\n",
            "\n",
            "trend_embeddings = get_embeddings(\n",
            "    trend_queries_seg, \n",
            "    model_name=MODEL_NAME, \n",
            "    trust_remote_code=True, \n",
            "    cache_dir=CACHE_DIR if USE_CACHE else None\n",
            ")\n",
            "\n",
            "print(f'✅ Post Embeddings: {post_embeddings.shape}')\n",
            "print(f'✅ Trend Embeddings: {trend_embeddings.shape}')\n"
        ]

# Save back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Notebook caching standardization complete.")
