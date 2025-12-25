import json
import re

notebook_path = 'notebooks/analysis-playground-v1.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

print(f"Loaded notebook with {len(nb['cells'])} cells.")

for cell in nb['cells']:
    if cell['cell_type'] != 'code':
        continue
    
    source = "".join(cell['source'])
    
    # 1. Add original_index to matches_hybrid
    if "matches_hybrid.append({" in source and "cluster_id" in source:
        print("Injecting original_index into matches_hybrid.")
        # Find where the dict is defined and inject
        new_source = []
        for line in cell['source']:
            if '"cluster_id": int(label),' in line:
                new_source.append(line)
                new_source.append('            "original_index": original_idx,\n')
            else:
                new_source.append(line)
        cell['source'] = new_source

    # 2. Optimize Step 4.5 (t-SNE) - Ensure it uses global post_embeddings
    # (It mostly does, but checking for re-calculation)
    
    # 3. Optimize Cross-Source Visualization (df_top10)
    if "top_embs = get_embeddings(top_texts" in source:
        print("Optimizing Cross-Source (top_embs) to use slicing.")
        cell['source'] = [
            "# [OPTIMIZED] Using semantic slicing instead of re-embedding\n",
            "if len(df_top10) < 5:\n",
            "    print('Not enough data for cross-source t-SNE.')\n",
            "else:\n",
            "    top_texts = df_top10['post_content'].tolist()\n",
            "    # Slice the global post_embeddings using the original indices\n",
            "    top_indices = df_top10['original_index'].tolist()\n",
            "    top_embs = post_embeddings[top_indices]\n",
            "\n",
            "    tsne_x = TSNE(n_components=2, perplexity=min(30, len(top_texts)-1), random_state=42)\n",
            "    coords_x = tsne_x.fit_transform(top_embs)\n",
            "    \n",
            "    df_vis_x = pd.DataFrame({\n",
            "        'x': coords_x[:, 0],\n",
            "        'y': coords_x[:, 1],\n",
            "        'Source': df_top10['source_type'].tolist(),\n",
            "        'Trend': df_top10['final_topic'].tolist(),\n",
            "        'Snippet': [t[:80] + '...' for t in top_texts]\n",
            "    })\n",
            "    \n",
            "    fig = px.scatter(df_vis_x, x='x', y='y', color='Source', symbol='Trend', \n",
            "                     hover_data=['Snippet', 'Trend'],\n",
            "                     title='Cross-Source Clusters: News vs Facebook Overlap')\n",
            "    fig.update_traces(marker=dict(size=10, opacity=0.8))\n",
            "    fig.show()\n"
        ]

    # 4. Optimize Integrated View (Shared space)
    if "joint_embs = get_embeddings(news_data" in source:
        print("Optimizing Integrated View (joint_embs) to use mapping.")
        cell['source'] = [
            "# [OPTIMIZED] Mapping to global embeddings\n",
            "news_indices = news_data['original_index'].tolist()\n",
            "fb_indices = fb_data['original_index'].tolist()\n",
            "joint_indices = news_indices + fb_indices\n",
            "joint_embs = post_embeddings[joint_indices]\n",
            "\n",
            "tsne_joint = TSNE(n_components=2, perplexity=min(30, len(joint_embs)-1), random_state=42)\n",
            "coords_joint = tsne_joint.fit_transform(joint_embs)\n",
            "\n",
            "df_joint_vis = pd.DataFrame({\n",
            "    'x': coords_joint[:, 0], 'y': coords_joint[:, 1],\n",
            "    'Trend': news_data['final_topic'].tolist() + fb_data['final_topic'].tolist(),\n",
            "    'Source': ['News article'] * len(news_data) + ['Facebook post'] * len(fb_data)\n",
            "})\n",
            "\n",
            "fig2 = px.scatter(df_joint_vis, x='x', y='y', color='Trend', symbol='Source',\n",
            "                  title='[Step 2] Integrated Clusters (Social Posts attached to News Seeds)',\n",
            "                  hover_data=['Trend', 'Source'])\n",
            "fig2.update_traces(marker=dict(size=10, opacity=0.7))\n",
            "print('✅ Successfully reused global embeddings via indexing.')\n",
            "fig2.show()\n"
        ]

    # 5. Optimize final t-SNE (embeddings = get_embeddings(texts, ...))
    if "embeddings = get_embeddings(texts" in source and "plot_df" in source:
        print("Optimizing final t-SNE cell.")
        cell['source'] = [
            "# [OPTIMIZED] Using global embeddings for final visualization\n",
            "print(f'Visualizing {len(plot_df)} clustered posts...')\n",
            "texts = plot_df['post_content'].tolist()\n",
            "labels = plot_df['final_topic'].tolist()\n",
            "types = plot_df['topic_type'].tolist()\n",
            "scores = plot_df['score'].tolist()\n",
            "\n",
            "plot_indices = plot_df['original_index'].tolist()\n",
            "embeddings = post_embeddings[plot_indices]\n",
            "\n",
            "print('Running t-SNE...')\n",
            "tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))\n",
            "coords = tsne.fit_transform(embeddings)\n",
            "\n",
            "df_vis = pd.DataFrame({\n",
            "    'x': coords[:, 0],\n",
            "    'y': coords[:, 1],\n",
            "    'Topic': labels,\n",
            "    'Type': types,\n",
            "    'Score': np.round(scores, 2),\n",
            "    'Snippet': [t[:100] + '...' for t in texts]\n",
            "})\n",
            "fig = px.scatter(df_vis, x='x', y='y', color='Topic', \n",
            "                 hover_data=['Snippet', 'Score', 'Type'],\n",
            "                 title='Final Clustered Visualization')\n",
            "fig.show()\n"
        ]

    # 6. Deprecate experimental encode calls
    if "emb_raw = model.encode(text_raw)" in source:
        print("Commenting out experimental encoding test.")
        cell['source'] = [f"# {line}" for line in cell['source']]

# Save back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, ensure_ascii=False, indent=2)

print("Double optimization complete.")
