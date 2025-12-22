import json
import os

NB_PATH = "notebooks/analysis-playground.ipynb"

VIZ_CELL = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# --- STEP 4.5: VISUALIZE CLUSTERS ---\n",
        "print(\"🎨 Visualizing Clusters with t-SNE (this may take a moment)...\")\n",
        "import pandas as pd\n",
        "import plotly.express as px\n",
        "from sklearn.manifold import TSNE\n",
        "\n",
        "# 1. Reduce Dimensions\n",
        "# Sample if too large to save time, but for playground usually <10k is fine\n",
        "n_samples = len(post_embeddings)\n",
        "perplexity = min(30, n_samples - 1) if n_samples > 1 else 1\n",
        "\n",
        "tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')\n",
        "projections = tsne.fit_transform(post_embeddings)\n",
        "\n",
        "# 2. Prepare Data for Plotly\n",
        "viz_df = pd.DataFrame({\n",
        "    'x': projections[:, 0],\n",
        "    'y': projections[:, 1],\n",
        "    'cluster': [str(l) if l != -1 else 'Noise' for l in cluster_labels],\n",
        "    'content': [p.get('content', '')[:100] + '...' for p in posts],\n",
        "    'source': [p.get('source', 'Unknown') for p in posts]\n",
        "})\n",
        "\n",
        "# Sort so 'Noise' is drawn first (background) or handling colors\n",
        "viz_df = viz_df.sort_values('cluster')\n",
        "\n",
        "# 3. Plot\n",
        "fig = px.scatter(\n",
        "    viz_df, x='x', y='y', color='cluster', \n",
        "    hover_data=['content', 'source'],\n",
        "    title='Cluster Visualization (t-SNE)',\n",
        "    template='plotly_dark',\n",
        "    color_discrete_sequence=px.colors.qualitative.Dark24\n",
        ")\n",
        "fig.update_traces(marker=dict(size=5, opacity=0.8))\n",
        "fig.show()"
    ]
}

def main():
    print(f"Reading {NB_PATH}...")
    with open(NB_PATH, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    new_cells = []
    inserted = False
    
    for cell in nb['cells']:
        new_cells.append(cell)
        src = "".join(cell.get('source', []))
        # Find Step 4 cell
        if "# --- STEP 4: SAHC Clustering ---" in src and not inserted:
            print("✅ Found Step 4. Inserting Viz Step 4.5...")
            new_cells.append(VIZ_CELL)
            inserted = True
            
    if inserted:
        nb['cells'] = new_cells
        with open(NB_PATH, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print("✅ Notebook updated with Cluster Visualization.")
    else:
        print("⚠️ Could not find Step 4 to insert visualization.")

if __name__ == "__main__":
    main()
