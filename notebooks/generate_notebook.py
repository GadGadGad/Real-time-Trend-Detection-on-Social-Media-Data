
import json
import os

notebook = {
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# 🧪 Project Trend Hunter: Analysis Playground\n",
    "\n",
    "Welcome to the interactive test bench! Here you can run the entire trend detection pipeline step-by-step, toggle different methods, and visualize the results immediately.\n",
    "\n",
    "### 🎯 Objectives:\n",
    "1.  **Compare Methods**: Semantic (Google Trends) vs. Hybrid (Cluster-First).\n",
    "2.  **Verify Reranking**: See the difference Cross-Encoder makes.\n",
    "3.  **Inspect Data**: View raw posts, clusters, and sentiment scores.\n",
    "4.  **Visualize**: Run t-SNE to see the clusters in 2D space.\n",
    "\n",
    "---"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Setup & Imports\n",
    "import sys\n",
    "import os\n",
    "import glob\n",
    "import pandas as pd\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import numpy as np\n",
    "from rich.console import Console\n",
    "from sklearn.manifold import TSNE\n",
    "\n",
    "# Ensure project root is in path\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "from crawlers.analyze_trends import find_matches, find_matches_hybrid, load_social_data, load_news_data, load_google_trends\n",
    "from crawlers.alias_normalizer import build_alias_dictionary\n",
    "from crawlers.vectorizers import get_embeddings\n",
    "\n",
    "console = Console()\n",
    "pd.set_option('display.max_colwidth', 100)\n",
    "%matplotlib inline"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ⚙️ Configuration\n",
    "Adjust these parameters to control the experiment."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "LIMIT_POSTS = 500  # Set to None for full run (~4600 posts), 500 for testing\n",
    "RERANK = True      # Enable Cross-Encoder Reranking (Slower but precise)\n",
    "USE_PHOBERT = True # Use PhoBERT for sentiment\n",
    "THRESHOLD = 0.5    # Similarity threshold\n",
    "MIN_CLUSTER_SIZE = 3 # HDBSCAN min cluster size (Lower = more micro-clusters, good for small data)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📂 1. Load Data"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Load Trends\n",
    "trend_files = glob.glob(\"../crawlers/trendings/*.csv\")\n",
    "trends = load_google_trends(trend_files)\n",
    "print(f\"Loaded {len(trends)} trends.\")\n",
    "\n",
    "# Load Social & News\n",
    "fb_files = glob.glob(\"../crawlers/facebook/*.json\")\n",
    "news_files = glob.glob(\"../crawlers/news/**/*.csv\", recursive=True)\n",
    "posts = load_social_data(fb_files) + load_news_data(news_files)\n",
    "\n",
    "if LIMIT_POSTS:\n",
    "    # Shuffle briefly before limiting to get mix? Or just take first.\n",
    "    posts = posts[:LIMIT_POSTS]\n",
    "    \n",
    "# Helper: Extract contents\n",
    "post_contents = [p.get('content', '') for p in posts]\n",
    "print(f\"Loaded {len(posts)} posts for analysis.\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🔬 2. Run Semantic Analysis (Baseline)\n",
    "Standard Bi-Encoder matching (fast, fuzzy)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(\"Running Semantic Matching...\")\n",
    "matches_semantic = find_matches(\n",
    "    posts, trends, \n",
    "    threshold=THRESHOLD, \n",
    "    model_name=\"paraphrase-multilingual-mpnet-base-v2\",\n",
    "    save_all=True  # Include unmatched\n",
    ")\n",
    "df_sem = pd.DataFrame(matches_semantic)\n",
    "print(\"Semantic Match Count:\", len(df_sem[df_sem['is_matched'] == True]))\n",
    "df_sem.head(3)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🚀 3. Run Hybrid Analysis (Cluster-First)\n",
    "This uses HDBSCAN + Cross-Encoder (if enabled).\n",
    "Note: This automatically filters noise and finds 'Discovery' topics."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "print(f\"Running Hybrid Analysis (Rerank={RERANK}, MinCluster={MIN_CLUSTER_SIZE})...\")\n",
    "matches_hybrid = find_matches_hybrid(\n",
    "    posts, trends, \n",
    "    threshold=THRESHOLD, \n",
    "    model_name=\"paraphrase-multilingual-mpnet-base-v2\",\n",
    "    rerank=RERANK,\n",
    "    min_cluster_size=MIN_CLUSTER_SIZE\n",
    ")\n",
    "df_hyb = pd.DataFrame(matches_hybrid)\n",
    "print(\"Hybrid Topics Found:\", df_hyb['final_topic'].nunique())\n",
    "df_hyb[['final_topic', 'topic_type', 'score', 'post_content']].head(5)"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 4. Comparison Stats\n",
    "Let's see the metrics side-by-side."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Comparison Data\n",
    "stats = {\n",
    "    'Method': ['Semantic', 'Hybrid'],\n",
    "    'Total Matched/Clustered': [\n",
    "        len(df_sem[df_sem['is_matched'] == True]),\n",
    "        len(df_hyb[df_hyb['final_topic'] != 'Unassigned'])\n",
    "    ],\n",
    "    'Unique Topics': [\n",
    "        df_sem[df_sem['is_matched'] == True]['trend'].nunique(),\n",
    "        df_hyb[df_hyb['final_topic'] != 'Unassigned']['final_topic'].nunique()\n",
    "    ]\n",
    "}\n",
    "df_stats = pd.DataFrame(stats)\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(12, 5))\n",
    "sns.barplot(data=df_stats, x='Method', y='Total Matched/Clustered', ax=ax[0], palette='viridis')\n",
    "ax[0].set_title(\"Coverage (Total Matched Posts)\")\n",
    "\n",
    "sns.barplot(data=df_stats, x='Method', y='Unique Topics', ax=ax[1], palette='magma')\n",
    "ax[1].set_title(\"Diversity (Unique Topics)\")\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🎨 5. t-SNE Visualization\n",
    "Let's visualize the clusters found by the **Hybrid Method** in 2D space.\n",
    "Points are colored by their assigned topic (Top 10 largest topics)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Filter data (remove 'Unassigned' or 'Noise' for clearer plot)\n",
    "plot_df = df_hyb[df_hyb['topic_type'] != 'Noise'].copy()\n",
    "\n",
    "if len(plot_df) < 5:\n",
    "    print(\"Not enough data points for t-SNE.\")\n",
    "else:\n",
    "    print(f\"Visualizing {len(plot_df)} clustered posts...\")\n",
    "    texts = plot_df['processed_content'].tolist()\n",
    "    labels = plot_df['final_topic'].tolist()\n",
    "    \n",
    "    # 2. Get Embeddings (using sentence-transformer via helper)\n",
    "    print(\"Generating embeddings... (This might take a moment)\")\n",
    "    embeddings = get_embeddings(texts, method=\"sentence-transformer\", \n",
    "                                model_name=\"paraphrase-multilingual-mpnet-base-v2\")\n",
    "    \n",
    "    # 3. Running t-SNE\n",
    "    print(\"Running t-SNE...\")\n",
    "    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))\n",
    "    coords = tsne.fit_transform(embeddings)\n",
    "    \n",
    "    # 4. Plotting\n",
    "    # Identification of Top 10 topics\n",
    "    top_topics = plot_df['final_topic'].value_counts().head(10).index.tolist()\n",
    "    \n",
    "    # Map labels to colors (Top 10 get color, others gray)\n",
    "    color_map = {t: i for i, t in enumerate(top_topics)}\n",
    "    colors = [color_map[l] if l in color_map else -1 for l in labels]\n",
    "    \n",
    "    plt.figure(figsize=(12, 8))\n",
    "    scatter = plt.scatter(coords[:, 0], coords[:, 1], c=colors, cmap='tab10', alpha=0.7, s=50)\n",
    "    \n",
    "    # Legend/Annotation\n",
    "    for i, topic in enumerate(top_topics):\n",
    "        # Find centroid\n",
    "        topic_indices = [idx for idx, t in enumerate(labels) if t == topic]\n",
    "        if topic_indices:\n",
    "            centroid = coords[topic_indices].mean(axis=0)\n",
    "            plt.annotate(topic, (centroid[0], centroid[1]), \n",
    "                         bbox=dict(boxstyle=\"round,pad=0.3\", fc=\"white\", ec=\"black\", alpha=0.8),\n",
    "                         fontsize=9)\n",
    "            \n",
    "    plt.title(\"t-SNE Visualization of Hybrid Clusters (Top 10 Topics)\")\n",
    "    plt.xlabel(\"t-SNE Dimension 1\")\n",
    "    plt.ylabel(\"t-SNE Dimension 2\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🌟 6. Discovery Viewer\n",
    "Top 'Discovery' topics (New trends not in Google Trends)."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "discoveries = df_hyb[df_hyb['topic_type'] == 'Discovery']\n",
    "top_discoveries = discoveries['final_topic'].value_counts().head(10)\n",
    "\n",
    "print(\"Top 10 New Discoveries:\")\n",
    "print(top_discoveries)\n",
    "\n",
    "# Show samples\n",
    "if not top_discoveries.empty:\n",
    "    top_topic = top_discoveries.index[0]\n",
    "    print(f\"\\nSample posts for top discovery '{top_topic}':\")\n",
    "    print(discoveries[discoveries['final_topic'] == top_topic]['post_content'].head(3).values)"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}

with open("notebooks/Analysis_Playground.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=1)
