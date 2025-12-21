
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
    "1.  **Inspect Data**: EDA on sources, timing, and content length.\n",
    "2.  **Compare Methods**: Semantic (Google Trends) vs. Hybrid (Cluster-First).\n",
    "3.  **Verify Reranking**: See the difference Cross-Encoder makes.\n",
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
    "# Install wordcloud if missing\n",
    "!pip install wordcloud --quiet"
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
    "import plotly.express as px\n",
    "import numpy as np\n",
    "from rich.console import Console\n",
    "from sklearn.manifold import TSNE\n",
    "from wordcloud import WordCloud\n",
    "from sklearn.feature_extraction.text import TfidfVectorizer\n",
    "\n",
    "# Ensure project root is in path\n",
    "sys.path.append(os.path.abspath('..'))\n",
    "\n",
    "from crawlers.analyze_trends import find_matches, find_matches_hybrid, load_social_data, load_news_data, load_google_trends, refine_trends_preprocessing\n",
    "from crawlers.alias_normalizer import build_alias_dictionary, normalize_with_aliases\n",
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
    "USE_PHOBERT = True # Use PhoBERT for sentiment\n",
    "THRESHOLD = 0.5    # Similarity threshold\n",
    "\n",
    "REFINE_TRENDS = True # [NEW] Phase 6: Use LLM to clean Google Trends before matching\n",
    "NO_DEDUP = False      # [NEW] Phase 4: Skip semantic deduplication if too aggressive\n",
    "\n",
    "# Recommendations for Vietnamese:\n",
    "# Bi-Encoder: 'keepitreal/vietnamese-sbert' or 'dangvantuan/vietnamese-embedding'\n",
    "MODEL_NAME = \"paraphrase-multilingual-mpnet-base-v2\"\n",
    "\n",
    "# Cross-Encoder: 'DiTy/cross-encoder-vietnamese-mobilebert'\n",
    "CROSS_ENCODER_MODEL = \"cross-encoder/ms-marco-MiniLM-L-6-v2\"\n",
    "\n",
    "EMBEDDING_MODEL = 'dangvantuan/vietnamese-embedding' # Or 'keepitreal/vietnamese-sbert'\n",
    "EMBEDDING_METHOD = 'sentence-transformer' # 'tfidf', 'bow', 'sentence-transformer'\n",
    "LABELING_METHOD = 'semantic'              # 'tfidf', 'semantic'\n",
    "RERANK = True                             # Use Cross-Encoder for precision\n",
    "MIN_CLUSTER_SIZE = 5,                     # Min posts to form a trend\n",
    "\n",
    "# LLM Refinement\n",
    "USE_LLM = False                           # Set to True to enable Refinement\n",
    "LLM_PROVIDER = 'gemini'                   # 'gemini' or 'kaggle'\n",
    "GEMINI_API_KEY = \"\"                       # For Gemini\n",
    "LLM_MODEL_PATH = \"google/gemma-2-2b-it\"   # For Kaggle\n",
    "USE_CACHE = True                          # Save/Load embeddings to disk\n",
    "DEBUG_LLM = False                         # Print raw LLM responses on error\n",
    "SUMMARIZE_ALL = False                     # Set True to summarize ALL posts with ViT5 (slow!)\n",
    "\n",
    "# Custom Prompt for Cluster Refinement\n",
    "LLM_CUSTOM_INSTRUCTION = \"\"\"For each cluster ID, provide a professional title, category, and reasoning.\n",
    "Categories:\n",
    "- A: Critical (Accidents, Disasters, Safety)\n",
    "- B: Social (Policy, controversy, public sentiment)\n",
    "- C: Market (Commerce, Tech, Entertainment)\"\"\"\n"
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
    "## 🧹 1.1 Phase 6: Google Trends Refinement (Optional)\n",
    "Clean and merge trends before analysis using instructions defined in Configuration."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "if REFINE_TRENDS:\n",
    "    trends = refine_trends_preprocessing(\n",
    "        trends, \n",
    "        llm_provider=LLM_PROVIDER, \n",
    "        gemini_api_key=GEMINI_API_KEY, \n",
    "        llm_model_path=LLM_MODEL_PATH, \n",
    "        debug_llm=DEBUG_LLM, \n",
    "        source_files=trend_files  # Enables caching\n",
    "    )\n",
    "else:\n",
    "    print(\"Skipping Trend Refinement (using raw trends).\")"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 📊 1.2 General Stats\n",
    "Let's understand our dataset volume."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Convert to DataFrame for EDA\n",
    "df_raw = pd.DataFrame(posts)\n",
    "\n",
    "# 1. Clean Time field\n",
    "df_raw['time'] = pd.to_datetime(df_raw['time'], errors='coerce')\n",
    "\n",
    "# 2. Source Categories\n",
    "df_raw['source_type'] = df_raw['source'].apply(lambda x: 'Facebook' if 'Face:' in x else 'News')\n",
    "df_raw['content_length'] = df_raw['content'].apply(len)\n",
    "\n",
    "fig, ax = plt.subplots(1, 2, figsize=(14, 5))\n",
    "\n",
    "# A. Source Type Distribution\n",
    "sns.countplot(data=df_raw, x='source_type', ax=ax[0], palette='pastel')\n",
    "ax[0].set_title(\"Distribution of Data Types\")\n",
    "\n",
    "# B. Post Counts over Time\n",
    "if df_raw['time'].notnull().any():\n",
    "    df_raw[df_raw['time'].notnull()].set_index('time').resample('D').size().plot(ax=ax[1], color='teal', marker='o')\n",
    "    ax[1].set_title(\"Daily Post Volume\")\n",
    "    ax[1].set_ylabel(\"Number of Posts\")\n",
    "else:\n",
    "    ax[1].text(0.5, 0.5, \"No Valid Time Data\", ha='center')\n",
    "\n",
    "plt.tight_layout()\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## ☁️ 1.2 Deep Dive: Sources and Content\n",
    "Which specific pages are most active? What are they talking about?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# A. Top 20 specific sources\n",
    "def clean_source_name(s):\n",
    "    return s.replace(\"Face: \", \"\")\n",
    "\n",
    "df_raw['clean_source'] = df_raw['source'].apply(clean_source_name)\n",
    "top_sources = df_raw['clean_source'].value_counts().head(20)\n",
    "\n",
    "plt.figure(figsize=(12, 6))\n",
    "sns.barplot(x=top_sources.values, y=top_sources.index, palette='viridis')\n",
    "plt.title(\"Top 20 Active Sources\")\n",
    "plt.xlabel(\"Number of Posts\")\n",
    "plt.show()\n",
    "\n",
    "# B. Word Cloud\n",
    "# Simple stopwords list for Vietnamese (basic)\n",
    "stops = {'và', 'của', 'là', 'có', 'trong', 'đã', 'ngày', 'theo', 'với', 'cho', 'người', 'những', 'tại', 'về', 'các', 'được'}\n",
    "text_corpus = \" \".join(df_raw['content'].dropna().tolist())\n",
    "\n",
    "wc = WordCloud(width=800, height=400, background_color='white', stopwords=stops, max_words=100).generate(text_corpus)\n",
    "\n",
    "plt.figure(figsize=(14, 7))\n",
    "plt.imshow(wc, interpolation='bilinear')\n",
    "plt.axis(\"off\")\n",
    "plt.title(\"Most Common Words (Word Cloud)\")\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "## 🛠️ 1.3 Preprocessing Demo (Before vs After)\n",
    "See how our **Alias Normalization** and **TF-IDF Tokenizer** process the raw text."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# 1. Initialize Alias Dictionary (Custom Layer)\n",
    "build_alias_dictionary(trends)\n",
    "\n",
    "# 2. Initialize TF-IDF (Scikit-Learn Layer)\n",
    "tfidf_demo = TfidfVectorizer(ngram_range=(1, 2), max_features=20)\n",
    "\n",
    "# 3. Pick a sample text (Try to find one with a potential alias)\n",
    "# Let's look for a post mentioning \"số 3\" (common alias for storm)\n",
    "sample_text = \"Cơn bão số 3 đang gây mưa lớn tại Hà Nội.\"\n",
    "candidates = df_raw[df_raw['content'].str.contains(\"số 3\", case=False, na=False)]\n",
    "if not candidates.empty:\n",
    "    sample_text = candidates.iloc[0]['content'][:100] + \"...\"\n",
    "\n",
    "print(\"--- STEP 1: RAW INPUT ---\")\n",
    "print(f\"Original: '{sample_text}'\")\n",
    "\n",
    "print(\"\\n--- STEP 2: OUR ALIAS NORMALIZATION (Augmentation) ---\")\n",
    "normalized_text = normalize_with_aliases(sample_text)\n",
    "print(f\"Processed: '{normalized_text}'\")\n",
    "print(\"(Notice how relevant trend names are PREPENDED to the text)\")\n",
    "\n",
    "print(\"\\n--- STEP 3: TF-IDF TOKENIZATION (Cleaning) ---\")\n",
    "tfidf_demo.fit([normalized_text])\n",
    "tokens = tfidf_demo.get_feature_names_out()\n",
    "print(f\"Final Tokens: {list(tokens)}\")\n",
    "print(\"(Lowercase, Punctuation Removed, Bigrams Created)\")"
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
    "### 🎨 Visualize Semantic Matches\n",
    "How do the posts group when assigned directly to trends?"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "metadata": {},
   "outputs": [],
   "source": [
    "# Filter for matched posts only\n",
    "sem_plot_df = df_sem[df_sem['is_matched'] == True].copy()\n",
    "\n",
    "if len(sem_plot_df) < 5:\n",
    "    print(\"Not enough semantic matches to plot.\")\n",
    "else:\n",
    "    print(f\"Visualizing {len(sem_plot_df)} Semantic Matches...\")\n",
    "    sem_texts = sem_plot_df['post_content'].tolist()\n",
    "    sem_labels = sem_plot_df['trend'].tolist()\n",
    "\n",
    "    # Embeddings (Always use Sentence Transformer for visualization quality)\n",
    "    sem_embeddings = get_embeddings(sem_texts, method=\"sentence-transformer\", \n",
    "                                    model_name=\"paraphrase-multilingual-mpnet-base-v2\")\n",
    "\n",
    "    # t-SNE\n",
    "    tsne_sem = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sem_texts)-1))\n",
    "    coords_sem = tsne_sem.fit_transform(sem_embeddings)\n",
    "\n",
    "    # Create DataFrame for Plotly\n",
    "    df_vis_sem = pd.DataFrame({\n",
    "        'x': coords_sem[:, 0],\n",
    "        'y': coords_sem[:, 1],\n",
    "        'Label': sem_labels,\n",
    "        'Snippet': [t[:100] + '...' for t in sem_texts]\n",
    "    })\n",
    "\n",
    "    # Interactive Plot\n",
    "    fig = px.scatter(df_vis_sem, x='x', y='y', color='Label', \n",
    "                     hover_data=['Snippet'],\n",
    "                     title=\"Interactive t-SNE: Semantic Matches (Baseline)\")\n",
    "    fig.show()"
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
    "print(f\"Running Hybrid Analysis (Embedding={EMBEDDING_METHOD}, Labeling={LABELING_METHOD}, Rerank={RERANK})...\")\n",
    "matches_hybrid = find_matches_hybrid(\n",
    "    posts, trends, \n",
    "    threshold=THRESHOLD, \n",
    "    model_name=MODEL_NAME,\n",
    "    reranker_model_name=CROSS_ENCODER_MODEL,\n",
    "    embedding_method=EMBEDDING_METHOD,\n",
    "    labeling_method=LABELING_METHOD,\n",
    "    rerank=RERANK,\n",
    "    use_llm=USE_LLM,\n",
    "    gemini_api_key=GEMINI_API_KEY,\n",
    "    llm_provider=LLM_PROVIDER,\n",
    "    llm_model_path=LLM_MODEL_PATH,\n",
    "    llm_custom_instruction=LLM_CUSTOM_INSTRUCTION,\n",
    "    use_cache=USE_CACHE,\n",
    "    debug_llm=DEBUG_LLM,\n",
    "    summarize_all=SUMMARIZE_ALL,\n",
    "    no_dedup=NO_DEDUP,\n",
    "    save_all=True\n",
    ")\n",
    "df_hyb = pd.DataFrame(matches_hybrid)\n",
    "print(\"Hybrid Topics Found:\", df_hyb['final_topic'].nunique())\n",
    "\n",
    "    # improved display with new metrics\n",
    "    cols = ['final_topic', 'category', 'topic_type', 'trend_score', 'sentiment', 'llm_reasoning', 'post_content']\n",
    "    # Check if columns exist (graceful fallback)\n",
    "    available_cols = [c for c in cols if c in df_hyb.columns]\n",
    "    df_result = df_hyb[available_cols].copy()\n",
    "\n",
    "    # Sort by Score if available\n",
    "    if 'trend_score' in df_result.columns:\n",
    "        df_result = df_result.sort_values('trend_score', ascending=False)\n",
    "        \n",
    "    df_result.head(10)"
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
    "## 🎨 5. t-SNE Visualization with Plotly (Hybrid)\n",
    "Let's visualize the clusters found by the **Hybrid Method** in 2D space.\n",
    "**Hover over the blue dots** to discover what those small clusters are!"
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
    "    types = plot_df['topic_type'].tolist()\n",
    "    scores = plot_df['score'].tolist()\n",
    "    \n",
    "    # 2. Get Embeddings (Use SAME method as configured)\n",
    "    print(f\"Generating embeddings using {EMBEDDING_METHOD}...\")\n",
    "    embeddings = get_embeddings(texts, method=EMBEDDING_METHOD, \n",
    "                                model_name=\"paraphrase-multilingual-mpnet-base-v2\",\n",
    "                                max_features=2000) # For TF-IDF/BoW speed\n",
    "    \n",
    "    # 3. Running t-SNE\n",
    "    print(\"Running t-SNE...\")\n",
    "    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))\n",
    "    coords = tsne.fit_transform(embeddings)\n",
    "    \n",
    "    # 4. Interactive Plot with Plotly\n",
    "    df_vis = pd.DataFrame({\n",
    "        'x': coords[:, 0],\n",
    "        'y': coords[:, 1],\n",
    "        'Topic': labels,\n",
    "        'Type': types,\n",
    "        'Score': np.round(scores, 2),\n",
    "        'Snippet': [t[:100] + '...' for t in texts]\n",
    "    })\n",
    "    \n",
    "    # Only show Top 20 topics in legend, others grouped as 'Other' to avoid palette exhaustion\n",
    "    top_n_topics = df_vis['Topic'].value_counts().head(20).index.tolist()\n",
    "    df_vis['Legend_Group'] = df_vis['Topic'].apply(lambda x: x if x in top_n_topics else 'Other (Blue Clusters)')\n",
    "    \n",
    "    fig = px.scatter(df_vis, x='x', y='y', \n",
    "                     color='Legend_Group', \n",
    "                     symbol='Type',\n",
    "                     hover_data=['Topic', 'Type', 'Score', 'Snippet'],\n",
    "                     title=f\"Interactive t-SNE: Hybrid Clusters ({EMBEDDING_METHOD})\")\n",
    "    fig.show()"
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
