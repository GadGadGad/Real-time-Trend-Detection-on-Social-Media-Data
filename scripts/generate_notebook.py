import json
import os

# Define the additions
CLEANING_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 🧹 2. Data Cleaning & Hybrid Search Setup\n",
            "\n",
            "Implementing specialized cleaning for Facebook OCR noise and Hybrid Search (BM25 + Dense) for better retrieval."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [IMPROVEMENT] Data Cleaning Function\n",
            "import re\n",
            "\n",
            "def clean_facebook_content(text):\n",
            "    if not isinstance(text, str): return \"\"\n",
            "    # Remove common OCR / UI artifacts\n",
            "    noise_patterns = [\n",
            "        r\"May be an image of.*?\\n\",\n",
            "        r\"No photo description available.*?\\n\",\n",
            "        r\"\\+?\\d+ others\",\n",
            "        r\"Theanh28.*?\\n\", # Specific page noise\n",
            "        r\"\\d+K likes\",\n",
            "        r\"\\d+ comments\"\n",
            "    ]\n",
            "    cleaned = text\n",
            "    for pattern in noise_patterns:\n",
            "        cleaned = re.sub(pattern, \"\", cleaned, flags=re.IGNORECASE)\n",
            "    return cleaned.strip()\n",
            "\n",
            "# Apply cleaning\n",
            "print(\"🧹 Cleaning Facebook posts...\")\n",
            "count = 0\n",
            "for p in posts:\n",
            "    if p.get('source') == 'Facebook':\n",
            "        p['content'] = clean_facebook_content(p.get('content', ''))\n",
            "        count += 1\n",
            "        \n",
            "print(f\"✅ Cleaned {count} Facebook posts!\")"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# [IMPROVEMENT] Hybrid Search Implementation\n",
            "!pip install -q rank_bm25\n",
            "\n",
            "from rank_bm25 import BM25Okapi\n",
            "from sentence_transformers import util\n",
            "import numpy as np\n",
            "\n",
            "print(\"⚙️ Indexing for Hybrid Search (BM25)...\")\n",
            "# Pre-tokenize for BM25 (simple whitespace tokenization for now)\n",
            "# Using parsed contents from 'posts'\n",
            "post_contents_clean = [p.get('content', '') for p in posts]\n",
            "tokenized_corpus = [doc.split(\" \") for doc in post_contents_clean]\n",
            "bm25 = BM25Okapi(tokenized_corpus)\n",
            "\n",
            "def hybrid_search(query, top_k=5, alpha=0.5):\n",
            "    \"\"\"\n",
            "    Combines Dense (Semantic) and Sparse (BM25) scores.\n",
            "    final_score = alpha * dense_score + (1-alpha) * bm25_score\n",
            "    \"\"\"\n",
            "    # 1. Dense Score\n",
            "    query_emb = embedder.encode(query, convert_to_tensor=True)\n",
            "    corpus_embs = embedder.encode(post_contents_clean, convert_to_tensor=True, show_progress_bar=False)\n",
            "    dense_scores = util.cos_sim(query_emb, corpus_embs)[0].cpu().numpy()\n",
            "    \n",
            "    # 2. Sparse Score (BM25)\n",
            "    tokenized_query = query.split(\" \")\n",
            "    sparse_scores = np.array(bm25.get_scores(tokenized_query))\n",
            "    \n",
            "    # Normalize BM25 scores to 0-1 range approx for combination\n",
            "    if sparse_scores.max() > 0:\n",
            "        sparse_scores = sparse_scores / sparse_scores.max()\n",
            "        \n",
            "    # 3. Combine\n",
            "    final_scores = alpha * dense_scores + (1 - alpha) * sparse_scores\n",
            "    return final_scores\n",
            "\n",
            "print(\"✅ Hybrid Search Ready (BM25 + Dense)\")"
        ]
    }
]

SUMMARY_CELLS = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 📝 Batch Summarization (Optional)\n",
            "\n",
            "Pre-compute summaries for all posts/articles and cache for later use."
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==========================================\n",
            "# BATCH SUMMARIZE FACEBOOK POSTS\n",
            "# ==========================================\n",
            "import glob\n",
            "from scripts.batch_summarize import batch_summarize, merge_summaries_into_posts, load_posts\n",
            "\n",
            "FB_SUMMARY_OUTPUT = '/kaggle/working/fb_summaries.json'\n",
            "SUMMARY_MODEL = 'vit5-base'\n",
            "\n",
            "fb_files = glob.glob('/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json')\n",
            "if fb_files:\n",
            "    fb_summaries = batch_summarize(input_path=fb_files[0], output_path=FB_SUMMARY_OUTPUT, model_name=SUMMARY_MODEL, max_length=200, resume=True)\n",
            "    print(f'Summarized {len(fb_summaries)} FB posts')\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==========================================\n",
            "# BATCH SUMMARIZE NEWS ARTICLES\n",
            "# ==========================================\n",
            "import os\n",
            "from scripts.batch_summarize import batch_summarize\n",
            "\n",
            "NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']\n",
            "NEWS_DATA_DIR = '/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/news'\n",
            "NEWS_SUMMARY_DIR = '/kaggle/working/news_summaries'\n",
            "os.makedirs(NEWS_SUMMARY_DIR, exist_ok=True)\n",
            "\n",
            "for source in NEWS_SOURCES:\n",
            "    input_path = f'{NEWS_DATA_DIR}/{source}/articles.csv'\n",
            "    output_path = f'{NEWS_SUMMARY_DIR}/{source}_summaries.json'\n",
            "    if os.path.exists(input_path):\n",
            "        print(f'Processing: {source}')\n",
            "        batch_summarize(input_path=input_path, output_path=output_path, model_name=SUMMARY_MODEL, max_length=200, resume=True)\n"
        ]
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "# ==========================================\n",
            "# LOAD SUMMARIES INTO DATAFRAMES\n",
            "# ==========================================\n",
            "import pandas as pd\n",
            "from scripts.batch_summarize import load_summaries_for_use\n",
            "\n",
            "for source in NEWS_SOURCES:\n",
            "    csv_path = f'{NEWS_DATA_DIR}/{source}/articles.csv'\n",
            "    summary_path = f'{NEWS_SUMMARY_DIR}/{source}_summaries.json'\n",
            "    if os.path.exists(csv_path) and os.path.exists(summary_path):\n",
            "        df = pd.read_csv(csv_path)\n",
            "        summaries = load_summaries_for_use(summary_path)\n",
            "        df['summary'] = df['url'].apply(lambda u: summaries.get(str(u), ''))\n",
            "        print(f'{source}: {len(df)} articles, {(df[\"summary\"] != \"\").sum()} with summaries')\n"
        ]
    }
]

def generate_notebook():
    notebook_path = 'notebooks/analysis-playground.ipynb'
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    
    # 1. Strip existing "Cleaning", "Hybrid Search", "Batch Summarize" cells to avoid dupes
    # We filter by checking if specific unique strings are in the source
    clean_nb_cells = []
    for cell in nb['cells']:
        source_str = "".join(cell.get('source', []))
        if "clean_facebook_content" in source_str: continue 
        if "Hybrid Search Implementation" in source_str: continue 
        if "BATCH SUMMARIZE FACEBOOK POSTS" in source_str: continue
        if "BATCH SUMMARIZE NEWS ARTICLES" in source_str: continue
        if "LOAD SUMMARIES INTO DATAFRAMES" in source_str: continue
        clean_nb_cells.append(cell)
    
    nb['cells'] = clean_nb_cells
    
    # 2. Find insert position for cleaning (After loading data)
    insert_idx = 0
    for i, cell in enumerate(nb['cells']):
        if "1. Load Data" in "".join(cell.get('source', [])):
            insert_idx = i + 2 # After header and next cell
            break
            
    # 3. Insert Cleaning + Hybrid Search
    for cell in reversed(CLEANING_CELLS):
        nb['cells'].insert(insert_idx, cell)
        
    # 4. Append Summarization at the end
    nb['cells'].extend(SUMMARY_CELLS)
    
    # 5. Save
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    
    print(f"✅ Generated notebook with {len(nb['cells'])} cells.")
    print(f"   - Added Data Cleaning & Hybrid Search at index {insert_idx}")
    print(f"   - Appended Batch Summarization at the end")

if __name__ == "__main__":
    generate_notebook()
