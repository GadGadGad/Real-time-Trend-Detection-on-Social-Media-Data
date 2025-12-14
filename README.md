# Multi-Source Trend Detection and ANalysis System

Real-time trend detection từ multiple sources: Google Trends, Facebook, News sites.

## 📁 Project Structure

```
├── crawlers/                   # Core analysis modules
│   ├── analyze_trends.py       # Main trend matching pipeline
│   ├── evaluate_trends.py      # Visualization & scoring
│   ├── alias_normalizer.py     # Text normalization with aliases
│   ├── ner_extractor.py        # NER enrichment (optional)
│   ├── trend_scoring.py        # G/F/N score calculator
│   ├── vnexpress_crawler.py    # VNExpress news crawler
│   ├── thanhnien_crawler.py    # Thanh Nien news crawler
│   └── facebook/               # Facebook page crawler
│
├── results/                    # Output files (gitignored)
│   ├── results.json            # Matched trends data
│   ├── trend_analysis.png      # Top trends chart
│   └── trend_tsne.png          # t-SNE visualization
│
├── notebooks/                  # Jupyter notebooks
│   └── kaggle_trend_analysis.ipynb  # Kaggle-ready notebook
│
├── data/                       # Crawled data storage
├── flow.mmd                    # Pipeline flow diagram
├── requirements.txt            # Python dependencies
└── run_crawlers.py             # Crawler orchestration
```

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
playwright install firefox
```

### 2. Run Trend Analysis

```bash
# Basic usage (alias normalization, recommended)
python crawlers/analyze_trends.py --output results.json

# With NER (alternative, requires underthesea)
python crawlers/analyze_trends.py --use-ner --output results.json

# Skip text enrichment
python crawlers/analyze_trends.py --no-aliases --output results.json
```

### 3. Evaluate & Visualize

```bash
# Default: Direct trend assignment (recommended)
python crawlers/evaluate_trends.py --input results.json

# Experimental: HDBSCAN clustering
python crawlers/evaluate_trends.py --input results.json --use-hdbscan

# Filter routine trends (weather, prices, etc.)
python crawlers/evaluate_trends.py --input results.json --filter-routine
```

## Pipeline Flow

```
Google Trends CSV → Build Aliases → Normalize Texts
                                         ↓
News + FB Posts → Normalize → Embed → Match → Valid Trends → Score → Classify
```

## Options

### analyze_trends.py

| Option | Default | Description |
|--------|---------|-------------|
| `--model` | `paraphrase-multilingual-mpnet-base-v2` | Embedding model |
| `--threshold` | `0.55` | Similarity threshold |
| `--use-ner` | `False` | Use NER instead of aliases |
| `--no-aliases` | `False` | Disable text normalization |
| `--save-all` | `False` | Include unmatched posts |

### evaluate_trends.py

| Option | Default | Description |
|--------|---------|-------------|
| `--min-posts` | `3` | Minimum posts for valid trend |
| `--use-hdbscan` | `False` | Use HDBSCAN clustering (experimental) |
| `--filter-routine` | `False` | Filter weather/price trends |

## Output

Each trend is scored and classified:

```json
{
  "trend": "Công Phượng",
  "Class": "Social-Driven",
  "Composite": 67.5,
  "G": 45, "F": 82, "N": 30,
  "posts": 156
}
```

**Classifications:**
- `Strong Multi-source`: High G + F + N
- `Social & News`: High F + N
- `Social-Driven`: High Facebook engagement
- `News-Driven`: High news coverage
- `Emerging`: Low scores across all

## Technical Notes

### Why Alias Normalization > NER?
- NER (underthesea) không nhận các tên quốc tế (e.g., "Yagi")
- Alias uses Google Trends keywords → higher match accuracy
- Test showed +16% improvement with aliases

### Why Direct Assignment > HDBSCAN?
- Data has 652+ small topics with no density peaks
- HDBSCAN classifies 84% as noise
- Direct trend assignment already provides meaningful clusters
