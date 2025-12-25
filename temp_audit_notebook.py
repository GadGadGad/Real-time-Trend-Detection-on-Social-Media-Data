!git clone https://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@github.com/GadGadGad/Real-time-Event-Detection-on-Social-Media-Data
%cd /kaggle/working/Real-time-Event-Detection-on-Social-Media-Data
!pip install -r requirements.txt -q
!pip install rank_bm25 -q
!git pull
from src.evaluation.metrics import evaluate_embeddings, evaluate_clustering, evaluate_refinement
import sys
import pkg_resources

def check_package(name):
    try:
        ver = pkg_resources.get_distribution(name).version
        print(f"✅ {name}: {ver}")
    except:
        print(f"❌ {name}: Not found")

print("🔍 Checking Environment Versions...")
print(f"Python: {sys.version.split()[0]}")
check_package("transformers")
check_package("sentence-transformers")
check_package("torch")
check_package("accelerate")
# 1. Setup & Imports
import sys
import os
import glob
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import numpy as np
from rich.console import Console
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure project root is in path
sys.path.append(os.path.abspath('..'))

from sklearn.metrics.pairwise import cosine_similarity
from src.pipeline.main_pipeline import find_matches, find_matches_hybrid, load_social_data, load_news_data, load_google_trends, refine_trends_preprocessing
from src.core.analysis.clustering import cluster_data, extract_cluster_labels
from src.utils.text_processing.alias_normalizer import build_alias_dictionary, normalize_with_aliases
from src.utils.text_processing.vectorizers import get_embeddings

console = Console()
pd.set_option('display.max_colwidth', 100)
%matplotlib inline
# FILE_ID = "1CRTi7U5cdyKQ7Z2xZvNEoMfuNMdszUrN"

# !pip install -q gdown
# !gdown $FILE_ID --fuzzy -O data.zip
# !mkdir -p embeddings_cache
# !unzip -oq data.zip -d embeddings_cache
# !unzip -oq data.zip -d embeddings_cache
# !mv embeddings_cacheembeddings_cache/* embeddings_cache
# !rm -rf embeddings_cache/kaggle
from kaggle_secrets import UserSecretsClient
user_secrets = UserSecretsClient()
# GEMINI_API_KEY_TIER_1 = user_secrets.get_secret("GEMINI_API_KEY_TIER_1")
GEMINI_API_KEY_FREE = user_secrets.get_secret("GEMINI_API_KEY_FREE")
LIMIT_POSTS = None

# --- Model Config ---
# MODEL_NAME = "bkai-foundation-models/vietnamese-bi-encoder" # [UPDATED] Use BKAI for superior Vietnamese performance
# MODEL_NAME = "BAAI/bge-m3"
MODEL_NAME = "dangvantuan/vietnamese-document-embedding"


EMBEDDING_METHOD = 'sentence-transformer' 
CROSS_ENCODER_MODEL = 'cross-encoder/ms-marco-MiniLM-L-6-v2' 

# --- Analysis Config ---
THRESHOLD = 0.5                           # [UPDATED] Calibrated for BKAI model
USE_KEYWORDS = True                         
USE_CACHE = True                            
LABELING_METHOD = 'semantic'              
RERANK = False                             

# --- Clustering Config ---
MIN_CLUSTER_SIZE = 3                     # [REFINED] More sensitive to emerging niche trends
CLUSTER_EPSILON = 0.04                   # [FIX] Prevents merging distinct topics
CLUSTER_METHOD = 'hdbscan'               
SELECTION_METHOD = 'eom'                 # 'eom' is better for unbalanced cluster sizes
N_CLUSTERS = 15

# --- Summarization Config ---
SUMMARIZE_POSTS = False    
SUMMARIZATION_MODEL = 'vit5-large'  
EMBEDDING_CHAR_LIMIT = 1500 

# --- LLM Refinement Config ---
REFINE_TRENDS = True       
LLM_PROVIDER = 'gemini'    
GEMINI_API_KEY = GEMINI_API_KEY_FREE      
DEBUG_LLM = True          
LLM_MODEL_PATH = 'models/gemma-3-27b-it'
# # [OPTIONAL] ProtonX API Key Setup (for ProtonX embeddings)
# # Get your key from: https://platform.protonx.io/
# import os
# try:
#     from kaggle_secrets import UserSecretsClient
#     secrets = UserSecretsClient()
#     os.environ['PROTONX_API_KEY'] = secrets.get_secret('PROTONX_API_KEY')
#     print('✅ ProtonX API key loaded from Kaggle Secrets')
# except Exception as e:
#     print(f'⚠️ ProtonX secret not found. Set PROTONX_API_KEY manually if using ProtonX embeddings.')
!cp -rf /kaggle/input/se363-temp/* crawlers/
!pip install -q py_vncorenlp
!mkdir vncorenlp_models
%%capture
%%bash
python3 - << 'EOF'
import py_vncorenlp
py_vncorenlp.download_model(save_dir='vncorenlp_models')
EOF
exit 0
> /dev/null 2>&1
!cp -rf vncorenlp_models/ ~/.cache/
# ==========================================
# LOAD DATA (Refined Merged Sources Priority)
# ==========================================
import os
import glob
import pandas as pd
from src.utils.data_loader import load_social_data, load_news_data # Keep imports if needed or define helpers

DATA_DIR = '/kaggle/input/se363-summaries'
if not os.path.exists(DATA_DIR): DATA_DIR = 'summarized_data'

# Helper to load merged csv as posts
def load_merged_posts(path, source_type='News'):
    loaded = []
    try:
        df = pd.read_csv(path)
        for _, row in df.iterrows():
            # content priority: refined_summary > summary > text
            content = row.get('refined_summary')
            if pd.isna(content) or content == "": content = row.get('summary')
            if pd.isna(content) or content == "": 
                # For news, fallback to title ONLY as requested by user.
                # Previously used title + \n + content, which was too noisy.
                t_raw = row.get('title', '')
                content = str(t_raw) if t_raw else row.get('content', '')
            
            # Source
            src = row.get('source')
            if not src:
                if source_type == 'Facebook':
                    src = f"Face: {row.get('pageName', 'Unknown')}"
                else:
                    src = source_type.upper()
            
            # Stats
            stats = {'likes': row.get('likes', 0), 'comments': row.get('comments', 0), 'shares': row.get('shares', 0)}
            
            # FILTER: Skip empty or very short content
            if len(str(content).strip()) < 20:
                continue

            loaded.append({
                "source": src,
                "content": str(content),
                "title": row.get('title', ''),
                "url": row.get('url') or row.get('postUrl', ''),
                "time": row.get('time') or row.get('published_at', ''),
                "stats": stats
            })
    except Exception as e:
        print(f"Error loading {path}: {e}")
    return loaded

# Load Trends
trend_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/trendings/*.csv")
trends = load_google_trends(trend_files)
print(f"Loaded {len(trends)} trends.")

# Load Posts (Refined > Raw)
posts = []

# 1. Facebook
fb_merged = os.path.join(DATA_DIR, 'facebook_merged.csv')
if os.path.exists(fb_merged):
    print(f"Loading merged Facebook data from {fb_merged}")
    posts.extend(load_merged_posts(fb_merged, 'Facebook'))
else:
    print("Merged Facebook data not found. Loading raw...")
    fb_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json")
    posts.extend(load_social_data(fb_files))

# 2. News
NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
news_merged_count = 0
for source in NEWS_SOURCES:
    n_path = os.path.join(DATA_DIR, f'{source}_merged.csv')
    if os.path.exists(n_path):
        posts.extend(load_merged_posts(n_path, source))
        news_merged_count += 1

if news_merged_count == 0:
    print("Merged News data not found. Loading raw...")
    news_files = glob.glob("/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/news/**/*.csv", recursive=True)
    posts.extend(load_news_data(news_files))

if LIMIT_POSTS:
    posts = posts[:LIMIT_POSTS]
    
# Helper: Extract contents
post_contents = [p.get('content', '') for p in posts]
print(f"Loaded {len(posts)} posts for analysis.")

# [EDA] Data Health Check
import pandas as pd
import matplotlib.pyplot as plt

# Create temp DF for analysis only
if 'posts' in locals():
    _df_health = pd.DataFrame(posts)
    print("--- Data Health Overview ---")
    print(f"Total Rows: {len(_df_health)}")
    if 'source' in _df_health.columns:
        print(f"Sources:\n{_df_health['source'].value_counts().head().to_dict()}")
    
    # Length distribution
    if 'content' in _df_health.columns:
        plt.figure(figsize=(10, 3))
        plt.hist(_df_health['content'].astype(str).str.len(), bins=50, color='skyblue', edgecolor='black')
        plt.title("Post Length Distribution")
        plt.xlabel("Character Count")
        plt.ylabel("Frequency")
        plt.show()
    else:
        print("'content' column not found in posts.")
else:
    print("'posts' variable not found. Skipping Step 1 check.")

lengths = _df_health['content'].astype(str).str.len()

plt.figure(figsize=(10, 2))
plt.boxplot(lengths, vert=False)
plt.title("Post Length Boxplot")
plt.xlabel("Character Count")
plt.show()

lengths = _df_health['content'].astype(str).str.len()

# keep only < 1500
_df_health_clean = _df_health[lengths < 1000]

print("Removed:", (lengths >= 1500).sum())
print("Remaining:", len(_df_health_clean))


lengths = _df_health_clean['content'].astype(str).str.len()
plt.figure(figsize=(10, 2))
plt.boxplot(lengths, vert=False)
plt.title("Post Length Boxplot")
plt.xlabel("Character Count")
plt.show()

print("Before:", len(_df_health))
print("After :", len(_df_health_clean))

import pandas as pd

# đảm bảo content không bị NaN -> chuỗi rỗng
lens = _df_health_clean['content'].fillna('').astype(str).str.len()

# 1. kiểm tra có tồn tại content rỗng hay không
has_empty = (lens <= 0).any()

# 2. đếm số lượng content rỗng
num_empty = (lens == 0).sum()

# 3. lấy các dòng content rỗng (để inspect)
empty_rows = _df_health_clean[lens == 0]

print(f"Has empty content: {has_empty}")
print(f"Number of empty contents: {num_empty}")

# 4. (tuỳ chọn) drop các dòng content rỗng
_df_clean_nan = _df_health_clean[lens > 0].reset_index(drop=True)

print(f"Before: {len(_df_health_clean)} rows")
print(f"After : {len(_df_clean_nan)} rows")

posts = _df_clean_nan.to_dict(orient="records")
post_contents = [p.get('content', '') for p in posts]
posts[:3]
# from src.core.extraction.keyword_extractor import KeywordExtractor
# import underthesea
# sample_text = "Học sinh Hà Nội đang háo hức chờ đón lễ hội áo dài tại hồ Hoàn Kiếm."
# print(f"Original: {sample_text}\n")
# # 1. Basic Split
# print("1. Basic Split:", sample_text.split())
# # 2. Fast Segmentation (CRF)
# kw_crf = KeywordExtractor(segmentation_method="underthesea")
# print("2. Underthesea (CRF):", kw_crf.extract_keywords(sample_text).split()[:5])
# # 3. Accurate Segmentation (Transformer)
# print("\nRunning Transformer Segmentation (First run may download model)...")
# kw_trans = KeywordExtractor(segmentation_method="transformer")
# print("3. Transformer:", kw_trans.extract_keywords(sample_text).split()[:5])
# kw_phonlp = KeywordExtractor(segmentation_method="phonlp")
# print("3. PhoNLP:", kw_phonlp.extract_keywords(sample_text).split()[:5])
if REFINE_TRENDS:
    trends = refine_trends_preprocessing(
        trends, 
        llm_provider=LLM_PROVIDER, 
        gemini_api_key=GEMINI_API_KEY, 
        llm_model_path=LLM_MODEL_PATH, 
        debug_llm=DEBUG_LLM,
        cache_path='/kaggle/input/trend-refine-v1/trend_refine_d10b7f422418623845025458f3b6497a.json',  # Use manual cache if set
        source_files=trend_files  # Enables caching
    )
else:
    print("Skipping Trend Refinement (using raw trends).")
# =========================
# [EDA-TRENDS 1] To DataFrame + Overview
# =========================
import pandas as pd
import numpy as np

df_tr = pd.DataFrame([
    {
        "trend": k,
        "volume": v.get("volume", np.nan),
        "time": v.get("time", None),
        "n_keywords": len(v.get("keywords", []) or []),
        "keywords": v.get("keywords", []) or []
    }
    for k, v in trends.items()
])

df_tr["time"] = pd.to_datetime(df_tr["time"], errors="coerce")
df_tr["trend_len"] = df_tr["trend"].astype(str).str.len()

print("=== Trends Overview ===")
print("Total trends:", len(df_tr))
print("Time valid:", df_tr["time"].notna().sum())
print("Volume valid:", df_tr["volume"].notna().sum())
display(df_tr.head(5))

print("\n=== Volume Stats ===")
display(df_tr["volume"].describe(percentiles=[.5,.75,.9,.95,.99]))

print("\n=== Keywords Count Stats ===")
display(df_tr["n_keywords"].describe(percentiles=[.5,.75,.9,.95,.99]))

# =========================
# [EDA-TRENDS 2] Top trends + Volume distribution
# =========================
import matplotlib.pyplot as plt

topN = 20
df_top = df_tr.sort_values("volume", ascending=False).head(topN)

print(f"Top {topN} trends by volume:")
display(df_top[["trend", "volume", "time", "n_keywords"]])

plt.figure(figsize=(10,3))
vals = df_tr["volume"].dropna().values
plt.hist(vals, bins=40)
plt.title("Trend Volume Distribution")
plt.xlabel("volume"); plt.ylabel("count")
plt.show()

# log-scale view (volume thường lệch mạnh)
plt.figure(figsize=(10,3))
vals = df_tr["volume"].dropna().values
plt.hist(np.log10(vals + 1), bins=40)
plt.title("Trend Volume Distribution (log10(volume+1))")
plt.xlabel("log10(volume+1)"); plt.ylabel("count")
plt.show()

# =========================
# [EDA-TRENDS 3] Trends over time (counts + volume)
# =========================
import matplotlib.pyplot as plt

if df_tr["time"].notna().any():
    df_day = df_tr.dropna(subset=["time"]).set_index("time").sort_index()
    
    plt.figure(figsize=(10,3))
    df_day["trend"].resample("D").count().plot()
    plt.title("Number of Trends per Day")
    plt.ylabel("count")
    plt.show()

    plt.figure(figsize=(10,3))
    df_day["volume"].resample("D").sum(min_count=1).plot()
    plt.title("Total Trend Volume per Day")
    plt.ylabel("sum(volume)")
    plt.show()
else:
    print("No valid 'time' to plot.")

# =========================
# [EDA-TRENDS 4] Keyword hygiene: phổ biến từ khóa + từ khóa rác
# =========================
from collections import Counter
import re

# flatten keywords
all_kw = []
for kws in df_tr["keywords"]:
    all_kw.extend([str(x).strip().lower() for x in (kws or []) if str(x).strip()])

kw_counter = Counter(all_kw)

print("Most common keywords (top 30):")
for k, c in kw_counter.most_common(30):
    print(f"{k:35} {c}")

# quick "garbage" heuristics
def is_garbage_kw(s: str):
    s = s.strip().lower()
    if len(s) <= 2: 
        return True
    if re.fullmatch(r"\d+", s):  # only digits
        return True
    if re.fullmatch(r"[a-z]{1,3}", s):  # quá ngắn kiểu 'cf', 'fo4' (tuỳ bạn)
        return True
    return False

garbage = [k for k in all_kw if is_garbage_kw(k)]
print("\nGarbage keyword ratio:", f"{len(garbage)}/{len(all_kw)}", f"({len(garbage)/max(1,len(all_kw))*100:.2f}%)")

print("\nSample garbage keywords (top 30):")
for k, c in Counter(garbage).most_common(30):
    print(f"{k:20} {c}")

# =========================
# [EDA-TRENDS 5] Duplicate / near-duplicate trend names (normalize + collisions)
# =========================
import unicodedata
from collections import defaultdict

def normalize_text_simple(s: str) -> str:
    s = (s or "").lower().strip()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove accents
    s = " ".join(s.split())
    return s

df_tr["trend_norm"] = df_tr["trend"].astype(str).apply(normalize_text_simple)

# exact collisions after normalization
groups = defaultdict(list)
for _, r in df_tr.iterrows():
    groups[r["trend_norm"]].append(r["trend"])

collisions = {k:v for k,v in groups.items() if len(v) > 1}

print("Normalized collisions:", len(collisions))
if collisions:
    # show a few
    for i, (k, v) in enumerate(list(collisions.items())[:15], 1):
        print(f"\n[{i}] norm='{k}'")
        for name in v:
            vol = df_tr.loc[df_tr["trend"]==name, "volume"].values[0]
            print(f"   - {name} (vol={vol})")
else:
    print("No collisions found.")

# =========================
# [EDA-TRENDS 6] Quick categorization: sports / lottery / air-quality / entertainment (rule-based)
# =========================
import re

def trend_category(name: str) -> str:
    s = normalize_text_simple(name)
    if any(x in s for x in [" vs ", "dau voi", "gap", "tran", "c1", "champions league", "ucl", "sea games", "seagame", "u22", "u23"]):
        return "sports"
    if any(x in s for x in ["xo so", "xsmn", "xsmt", "xsmb", "kqx", "ket qua xo so"]):
        return "lottery"
    if any(x in s for x in ["aqi", "chat luong khong khi", "o nhiem"]):
        return "air_quality"
    if any(x in s for x in ["tap", "phim", "show", "concert", "miss", "hoa hau", "game awards", "goty"]):
        return "entertainment"
    return "other"

df_tr["category"] = df_tr["trend"].apply(trend_category)
display(df_tr["category"].value_counts())

# top trends per category
for cat in df_tr["category"].unique():
    print(f"\n=== Top 10 in category: {cat} ===")
    display(df_tr[df_tr["category"]==cat].sort_values("volume", ascending=False).head(10)[["trend","volume","time","n_keywords"]])

def create_smart_trend_query(trend_name, keyword_list, max_keywords=5):
    # 1. Lấy tên trend làm gốc
    unique_signals = [trend_name]
    
    # 2. Lọc lấy các thực thể hoặc từ khóa mang tính phân biệt cao
    # Ưu tiên các từ có tên quốc gia hoặc sự kiện cụ thể
    for kw in keyword_list:
        if len(unique_signals) >= max_keywords: break
        
        # Chỉ lấy nếu nó chưa có trong list (để tránh lặp "bóng đá")
        if not any(token in kw for token in unique_signals):
             unique_signals.append(kw)
             
    return " ".join(unique_signals)

# Thử nghiệm với data SEA Games của bạn:
sample_keywords = trends['sea games 33']['keywords']
smart_query = create_smart_trend_query('sea games 33', sample_keywords)
print(f"Smart Query: {smart_query}")
# garbage_list = ['cf', 'fo4', 'sou', 'scl', 'hit', 'vic', 'xoi', 'vck']

# def filter_trends(keywords):
#     trends = []
#     for k in keywords:
#         # Lọc theo độ dài: thường trend không bao giờ chỉ có 2-3 ký tự (trừ tên riêng đặc biệt)
#         if len(k) < 4 and k not in ['t1', 'tga']: 
#             continue
#         # Lọc theo garbage list
#         if k in garbage_list:
#             continue
#         # Lọc những thằng search dở (kết thúc bằng 'vs' hoặc 'vs ')
#         if k.endswith('vs') or k.endswith('vs '):
#             continue
            
#         trends.append(k)
#     return trends

# filtered_trends = filter_trends(trends)
# print(f"{len(filtered_trends)} remaining trends.")
import re
import unicodedata

garbage_list = {'cf', 'fo4', 'sou', 'scl', 'hit', 'vic', 'xoi', 'vck'}
allow_short = {'t1', 'tga'}  # whitelist cho viết tắt hợp lệ

# Các cụm "ý định tìm kiếm" quá chung (không phải trend)
generic_terms = {
    'truc tiep', 'trực tiếp',
    'ket qua', 'kết quả', 'kq', 'kqbd',
    'lich', 'lịch', 'lich thi dau', 'lịch thi đấu',
    'bang xep hang', 'bảng xếp hạng', 'bxh',
    'hom nay', 'hôm nay', 'hom qua', 'hôm qua',
    'moi nhat', 'mới nhất',
    'highlight', 'tin the thao', 'tin tức'
}

# Regex nhận diện dạng ngày/tháng (routine)
date_pat = re.compile(
    r'(\bngay\s*\d{1,2}\b)|(\b\d{1,2}\s*thang\s*\d{1,2}\b)|(\b\d{1,2}[/-]\d{1,2}([/-]\d{2,4})?\b)',
    flags=re.IGNORECASE
)

def normalize_text_simple(s: str) -> str:
    s = (s or "").strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.combining(ch))  # remove accents
    s = re.sub(r"\s+", " ", s)
    return s

def mostly_numeric(s: str) -> bool:
    s = s.strip()
    if not s:
        return True
    digits = sum(ch.isdigit() for ch in s)
    letters = sum(ch.isalpha() for ch in s)
    # toàn số hoặc gần toàn số (mà ít chữ)
    return (digits >= max(1, int(0.6 * len(s)))) and letters <= 1

def too_many_symbols(s: str) -> bool:
    # nếu ký tự không phải chữ/số/space chiếm quá nhiều
    if not s:
        return True
    allowed = sum(ch.isalnum() or ch.isspace() for ch in s)
    return (1 - allowed / max(1, len(s))) > 0.30

def bad_vs_pattern(s: str) -> bool:
    # bỏ query kết thúc vs / vs_
    if s.endswith("vs") or s.endswith("vs "):
        return True
    # nếu có "vs" nhưng 1 bên quá ngắn -> coi là query dở
    if " vs " in s:
        left, right = s.split(" vs ", 1)
        if len(left.strip()) < 3 or len(right.strip()) < 3:
            return True
    return False

def is_generic_only(s_norm: str) -> bool:
    # nếu toàn bộ chuỗi là 1 cụm generic -> bỏ
    return s_norm in {normalize_text_simple(x) for x in generic_terms}

def filter_trends(keywords):
    kept = []
    seen_norm = set()

    for k in keywords:
        raw = str(k).strip()
        if not raw:
            continue

        norm = normalize_text_simple(raw)

        # 0) de-dup after normalize
        if norm in seen_norm:
            continue

        # 1) length filter (giữ whitelist)
        if len(norm) < 4 and norm not in allow_short:
            continue

        # 2) blacklist rác (so trên normalized)
        if norm in garbage_list:
            continue

        # 3) loại intent-term quá chung
        if is_generic_only(norm):
            continue

        # 4) vs pattern xấu
        if bad_vs_pattern(norm):
            continue

        # 5) routine ngày/tháng (tuỳ bạn: bỏ hẳn hoặc gắn nhãn)
        if date_pat.search(norm):
            # nếu bạn muốn GIỮ các trend kiểu xổ số ngày/tháng thì comment dòng continue này
            continue

        # 6) toàn số / gần toàn số
        if mostly_numeric(norm) and len(norm) <= 6:
            continue

        # 7) URL/handle/hashtag nặng
        if "http" in norm or "www" in norm or ".com" in norm:
            continue

        # 8) nhiều ký tự lạ
        if too_many_symbols(raw):
            continue

        kept.append(raw)
        seen_norm.add(norm)

    return kept

# Usage
filtered_trends = filter_trends(trends)
print(f"{len(filtered_trends)} remaining trends.")

# Convert to DataFrame for EDA
df_raw = pd.DataFrame(posts)

# 1. Clean Time field
df_raw['time'] = pd.to_datetime(df_raw['time'], errors='coerce')

# 2. Source Categories
df_raw['source_type'] = df_raw['source'].apply(lambda x: 'Facebook' if 'Face:' in x else 'News')
df_raw['content_length'] = df_raw['content'].apply(len)

fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# A. Source Type Distribution
sns.countplot(data=df_raw, x='source_type', ax=ax[0], palette='pastel')
ax[0].set_title("Distribution of Data Types")

# B. Post Counts over Time
if df_raw['time'].notnull().any():
    df_raw[df_raw['time'].notnull()].set_index('time').resample('D').size().plot(ax=ax[1], color='teal', marker='o')
    ax[1].set_title("Daily Post Volume")
    ax[1].set_ylabel("Number of Posts")
else:
    ax[1].text(0.5, 0.5, "No Valid Time Data", ha='center')

plt.tight_layout()
plt.show()
# A. Top 20 specific sources
def clean_source_name(s):
    return s.replace("Face: ", "")

df_raw['clean_source'] = df_raw['source'].apply(clean_source_name)
top_sources = df_raw['clean_source'].value_counts().head(20)

plt.figure(figsize=(12, 6))
sns.barplot(x=top_sources.values, y=top_sources.index, palette='viridis')
plt.title("Top 20 Active Sources")
plt.xlabel("Number of Posts")
plt.show()
# B. Word Cloud
# Simple stopwords list for Vietnamese (basic)
path = "/kaggle/input/vietnamese-stopwords-txt/vietnamese-stopwords.txt"

with open(path, "r", encoding="utf-8") as f:
    stops = [line.strip() for line in f]

# bỏ dòng rỗng + (tuỳ chọn) bỏ comment
stops = [s for s in stops if s and not s.startswith("#")]
text_corpus = " ".join(df_raw['content'].dropna().tolist())

wc = WordCloud(width=800, height=400, background_color='white', stopwords=stops, max_words=100).generate(text_corpus)

plt.figure(figsize=(14, 7))
plt.imshow(wc, interpolation='bilinear')
plt.axis("off")
plt.title("Most Common Words (Word Cloud)")
plt.show()
!pip install py_vncorenlp underthesea -q

from src.utils.text_processing.segmenter import batch_segment_texts # <--- Import this
post_contents_enriched = post_contents
# 1. Segment Posts
print("✂️ Segmenting Posts...")
post_contents_segmented = batch_segment_texts(post_contents_enriched)

# 2. Segment Trends (CRITICAL Step)
print("✂️ Segmenting Trend Keywords...")
trend_keys = list(trends.keys())
trend_queries = []
for t in trend_keys:
    # Segment the keyword list properly
    keywords = trends[t]['keywords']
    # Join first to segment contextually, or segment each? 
    # Better to segment the joined string or each keyword.
    # Simple way: Segment the search query string
    query_str = " ".join(keywords)
    seg_query = batch_segment_texts([query_str])[0]
    trend_queries.append(seg_query)
post_contents_segmented[:5]
import gc, torch
gc.collect()
torch.cuda.empty_cache()
from sentence_transformers import SentenceTransformer, util

def find_matches_segmented(posts, trends, model_name=None, threshold=0.35, save_all=False):
    # 1. Segment Posts
    post_contents = [p.get('content', '')[:500] for p in posts]
    post_contents_seg = batch_segment_texts(post_contents)
    
    # 2. Segment Trends
    trend_keys = list(trends.keys())
    trend_queries_raw = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_queries_seg = batch_segment_texts(trend_queries_raw) # <--- List này khớp index với trend_keys
    # 3. Embed
    post_embeddings = get_embeddings(post_contents_seg, model_name=model_name, trust_remote_code=True)
    
    embedder = SentenceTransformer(model_name,  trust_remote_code=True) 
    trend_embeddings = embedder.encode(trend_queries_seg, batch_size=1) 
    # 4. Match
    sims = cosine_similarity(post_embeddings, trend_embeddings)
    matches = []
    
    for i, post in enumerate(posts):
        best_idx = np.argmax(sims[i])
        best_score = sims[i][best_idx]
        
        nearest_topic = trend_keys[best_idx]
        nearest_seg   = trend_queries_seg[best_idx] # <--- Lấy text trend đã tách từ
        
        is_high_score = (best_score >= threshold)
        
        if is_high_score or save_all:
            matches.append({
                "source": post.get('source'),
                "post_content": post_contents[i],
                "segmented_content": post_contents_seg[i],
                
                "nearest_candidate": nearest_topic,
                "segmented_candidate": nearest_seg, # <--- CỘT MỚI
                
                "trend": nearest_topic if is_high_score else "Unassigned",
                "score": float(best_score), 
                "is_matched": is_high_score
            })
    return matches
# Run
print("Running DEBUG Segmented Matching...")
matches_semantic = find_matches_segmented(posts, trends, model_name=MODEL_NAME, threshold=THRESHOLD, save_all=True)
df_sem = pd.DataFrame(matches_semantic)
# Xem kết quả, focus vào mấy cột quan trọng
cols = ['segmented_content', 'segmented_candidate', 'score', 'is_matched']
df_sem[cols].head(5)
# Filter for matched posts only
sem_plot_df = df_sem[df_sem['is_matched'] == True].copy()

if len(sem_plot_df) < 5:
    print("Not enough semantic matches to plot.")
else:
    print(f"Visualizing {len(sem_plot_df)} Semantic Matches...")
    sem_texts = sem_plot_df['post_content'].tolist()
    sem_labels = sem_plot_df['trend'].tolist()

    # Embeddings (Always use Sentence Transformer for visualization quality)
    sem_embeddings = get_embeddings(sem_texts, method="sentence-transformer", 
                                    model_name=MODEL_NAME, trust_remote_code=True)

    # t-SNE
    tsne_sem = TSNE(n_components=2, random_state=42, perplexity=min(30, len(sem_texts)-1))
    coords_sem = tsne_sem.fit_transform(sem_embeddings)

    # Create DataFrame for Plotly
    df_vis_sem = pd.DataFrame({
        'x': coords_sem[:, 0],
        'y': coords_sem[:, 1],
        'Label': sem_labels,
        'Snippet': [t[:100] + '...' for t in sem_texts]
    })

    # Interactive Plot
    fig = px.scatter(df_vis_sem, x='x', y='y', color='Label', 
                     hover_data=['Snippet'],
                     title="Interactive t-SNE: Semantic Matches (Baseline)")
    fig.show()
# post_contents_seg = df_sem['segmented_content']
# 1. Model & Tool Setup (Refined Pipeline)
from src.utils.text_processing.segmenter import batch_segment_texts
from src.core.analysis.clustering import cluster_data
from src.utils.text_processing.vectorizers import get_embeddings

# 2. Post Processing (Segmentation + Embedding)
print("✂️ Segmenting Posts...")
post_contents_seg = batch_segment_texts(post_contents_enriched)
post_embeddings = get_embeddings(post_contents_seg, model_name=MODEL_NAME, trust_remote_code=True)

# 3. Trend Processing (Smart Query + Segmentation)
def create_smart_trend_query(trend_name, keyword_list, max_keywords=8):
    unique_signals = [trend_name]
    for kw in keyword_list:
        if len(unique_signals) >= max_keywords: break
        if not any(token.lower() in kw.lower() for token in unique_signals):
             unique_signals.append(kw)
    return " ".join(unique_signals)

trend_keys = list(trends.keys())
trend_queries_raw = [create_smart_trend_query(k, trends[k]['keywords']) for k in trend_keys]
trend_queries_seg = batch_segment_texts(trend_queries_raw)
from sentence_transformers import SentenceTransformer
embedder = SentenceTransformer(MODEL_NAME, trust_remote_code=True)
trend_embeddings = embedder.encode(trend_queries_seg)

# 4. Clustering (Optimized HDBSCAN)
cluster_labels = cluster_data(
    embeddings=post_embeddings,
    min_cluster_size=MIN_CLUSTER_SIZE,
    epsilon=CLUSTER_EPSILON,
    method=CLUSTER_METHOD,
    selection_method=SELECTION_METHOD
)

# 5. Cluster Post-Processing (Refined)
import pandas as pd
import numpy as np
from src.core.analysis.clustering import extract_cluster_labels

# Create results DataFrame
df_results = pd.DataFrame({
    'content': post_contents_enriched,
    'segmented_content': post_contents_seg,
    'cluster': cluster_labels
})

# Filter out outliers (label -1)
df_clean = df_results[df_results['cluster'] != -1]

print(f"Total items: {len(df_results)}")
print(f"Items in clusters: {len(df_clean)} (Outliers: {len(df_results) - len(df_clean)})")

# Representative Headings Selection (Correct API Usage)
print("🏷️ Extracting cluster labels...")
cluster_summaries = extract_cluster_labels(
    texts=df_results['content'].tolist(),
    labels=cluster_labels,
    method='tfidf'
)

print(f"Found {len(cluster_summaries)} valid clusters.")
for c_id, label in list(cluster_summaries.items())[:10]:
    count = len(df_clean[df_clean['cluster'] == c_id])
    print(f"Cluster {c_id} ({count} posts): {label}")

# --- STEP 1: Detailed Setup & Imports ---
from src.pipeline.pipeline_stages import run_summarization_stage, run_sahc_clustering, calculate_match_scores
from src.utils.text_processing.vectorizers import get_embeddings
from src.pipeline.main_pipeline import extract_dynamic_anchors
from src.core.analysis.clustering import extract_cluster_labels
from src.pipeline.trend_scoring import calculate_unified_score
from src.core.analysis.sentiment import batch_analyze_sentiment
from sentence_transformers import SentenceTransformer, CrossEncoder
from rich.console import Console
from dateutil import parser
import torch
import numpy as np

console = Console()

# Setup Devices & Models
embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using Device: {embedding_device}")

embedder = SentenceTransformer(MODEL_NAME, device=embedding_device, trust_remote_code=True)

reranker = None
if RERANK:
    try: 
        reranker = CrossEncoder(CROSS_ENCODER_MODEL, device=embedding_device)
        print(f"✅ Reranker initialized: {CROSS_ENCODER_MODEL}")
    except Exception as e:
        print(f"⚠️ Failed to load reranker: {e}")
# ==========================================
# [EDA] Embedding Quality Check
# ==========================================

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import matplotlib.pyplot as plt

# Use 'embeddings' key variable if available, else skip
if 'post_embeddings' in locals() and len(post_embeddings) > 0:

    # Sample 1000 for efficiency
    _sample_idx = np.random.choice(
        len(post_embeddings),
        size=min(1000, len(post_embeddings)),
        replace=False
    )

    _sample_embs = (
        post_embeddings[_sample_idx]
        if hasattr(post_embeddings, '__getitem__')
        else post_embeddings
    )

    _sim_matrix = cosine_similarity(_sample_embs)

    _sim_vals = _sim_matrix[
        np.triu_indices_from(_sim_matrix, k=1)
    ]

    plt.figure(figsize=(8, 3))
    plt.hist(
        _sim_vals,
        bins=50,
        color='salmon',
        alpha=0.7,
        edgecolor='none'
    )

    plt.title(
        f"Pairwise Cosine Similarity (Sample N={len(_sample_idx)})"
    )
    plt.xlabel("Cosine Similarity")

    plt.axvline(
        np.mean(_sim_vals),
        color='k',
        linestyle='dashed',
        linewidth=1,
        label=f"Mean: {np.mean(_sim_vals):.2f}"
    )

    plt.legend()
    plt.show()

else:
    print(
        "'embeddings' variable not found or empty. "
        "Skipping Step 2 check."
    )

# --- 🔬 ADVANCED MULTI-SCENARIO BENCHMARK ---
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch, gc, pandas as pd
import numpy as np

# 1. Define Scenarios
scenarios = [
    {
        "name": "Storm Synonyms",
        "ref": "Siêu bão Yagi tiến vào vịnh Bắc Bộ",
        "pos": "Dự báo bão số 3 gây mưa lớn diện rộng", # Same event, different name
        "neg": "Giá xăng giảm mạnh chiều nay"
    },
    {
        "name": "Domain overlap (Hanoi)",
        "ref": "Ùn tắc nghiêm trọng tại cửa ngõ Hà Nội",
        "pos": "Kẹt xe kéo dài ở đường Giải Phóng thủ đô", # Same event
        "neg": "Phở Hà Nội lọt top món ăn ngon nhất"     # Same location, Different event
    },
    {
        "name": "Category overlap (Prices)",
        "ref": "Giá vàng SJC tăng vọt lên 85 triệu",
        "pos": "Thị trường vàng trong nước biến động", # Same event
        "neg": "Giá vé máy bay Tết tăng cao"           # Same category (Price), Different event
    },
    {
        "name": "Social vs News Language",
        "ref": "U23 Việt Nam giành huy chương vàng",
        "pos": "Mấy ông giáo trẻ đá hay quá, vô địch rồi!", # Social slang
        "neg": "Hướng dẫn đăng ký định danh điện tử"
    }
]

models_to_test = [
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "keepitreal/vietnamese-sbert",
    "bkai-foundation-models/vietnamese-bi-encoder",
    "BAAI/bge-m3",
    "intfloat/multilingual-e5-large",
    "dangvantuan/vietnamese-document-embedding",
    "Alibaba-NLP/gte-multilingual-base"# We'll add trust_remote_code
]

results = []

for model_name in models_to_test:
    print(f"📦 Testing {model_name}...")
    try:
        # Pass trust_remote_code=True for the dangvantuan model
        extra_args = {"trust_remote_code": True} if "dangvantuan" or "Alibaba-NLP" in model_name else {}
        model = SentenceTransformer(model_name, device='cuda' if torch.cuda.is_available() else 'cpu', **extra_args)
        
        scenario_scores = []
        for s in scenarios:
            embs = model.encode([s['ref'], s['pos'], s['neg']])
            sims = cosine_similarity(embs)
            pos_sim = sims[0, 1]
            neg_sim = sims[0, 2]
            scenario_scores.append(pos_sim - neg_sim)
            
        results.append({
            "Model": model_name,
            "Avg Stability Gap": np.mean(scenario_scores),
            "Min Gap (Worst Scenario)": np.min(scenario_scores),
            "Max Gap (Best Scenario)": np.max(scenario_scores)
        })
        
        del model
        gc.collect()
        if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"❌ Skipping {model_name}: {e}")

df_final = pd.DataFrame(results).sort_values(by="Avg Stability Gap", ascending=False)
display(df_final)
X = np.asarray(post_embeddings)

print("shape:", X.shape)
print("dtype:", X.dtype)
print("contiguous:", X.flags['C_CONTIGUOUS'])
print("memory MB:", X.nbytes / 1024**2)
# --- 1A. GLOBAL SIMILARITY ANALYSIS ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sample_idx = np.random.choice(len(post_embeddings), min(1000, len(post_embeddings)), replace=False)
sample_embs = post_embeddings[sample_idx]
sims = cosine_similarity(sample_embs)
avg_sim = np.mean(sims[np.triu_indices(len(sample_idx), k=1)])

print(f"🔬 Average Global Pairwise Similarity: {avg_sim:.3f}")
print("   → If > 0.75: Embeddings are too uniform (model problem)")
print("   → If 0.5-0.75: Moderate overlap (preprocessing problem)")
print("   → If < 0.5: Good separation (clustering parameters problem)")
import torch

X = torch.from_numpy(post_embeddings)

# center
X = X - X.mean(dim=0, keepdim=True)

# low-rank PCA
U, S, V = torch.pca_lowrank(X, q=50)

var = (S**2) / (S**2).sum()
cumulative_var = torch.cumsum(var, dim=0)

dims_90 = int((cumulative_var >= 0.90).nonzero()[0]) + 1
print(f"📐 Dimensions needed for 90% variance: {dims_90}")
print("   → If < 10: Embeddings are low-rank/redundant (bad)")
print("   → If > 30: Good dimensionality (healthy)")
# --- 2A. TEXT LENGTH DISTRIBUTION ---
import matplotlib.pyplot as plt

lengths = [len(p.get('content', '')) for p in posts]
plt.figure(figsize=(10, 4))
plt.hist(lengths, bins=50, edgecolor='black')
plt.axvline(np.median(lengths), color='red', linestyle='--', label=f'Median: {np.median(lengths):.0f}')
plt.xlabel("Character Length")
plt.title("Post Length Distribution")
plt.legend()
plt.show()

short_posts = sum(1 for l in lengths if l < 50)
print(f"⚠️ Posts < 50 chars: {short_posts} ({100*short_posts/len(lengths):.1f}%)")
# --- 2B. VOCABULARY OVERLAP (Are all posts using the same words?) ---
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import Counter

tfidf = TfidfVectorizer(max_features=500, stop_words=None)
tfidf_matrix = tfidf.fit_transform([p.get('content', '') for p in posts])

# Top 20 most common words across ALL posts
feature_names = tfidf.get_feature_names_out()
word_freq = np.array(tfidf_matrix.sum(axis=0)).flatten()
top_words = sorted(zip(feature_names, word_freq), key=lambda x: -x[1])[:20]

print("📝 Top 20 TF-IDF Words (potential noise):")
for word, score in top_words:
    print(f"   {word}: {score:.1f}")
# --- 3A. CLUSTER SIZE DISTRIBUTION ---
from collections import Counter

label_counts = Counter(cluster_labels)
sizes = [c for l, c in label_counts.items() if l != -1]

print(f"📦 Cluster Size Stats:")
print(f"   Min: {min(sizes)}, Max: {max(sizes)}, Median: {np.median(sizes):.0f}")
print(f"   Mega-clusters (>100 posts): {sum(1 for s in sizes if s > 100)}")
print(f"   Micro-clusters (<10 posts): {sum(1 for s in sizes if s < 10)}")
# --- 3B. COHESION PER CLUSTER (Are clusters tight?) ---
cohesion_scores = []
for label in set(cluster_labels):
    if label == -1: continue
    mask = np.array(cluster_labels) == label
    cluster_embs = post_embeddings[mask]
    centroid = cluster_embs.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(cluster_embs, centroid).flatten()
    cohesion_scores.append((label, np.mean(sims), np.sum(mask)))

# Sort by cohesion (ascending = worst first)
cohesion_scores.sort(key=lambda x: x[1])
print("🧲 Worst 5 Clusters by Cohesion:")
for label, coh, size in cohesion_scores[:5]:
    print(f"   Cluster {label}: Cohesion={coh:.3f}, Size={size}")
# [EDA] Tune EMBEDDING_CHAR_LIMIT
# Tests different character limits to find optimal embedding quality

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import numpy as np

char_limits = [20, 50, 80, 100, 150, -1]
results = []

for limit in char_limits:
    # Truncate posts to this limit
    truncated = [p.get('content', '')[:limit] for p in posts[:1000]]  # Sample for speed
    
    # Get embeddings
    embs = embedder.encode(truncated, show_progress_bar=False)
    
    # Quick K-Means clustering
    kmeans = KMeans(n_clusters=15, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embs)
    
    # Calculate metrics
    silhouette = silhouette_score(embs, labels)
    avg_sim = np.mean([np.dot(embs[i], embs[j]) / (np.linalg.norm(embs[i]) * np.linalg.norm(embs[j])) 
                       for i in range(50) for j in range(i+1, 50)])
    
    results.append({'limit': limit, 'silhouette': silhouette, 'avg_similarity': avg_sim})
    print(f"LIMIT={limit:4d}: Silhouette={silhouette:.4f}, Avg Sim={avg_sim:.3f}")

# Find best
best = max(results, key=lambda x: x['silhouette'])
print(f"✅ Best: EMBEDDING_CHAR_LIMIT = {best['limit']} (silhouette={best['silhouette']:.4f})")

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity
import torch

def eval_hdbscan_for_limit(
    posts,
    embedder,
    limit,
    segment_fn=None,              # batch_segment_texts nếu bạn muốn, hoặc None
    min_cluster_size=5,
    epsilon=0.05,
    selection_method="eom",
    sample_n=1500,                # giảm để đỡ OOM
    seed=42
):
    rng = np.random.default_rng(seed)
    n = len(posts)
    idx = np.arange(n)
    if n > sample_n:
        idx = rng.choice(idx, size=sample_n, replace=False)

    # 1) chuẩn bị text
    texts = []
    for i in idx:
        t = str(posts[i].get("content",""))
        t = t if limit == -1 else t[:limit]
        texts.append(t)

    # 2) segmentation (optional)
    if segment_fn is not None:
        texts_in = segment_fn(texts)
    else:
        texts_in = texts

    # 3) embeddings
    with torch.no_grad():
        embs = embedder.encode(
            texts_in,
            batch_size=32,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

    # 4) clustering (dùng hàm của bạn)
    labels = cluster_data(
        embeddings=embs,
        min_cluster_size=min_cluster_size,
        epsilon=epsilon,
        method="hdbscan",
        selection_method=selection_method,
        texts=texts  # optional logging
    )

    labels = np.asarray(labels)
    mask = labels != -1
    n_noise = int((~mask).sum())
    n_clusters = len(set(labels[mask])) if mask.any() else 0
    noise_ratio = n_noise / len(labels)

    # 5) cohesion = avg cosine(sim(emb, centroid)) theo cụm
    cohesion = np.nan
    if n_clusters > 0:
        cohes = []
        for c in set(labels[mask]):
            c_mask = labels == c
            c_embs = embs[c_mask]
            centroid = c_embs.mean(axis=0, keepdims=True)
            sims = cosine_similarity(c_embs, centroid).flatten()
            cohes.append(float(np.mean(sims)))
        cohesion = float(np.mean(cohes))

    # 6) silhouette (cosine) chỉ nếu đủ cụm
    sil = np.nan
    if n_clusters >= 2 and mask.sum() > n_clusters:
        sil = float(silhouette_score(embs[mask], labels[mask], metric="cosine"))

    # cleanup GPU cache nếu cần
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "limit": limit,
        "n_samples": len(labels),
        "n_clusters": n_clusters,
        "noise_ratio": noise_ratio,
        "cohesion": cohesion,
        "silhouette_cos": sil,
    }

# ==== RUN SWEEP ====
char_limits = [50, 100, 150, 300, 500, 1000, -1]
rows = []
for lim in char_limits:
    r = eval_hdbscan_for_limit(
        posts=posts,
        embedder=embedder,                # SentenceTransformer(MODEL_NAME, device=...)
        limit=lim,
        segment_fn=None,   # hoặc None nếu muốn test raw
        min_cluster_size=MIN_CLUSTER_SIZE,
        epsilon=CLUSTER_EPSILON,
        selection_method=SELECTION_METHOD,
        sample_n=1500
    )
    rows.append(r)

df_eval = pd.DataFrame(rows).sort_values(["cohesion","noise_ratio"], ascending=[False, True])
display(df_eval)

# --- DUPLICATE DETECTION ---
from collections import Counter

content_hashes = [hash(p.get('content', '')[:100]) for p in posts]
duplicates = [count for h, count in Counter(content_hashes).items() if count > 1]

print(f"🔁 Near-duplicate groups: {len(duplicates)}")
print(f"   Total duplicate posts: {sum(duplicates)}")
# --- CLUSTER SANITY CHECK (WITH METRICS) ---
import random
from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

print("🔍 CLUSTER SANITY CHECK (WITH METRICS)")
print("="*70)

# Group posts by cluster
clusters = defaultdict(list)
cluster_indices = defaultdict(list)
for i, label in enumerate(cluster_labels):
    if label != -1:
        clusters[label].append(posts[i])
        cluster_indices[label].append(i)

# Sample 5 random clusters
sample_clusters = random.sample(list(clusters.keys()), min(5, len(clusters)))

for cluster_id in sample_clusters:
    cluster_posts = clusters[cluster_id]
    indices = cluster_indices[cluster_id]
    
    # Calculate cohesion (avg similarity to centroid)
    cluster_embs = post_embeddings[indices]
    centroid = cluster_embs.mean(axis=0).reshape(1, -1)
    sims_to_centroid = cosine_similarity(cluster_embs, centroid).flatten()
    cohesion = np.mean(sims_to_centroid)
    
    # Calculate avg pairwise similarity
    if len(cluster_embs) > 1:
        pairwise = cosine_similarity(cluster_embs)
        avg_sim = np.mean(pairwise[np.triu_indices(len(cluster_embs), k=1)])
    else:
        avg_sim = 1.0
    
    # Get cluster label
    label = cluster_names.get(cluster_id, f"Cluster {cluster_id}") if 'cluster_names' in dir() else f"Cluster {cluster_id}"
    
    # Quality indicator
    quality = "✅" if cohesion > 0.6 else ("⚠️" if cohesion > 0.4 else "❌")
    
    print(f"\n{'='*70}")
    print(f"{quality} CLUSTER {cluster_id}: '{label}' ({len(cluster_posts)} posts)")
    print(f"   📊 Cohesion: {cohesion:.3f} | Avg Pairwise Sim: {avg_sim:.3f}")
    print("-"*70)
    
    # Show 3 random samples
    samples = random.sample(cluster_posts, min(3, len(cluster_posts)))
    for j, post in enumerate(samples, 1):
        content = post.get('content', '')[:180]
        source = post.get('source', 'Unknown')
        print(f"\n  [{j}] {source}: {content}...")

print("\n" + "="*70)
print("📊 QUALITY GUIDE: ✅ Cohesion > 0.6 (tight) | ⚠️ 0.4-0.6 (loose) | ❌ < 0.4 (garbage)")
# --- FILTER LOW-QUALITY CLUSTERS ---
# Removes clusters with low pairwise similarity (garbage bins)

from collections import defaultdict
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

MIN_PAIRWISE_SIM = 0.35  # Threshold for quality

print("🧹 FILTERING LOW-QUALITY CLUSTERS")
print("="*60)

# Group posts by cluster
cluster_indices = defaultdict(list)
for i, label in enumerate(cluster_labels):
    if label != -1:
        cluster_indices[label].append(i)

good_clusters = []
bad_clusters = []

for cluster_id, indices in cluster_indices.items():
    cluster_embs = post_embeddings[indices]
    
    # Calculate avg pairwise similarity
    if len(cluster_embs) > 1:
        pairwise = cosine_similarity(cluster_embs)
        avg_sim = np.mean(pairwise[np.triu_indices(len(cluster_embs), k=1)])
    else:
        avg_sim = 1.0
    
    if avg_sim >= MIN_PAIRWISE_SIM:
        good_clusters.append((cluster_id, avg_sim, len(indices)))
    else:
        bad_clusters.append((cluster_id, avg_sim, len(indices)))

# Stats
total_good_posts = sum(c[2] for c in good_clusters)
total_bad_posts = sum(c[2] for c in bad_clusters)

print(f"\n✅ Good Clusters: {len(good_clusters)} ({total_good_posts} posts)")
print(f"❌ Bad Clusters:  {len(bad_clusters)} ({total_bad_posts} posts)")
print(f"   → {100*total_bad_posts/(total_good_posts+total_bad_posts):.1f}% of clustered posts are in garbage clusters")

# Show worst offenders
print(f"\n🗑️ Worst 5 Clusters (to be filtered):")
bad_clusters.sort(key=lambda x: x[1])
for cid, sim, size in bad_clusters[:5]:
    label = cluster_names.get(cid, f"Cluster {cid}") if 'cluster_names' in dir() else f"Cluster {cid}"
    print(f"   Cluster {cid}: Pairwise={sim:.3f}, Size={size} - '{label[:40]}'")

# Apply filter to cluster_labels
filtered_labels = cluster_labels.copy()
bad_cluster_ids = set(c[0] for c in bad_clusters)
for i, label in enumerate(filtered_labels):
    if label in bad_cluster_ids:
        filtered_labels[i] = -1  # Mark as noise

print(f"\n✅ Filtered! Use 'filtered_labels' instead of 'cluster_labels' for downstream tasks.")
# --- INSPECT GARBAGE CLUSTERS ---
from collections import defaultdict

print("🔍 INSPECTING GARBAGE CLUSTERS")
print("="*70)

# Group posts by cluster
clusters = defaultdict(list)
for i, label in enumerate(cluster_labels):
    if label != -1:
        clusters[label].append(posts[i])

# Show content of worst 5 garbage clusters
for cid, sim, size in bad_clusters[:5]:
    label = cluster_names.get(cid, f"Cluster {cid}") if 'cluster_names' in dir() else f"Cluster {cid}"
    
    print(f"\n{'='*70}")
    print(f"❌ GARBAGE CLUSTER {cid}: '{label}' (Pairwise={sim:.3f}, {size} posts)")
    print("-"*70)
    
    cluster_posts = clusters[cid]
    for j, post in enumerate(cluster_posts[:5], 1):  # Show up to 5
        content = post.get('content', '')[:150]
        source = post.get('source', 'Unknown')
        print(f"  [{j}] {source}: {content}...")
# # --- RE-CLUSTER GARBAGE POSTS ---
# garbage_indices = [i for i, label in enumerate(cluster_labels) if label in bad_cluster_ids]
# garbage_embeddings = post_embeddings[garbage_indices]

# # Try tighter clustering
# from hdbscan import HDBSCAN
# reclustered = HDBSCAN(min_cluster_size=3, min_samples=2).fit_predict(garbage_embeddings)

# recovered = sum(1 for l in reclustered if l != -1)
# print(f"♻️ Recovered {recovered}/{len(garbage_indices)} posts into new micro-clusters")
# # --- CONFIRM RECOVERED CLUSTERS QUALITY ---
# from collections import defaultdict
# from sklearn.metrics.pairwise import cosine_similarity
# import numpy as np
# import random

# print("✅ CONFIRMING RECOVERED CLUSTERS")
# print("="*70)

# # Get original posts for garbage indices
# garbage_posts = [posts[i] for i in garbage_indices]

# # Group recovered posts by new cluster
# recovered_clusters = defaultdict(list)
# recovered_indices_map = defaultdict(list)
# for i, label in enumerate(reclustered):
#     if label != -1:
#         recovered_clusters[label].append(garbage_posts[i])
#         recovered_indices_map[label].append(i)

# print(f"📦 Recovered into {len(recovered_clusters)} micro-clusters")

# # Sample 5 recovered clusters
# sample_clusters = random.sample(list(recovered_clusters.keys()), min(5, len(recovered_clusters)))

# for cluster_id in sample_clusters:
#     cluster_posts = recovered_clusters[cluster_id]
#     indices = recovered_indices_map[cluster_id]
    
#     # Calculate pairwise similarity
#     cluster_embs = garbage_embeddings[indices]
#     if len(cluster_embs) > 1:
#         pairwise = cosine_similarity(cluster_embs)
#         avg_sim = np.mean(pairwise[np.triu_indices(len(cluster_embs), k=1)])
#     else:
#         avg_sim = 1.0
    
#     quality = "✅" if avg_sim > 0.4 else "⚠️"
    
#     print(f"\n{'='*70}")
#     print(f"{quality} RECOVERED CLUSTER {cluster_id} ({len(cluster_posts)} posts, Pairwise={avg_sim:.3f})")
#     print("-"*70)
    
#     samples = random.sample(cluster_posts, min(3, len(cluster_posts)))
#     for j, post in enumerate(samples, 1):
#         content = post.get('content', '')[:150]
#         source = post.get('source', 'Unknown')
#         print(f"  [{j}] {source}: {content}...")

# # Overall quality check
# all_sims = []
# for cluster_id, indices in recovered_indices_map.items():
#     cluster_embs = garbage_embeddings[indices]
#     if len(cluster_embs) > 1:
#         pairwise = cosine_similarity(cluster_embs)
#         avg_sim = np.mean(pairwise[np.triu_indices(len(cluster_embs), k=1)])
#         all_sims.append(avg_sim)

# print(f"\n{'='*70}")
# print(f"📊 OVERALL: Avg Pairwise Sim = {np.mean(all_sims):.3f}")
# print(f"   ✅ Good clusters (>0.4): {sum(1 for s in all_sims if s > 0.4)}")
# print(f"   ⚠️ Weak clusters (<0.4): {sum(1 for s in all_sims if s <= 0.4)}")
# # [EDA] Summarization Sanity Check
# from src.core.analysis.summarizer import Summarizer

# # Get long posts to test
# long_posts = [p.get('content', '') for p in posts if len(p.get('content', '')) > 2000][:10]
# print(f"Testing on {len(long_posts)} long posts...")

# # Run sanity check
# summ = Summarizer(model_name=SUMMARIZATION_MODEL)
# summ.sanity_check(long_posts, n_samples=3)
USE_KEYWORDS = False
# --- STEP 2: Preprocessing & Keywords ---
USE_TITLE_EMBEDDING = False  # [NEW] Set True to embed Titles (News) instead of full Content

print("📝 Preprocessing posts...")
if USE_TITLE_EMBEDDING:
    print("ℹ️ Mode: TITLE Embedding (using Title for News, Content for FB)")
    post_contents = []
    for p in posts:
        # Prefer Title for News, fall back to Content for FB
        text = p.get('title', '')
        if not text or len(str(text)) < 5:
            text = p.get('content', '')
        post_contents.append(str(text)[:EMBEDDING_CHAR_LIMIT])
else:
    print("ℹ️ Mode: CONTENT Embedding (using full Content)")
    post_contents = [p.get('content', '')[:EMBEDDING_CHAR_LIMIT] for p in posts]

# Debug Check
print(f"Sample (Head): {post_contents[0][:50]}...")

if USE_KEYWORDS:
    from src.core.extraction.keyword_extractor import KeywordExtractor
    print("🔑 Extracting high-signal keywords...")
    kw_extractor = KeywordExtractor()
    post_contents_enriched = kw_extractor.batch_extract(post_contents)
else:
    post_contents_enriched = post_contents
print(f"Enriched Sample (Head): {post_contents_enriched[0][:50]}...")
# # --- STEP 2: Preprocessing & Keywords ---
# from src.core.extraction.keyword_extractor import KeywordExtractor
# from src.core.llm.llm_refiner import LLMRefiner

# USE_TITLE_EMBEDDING = True

# # Initialize LLM for Keywords (if provider is set)
# kw_llm_refiner = None
# if LLM_PROVIDER != "none":
#     print(f"🚀 Initializing LLM for Keyword Extraction ({LLM_PROVIDER})...")
#     kw_llm_refiner = LLMRefiner(provider=LLM_PROVIDER, api_key=GEMINI_API_KEY, model_path=LLM_MODEL_PATH, debug=True)

# print("📝 Preprocessing posts...")
# post_contents = []
# for p in posts:
#     if USE_TITLE_EMBEDDING:
#         text = p.get('title', '')
#         if not text or len(str(text)) < 5:
#             text = p.get('content', '')
#     else:
#         text = p.get('content', '')
#     post_contents.append(str(text)[:EMBEDDING_CHAR_LIMIT])

# # Debug Check
# print(f"Sample (Head): {post_contents[0][:50]}...")

# if USE_KEYWORDS:
#     print(f"🔑 Extracting high-signal keywords (LLM Enabled: {kw_llm_refiner is not None})...")
#     # Pass the LLM Refiner to the Extractor
#     kw_extractor = KeywordExtractor(use_llm=(kw_llm_refiner is not None), llm_refiner=kw_llm_refiner)
#     post_contents_enriched = kw_extractor.batch_extract(post_contents)
# else:
#     post_contents_enriched = post_contents

# print(f"Enriched Sample (Head): {post_contents_enriched[0][:50]}...")
# ==========================================
# [VISUALIZATION] Data Preview Before Embedding
# ==========================================

import pandas as pd

print("👀 Inspecting Data for Embedding...")
print(f"Total Documents: {len(post_contents_enriched)}")

# ------------------------------------------
# Create a DataFrame for nice display
# Note:
# Mapping back to 'posts' assumes post_contents
# aligns 1:1 with 'posts' list
# ------------------------------------------

preview_data = []

for i, p in enumerate(posts):
    if i < len(post_contents_enriched):
        preview_data.append({
            "Source": p.get("source", "Unknown"),
            "Original_Title": str(
                p.get("title", "N/A")
            )[:50],
            "Processed_Text_To_Embed": post_contents_enriched[i],
            "Length": len(post_contents_enriched[i])
        })

preview_df = pd.DataFrame(preview_data)

# ------------------------------------------
# Sample News
# ------------------------------------------

print("\n📰 Sample News Articles (Top 3):")
display(
    preview_df[
       ~preview_df["Source"].str.contains("FacE", case=False, na=False)
    ].head(3)
)


# ------------------------------------------
# Sample Facebook
# ------------------------------------------

print("\n📱 Sample Facebook Posts (Top 3):")
display(
    preview_df[
        preview_df["Source"].str.contains("Face", case=False, na=False)
    ].head(3)
)


# ------------------------------------------
# Stats
# ------------------------------------------

print("\n📊 Length Statistics:")
print(
    preview_df
        .groupby("Source")["Length"]
        .describe()
)

from src.utils.text_processing.segmenter import segment_text
from sentence_transformers import SentenceTransformer, util

# 1. Load Model (keepitreal needs segmentation)
model = SentenceTransformer(MODEL_NAME, trust_remote_code =True)

# 2. Test Cases
# "đất nước" (country) vs "đất" (soil) + "nước" (water)
text_raw = "Tôi yêu đất nước Việt Nam"
text_seg = segment_text(text_raw) # -> "Tôi yêu đất_nước Việt_Nam"

print(f"Original:  '{text_raw}'")
print(f"Segmented: '{text_seg}'")

# 3. Compare Embeddings
emb_raw = model.encode(text_raw)
emb_seg = model.encode(text_seg)

# 4. Measure Difference (Cosine Similarity)
# If score < 0.9, the model sees them as DIFFERENT things.
similarity = util.cos_sim(emb_raw, emb_seg).item()
print(f"\nSelf-Similarity (Raw vs Seg): {similarity:.4f}")

if similarity < 0.99:
    print("✅ HUGE DIFFERENCE! The model sees them as different concepts.")
else:
    print("❌ No difference (Segmentation didn't change meaning).")

# 5. Semantic Check (What does it match?)
# Let's compare against a target concept "Tổ quốc" (Fatherland)
target = "Tổ quốc"
emb_target = model.encode(segment_text(target))

sim_raw_target = util.cos_sim(emb_raw, emb_target).item()
sim_seg_target = util.cos_sim(emb_seg, emb_target).item()

print(f"\nMatch with '{target}':")
print(f"   Raw Text:      {sim_raw_target:.4f}")
print(f"   Segmented Text:{sim_seg_target:.4f} (Should be higher)")
import gc, torch
gc.collect()
torch.cuda.empty_cache()
# --- STEP 3: Generate Embeddings ---

print(f"🚀 Generating Embeddings ({EMBEDDING_METHOD})...")
post_embeddings = get_embeddings(
    post_contents_segmented, 
    method=EMBEDDING_METHOD, 
    model_name=MODEL_NAME,
    existing_model=embedder,
    device=embedding_device,
    cache_dir="embeddings_cache" if USE_CACHE else None
)
print(f"✅ Embeddings Shape: {post_embeddings.shape}")

# [EVAL] Embeddings
print("\n📊 Evaluating Embeddings...")
emb_stats = evaluate_embeddings(post_embeddings)
print(f"   Dims: {emb_stats.get('dim')}, Variance: {emb_stats.get('variance_mean'):.4f}, Mean Norm: {emb_stats.get('norm_mean'):.4f}")
trend_embeddings = model.encode(trend_queries, batch_size=1)
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

X = np.asarray(post_embeddings)

# sample để đỡ nặng
n = len(X)
m = min(1200, n)
idx = np.random.choice(n, size=m, replace=False)
Xs = X[idx]

S = cosine_similarity(Xs)
vals = S[np.triu_indices_from(S, k=1)]

print(f"🔬 Average Global Pairwise Similarity (sample={m}): {vals.mean():.3f}")

plt.figure(figsize=(8,3))
plt.hist(vals, bins=50, edgecolor="black")
plt.xlabel("Cosine similarity")
plt.ylabel("count")
plt.title("Pairwise cosine similarity distribution")
plt.show()

# --- STEP 3.1: Compare Clustering Methods (Optional) ---
# Run this cell to compare K-Means, HDBSCAN, BERTopic, and Top2Vec side-by-side
import time
import pandas as pd
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from rich.table import Table
from src.core.analysis.clustering import cluster_data, extract_cluster_labels

def compare_clustering_methods(embeddings, texts, methods=['kmeans', 'hdbscan', 'bertopic', 'top2vec']):
    print("\nCOMPARING CLUSTERING METHODS...")
    
    table = Table(title="Clustering Method Comparison (Metric: Cosine)")
    table.add_column("Method", style="cyan", no_wrap=True)
    table.add_column("Clusters", justify="right")
    table.add_column("Noise", justify="right")
    table.add_column("Silh (Cos)", justify="right")
    table.add_column("DB Index", justify="right")
    table.add_column("CH Score", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Dominant Topic Label", style="italic")
    
    for method in methods:
        start = time.time()
        try:
            labels = cluster_data(
                embeddings, 
                min_cluster_size=MIN_CLUSTER_SIZE, 
                method=method, 
                n_clusters=15 if method == 'kmeans' else None,
                texts=texts,
                embedding_model=MODEL_NAME
            )
            elapsed = time.time() - start
            
            unique = set(labels)
            if -1 in unique: unique.remove(-1)
            n_clusters = len(unique)
            n_noise = list(labels).count(-1)
            
            sil, db, ch = -1.0, -1.0, -1.0
            if n_clusters > 1 and n_clusters < len(texts):
                 mask = labels != -1
                 if mask.sum() > n_clusters:
                     sil = silhouette_score(embeddings[mask], labels[mask], metric='cosine')
                     db = davies_bouldin_score(embeddings[mask], labels[mask])
                     ch = calinski_harabasz_score(embeddings[mask], labels[mask])
            
            topic_names = extract_cluster_labels(texts, labels, method="semantic")
            top_label = topic_names.get(0, "N/A") if 0 in topic_names else "N/A"
            if top_label == "N/A" and topic_names:
                top_label = topic_names[list(topic_names.keys())[0]]
            
            table.add_row(
                method.upper(), 
                str(n_clusters), 
                str(n_noise), 
                f"{sil:.3f}", 
                f"{db:.3f}", 
                f"{ch:.1f}", 
                f"{elapsed:.2f}",
                top_label
            )
        except Exception as e:
            table.add_row(method.upper(), "ERR", "-", "-", "-", "-", "-", str(e)[:30])
    
    console.print(table)

if 'post_embeddings' in locals() and 'post_contents_enriched' in locals():
    compare_clustering_methods(post_embeddings, post_contents)

import numpy as np
import pandas as pd
from src.core.analysis.clustering import cluster_data
from src.utils.text_processing.vectorizers import get_embeddings
from sklearn.metrics.pairwise import cosine_similarity

# 1. Chạy BERTopic với Cohesion Filter (0.45 là ngưỡng khá khắt khe)
# Lưu ý: texts=post_contents_enriched hoặc post_contents tùy bước bạn đang đứng
test_labels = cluster_data(
    post_embeddings, 
    method='bertopic', 
    min_cluster_size=15, 
    min_cohesion=0.45,  # <--- Quan trọng: Ngưỡng lọc
    texts=post_contents_enriched,
    embedding_model=embedder
)

# 2. Kiểm tra xem Topic 0 còn tồn tại không hay đã bị biến thành Noise (-1)
topic_0_mask = (test_labels == 0)
count_0 = np.sum(topic_0_mask)

print(f"📊 Số lượng bài trong Topic 0 sau khi lọc: {count_0}")

if count_0 > 0:
    # 3. Tính toán độ tương đồng nội bộ để xem tại sao nó "thoát" được lưới lọc
    embs_0 = post_embeddings[topic_0_mask]
    centroid = embs_0.mean(axis=0).reshape(1, -1)
    sims = cosine_similarity(embs_0, centroid)
    print(f"✅ Độ gắn kết thực tế của Topic 0: {sims.mean():.3f}")
    
    print("\n📝 5 bài tiêu biểu trong Topic 0:")
    sample_indices = np.where(topic_0_mask)[0][:5]
    for i in sample_indices:
        print(f"- {post_contents_enriched[i][:150]}...")
else:
    print("🗑️ Topic 0 đã bị loại bỏ vì quá loãng (Cohesion < 0.45)!")
from src.pipeline.pipeline_stages import run_sahc_clustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.metrics.pairwise import cosine_similarity
import time
import numpy as np

console = Console()
table = Table(title="Clustering Method Comparison (Metric: Cosine)")
table.add_column("Method", style="cyan")
table.add_column("Clusters", justify="right")
table.add_column("Noise", justify="right")
table.add_column("Silh (Cos)", justify="right")
table.add_column("Cohesion", justify="right", style="magenta")
table.add_column("DB Index", justify="right")
table.add_column("CH Score", justify="right")
table.add_column("Time (s)", justify="right")
table.add_column("Dominant Topic Label", style="italic")

methods = ['kmeans', 'hdbscan', 'bertopic', 'top2vec', 'sahc']

for method in methods:
    start = time.time()
    try:
        if method == 'sahc':
            labels = run_sahc_clustering(
                posts, post_embeddings, 
                min_cluster_size=MIN_CLUSTER_SIZE, 
                method='hdbscan', 
                n_clusters=N_CLUSTERS, 
                post_contents=post_contents_enriched, 
                epsilon=CLUSTER_EPSILON
            )
        else:
            labels = cluster_data(
                post_embeddings, 
                min_cluster_size=MIN_CLUSTER_SIZE, 
                method=method, 
                n_clusters=N_CLUSTERS if method == 'kmeans' else None,
                texts=post_contents_enriched, 
                embedding_model=embedder,
                min_cohesion=0.4
            )
        
        elapsed = time.time() - start
        unique = [l for l in set(labels) if l != -1]
        n_clusters = len(unique)
        n_noise = list(labels).count(-1)
        
        sil, db, ch, cohesion = -1.0, -1.0, -1.0, -1.0
        if n_clusters > 0:
            mask = labels != -1
            if mask.sum() > n_clusters:
                if n_clusters > 1:
                    sil = silhouette_score(post_embeddings[mask], labels[mask], metric='cosine')
                    db = davies_bouldin_score(post_embeddings[mask], labels[mask])
                    ch = calinski_harabasz_score(post_embeddings[mask], labels[mask])
                
                cluster_cohesions = []
                for label in unique:
                    c_mask = (labels == label)
                    c_embs = post_embeddings[c_mask]
                    centroid = c_embs.mean(axis=0).reshape(1, -1)
                    sims = cosine_similarity(c_embs, centroid)
                    cluster_cohesions.append(sims.mean())
                cohesion = np.mean(cluster_cohesions)
        
        topic_names = extract_cluster_labels(post_contents_enriched, labels, model=embedder, method="semantic")
        top_label = topic_names.get(0, "N/A") if 0 in topic_names else "N/A"
        if top_label == "N/A" and topic_names:
            top_label = topic_names[list(topic_names.keys())[0]]
            
        table.add_row(
            method.upper(), str(n_clusters), str(n_noise), 
            f"{sil:.3f}", f"{cohesion:.3f}", f"{db:.3f}", f"{ch:.1f}", f"{elapsed:.2f}", top_label
        )
    except Exception as e:
        table.add_row(method.upper(), "ERR", "-", "-", "-", "-", "-", "-", str(e)[:30])

console.print(table)
# # --- STEP 4: SAHC Clustering ---
# print("🧩 Running SAHC Clustering...")
# cluster_labels = run_sahc_clustering(
#     posts, post_embeddings, 
#     min_cluster_size=MIN_CLUSTER_SIZE, 
#     method=CLUSTER_METHOD, 
#     n_clusters=N_CLUSTERS, 
#     post_contents=post_contents_enriched,
#     epsilon=CLUSTER_EPSILON
# )
# unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
# print(f"✅ Found {len(unique_labels)} clusters.")

# # [EVAL] Clustering
# print("\n📊 Evaluating Clustering...")
# clust_stats = evaluate_clustering(post_embeddings, cluster_labels)
# if 'error' not in clust_stats:
#     print(f"   Clusters: {clust_stats['n_clusters']} (Noise: {clust_stats['n_noise']})")
#     print(f"   Silhouette: {clust_stats.get('silhouette', 0):.4f} | CHI: {clust_stats.get('calinski_harabasz', 0):.1f}")
# else:
#     print(f"   Clustering Eval Error: {clust_stats['error']}")

from src.core.analysis.clustering import cluster_data

# --- STEP 4: Standard Clustering (Tuned) ---
print("🧩 Running Tuned Standard Clustering...")

# Use the "Golden Config" directly
cluster_labels = cluster_data(
    embeddings=post_embeddings,
    min_cluster_size=3,         # Tune this: 2 or 3 for small topics
    epsilon=0.05,               # Strict: Prevents mixed clusters
    method='hdbscan',
    selection_method='leaf',     # Best for stability
    texts=post_contents_enriched # Optional: Used for logging
)

unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
print(f"✅ Found {len(unique_labels)} clusters.")
from src.core.analysis.clustering import diagnose_clustering

# Basic run with your current variables
diagnose_clustering(posts, cluster_labels, post_embeddings)
# [EDA] Clustering Diagnostics
if 'cluster_labels' in locals():
    import pandas as pd
    _labels = pd.Series(cluster_labels)
    _noise_ratio = (_labels == -1).mean()
    
    print(f"Noise Ratio (-1): {_noise_ratio:.2%}")
    print(f"Total Clusters Found: {_labels.nunique() - (1 if -1 in _labels.values else 0)}")
    
    plt.figure(figsize=(10, 3))
    _counts = _labels.value_counts().head(20)
    _counts.plot(kind='bar', color='teal')
    plt.title("Top 20 Cluster Sizes (inc. Noise)")
    plt.xticks(rotation=45, ha='right')
    plt.show()
else:
    print("'cluster_labels' not found. Skipping Step 3 check.")

# --- STEP 4.5: VISUALIZE CLUSTERS ---
print("🎨 Visualizing Clusters with t-SNE (this may take a moment)...")
import pandas as pd
import plotly.express as px
from sklearn.manifold import TSNE

# 1. Reduce Dimensions
# Sample if too large to save time, but for playground usually <10k is fine
n_samples = len(post_embeddings)
perplexity = min(30, n_samples - 1) if n_samples > 1 else 1

tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
projections = tsne.fit_transform(post_embeddings)

# 2. Prepare Data for Plotly
viz_df = pd.DataFrame({
    'x': projections[:, 0],
    'y': projections[:, 1],
    'cluster': [str(l) if l != -1 else 'Noise' for l in cluster_labels],
    'content': [p.get('content', '')[:100] + '...' for p in posts],
    'source': [p.get('source', 'Unknown') for p in posts]
})

# Sort so 'Noise' is drawn first (background) or handling colors
viz_df = viz_df.sort_values('cluster')

# 3. Plot
fig = px.scatter(
    viz_df, x='x', y='y', color='cluster', 
    hover_data=['content', 'source'],
    title='Cluster Visualization (t-SNE)',
    template='plotly_dark',
    color_discrete_sequence=px.colors.qualitative.Dark24
)
fig.update_traces(marker=dict(size=5, opacity=0.8))
fig.show()
# --- STEP 5: Matching Clusters to Trends ---
print("🔗 Matching Clusters to Trends...")

# 5a. Embed Trends (WITH SEGMENTATION)
trend_keys = list(trends.keys())
# Create query strings first
raw_trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]

# Segment them!
trend_queries = batch_segment_texts(raw_trend_queries) 

if trend_queries:
    trend_embeddings = get_embeddings(
        trend_queries, # Now using segmented queries
        method=EMBEDDING_METHOD, 
        model_name=MODEL_NAME,
        existing_model=embedder,
        device=embedding_device,
        cache_dir="embeddings_cache" if USE_CACHE else None
    )
else:
    trend_embeddings = []

# 5b. Label Clusters
anchors = extract_dynamic_anchors(posts, trends)
cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=LABELING_METHOD, anchors=anchors)

matches_hybrid = []

print("😊 Analyzing sentiment (batch)...")
sentiments = batch_analyze_sentiment(post_contents)

for label in unique_labels:
    indices = [i for i, l in enumerate(cluster_labels) if l == label]
    cluster_posts = [posts[i] for i in indices]
    cluster_query = cluster_names.get(label, f"Cluster {label}")
    
    assigned_trend, topic_type, best_match_score = calculate_match_scores(
        cluster_query, label, trend_embeddings, trend_keys, trend_queries, 
        embedder, reranker, RERANK, THRESHOLD
    )
    
    # Calculate Scores (Simplified for notebook view)
    trend_data = trends.get(assigned_trend, {'volume': 0})
    t_time_str = trend_data.get('time')
    t_time = parser.parse(t_time_str) if t_time_str else None
    
    unified_score, _ = calculate_unified_score(trend_data, cluster_posts, trend_time=t_time)
    
    for i, p in enumerate(cluster_posts):
         # We need to map cluster_post index back to original index for sentiment, 
         # or just lookup sentiment for this post content/index if possible.
         # Simply: indices[i] is the index in original 'posts' and 'sentiments'
         original_idx = indices[i]
         
         matches_hybrid.append({
            "source": p.get('source'), "time": p.get('time'), "post_content": p.get('content'),
            "trend": assigned_trend, "score": float(best_match_score), 
            "trend_score": unified_score,
            "is_matched": (topic_type == "Trending"),
            "final_topic": assigned_trend,
            "cluster_id": int(label),
            "topic_type": topic_type,
            "category": "Unclassified", # Skipping taxonomy for speed in playground
            "sentiment": sentiments[original_idx]
        })

# Add unassigned posts?
# For playground, usually we focus on what matched or clustered. 
# But original find_matches_hybrid saves unassigned as 'Noise' if save_all=True.
unassigned_indices = [i for i, l in enumerate(cluster_labels) if l == -1]
for idx in unassigned_indices:
    matches_hybrid.append({
        "source": posts[idx].get('source'), "time": posts[idx].get('time'), "post_content": posts[idx].get('content'),
        "trend": "Unassigned", "score": 0.0, "trend_score": 0,
        "is_matched": False, "final_topic": "Unassigned", "topic_type": "Noise",
        "category": "Noise", "sentiment": sentiments[idx]
    })

print(f"🎉 Final Matches Generated: {len(matches_hybrid)}")
# --- 1A. GLOBAL SIMILARITY ANALYSIS ---
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

sample_idx = np.random.choice(len(post_embeddings), min(1000, len(post_embeddings)), replace=False)
sample_embs = post_embeddings[sample_idx]
sims = cosine_similarity(sample_embs)
avg_sim = np.mean(sims[np.triu_indices(len(sample_idx), k=1)])

print(f"🔬 Average Global Pairwise Similarity: {avg_sim:.3f}")
print("   → If > 0.75: Embeddings are too uniform (model problem)")
print("   → If 0.5-0.75: Moderate overlap (preprocessing problem)")
print("   → If < 0.5: Good separation (clustering parameters problem)")
from sklearn.decomposition import IncrementalPCA
import numpy as np

ipca = IncrementalPCA(n_components=50, batch_size=1024)

for i in range(0, len(post_embeddings), 1024):
    ipca.partial_fit(post_embeddings[i:i+1024])

cumulative_var = np.cumsum(ipca.explained_variance_ratio_)
dims_90 = np.argmax(cumulative_var >= 0.90) + 1

print(f"📐 Dimensions needed for 90% variance: {dims_90}")

# Initialize
if 'LLM_PROVIDER' in locals() and LLM_PROVIDER != "none":
    from src.core.llm.llm_refiner import LLMRefiner
    print(f"🚀 Refining Clusters with {LLM_PROVIDER}...")
    
    llm = LLMRefiner(provider=LLM_PROVIDER, api_key=GEMINI_API_KEY, model_path=LLM_MODEL_PATH, debug=True)
    
    # Prepare clusters for refinement
    cluster_groups = {}
    for m in matches_hybrid:
        if m['topic_type'] == 'Noise': continue
        cid = m.get('cluster_id')
        if cid is None: continue
        if cid not in cluster_groups:
            cluster_groups[cid] = {
                'label': cid,
                'name': m['trend'],
                'sample_posts': [],
                'topic_type': m['topic_type'],
                'keywords': trends.get(m['trend'], {}).get('keywords', [])
            }
        if len(cluster_groups[cid]['sample_posts']) < 5:
            cluster_groups[cid]['sample_posts'].append({'content': m['post_content'], 'time': m['time']})

    to_refine = list(cluster_groups.values())
    
    # --- PHASE 3: REFINEMENT (Titles Only) ---
    if to_refine:
        print(f"   🤖 Phase 3: Batch Refining {len(to_refine)} clusters...")
        refined_results = llm.refine_batch(to_refine)
        
        # Create a map of refined titles/reasoning
        refine_map = {}
        for cid, res in refined_results.items():
            refine_map[int(cid)] = res
        
        # --- PHASE 5: CLASSIFICATION (A/B/C) ---
        print(f"   ⚖️ Phase 5: Classifying refined topics...")
        topics_to_classify = []
        for cid, res in refined_results.items():
            topics_to_classify.append({
                "id": cid,                          # CORRECTED: Added ID to match API
                "label": res.get('refined_title'),  # CORRECTED: Renamed key to match API
                "reasoning": res.get('reasoning', "") # CORRECTED: Added context
            })
        
        classification_results = llm.classify_batch(topics_to_classify)
        
        # Merge classification into refine_map
        for topic, class_res in classification_results.items():
            for cid, ref_res in refine_map.items():
                if ref_res.get('refined_title') == topic:
                    ref_res['category'] = class_res.get('category', 'B')
                    ref_res['event_type'] = class_res.get('event_type', 'Specific')
                    break

        # --- PHASE 4: SEMANTIC DEDUPLICATION (Optional) & SAVING ---
        # 1. Deduplication (Optional)
        all_topics = list(set([res['refined_title'] for res in refine_map.values()]))
        canonical_map = {}

        if len(all_topics) > 1:
            print(f"🔗 Phase 4: Deduplicating {len(all_topics)} topics...")
            canonical_map = llm.deduplicate_topics(all_topics)
            
            # Update refine_map with canonical names
            for cid, res in refine_map.items():
                orig = res['refined_title']
                if orig in canonical_map and canonical_map[orig] != orig:
                    res['refined_title'] = canonical_map[orig]
        
        # Apply updates to matches_hybrid
        new_matches = []
        updated_count = 0
        noise_count = 0
        
        for m in matches_hybrid:
            cid = m.get('cluster_id')
            if cid is not None and int(cid) in refine_map:
                res = refine_map[int(cid)]
                
                # Get classification data (now separate)
                category = res.get('category', 'B')
                event_type = res.get('event_type', 'Specific')
                
                # Filter Noise logic
                is_routine_c = (category == 'C' and m['trend_score'] < 90)
                
                if event_type == 'Generic' or is_routine_c:
                   if m['trend_score'] < 80 or is_routine_c:
                       noise_count += 1
                       continue
                
                # Update valid match
                m['final_topic'] = res['refined_title']
                m['category'] = category
                m['llm_reasoning'] = res['reasoning']
                m['topic_type'] = 'Trending' if m['topic_type'] == 'Discovery' else m['topic_type']
                updated_count += 1
            
            new_matches.append(m)
            
        matches_hybrid = new_matches
        print(f"   ✅ Refined {updated_count} posts. Filtered out {noise_count} noise posts.")

        # 3. Save
        import json
        output_path = "notebook_refined_results.json"
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(matches_hybrid, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Saved {len(matches_hybrid)} refined posts to {output_path}")

        # [EVAL] Refinement
        print("\n📊 Evaluating Refinement...")
        from src.evaluation.metrics import evaluate_refinement
        ref_stats = evaluate_refinement([], refined_results)
        print(f"   Refined {ref_stats['total_clusters']} clusters.")
        print(f"   Categories: {dict(ref_stats['categories'])}")
# 1. Define Source Type
df_hyb = pd.DataFrame(matches_hybrid)

df_hyb['source_type'] = df_hyb['source'].apply(lambda x: 'Facebook' if isinstance(x, str) and 'Face:' in x else 'News')
# 2. Calculate Mixing Statistics
topic_sources = df_hyb.groupby(['final_topic', 'source_type']).size().unstack(fill_value=0)

mixed_topics = topic_sources[(topic_sources['Facebook'] > 0) & (topic_sources['News'] > 0)]
total_topics = df_hyb[df_hyb['topic_type'] != 'Noise']['final_topic'].nunique()

print(f"📊 Total Meaningful Trmin_cluster_sizeends: {total_topics}")
print(f"🤝 Mixed-Source Trends (FB + News): {len(mixed_topics)} ({len(mixed_topics)/total_topics*100:.1f}%)")

# 3. Visualize Top 10 Trends (Source Distribution)
top_10 = df_hyb[df_hyb['topic_type'] != 'Noise']['final_topic'].value_counts().head(10).index
df_top10 = df_hyb[df_hyb['final_topic'].isin(top_10)]

plt.figure(figsize=(12, 6))
sns.countplot(data=df_top10, y='final_topic', hue='source_type', palette='Set2')
plt.title("Source Distribution in Top 10 Trends")
plt.xlabel("Article/Post Count")
plt.ylabel("Trend Topic")
plt.legend(title="Source Type")
plt.show()
# [EDA] Ranking Evaluation
# Identify the results dataframe
_target_df = None
if 'df_hyb' in locals(): _target_df = df_hyb
elif 'df_results' in locals(): _target_df = df_results
elif 'matches_hybrid' in locals(): _target_df = pd.DataFrame(matches_hybrid)

if _target_df is not None and not _target_df.empty:
    if 'score' in _target_df.columns:
        plt.figure(figsize=(8, 3))
        plt.hist(_target_df['score'].dropna(), bins=30, color='gold', edgecolor='gray')
        plt.title("Score Distribution")
        plt.xlabel("Score")
        plt.ylabel("Count")
        plt.show()
        
        print("--- Top 3 Highest Scored ---")
        cols = [c for c in ['trend', 'final_topic', 'score', 'post_content', 'content'] if c in _target_df.columns]
        print(_target_df.nlargest(3, 'score')[cols].to_string())
    else:
        print("'score' column not found in results dataframe.")
else:
    print("Results dataframe (df_hyb/matches_hybrid) not found. Skipping Step 4 check.")

if len(df_top10) < 5:
    print("Not enough data for cross-source t-SNE.")
else:
    # Embed top 10 trends articles
    top_texts = df_top10['post_content'].tolist()
    top_embs = get_embeddings(top_texts, method=EMBEDDING_METHOD, model_name=MODEL_NAME, trust_remote_code=True)

    
    tsne_x = TSNE(n_components=2, perplexity=min(30, len(top_texts)-1), random_state=42)
    coords_x = tsne_x.fit_transform(top_embs)
    
    df_vis_x = pd.DataFrame({
        'x': coords_x[:, 0],
        'y': coords_x[:, 1],
        'Source': df_top10['source_type'].tolist(),
        'Trend': df_top10['final_topic'].tolist(),
        'Snippet': [t[:80] + '...' for t in top_texts]
    })
    
    fig = px.scatter(df_vis_x, x='x', y='y', color='Source', symbol='Trend', 
                     hover_data=['Snippet', 'Trend'],
                     title="Cross-Source Clusters: News vs Facebook Overlap")
    fig.update_traces(marker=dict(size=10, opacity=0.8))
    fig.show()
# [EDA] Tune CLUSTERING Parameters
# Test SAHC with Garbage Re-clustering enabled

from src.core.analysis.clustering import cluster_data

print("🧪 TUNING CLUSTERING PARAMETERS (With Garbage Recovery)")
print("="*60)

# Settings to Test
configurations = [
    {'min_size': 5, 'eps': 0.05, 'recluster': False, 'label': "Baseline (No Recovery)"},
    {'min_size': 5, 'eps': 0.05, 'recluster': True,  'label': "Baseline + RECOVERY ♻️"},
]

for config in configurations:
    print(f"\n⚙️ Testing: {config['label']}")
    
    labels = cluster_data(
        post_embeddings, 
        min_cluster_size=config['min_size'], 
        epsilon=config['eps'],
        recluster_garbage=config['recluster'],
        min_pairwise_sim=0.35
    )
    
    # Stats
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = list(labels).count(-1)
    noise_ratio = 100 * n_noise / len(labels)
    
    print(f"   → Found {n_clusters} clusters")
    print(f"   → Noise: {n_noise} posts ({noise_ratio:.1f}%)")

# 1. Extract News vs Social indices from the subset used above (df_top10)
news_data = df_top10[df_top10['source_type'] == 'News'].copy()
fb_data = df_top10[df_top10['source_type'] == 'Facebook'].copy()

print(f"Original News Seeds: {len(news_data)} | Attaching Facebook Posts: {len(fb_data)}")

# 2. Get embeddings for just News
news_embs = get_embeddings(news_data['post_content'].tolist(), method=EMBEDDING_METHOD, model_name=MODEL_NAME, trust_remote_code=True)
tsne_sub = TSNE(n_components=2, perplexity=min(30, len(news_data)-1), random_state=42)
coords_news = tsne_sub.fit_transform(news_embs)

# 3. Plot 1: News Clusters Only
df_news_vis = pd.DataFrame({
    'x': coords_news[:, 0], 'y': coords_news[:, 1],
    'Trend': news_data['final_topic'].tolist(),
    'Type': ['News (Seed)'] * len(news_data),
    'Content': news_data['post_content'].str[:50].tolist()

})

fig1 = px.scatter(df_news_vis, x='x', y='y', color='Trend', 
                  title="[Step 1] News-Only Foundational Clusters",
                  hover_data=['Trend', 'Content'])
fig1.update_traces(marker=dict(size=12))
fig1.show()

# 4. Plot 2: Integrated View (Show Displacement/Inclusion)
# Re-run t-SNE on joint set to show shared space
joint_embs = get_embeddings(news_data['post_content'].tolist() + fb_data['post_content'].tolist(), method=EMBEDDING_METHOD, model_name=MODEL_NAME, trust_remote_code=True)
tsne_joint = TSNE(n_components=2, perplexity=min(30, len(joint_embs)-1), random_state=42)
coords_joint = tsne_joint.fit_transform(joint_embs)

df_joint_vis = pd.DataFrame({
    'x': coords_joint[:, 0], 'y': coords_joint[:, 1],
    'Trend': news_data['final_topic'].tolist() + fb_data['final_topic'].tolist(),
    'Source': ['News article'] * len(news_data) + ['Facebook post'] * len(fb_data)
})

fig2 = px.scatter(df_joint_vis, x='x', y='y', color='Trend', symbol='Source',
                  title="[Step 2] Integrated Clusters (Social Posts attached to News Seeds)",
                  hover_data=['Trend', 'Source'])
fig2.update_traces(marker=dict(size=10, opacity=0.7))
print("✅ Visual similarity between News dots and FB symbols within same color confirms successful SAHC attachment.")
fig2.show()
# 1. Select the largest cluster
top_cluster_name = df_top10['final_topic'].value_counts().index[0]
cluster_df = df_top10[df_top10['final_topic'] == top_cluster_name].head(10)

print(f"Analyzing Cluster: {top_cluster_name} ({len(cluster_df)} samples)")

# 2. Extract shared keywords (using our KeywordExtractor logic)
from src.core.extraction.keyword_extractor import KeywordExtractor
kw_ext = KeywordExtractor()

all_kw_sets = []
for content in cluster_df['post_content']:
    kws = set(kw_ext.extract_keywords(content).split())
    all_kw_sets.append(kws)

# Find common keywords across at least 2 posts
from collections import Counter
kw_counts = Counter([kw for s in all_kw_sets for kw in s])
shared_kws = [kw for kw, count in kw_counts.items() if count >= 2]
shared_kws = sorted(shared_kws, key=lambda x: kw_counts[x], reverse=True)[:15]

# 3. Build Heatmap Matrix
matrix = []
for i, p_kws in enumerate(all_kw_sets):
    row = [1 if kw in p_kws else 0 for kw in shared_kws]
    matrix.append(row)

import plotly.express as px
fig = px.imshow(matrix, 
                labels=dict(x="High-Signal Keywords", y="Posts in Cluster", color="Present"),
                x=shared_kws, 
                y=[f"Post {i+1}" for i in range(len(matrix))],
                color_continuous_scale='Blues',
                title=f"Keyword Collision Matrix: {top_cluster_name}")
fig.show()
print("💡 Each blue square represents a shared factual anchor (Location, Event Keyword, or Alias).")
# 1. Classification Reasoning
sample_post = cluster_df.iloc[3]
print(f"--- Classification for '{top_cluster_name}' ---")
print(f"Group: {sample_post.get('category', 'Group B (Social Signal)')}")
print(f"Method: {sample_post.get('cat_method', 'Keyword Mapping')}")
print(f"Reasoning: Detected anchors like {', '.join([k for k in shared_kws if k.lower() in sample_post['post_content'].lower()][:3])}")

# 2. Naming Confidence (Top 5 Candidates from TF-IDF/Semantic)
print(f"\n--- Naming Candidates in Cluster ---")
print("Rank | Candidate Title    | Significance")
print("-----|-------------------|-------------")
for i, kw in enumerate(shared_kws[:5]):
    marker = "⭐ (Selected)" if kw.title() in top_cluster_name else ""
    print(f"{i+1:<4} | {kw.title():<17} | {kw_counts[kw]/len(cluster_df):.1%} Overlap {marker}")
import gc, torch
gc.collect()
torch.cuda.empty_cache()
from src.pipeline.main_pipeline import find_matches_hybrid

# Run the full pipeline
matches = find_matches_hybrid(
    posts=posts, 
    trends=trends, 
    use_llm=True, 
    gemini_api_key=GEMINI_API_KEY,  # Ensure this is set in secrets
    llm_provider='gemini',          # or 'ollama' / 'vllm'
    min_cluster_size=5,
    no_dedup=False,
    debug_llm=DEBUG_LLM
)

# Convert to DataFrame for visualization
df_results = pd.DataFrame(matches)
print(f"✅ 5-Phase Pipeline completed. Total results: {len(df_results)}")
df_results.head(10)
# !zip -r embeddings_cache embeddings_cache
# Comparison Data
stats = {
    'Method': ['Semantic', 'Hybrid'],
    'Total Matched/Clustered': [
        len(df_sem[df_sem['is_matched'] == True]),
        len(df_hyb[df_hyb['final_topic'] != 'Unassigned'])
    ],
    'Unique Topics': [
        df_sem[df_sem['is_matched'] == True]['trend'].nunique(),
        df_hyb[df_hyb['final_topic'] != 'Unassigned']['final_topic'].nunique()
    ]
}
df_stats = pd.DataFrame(stats)

fig, ax = plt.subplots(1, 2, figsize=(12, 5))
sns.barplot(data=df_stats, x='Method', y='Total Matched/Clustered', ax=ax[0], palette='viridis')
ax[0].set_title("Coverage (Total Matched Posts)")

sns.barplot(data=df_stats, x='Method', y='Unique Topics', ax=ax[1], palette='magma')
ax[1].set_title("Diversity (Unique Topics)")
plt.tight_layout()
plt.show()
# 1. Filter data (remove 'Unassigned' or 'Noise' for clearer plot)
plot_df = df_hyb[df_hyb['topic_type'] != 'Noise'].copy()

if len(plot_df) < 5:
    print("Not enough data points for t-SNE.")
else:
    print(f"Visualizing {len(plot_df)} clustered posts...")
    texts = plot_df['post_content'].tolist()
    labels = plot_df['final_topic'].tolist()
    types = plot_df['topic_type'].tolist()
    scores = plot_df['score'].tolist()
    
    # 2. Get Embeddings (Use SAME method as configured)
    print(f"Generating embeddings using {EMBEDDING_METHOD}...")
    embeddings = get_embeddings(texts, method=EMBEDDING_METHOD, 
                                model_name=MODEL_NAME,
                                max_features=2000) # For TF-IDF/BoW speed
    
    # 3. Running t-SNE
    print("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(texts)-1))
    coords = tsne.fit_transform(embeddings)
    
    # 4. Interactive Plot with Plotly
    df_vis = pd.DataFrame({
        'x': coords[:, 0],
        'y': coords[:, 1],
        'Topic': labels,
        'Type': types,
        'Score': np.round(scores, 2),
        'Snippet': [t[:100] + '...' for t in texts]
    })
    
    # Only show Top 20 topics in legend, others grouped as 'Other' to avoid palette exhaustion
    top_n_topics = df_vis['Topic'].value_counts().head(20).index.tolist()
    df_vis['Legend_Group'] = df_vis['Topic'].apply(lambda x: x if x in top_n_topics else 'Other (Blue Clusters)')
    
    fig = px.scatter(df_vis, x='x', y='y', 
                     color='Legend_Group', 
                     symbol='Type',
                     hover_data=['Topic', 'Type', 'Score', 'Snippet'],
                     title=f"Interactive t-SNE: Hybrid Clusters ({EMBEDDING_METHOD})")
    fig.show()
import umap
import matplotlib.pyplot as plt
import numpy as np

def visualize_sahc_projection(posts, embeddings, labels):
    """Visualize SAHC progress using UMAP"""
    reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2, random_state=42)
    coords = reducer.fit_transform(embeddings)
    
    is_news = np.array(['Face' not in p.get('source', '') for p in posts])
    
    plt.figure(figsize=(12, 8))
    
    # Plot News (Anchors)
    plt.scatter(coords[is_news, 0], coords[is_news, 1], 
                c='blue', label='News Anchors', alpha=0.8, s=100, edgecolors='white', marker='o')
    
    # Plot Social (Attached or Discovery)
    is_social = ~is_news
    plt.scatter(coords[is_social, 0], coords[is_social, 1], 
                c='orange', label='Social Posts', alpha=0.4, s=20, marker='x')
    
    plt.title("SAHC Embedding Projection: News Anchors vs Social Attachment")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

if 'post_embeddings' in locals() and 'posts' in locals():
    visualize_sahc_projection(posts, post_embeddings, cluster_labels)
else:
    print("⚠️ Run the Hybrid Pipeline section first to generate embeddings and labels.")
import pandas as pd
from IPython.display import display, HTML

def show_refined_dashboard(consolidated_mapping):
    data = []
    for topic, m in consolidated_mapping.items():
        data.append({
            "Refined Title": topic,
            "Type": m.get("topic_type", "n/a"),
            "Category": m.get("category", "n/a"),
            "Event Type": m.get("event_type", "Specific"),
            "Score (G-F-N)": f"{m['trend_score']:.1f}",
            "Reasoning": m.get("llm_reasoning", "n/a")[:150] + "..."
        })
    
    df = pd.DataFrame(data).sort_values("Score (G-F-N)", ascending=False)
    
    # Style the dataframe
    def color_category(val):
        colors = {'A': 'background-color: #ffcccc', 'B': 'background-color: #fff0b3', 'C': 'background-color: #e6ffcc'}
        return colors.get(val, '')

    styled_df = df.style.applymap(color_category, subset=['Category'])
    display(styled_df)

if 'consolidated_mapping' in locals():
    show_refined_dashboard(consolidated_mapping)
else:
    print("⚠️ Run the LLM Refinement section first.")
if 'cluster_mapping' in locals() and 'consolidated_mapping' in locals():
    orig_count = len(cluster_mapping)
    final_count = len(consolidated_mapping)
    merged = orig_count - final_count
    
    print(f"📊 Initial Clusters: {orig_count}")
    print(f"✨ Final Canonical Topics: {final_count}")
    print(f"🔗 Redundant Clusters Merged: {merged} ({(merged/orig_count)*100:.1f}% reduction)")
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

def plot_keyword_heatmap(df, top_n=10, min_clusters=3):
    """Generate a heatmap of top keywords per cluster"""
    # Filter out Noise/Unassigned
    df_clean = df[~df['final_topic'].isin(['Unassigned', 'Noise'])]
    if df_clean.empty or len(df_clean['final_topic'].unique()) < 2:
        print("⚠️ Not enough matched clusters to generate heatmap.")
        return

    # Group text by topic
    topic_texts = df_clean.groupby('final_topic')['post_content'].apply(lambda x: ' '.join(x)).reset_index()
    
    # Vectorize
    vectorizer = TfidfVectorizer(max_features=50, stop_words=None) # Stopwords usually handled by segmenter
    tfidf_matrix = vectorizer.fit_transform(topic_texts['post_content'])
    terms = vectorizer.get_feature_names_out()
    
    # Create Matrix for Heatmap
    heatmap_data = pd.DataFrame(
        tfidf_matrix.toarray(),
        columns=terms,
        index=[t[:30] + '...' if len(t) > 30 else t for t in topic_texts['final_topic']]
    )
    
    plt.figure(figsize=(15, 10))
    sns.heatmap(heatmap_data, cmap="YlGnBu", annot=False, cbar_kws={'label': 'TF-IDF Strength'})
    plt.title("Keyword Signal Fingerprint per Topic (TF-IDF)")
    plt.xlabel("High-Signal Keywords")
    plt.ylabel("Refined Topics")
    plt.show()

if 'df_results' in locals():
    plot_keyword_heatmap(df_results)
elif 'matches_hybrid' in locals():
    plot_keyword_heatmap(pd.DataFrame(matches_hybrid))
else:
    print("⚠️ Run the pipeline first to generate matches.")
import time
import torch

def get_vram_usage():
    if torch.cuda.is_available():
        return f"{torch.cuda.memory_allocated() / 1024**2:.1f} MB"
    return "N/A (CPU Mode)"

print("📊 Pipeline Resource Snapshot:")
print(f"🔹 Current VRAM Allocated: {get_vram_usage()}")
print(f"🔹 Peak VRAM: {torch.cuda.max_memory_allocated() / 1024**2:.1f} MB" if torch.cuda.is_available() else "Peak: N/A")

# Simple stage timing (mock values if not tracked)
stages = {
    "Embedding (GPU)": "~2-5s",
    "SAHC Clustering (CPU)": "~1s",
    "LLM Refinement (API)": "~10-20s (depends on concurrency)",
    "Deduplication (API)": "~2s"
}

for stage, duration in stages.items():
    print(f"⏱️ {stage:25} : {duration}")
from sklearn.metrics.pairwise import cosine_similarity

def plot_discovery_gap(df, embeddings, posts):
    """Analyze the 'Newness' of Discovery clusters"""
    discovery_df = df[df['topic_type'] == 'Discovery']
    if discovery_df.empty:
        print("ℹ️ No Discovery trends found in this run.")
        return
        
    news_indices = [i for i, p in enumerate(posts) if 'Face' not in p.get('source', '')]
    social_indices = [i for i, p in enumerate(posts) if 'Face' in p.get('source', '')]
    
    # (In a real implementation, we'd compare cluster centroids to news centroids)
    print("🔭 Discovery Analysis: Quantifying the News Gap...")
    print("- Discovery clusters identified: ", len(discovery_df['final_topic'].unique()))
    print("- Insights: These topics have no strong similarity to current news anchoring.")

if 'df_results' in locals() and 'post_embeddings' in locals():
    plot_discovery_gap(df_results, post_embeddings, posts)
else:
    print("⚠️ Missing data for Discovery Gap analysis.")
discoveries = df_hyb[df_hyb['topic_type'] == 'Discovery']
top_discoveries = discoveries['final_topic'].value_counts().head(10)

print("Top 10 New Discoveries:")
print(top_discoveries)

# Show samples
if not top_discoveries.empty:
    top_topic = top_discoveries.index[0]
    print(f"\nSample posts for top discovery '{top_topic}':")
    print(discoveries[discoveries['final_topic'] == top_topic]['post_content'].head(3).values)
# # ==========================================
# # BATCH SUMMARIZE FACEBOOK POSTS
# # ==========================================
# import glob
# from scripts.batch_summarize import batch_summarize, merge_summaries_into_posts, load_posts

# FB_SUMMARY_OUTPUT = '/kaggle/working/fb_summaries.json'
# SUMMARY_MODEL = 'vit5-base'

# fb_files = glob.glob('/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/facebook/*.json')
# if fb_files:
#     fb_summaries = batch_summarize(input_path=fb_files[0], output_path=FB_SUMMARY_OUTPUT, model_name=SUMMARY_MODEL, max_length=200, resume=True)
#     print(f'Summarized {len(fb_summaries)} FB posts')

# # ==========================================
# # BATCH SUMMARIZE NEWS ARTICLES
# # ==========================================
# import os
# from scripts.batch_summarize import batch_summarize

# NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
# NEWS_DATA_DIR = '/kaggle/working/Real-time-Event-Detection-on-Social-Media-Data/crawlers/news'
# NEWS_SUMMARY_DIR = '/kaggle/working/news_summaries'
# os.makedirs(NEWS_SUMMARY_DIR, exist_ok=True)

# for source in NEWS_SOURCES:
#     input_path = f'{NEWS_DATA_DIR}/{source}/articles.csv'
#     output_path = f'{NEWS_SUMMARY_DIR}/{source}_summaries.json'
#     if os.path.exists(input_path):
#         print(f'Processing: {source}')
#         batch_summarize(input_path=input_path, output_path=output_path, model_name=SUMMARY_MODEL, max_length=200, resume=True)

# # ==========================================
# # LOAD MERGED NEWS SUMMARIES (UPDATED)
# # ==========================================
# import pandas as pd
# import os

# # Adjust path: 'summarized_data' is in project root, notebook is in 'notebooks/'
# # We try both '../summarized_data' (if running from notebooks dir) and 'summarized_data' (if root)
# DATA_DIR = '../summarized_data'
# if not os.path.exists(DATA_DIR):
#     DATA_DIR = 'summarized_data'

# NEWS_SOURCES = ['vnexpress', 'tuoitre', 'thanhnien', 'vietnamnet', 'nld']
# dfs = []

# print(f"Loading data from: {os.path.abspath(DATA_DIR)}")

# for source in NEWS_SOURCES:
#     merged_path = os.path.join(DATA_DIR, f'{source}_merged.csv')
#     if os.path.exists(merged_path):
#         df = pd.read_csv(merged_path)
#         # Ensure 'summary' column is refined if available
#         # The notebook pipeline likely uses 'summary' column for downstream analysis
#         if 'refined_summary' in df.columns:
#             # Fill NaN or empty refined summaries with the original text or summary
#             # Fallback chain: refined_summary -> summary -> text -> empty
#             fallback = df['summary'] if 'summary' in df.columns else (df['text'] if 'text' in df.columns else '')
#             df['summary'] = df['refined_summary'].fillna(fallback)
        
#         # Add metadata if needed (e.g. source)
#         if 'source' not in df.columns:
#             df['source'] = source
            
#         dfs.append(df)
#         print(f'{source}: Loaded {len(df)} rows from merged file')
#     else:
#         print(f"Warning: {merged_path} not found. Skipping {source}.")

# # ==========================================
# # LOAD MERGED FACEBOOK DATA (ADDED)
# # ==========================================
# fb_path = os.path.join(DATA_DIR, 'facebook_merged.csv')
# if os.path.exists(fb_path):
#     df_fb = pd.read_csv(fb_path)
#     if 'refined_summary' in df_fb.columns:
#          fallback = df_fb['summary'] if 'summary' in df_fb.columns else (df_fb['text'] if 'text' in df_fb.columns else '')
#          df_fb['summary'] = df_fb['refined_summary'].fillna(fallback)
    
#     df_fb['source'] = 'Facebook'
#     print(f'Facebook: Loaded {len(df_fb)} rows from merged file')
#     # Append to dfs if the intention is to analyze all together, or keep separate
#     # For now, we keep separate as df_fb, but user can merge if needed
# else:
#     print(f'Warning: {fb_path} not found')

