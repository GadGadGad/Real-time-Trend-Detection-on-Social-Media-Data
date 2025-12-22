"""
Multi-Source Trend Analysis Pipeline
Matches social & news posts to Google Trends using semantic similarity.
"""

import json
import csv
import re
import os
import glob
import unicodedata
import numpy as np
from datetime import datetime, timedelta
from dateutil import parser
from rich.console import Console
from rich.table import Table
import argparse
from sentence_transformers import SentenceTransformer, CrossEncoder
from sklearn.metrics.pairwise import cosine_similarity
DEFAULT_MODEL = "paraphrase-multilingual-mpnet-base-v2"
console = Console()

# --- PROJECT IMPORTS ---
import sys
import os

# Ensure the parent directory is in path for package imports
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
src_dir = os.path.dirname(current_dir)
project_root = os.path.dirname(src_dir)

if project_root not in sys.path:
    sys.path.insert(0, project_root)
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from src.core.analysis.clustering import cluster_data, extract_cluster_labels
    from src.utils.text_processing.alias_normalizer import normalize_with_aliases, build_alias_dictionary, batch_normalize_texts, normalize_text
    from src.core.extraction.ner_extractor import enrich_text_with_entities, batch_enrich_texts, HAS_NER
    from src.core.analysis.sentiment import batch_analyze_sentiment, clear_sentiment_analyzer
    from src.utils.text_processing.vectorizers import get_embeddings
    from src.core.extraction.taxonomy_classifier import TaxonomyClassifier
    from src.core.llm.llm_refiner import LLMRefiner
    from src.pipeline.trend_scoring import calculate_unified_score
    from src.core.extraction.keyword_extractor import KeywordExtractor
except (ImportError, ModuleNotFoundError) as e:
    console.print(f"[red]Import Error: {e}[/red]")
    sys.exit(1)

def clean_text(text):
    if not text: return ""
    patterns = [r'(?i)\b(cre|credit|via|nguồn)\s*[:.-]\s*.*$', r'(?i)\b(cre|credit)\s+by\s*[:.-]?\s*.*$']
    cleaned = text
    for p in patterns: cleaned = re.sub(p, '', cleaned)
    return cleaned.strip()


def filter_obvious_noise(trends):
    """
    Stage 1: Pre-Filter. Removes lottery, price charts, and generic dates.
    """
    noise_keywords = [
        # Lottery & Betting (High noise)
        'xo so', 'xo so mb', 'xo so mn', 'xo so mt', 'xsmb', 'xsmn', 'xsmt', 'vietlott', 
        'so mien bac', 'so mien nam', 'so mien dong', 'so mb', 'so mn', 'so mt',
        'thống kề lô', 'thong ke lo', 'đề hôm nay', 'xspy', 'xshcm', 'xsbd',
        'bet', '88', 'bong88', 'fun88', 'new88', 's666', 'ee88', '188bet', '8xbet', 'w88',
        
        # Finance & Market Indicators (Generic)
        'gia vang', 'ti gia', 'lãi suất', 'lai suat', 'thuế thu nhập', 'thue thu nhap', 
        'vnindex', 'chung khoan', 'co phieu', 'giá bạc', 'gia bac', 'giá heo', 'gia heo',
        'crypto', 'bitcoin', 'eth', 'usdt',

        # Weather & Env Features (Generic)
        'weather', 'thoi tiet', 'nhiệt độ', 'nhiet do', 'nhiet do tphcm', 'nhiet do hcm', 'nhiet do hcm city', 'nhiet do hcm city'
        'nhiet do ha noi', 'nhiet do da nang',
        'mưa không', 'mua khong', 'có mưa không', 'dự báo thời tiết', 'du bao thoi tiet',
        'áp thấp nhiệt đới', 'ap thap nhiet doi', 'bão mặt trời', 'bao mat troi',
        'cúp điện', 'cup dien', 'lịch cúp điện',

        # Platforms & Generic Terms
        'code', 'wiki', 'spotify', 'youtube', 'netflix', 'twitch', 'discord', 'instagram', 'facebook', 'tiktok',
        'google', 'gemini', 'claude', 'meta', 'twitter', 'x.com', 'reddit', 'thread',
        'cloudflare', 'disney+', 'k+', 'vtv', 'fpt play', 'tv360', 'my tv', 'vieon',
        'live', 'online', 'stream', 'xem', 'truc tiep', 'ket qua', 'lich thi dau',
        'bxh', 'bang xep hang', 'kqbd', 'livescore', 'socolive', 'xoilac',
        'time', 'date', 'doc', 'prep', 'test', 'demo', 'kq', 'cancel', 'bk8',
        'tết', 'nghỉ tết', 'lịch nghỉ',

        # News Outlets (Source names often appear as trends)
        'vnexpress', 'tuoi tre', 'thanh nien', 'dan tri', 'kenh14', 'zing', 'bao moi', 
        'vietnamnet', 'vtv', 'tien phong', 'sggp', 'nld', 'nguoi lao dong', 'lao dong', 
        'soha', 'vtc', '24h', 'cafea', 'cafef', 'cafebiz', 'yan', 'znews' 'dan viet'
    ]

    
    
    # Sports Filtering Config
    # Keep matches involving national interest or major events
    sports_whitelist = [
        # Việt Nam – cách gọi
        'viet nam', 'việt nam', 'vietnam', 'vn',
        'đtvn', 'dtvn', 'doi tuyen viet nam', 'đội tuyển việt nam',
        'tuyen viet nam',

        # Đội trẻ & nữ
        'u15', 'u16', 'u17', 'u18', 'u19', 'u20', 'u21', 'u22', 'u23',
        'tuyển nữ', 'đội tuyển nữ', 'nu viet nam', 'nữ việt nam',

        # Giải khu vực & châu lục (VN rất hay viết)
        'sea games', 'seagame', 'seagames',
        'aff cup', 'asean cup', 'asean championship',
        'vck u23', 'vck u23 chau a',
        'u23 chau a', 'u23 châu á',
        'asian cup', 'vck asian cup',
        'vong loai asian cup',

        # World Cup
        'world cup', 'vong loai world cup',
        'world cup chau a', 'world cup nữ',

        # Olympic
        'olympic', 'olympic paris', 'olympic 2024', 'olympic 2028',
        'vong loai olympic',

        # Euro (báo VN hay viết dù không có VN)
        'euro', 'euro 2024', 'euro 2020', 'euro 2021',

        # Cách viết match báo VN
        'viet nam vs', 'việt nam vs', 'vietnam vs', 'vn vs',
    ]
    # Filter out common weekly club matches (unless they play against VN teams)
    league_blacklist = [
    'premier league', 'epl',
    'la liga',
    'serie a',
    'bundesliga',
    'ligue 1',
    'champions league', 'ucl',
    'europa league',
    'conference league',
]


    club_blacklist = [
        # England
        'man city', 'manchester city', 'mc',
        'man utd', 'man united', 'manchester united', 'mu', 'man u',
        'arsenal', 'chelsea', 'liverpool', 'tottenham', 'spurs',
        'aston villa', 'newcastle', 'west ham', 'fulham',
        'brighton', 'wolves', 'everton', 'bournemouth',
        'crystal palace', 'nottingham forest', 'nottm forest',
        'leicester', 'leeds', 'southampton', 'burnley', 'brentford',

        # Spain
        'real madrid', 'barcelona', 'barca',
        'atletico', 'atlético', 'betis',
        'sevilla', 'valencia', 'villarreal',
        'athletic bilbao', 'ath bilbao',
        'real sociedad', 'getafe', 'osasuna', 'alaves', 'alavés',

        # Italy
        'juventus', 'juve',
        'inter', 'inter milan',
        'milan', 'ac milan',
        'roma', 'as roma',
        'napoli', 'lazio', 'atalanta',
        'bologna', 'fiorentina', 'torino',
        'udinese', 'genoa', 'cagliari', 'sassuolo',

        # Germany
        'bayern', 'bayern munich',
        'dortmund', 'borussia dortmund',
        'leipzig', 'rb leipzig',
        'leverkusen', 'bayer leverkusen',
        'stuttgart', 'werder bremen',
        'union berlin', 'freiburg', 'hoffenheim',

        # France
        'psg', 'paris saint germain',
        'marseille', 'monaco', 'lyon', 'lille', 'nice', 'brest',

        # Others
        'benfica', 'porto', 'sporting', 'ajax', 'psv',
        'galatasaray', 'fenerbahçe', 'celtic',
        'al nassr', 'al-nassr', 'alnassr',
        'inter miami',
    ]


    filtered_trends = {}
    for k, v in trends.items():
        norm_k = normalize_text(k)
        if any(nk in norm_k for nk in noise_keywords):
            continue
        # Remove standalone numbers or pure date patterns
        if re.match(r'^\d+$', norm_k.replace(' ', '')):
            continue
            
        # Sports Logic (Aggressive)
        
        # 1. League Filtering
        if any(l in norm_k for l in league_blacklist):
             if not any(w in norm_k for w in sports_whitelist):
                 continue

        # 2. Club Filtering (Aggressive: Any mention of a blacklisted club)
        # Check against club_blacklist
        found_club = False
        for c in club_blacklist:
            # Use space boundaries for short names to avoid false positives (e.g. "mu" in "music")
            # But for "juve" or "chelsea", substring is usually fine in this context.
            # We'll use simple substring for now as per user intent for "juve"
            if c in norm_k:
                found_club = True
                break
        
        if found_club:
            if not any(w in norm_k for w in sports_whitelist):
                continue

        filtered_trends[k] = v
    
    removed = len(trends) - len(filtered_trends)
    if removed > 0:
        console.print(f"   🧹 Pre-Filter: Removed {removed} noise terms (including club sports).")
    return filtered_trends

def normalize_sports_matches(text):
    """
    Stage 2: Heuristic Normalization for Sports.
    Unifies match formats to facilitate easier merging.
    """
    if not text: return ""
    # "A đấu với B" -> "A vs B"
    text = re.sub(r'(?i)\s+đấu với\s+', ' vs ', text)
    # "atletico madrid" -> "atlético madrid" (canonicalize common misspellings)
    text = re.sub(r'(?i)atletico', 'atlético', text)
    # "real" -> "real madrid" if it looks like a match, only if not already there
    if ' vs ' in text:
        # Use negative lookahead to prevent "real madrid" -> "real madrid madrid"
        text = re.sub(r'\breal\b(?!\s+madrid)', 'real madrid', text, flags=re.IGNORECASE)
        text = re.sub(r'\bars\b(?!\s+enal)', 'arsenal', text, flags=re.IGNORECASE)
        text = re.sub(r'\bmu\b(?!\s+utd|\s+manchester)', 'manchester united', text, flags=re.IGNORECASE)
    return text.strip()

def post_refinement_clean(trends, custom_blacklist=None):
    """
    Stage 3: Post-Filter (User Suggestion). 
    Final cleanup pass using keywords.
    """
    blacklist = custom_blacklist or [
        'xo so', 'tieu dung', 'gia ca', 'thoi tiet', 'tu vi', 
        'lich cup dien', 'ket qua xo so', 'kqxs', 'recap youtube',
        'spotify wrapped', 'xsmb', 'xsmn', 'xsmt', 'vietlott',
        # Platforms & generic services
        'spotify', 'youtube', 'netflix', 'twitch', 'instagram', 'meta', 'gemini', 'claude',
        'cloudflare', 'disney', 'riot', 'epic', 'steam', 'kick',
        # Schedule/results queries (not events themselves)
        'lich thi dau', 'lich', 'schedule', 'bxh', 'bang xep hang', 'kqbd', 'ket qua',
        'livescore', 'socolive', 'xoilac', 'truc tiep', 'xem', 'live'
    ]
    
    final_trends = {}
    for k, v in trends.items():
        norm_k = normalize_text(k)
        if any(b in norm_k for b in blacklist):
            continue
        final_trends[k] = v
        
    removed = len(trends) - len(final_trends)
    if removed > 0:
        console.print(f"   🧹 Post-Filter: Cleaned up {removed} remaining noise terms.")
    return final_trends

def load_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
            unified = []
            for item in data:
                text = item.get('text') or item.get('content') or ''
                time_str = item.get('time') or item.get('time_label') or ''
                # Prioritize structured timestamp if available
                p_timestamp = item.get('timestamp')
                published_at = p_timestamp if p_timestamp else time_str
                
                page_name = item.get('pageName') or item.get('page_name') or 'Unknown'
                unified.append({
                    "source": f"Face: {page_name}",
                    "content": clean_text(text),
                    "title": "",
                    "url": item.get('url') or item.get('postUrl') or '',
                    "stats": item.get('stats') or {'likes': item.get('likes', 0), 'comments': item.get('comments', 0), 'shares': item.get('shares', 0)},
                    "time": time_str,
                    "published_at": published_at,
                    "timestamp": p_timestamp
                })
            return unified
    except Exception as e:
        console.print(f"[red]Error loading JSON {filepath}: {e}[/red]")
        return []

def load_trends(csv_files):
    trends = {}
    for filepath in csv_files:
        # Extract timestamp from filename (e.g., 20251208-1452)
        file_time = None
        match = re.search(r'(\d{8}-\d{4})', os.path.basename(filepath))
        if match:
            try:
                file_time = datetime.strptime(match.group(1), "%Y%m%d-%H%M")
            except: pass
            
        # Support pre-refined JSON
        if filepath.endswith('.json'):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    loaded = json.load(f)
                    # If loaded items don't have time, add file_time
                    for k, v in loaded.items():
                        if 'time' not in v and file_time:
                            v['time'] = file_time.isoformat()
                    trends.update(loaded)
                    console.print(f"[dim]📦 Loaded {len(loaded)} pre-refined trends from {filepath}[/dim]")
                    continue
            except Exception as e:
                console.print(f"[red]Error loading JSON {filepath}: {e}[/red]")
                continue
        
        # Original CSV parsing
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader) 
                for row in reader:
                    if len(row) < 5: continue
                    main_trend = row[0].strip()
                    vol_str = row[1].strip()
                    # Parse volume
                    clean_vol = vol_str.upper().replace(',', '').replace('.', '')
                    multiplier = 1000 if 'N' in clean_vol or 'K' in clean_vol else (1000000 if 'M' in clean_vol or 'TR' in clean_vol else 1)
                    num_parts = re.findall(r'\d+', clean_vol)
                    volume = int(num_parts[0]) * multiplier if num_parts else 0
                    
                    keywords = [k.strip() for k in row[4].split(',') if k.strip()]
                    if main_trend not in keywords: keywords.insert(0, main_trend)
                    trends[main_trend] = {
                        "keywords": keywords, 
                        "volume": volume,
                        "time": file_time.isoformat() if file_time else None
                    }
        except Exception: pass
    return trends

def refine_trends_preprocessing(trends, llm_provider, gemini_api_key, llm_model_path, debug_llm, source_files=None, cache_path=None):
    """
    Dedicated preprocessing step for Google Trends.
    Checks for cache based on input files.
    """
    # 1. Check for cache
    if cache_path and os.path.exists(cache_path):
         try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                console.print(f"[green]📂 Loading trend refinement from MANUAL cache: {os.path.basename(cache_path)}[/green]")
                return json.load(f)
         except Exception as e:
             console.print(f"[red]Error loading manual cache: {e}[/red]")

    computed_cache_path = None
    if source_files:
        from hashlib import md5
        # Normalize to absolute paths for stability against CWD
        abs_files = sorted([os.path.abspath(f) for f in source_files])
        if debug_llm:
            console.print(f"[dim]DEBUG: Hashing files for cache: {abs_files}[/dim]")
        
        combined_path = "".join(abs_files)
        cache_id = md5(combined_path.encode()).hexdigest()
        computed_cache_path = os.path.join(project_root, "cache", f"trend_refine_{cache_id}.json")
        if os.path.exists(computed_cache_path):
            try:
                with open(computed_cache_path, 'r', encoding='utf-8') as f:
                    console.print(f"[green]📂 Loading trend refinement from cache: {os.path.basename(computed_cache_path)}[/green]")
                    return json.load(f)
            except Exception: pass
    
    # Use computed path for saving later if manual path not provided
    save_cache_path = cache_path if cache_path else computed_cache_path

    # --- Start Multi-Stage Filtering ---
    
    # Stage 1: Pre-Filter Noise
    trends = filter_obvious_noise(trends)
    
    # Stage 2: Heuristic Normalization (Normalize keys before LLM)
    normalized_trends = {}
    for k, v in trends.items():
        norm_key = normalize_sports_matches(k)
        if norm_key in normalized_trends:
            # Merge if normalization creates a collision
            normalized_trends[norm_key]['volume'] += v['volume']
            normalized_trends[norm_key]['keywords'] = list(set(normalized_trends[norm_key]['keywords'] + v['keywords']))
        else:
            normalized_trends[norm_key] = v
    trends = normalized_trends

    # 2. Refine using LLM
    refiner = LLMRefiner(
        provider=llm_provider,
        api_key=gemini_api_key,
        model_path=llm_model_path,
        debug=debug_llm
    )
    
    if not refiner.enabled:
        return post_refinement_clean(trends) # Still clean if LLM disabled

    # Map normalized names to original keys for robust matching
    norm_map = {normalize_text(k): k for k in trends.keys()}
    
    result = refiner.refine_trends(trends)
    if result:
        filtered = result.get("filtered", [])
        merged = result.get("merged", {})
        
        refined_trends = trends.copy()
        
        # Apply Filtering (Case-insensitive & Diacritic-robust)
        for f_term in filtered:
            norm_f = normalize_text(f_term)
            target_key = norm_map.get(norm_f)
            if not target_key:
                # Check for partial match
                for norm_k, orig_k in norm_map.items():
                    if norm_f in norm_k or norm_k in norm_f:
                        target_key = orig_k
                        break
            if target_key and target_key in refined_trends:
                del refined_trends[target_key]

        # Apply Merging (Case-insensitive & Diacritic-robust)
        norm_key_map = {normalize_text(k): k for k in refined_trends.keys()}

        for variant, canonical in merged.items():
            if isinstance(canonical, list):
                canonical = canonical[0] if canonical else ""
            if not isinstance(canonical, str): continue
            
            norm_v = normalize_text(variant)
            norm_c = normalize_text(canonical)
            var_key = norm_key_map.get(norm_v)
            can_key = norm_key_map.get(norm_c)
            
            if not var_key:
                for norm_k, orig_k in norm_key_map.items():
                    if norm_v in norm_k:
                        var_key = orig_k
                        break

            if var_key and can_key and var_key != can_key:
                if var_key in refined_trends and can_key in refined_trends:
                    refined_trends[can_key]['volume'] += refined_trends[var_key]['volume']
                    refined_trends[can_key]['keywords'] = list(set(refined_trends[can_key]['keywords'] + refined_trends[var_key]['keywords']))
                    del refined_trends[var_key]
                    norm_key_map = {normalize_text(k): k for k in refined_trends.keys()}

        # Stage 3: Post-Refinement Cleanup (User Suggestion)
        refined_trends = post_refinement_clean(refined_trends)
        
        console.print(f"\n   ✨ Refinement Complete: {len(trends)} -> {len(refined_trends)} trends.")
        
        # 3. Save to cache
        # 3. Save to cache
        if save_cache_path:
            os.makedirs(os.path.dirname(save_cache_path), exist_ok=True)
            with open(save_cache_path, 'w', encoding='utf-8') as f:
                json.dump(refined_trends, f, ensure_ascii=False, indent=2)
            console.print(f"   💾 Saved refinement result to cache.")
            
        return refined_trends

    return trends

def extract_dynamic_anchors(posts, trends, top_n=20, include_locations=True):
    from sklearn.feature_extraction.text import CountVectorizer
    from src.utils.config.locations import get_known_locations
    trend_kws = set()
    for t in trends.values():
        for kw in t.get('keywords', []): trend_kws.add(kw.lower())
    
    # Optional: Add major provinces as potential anchors
    if include_locations:
        for loc in get_known_locations():
            if len(loc) > 3: trend_kws.add(loc.lower())
    texts = [p.get('content', '').lower() for p in posts]
    vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=1000)
    try:
        X = vectorizer.fit_transform(texts)
        post_words = vectorizer.get_feature_names_out()
        word_counts = X.toarray().sum(axis=0)
        anchors = sorted([(w, c) for w, c in zip(post_words, word_counts) if w in trend_kws], key=lambda x: -x[1])
        return [a[0] for a in anchors[:top_n]]
    except: return []

def apply_guidance_enrichment(text, anchors):
    found = [a for a in anchors if a in text.lower()]
    if found: return f"{' '.join(found * 2)} | {text}"
    return text

def find_matches_hybrid(posts, trends, model_name=None, threshold=0.5, 
                        use_aliases=True, use_ner=False, 
                        embedding_method="sentence-transformer", save_all=False,
                        rerank=True, min_cluster_size=5, labeling_method="semantic",
                        reranker_model_name=None, use_llm=False, gemini_api_key=None,
                        llm_provider="gemini", llm_model_path=None,
                        llm_custom_instruction=None, use_cache=True,
                        debug_llm=False, summarize_all=False, no_dedup=False,
                        use_keywords=False, use_llm_keywords=False,
                        cluster_epsilon=0.15):
    if not posts: return []
    
    # KeywordExtractor is already imported at top level
    
    # In Sequential mode, we use CUDA for everything, but one at a time.
    import torch
    embedding_device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    console.print(f"🚀 [cyan]Phase 1: High-Speed Embeddings & Sentiment on {embedding_device}...[/cyan]")
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL, device=embedding_device)
    
    # --- PHASE 1: EMBEDDINGS ---
    
    taxonomy_clf = TaxonomyClassifier(embedding_model=embedder) if TaxonomyClassifier else None
    reranker = None
    if rerank:
        ce_model = reranker_model_name or 'cross-encoder/ms-marco-MiniLM-L-6-v2'
        try: reranker = CrossEncoder(ce_model, device=embedding_device)
        except: pass

    post_contents = [p.get('content', '')[:500] for p in posts]
    anchors = extract_dynamic_anchors(posts, trends)
    
    if use_aliases:
        build_alias_dictionary({k: v['keywords'] for k, v in trends.items()})
        post_contents_enriched = batch_normalize_texts(post_contents)
    else:
        post_contents_enriched = post_contents

    if anchors:
        post_contents_enriched = [apply_guidance_enrichment(t, anchors) for t in post_contents_enriched]

    if use_keywords:
        console.print("[cyan]🔑 Phase 0.5: Extracting high-signal keywords...[/cyan]")
        if use_llm_keywords and use_llm:
             from src.core.llm.llm_refiner import LLMRefiner
             kw_llm = LLMRefiner(provider=llm_provider, api_key=gemini_api_key, model_path=llm_model_path)
             kw_extractor = KeywordExtractor(use_llm=True, llm_refiner=kw_llm)
        else:
             kw_extractor = KeywordExtractor()
        extracted_kws = kw_extractor.batch_extract(post_contents_enriched)
        # Concatenate keywords with original text for richer embeddings (Context + Signal)
        post_contents_enriched = [f"{k}. {t}" if k else t for k, t in zip(extracted_kws, post_contents_enriched)]

    # --- PHASE 0: SUMMARIZATION ---
    from src.pipeline.pipeline_stages import run_summarization_stage, run_sahc_clustering, calculate_match_scores
    
    post_contents_enriched = run_summarization_stage(post_contents_enriched, use_llm, summarize_all)

    post_embeddings = get_embeddings(
        post_contents_enriched, 
        method=embedding_method, 
        model_name=model_name,
        existing_model=embedder,
        device=embedding_device,
        cache_dir="embeddings_cache" if use_cache else None
    )

    # --- SAHC PHASE 1-3: CLUSTERING ---
    cluster_labels = run_sahc_clustering(posts, post_embeddings, min_cluster_size=min_cluster_size, epsilon=cluster_epsilon)
    unique_labels = sorted([l for l in set(cluster_labels) if l != -1])
    sentiments = batch_analyze_sentiment(post_contents)

    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    trend_embeddings = get_embeddings(
        trend_queries, 
        method=embedding_method, 
        model_name=model_name,
        existing_model=embedder,
        device=embedding_device,
        cache_dir="embeddings_cache" if use_cache else None
    ) if trend_queries else []
    
    cluster_names = extract_cluster_labels(post_contents, cluster_labels, model=embedder, method=labeling_method, anchors=anchors)
    cluster_mapping = {}

    for label in unique_labels:
        indices = [i for i, l in enumerate(cluster_labels) if l == label]
        cluster_posts = [posts[i] for i in indices]
        cluster_query = cluster_names.get(label, f"Cluster {label}")
        assigned_trend, topic_type, best_match_score = calculate_match_scores(
            cluster_query, label, trend_embeddings, trend_keys, trend_queries, 
            embedder, reranker, rerank, threshold
        )

        trend_data = trends.get(assigned_trend, {'volume': 0})
        # Extract trend time
        t_time_str = trend_data.get('time')
        t_time = parser.parse(t_time_str) if t_time_str else None
        
        unified_score, components = calculate_unified_score(trend_data, cluster_posts, trend_time=t_time)
        category, category_method = taxonomy_clf.classify(cluster_query + " " + assigned_trend) if taxonomy_clf else ("Unclassified", "None")
        
        llm_reasoning = ""
        final_topic_name = assigned_trend if assigned_trend != "Discovery" else f"New: {cluster_query}"
        
        cluster_mapping[label] = {
            "final_topic": final_topic_name, "topic_type": topic_type, "cluster_name": cluster_query,
            "category": category, "category_method": category_method, "match_score": best_match_score,
            "trend_score": unified_score, "score_components": components, "llm_reasoning": "",
            "posts": cluster_posts # Temporary for LLM
        }

    # --- PHASE 2: SEQUENTIAL GPU CLEANUP ---
    if use_llm and embedding_device == 'cuda' and llm_provider != 'gemini':
        console.print("[yellow]🧹 Phase 2: Unloading Phase 1 models to free VRAM for LLM...[/yellow]")
        if 'embedder' in locals(): del embedder
        if 'reranker' in locals(): del reranker
        if 'taxonomy_clf' in locals(): del taxonomy_clf
        clear_sentiment_analyzer()
        import gc
        gc.collect()
        torch.cuda.empty_cache()

    # --- PHASE 3: BATCH LLM REFINEMENT ---
    console.print(f"🚀 [cyan]Phase 3: LLM Refinementpass using {llm_provider}...[/cyan]")
    llm_refiner = LLMRefiner(provider=llm_provider, api_key=gemini_api_key, model_path=llm_model_path, debug=debug_llm) if use_llm else None
    if llm_refiner:
        to_refine = []
        for l, m in cluster_mapping.items():
            if m["topic_type"] == "Discovery" or m["trend_score"] > 30:
                # Include trend keywords if available to help grounding
                keywords = []
                if m["final_topic"] in trends:
                    keywords = trends[m["final_topic"]].get('keywords', [])
                
                to_refine.append({
                    "label": l, "name": m["cluster_name"], "topic_type": m["topic_type"],
                    "category": m["category"], "sample_posts": m["posts"],
                    "keywords": keywords
                })
        
        if to_refine:
            console.print(f"   🤖 [cyan]Batch Refining {len(to_refine)} clusters with {llm_provider}...[/cyan]")
            batch_results = llm_refiner.refine_batch(to_refine, custom_instruction=llm_custom_instruction)
            
            for l, res in batch_results.items():
                label_key = int(l) if isinstance(l, (str, int)) else l
                if label_key in cluster_mapping:
                    m = cluster_mapping[label_key]
                    if m["topic_type"] == "Discovery":
                        m["final_topic"] = f"New: {res['refined_title']}"
                    m["category"] = res["category"]
                    m["event_type"] = res.get("event_type", "Specific") # Default to specific if missing
                    m["category_method"] = "LLM"
                    m["llm_reasoning"] = res["reasoning"]

                    # FILTER: Downgrade "Generic" events or routine Category C unless they are massive viral hits
                    is_routine_c = (res["category"] == "C" and m["trend_score"] < 90)
                    if m["event_type"] == "Generic" or is_routine_c:
                        if m["trend_score"] < 80 or is_routine_c:
                            m["topic_type"] = "Noise"
                            reason = "Routine Category C" if is_routine_c else "Generic"
                            m["category"] = f"{reason}/Routine"
                            m["final_topic"] = f"[{reason}] {res['refined_title']}"
                            if debug_llm:
                                console.print(f"      🗑️  Downgraded cluster {l} to Noise ({reason}): {res['refined_title']}")
                        else:
                            # Keep it but mark it
                            m["final_topic"] = f"Viral: {res['refined_title']}"
            
            success_count = len(batch_results)
            console.print(f"   ✅ [bold green]LLM Pass Complete: Successfully refined {success_count}/{len(to_refine)} clusters.[/bold green]")

        # --- PHASE 4: SEMANTIC DEDUPLICATION ---
        if not no_dedup:
            console.print("🔗 [cyan]Phase 4: Semantic Topic Deduplication...[/cyan]")
        all_topics = [m["final_topic"] for m in cluster_mapping.values() if m["topic_type"] != "Discovery"]
        if all_topics:
            canonical_map = llm_refiner.deduplicate_topics(all_topics)
            dedup_count = 0
            for label, m in cluster_mapping.items():
                orig = m["final_topic"]
                if orig in canonical_map and canonical_map[orig] != orig:
                    m["final_topic"] = canonical_map[orig]
                    dedup_count += 1
            
            if dedup_count > 0:
                console.print(f"   ✨ [green]Merged {dedup_count} clusters into canonical topics.[/green]")

    # Consolidated Mapping
    consolidated_mapping = {}
    topic_groups = {}
    for l, m in cluster_mapping.items():
        t = m["final_topic"]
        if t not in topic_groups: topic_groups[t] = []
        topic_groups[t].append(l)

    for topic, labels in topic_groups.items():
        all_posts = []
        for l in labels:
            idx = [i for i, val in enumerate(cluster_labels) if val == l]
            all_posts.extend([posts[i] for i in idx])
        
        m = cluster_mapping[labels[0]]
        t_data = trends.get(topic, {'volume': 0})
        t_time_str = t_data.get('time')
        t_time = parser.parse(t_time_str) if t_time_str else None
        
        combined_score, combined_comp = calculate_unified_score(t_data, all_posts, trend_time=t_time)
        consolidated_mapping[topic] = {**m, "trend_score": combined_score, "score_components": combined_comp}

    matches = []
    for i, post in enumerate(posts):
        label = cluster_labels[i]
        if label != -1:
            t_name = cluster_mapping[label]["final_topic"]
            m = consolidated_mapping[t_name]
            
            # Filter Noise unless debug
            if m["topic_type"] == "Noise" and not save_all:
                continue

            matches.append({
                "source": post.get('source'), "time": post.get('time'), "post_content": post.get('content'),
                "final_topic": m["final_topic"], "topic_type": m["topic_type"], "category": m["category"],
                "score": m["match_score"], "trend_score": m["trend_score"], "llm_reasoning": m["llm_reasoning"],
                "sentiment": sentiments[i], "is_matched": (m["topic_type"] == "Trending"), "trend": m["final_topic"],
                "embeddings": post_embeddings[i], "trend_embeddings": trend_embeddings[best_idx]
            })
        elif save_all:
            matches.append({"final_topic": "Unassigned", "topic_type": "Noise", "sentiment": sentiments[i], "is_matched": False,
                            "embeddings": post_embeddings[i], "trend_embeddings": trend_embeddings[best_idx]})

    return matches

def find_matches(posts, trends, model_name=None, threshold=0.35, 
                 use_aliases=True, embedding_method="sentence-transformer", save_all=False):
    """Baseline semantic matching without clustering."""
    embedder = SentenceTransformer(model_name or DEFAULT_MODEL)
    trend_keys = list(trends.keys())
    trend_queries = [" ".join(trends[t]['keywords']) for t in trend_keys]
    
    post_contents = [p.get('content', '')[:500] for p in posts]
    post_embeddings = get_embeddings(post_contents, method=embedding_method, model_name=model_name)
    trend_embeddings = embedder.encode(trend_queries)
    
    sims = cosine_similarity(post_embeddings, trend_embeddings)
    matches = []
    for i, post in enumerate(posts):
        best_idx = np.argmax(sims[i])
        best_score = sims[i][best_idx]
        topic = trend_keys[best_idx] if best_score >= threshold else "Unassigned"
        if topic != "Unassigned" or save_all:
            matches.append({
                "source": post.get('source'), "time": post.get('time'), "post_content": post.get('content'),
                "trend": topic, "score": float(best_score), "is_matched": (topic != "Unassigned"),
                "final_topic": topic, "embeddings": post_embeddings[i], "trend_embeddings": trend_embeddings[best_idx]
            })
    return matches

# Aliases for notebook compatibility
def load_social_data(files):
    all_data = []
    for f in files: all_data.extend(load_json(f))
    return all_data

def load_news_data(files):
    all_data = []
    for f in files:
        try:
            with open(f, 'r', encoding='utf-8') as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    content_str = f"{row.get('title', '')}\n{row.get('content', '')}"
                    if not content_str.strip(): # Skip empty content
                        continue
                        
                    all_data.append({
                        "source": os.path.basename(os.path.dirname(f)).upper(),
                        "content": content_str,
                        "title": row.get('title', ''), "url": row.get('url', ''),
                        "stats": {'likes': 0, 'comments': 0, 'shares': 0},
                        "time": row.get('published_at', '')
                    })
        except: pass
    return all_data

load_google_trends = load_trends

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Source Trend Analysis")
    parser.add_argument("--social", type=str, nargs="+", help="Social JSON files")
    parser.add_argument("--news", type=str, nargs="+", help="News CSV files")
    parser.add_argument("--trends", type=str, nargs="+", help="Google Trends CSV files")
    parser.add_argument("--llm", action="store_true", help="Enable LLM refinement")
    parser.add_argument("--summarize-all", action="store_true", help="Summarize ALL posts before clustering")
    parser.add_argument("--llm-provider", type=str, default="gemini", choices=["gemini", "kaggle", "local"], help="LLM Provider")
    parser.add_argument("--llm-model-path", type=str, help="Local path or HF ID for local LLM")
    parser.add_argument("--llm-instruction", type=str, help="Custom instructions for LLM refinement")
    
    parser.add_argument("--no-dedup", action="store_true", help="Disable Semantic Deduplication (Phase 4)")
    parser.add_argument("--refine-trends", action="store_true", help="Refine Google Trends inputs before matching")
    parser.add_argument("--use-keywords", action="store_true", help="Extract high-signal keywords before clustering")
    parser.add_argument("--use-llm-keywords", action="store_true", help="Use LLM for keyword extraction (requires --llm)")
    parser.add_argument("--trend-cache-path", type=str, help="Manually specify cache file for trend refinement")
    
    args = parser.parse_args()
    
    if args.social and args.trends:
        # Sort inputs for deterministic processing and cache stability
        if args.social: args.social = sorted(args.social)
        if args.trends: args.trends = sorted(args.trends)
        if args.news: args.news = sorted(args.news)

        social_posts = []
        for f in args.social: social_posts.extend(load_json(f))
        trends = load_trends(args.trends)
        
        if args.refine_trends:
            # Pass args.trends to the refiner for caching
            trends = refine_trends_preprocessing(
                trends, 
                args.llm_provider, 
                os.getenv("GEMINI_API_KEY"), # Assuming API key might be in env or passed
                args.llm_model_path, 
                False, # debug
                source_files=args.trends,
                cache_path=args.trend_cache_path
            )

        results = find_matches_hybrid(
            social_posts, trends, 
            use_llm=args.llm, 
            llm_provider=args.llm_provider, 
            llm_model_path=args.llm_model_path,
            llm_custom_instruction=args.llm_instruction,
            summarize_all=args.summarize_all,
            no_dedup=args.no_dedup,
            use_keywords=args.use_keywords,
            use_llm_keywords=args.use_llm_keywords
        )
        print(f"Analyzed {len(social_posts)} posts. Found {len(set(r['final_topic'] for r in results))} trends.")
