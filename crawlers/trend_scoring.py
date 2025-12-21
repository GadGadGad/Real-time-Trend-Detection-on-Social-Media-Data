import csv
import math
import re
from collections import defaultdict
from rich.console import Console

console = Console()

class ScoreCalculator:
    def __init__(self, trends_file=None):
        self.google_trends_volume = {}
        if trends_file:
            self.load_google_trends(trends_file)
            
        # Weights (Configurable)
        self.w_g = 0.4  # Google
        self.w_f = 0.35 # Facebook/Social
        self.w_n = 0.25 # News
        
    def load_google_trends(self, filepath):
        """
        Load Google Trends data from CSV.
        Expected format: "Xu hướng","Lượng tìm kiếm",...
        Example volume: "50 N+" -> 50000, "100+" -> 100
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                reader = csv.reader(f)
                header = next(reader) # Skip header
                
                for row in reader:
                    if len(row) < 2: continue
                    trend_name = row[0].strip()
                    volume_str = row[1].strip()
                    
                    self.google_trends_volume[trend_name.lower()] = self._parse_volume(volume_str)
                    
            console.print(f"[green]Loaded {len(self.google_trends_volume)} trends from Google Trends file.[/green]")
        except Exception as e:
            console.print(f"[red]Error loading Google Trends file: {e}[/red]")

    def _parse_volume(self, quantity_str):
        """Parse volume string like '50 N+' or '200+' to integer."""
        # Remove non-numeric chars except 'N' and '+'
        clean_str = quantity_str.lower().replace(',', '').replace('.', '')
        
        multiplier = 1
        if 'n' in clean_str: # 'ngìn' or 'N' -> thousand
            multiplier = 1000
            clean_str = clean_str.replace('n', '')
        elif 'tr' in clean_str: # 'triệu' -> million (just in case)
             multiplier = 1000000
             clean_str = clean_str.replace('tr', '')
             
        # Extract number
        digits = re.findall(r'\d+', clean_str)
        if not digits:
            return 0
        
        number = int(digits[0])
        return number * multiplier

    def compute_scores(self, trend_name, cluster_items, sources_list):
        """
        Compute G, F, N and Composite scores.
        Returns dictionary of scores and classification.
        """
        # 1. Google Score (G)
        # Log-scale normalization: log(vol) / log(max_vol)
        # Assuming max possible vol is roughly 1M+ for a daily trend in VN, or we track max in dataset.
        # For now, let's use a fixed realistic max ceiling like 2,000,000 (2M) to avoid skew.
        # "50 N+" = 50,000.
        
        vol = self.google_trends_volume.get(trend_name.lower(), 0)
        
        # Max reasonable volume reference (can be adjusted)
        MAX_VOL = 1000000 
        
        if vol > 0:
            # Use log10 to dampen impact of massive spikes
            # Score 0-100
            g_score = (math.log10(vol + 1) / math.log10(MAX_VOL + 1)) * 100
            g_score = min(100, g_score)
        else:
            g_score = 0
            
        # 2. Facebook Score (F)
        # Based on interactions (Likes + Comments + Shares)
        # Normalized by... assuming a viral post can have 10k-100k interactions.
        total_interactions = 0
        fb_sources = [s for s in sources_list if 'face' in s.lower()]
        
        for item in cluster_items:
            # Only count stats if it's social? Or all? FB items have stats.
            if 'stats' in item:
                stats = item['stats']
                total_interactions += stats.get('likes', 0)
                total_interactions += stats.get('comments', 0) * 2 # Weight comments higher?
                total_interactions += stats.get('shares', 0) * 3   # Weight shares higher?
        
        # Normalize F
        # Assume a strong trend has ~10,000 weighted interactions in the crawled window
        MAX_INTERACTIONS = 20000 
        f_score = (math.log10(total_interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
        f_score = min(100, f_score)

        # 3. News Score (N)
        # Based on unique article count
        news_files = set()
        for item in cluster_items:
            if 'Face' not in item.get('source', ''):
                news_files.add(item.get('post_content', '')[:100]) 
        
        news_count = len(news_files)
        
        # Normalize N using log10
        MAX_ARTICLES = 100
        n_score = (math.log10(news_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if news_count > 0 else 0
        n_score = min(100, n_score)
        
        # 4. Synergy Bonus
        # A trend is stronger if it appears in multiple silos.
        active_sources = 0
        if g_score > 10: active_sources += 1
        if f_score > 10: active_sources += 1
        if n_score > 10: active_sources += 1
        
        synergy_mult = 1.0
        if active_sources == 3: synergy_mult = 1.2
        elif active_sources == 2: synergy_mult = 1.1

        # 5. Composite Score
        composite = (self.w_g * g_score) + (self.w_f * f_score) + (self.w_n * n_score)
        final_score = min(100, composite * synergy_mult)
        
        # 6. Classification
        classification = self._classify(g_score, f_score, n_score)
        
        return {
            "G": round(g_score, 1),
            "F": round(f_score, 1),
            "N": round(n_score, 1),
            "Composite": round(final_score, 1),
            "Synergy": synergy_mult,
            "Class": classification,
            "Volume": vol,
            "Interactions": total_interactions,
            "NewsCount": news_count
        }

    def _classify(self, g, f, n):
        """Classify trend based on scores (0-100)."""
        # Thresholds
        HIGH = 40  # Meaningful presence
        LOW = 15   # Minimal presence
        
        if f > HIGH or n > HIGH:
            return "Trending"
        elif g > HIGH and (f > LOW or n > LOW):
            return "Trending"
        else:
            return "Emerging / Noise"

def calculate_unified_score(trend_data, cluster_posts, trend_time=None):
    """
    Simplified scoring adapter for analyze_trends.py
    trend_data: {'keywords': [], 'volume': float}
    cluster_posts: list of posts
    trend_time: datetime object
    """
    import datetime
    from dateutil import parser
    
    calc = ScoreCalculator()
    
    # 1. Google (G)
    vol = trend_data.get('volume', 0)
    MAX_VOL = 1000000 
    g_score = (math.log10(vol + 1) / math.log10(MAX_VOL + 1)) * 100 if vol > 0 else 0
    g_score = min(100, g_score)

    # 2. Social (F)
    total_interactions = 0
    post_times = []
    
    for p in cluster_posts:
        stats = p.get('stats', {})
        total_interactions += stats.get('likes', 0)
        total_interactions += stats.get('comments', 0) * 2
        total_interactions += stats.get('shares', 0) * 3
        
        # Collect timestamps
        try:
            pt = p.get('published_at')
            if pt:
                if isinstance(pt, str):
                    post_times.append(parser.parse(pt))
                elif isinstance(pt, datetime.datetime):
                    post_times.append(pt)
        except: pass
    
    MAX_INTERACTIONS = 20000 
    f_score = (math.log10(total_interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
    f_score = min(100, f_score)

    # 3. News (N)
    news_count = len([p for p in cluster_posts if 'Face' not in p.get('source', '')])
    MAX_ARTICLES = 100
    n_score = (math.log10(news_count + 1) / math.log10(MAX_ARTICLES + 1)) * 100 if news_count > 0 else 0
    n_score = min(100, n_score)

    # 4. Synergy Multiplier
    active_sources = 0
    if g_score > 10: active_sources += 1
    if f_score > 10: active_sources += 1
    if n_score > 10: active_sources += 1
    
    synergy_mult = 1.0
    if active_sources == 3: synergy_mult = 1.2
    elif active_sources == 2: synergy_mult = 1.1

    # 5. Temporal Decay
    temporal_mult = 1.0
    time_reason = ""
    if trend_time and post_times:
        # Calculate median post time
        sorted_times = sorted([t.replace(tzinfo=None) for t in post_times])
        median_post_time = sorted_times[len(sorted_times)//2]
        
        delta = abs((trend_time.replace(tzinfo=None) - median_post_time).total_seconds())
        delta_days = delta / 86400
        
        # Decay: 0.8^days (lose 20% relevance per day of difference)
        temporal_mult = math.pow(0.8, delta_days)
        time_reason = f"Relevance decay: {round(temporal_mult, 2)} (Δ{round(delta_days, 1)} days)"
    elif not post_times and trend_time:
        # Penalize if we have an old trend but NO date data in posts (uncertainty)
        temporal_mult = 0.9

    # Composite
    base_score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
    final_score = min(100, base_score * synergy_mult * temporal_mult)
    
    return round(final_score, 1), {
        "G": round(g_score, 1),
        "F": round(f_score, 1),
        "N": round(n_score, 1),
        "Synergy": synergy_mult,
        "Temporal": round(temporal_mult, 2),
        "TimeNote": time_reason,
        "total_posts": len(cluster_posts)
    }
