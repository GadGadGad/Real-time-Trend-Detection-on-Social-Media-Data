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
            # Count distinct urls or contents if URL empty?
            # Assuming sources without 'Face' are news
            if 'Face' not in item.get('source', ''):
                # Use content hash/snippet as proxy for unique article if url missing
                news_files.add(item.get('post_content')[:50]) 
        
        news_count = len(news_files)
        
        # Normalize N
        # A big event might have 50-100 articles.
        MAX_ARTICLES = 50
        n_score = (news_count / MAX_ARTICLES) * 100
        n_score = min(100, n_score)
        
        # 4. Composite Score
        composite = (self.w_g * g_score) + (self.w_f * f_score) + (self.w_n * n_score)
        
        # 5. Classification
        classification = self._classify(g_score, f_score, n_score)
        
        return {
            "G": round(g_score, 1),
            "F": round(f_score, 1),
            "N": round(n_score, 1),
            "Composite": round(composite, 1),
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
        
        if g > HIGH and f > HIGH and n > HIGH:
            return "Strong Multi-source"
        elif g > HIGH and f > HIGH:
            return "Search & Social"
        elif g > HIGH and n > HIGH:
            return "Search & News"
        elif f > HIGH and n > HIGH:
            return "Social & News"
        elif g > HIGH:
            return "Search-Driven"
        elif f > HIGH:
            return "Social-Driven"
        elif n > HIGH:
            return "News-Driven"
        else:
            return "Emerging / Noise"

def calculate_unified_score(trend_data, cluster_posts):
    """
    Simplified scoring adapter for analyze_trends.py
    trend_data: {'keywords': [], 'volume': float}
    cluster_posts: list of posts
    """
    calc = ScoreCalculator()
    
    # 1. Google (G)
    vol = trend_data.get('volume', 0)
    MAX_VOL = 1000000 
    g_score = (math.log10(vol + 1) / math.log10(MAX_VOL + 1)) * 100 if vol > 0 else 0
    g_score = min(100, g_score)

    # 2. Social (F)
    total_interactions = 0
    for p in cluster_posts:
        stats = p.get('stats', {})
        total_interactions += stats.get('likes', 0)
        total_interactions += stats.get('comments', 0) * 2
        total_interactions += stats.get('shares', 0) * 3
    
    MAX_INTERACTIONS = 20000 
    f_score = (math.log10(total_interactions + 1) / math.log10(MAX_INTERACTIONS + 1)) * 100
    f_score = min(100, f_score)

    # 3. News (N)
    news_count = len([p for p in cluster_posts if 'Face' not in p.get('source', '')])
    MAX_ARTICLES = 50
    n_score = (news_count / MAX_ARTICLES) * 100
    n_score = min(100, n_score)

    # Composite
    score = (0.4 * g_score) + (0.35 * f_score) + (0.25 * n_score)
    
    return round(score, 1), {
        "G": round(g_score, 1),
        "F": round(f_score, 1),
        "N": round(n_score, 1),
        "total_posts": len(cluster_posts)
    }
