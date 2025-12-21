import re
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

console = Console()

# --- 1. KEYWORD DEFINITIONS ---
# Group A: Critical Alerts (High Urgency)
KEYWORDS_A = [
    "tai nạn", "cháy", "nổ", "sập", "cứu hỏa", "cứu thương", 
    "bão", "lũ", "lụt", "sạt lở", "động đất", "thiên tai",
    "biểu tình", "bạo loạn", "khủng bố", "giết người", "cướp",
    "dịch bệnh", "covid", "ngộ độc", "khẩn cấp", "cảnh báo",
    "truy nã", "mất tích"
]

# Group B: Social Signals (Monitoring)
KEYWORDS_B = [
    "chính sách", "luật mới", "nghị định", "bầu cử", "tuyên bố",
    "tranh cãi", "phốt", "tẩy chay", "drama", "scandal", "lừa đảo",
    "phản đối", "ý kiến", "góp ý", "cộng đồng mạng", "xôn xao",
    "bức xúc", "khiếu nại", "tố cáo"
]

# Group C: Market Trends (Opportunity)
KEYWORDS_C = [
    "món mới", "ra mắt", "khai trương", "giảm giá", "khuyến mãi",
    "du lịch", "check-in", "review", "trải nghiệm", "hot trend",
    "thời trang", "công nghệ", "điện ảnh", "âm nhạc", "concert",
    "show", "mv", "sản phẩm", "bán chạy", "cháy hàng"
]

CATEGORY_MAP = {
    "Group A": KEYWORDS_A,
    "Group B": KEYWORDS_B,
    "Group C": KEYWORDS_C
}

CATEGORY_DESCRIPTIONS = {
    "Group A": "Tai nạn, thảm họa thiên nhiên, tội phạm, dịch bệnh, khẩn cấp.",
    "Group B": "Chính trị, xã hội, tranh cãi, quan điểm công chúng, drama.",
    "Group C": "Giải trí, tiêu dùng, mua sắm, du lịch, văn hóa, thị trường."
}

def classify_by_keywords(text):
    """
    Classify based on keyword presence.
    Returns: Category Name or None
    """
    text_lower = text.lower()
    
    # Check A (High Priority)
    for kw in KEYWORDS_A:
        if kw in text_lower:
            return "Group A (Critical Alert)"
            
    # Check B
    for kw in KEYWORDS_B:
        if kw in text_lower:
            return "Group B (Social Signal)"
            
    # Check C
    for kw in KEYWORDS_C:
        if kw in text_lower:
            return "Group C (Market Trend)"
            
    return None

class TaxonomyClassifier:
    def __init__(self, embedding_model=None):
        self.embedder = embedding_model
        self.category_embeddings = None
        self.categories = list(CATEGORY_DESCRIPTIONS.keys())
        
        if self.embedder:
            self._precompute_embeddings()
            
    def _precompute_embeddings(self):
        """Pre-compute embeddings for category descriptions"""
        console.print("[dim]🧠 Pre-computing Taxonomy Embeddings...[/dim]")
        descriptions = [CATEGORY_DESCRIPTIONS[c] for c in self.categories]
        self.category_embeddings = self.embedder.encode(descriptions)
        
    def classify(self, text, threshold=0.25):
        """
        Hybrid Classification:
        1. Check Keywords (Fast, High Precision).
        2. If None, use Semantic Similarity (High Recall).
        """
        # 1. Keyword Check
        kw_result = classify_by_keywords(text)
        if kw_result:
            return kw_result, "Keyword"
            
        # 2. Semantic Check (if model available)
        if self.embedder and self.category_embeddings is not None:
            text_emb = self.embedder.encode([text])
            sims = cosine_similarity(text_emb, self.category_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            
            if best_score > threshold:
                cat_name = self.categories[best_idx]
                if cat_name == "Group A": return "Group A (Critical Alert)", "Semantic"
                if cat_name == "Group B": return "Group B (Social Signal)", "Semantic"
                if cat_name == "Group C": return "Group C (Market Trend)", "Semantic"
                
        return "Unclassified", "None"
