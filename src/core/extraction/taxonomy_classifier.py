import re
from rich.console import Console
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

console = Console()

from src.core.extraction.taxonomy_keywords import get_flattened_keywords_by_group

# --- 1. KEYWORD DEFINITIONS (T1-T7) ---
KEYWORDS_T1 = get_flattened_keywords_by_group("T1_CRISIS")
KEYWORDS_T2 = get_flattened_keywords_by_group("T2_GOVERNANCE")
KEYWORDS_T3 = get_flattened_keywords_by_group("T3_REPUTATION")
KEYWORDS_T4 = get_flattened_keywords_by_group("T4_MARKET")
KEYWORDS_T5 = get_flattened_keywords_by_group("T5_CULTURE")
KEYWORDS_T6 = get_flattened_keywords_by_group("T6_OPERATIONAL")
KEYWORDS_T7 = get_flattened_keywords_by_group("T7_ROUTINE")

CATEGORY_MAP = {
    "T1": KEYWORDS_T1,
    "T2": KEYWORDS_T2,
    "T3": KEYWORDS_T3,
    "T4": KEYWORDS_T4,
    "T5": KEYWORDS_T5,
    "T6": KEYWORDS_T6,
    "T7": KEYWORDS_T7
}

CATEGORY_DESCRIPTIONS = {
    "T1": "Crisis & Public Risk: Tai nạn, thảm họa thiên nhiên, tội phạm, dịch bệnh, khẩn cấp.",
    "T2": "Policy & Governance: Nghị định, thông tư, chính sách mới, phát biểu của chính phủ.",
    "T3": "Reputation & Trust: Scandal, lùm xùm, tẩy chay, tranh cãi công luận.",
    "T4": "Market Opportunity: Xu hướng tiêu dùng, sản phẩm mới, kinh tế, công nghệ AI.",
    "T5": "Cultural Trend: Meme, người nổi tiếng, giải trí, viral mạng xã hội.",
    "T6": "Operational Pain: Kẹt xe, mất điện, quá tải dịch vụ công, tăng giá xăng.",
    "T7": "Routine Signals: Thời tiết hàng ngày, xổ số, kết quả thể thao định kỳ."
}

def classify_by_keywords(text):
    """
    Classify based on keyword presence in priority order.
    Returns: Category Name (T1..T7) or None
    """
    text_lower = text.lower()
    
    # Priority Order: T1 > T2 > T3 > T6 > T4 > T5 > T7
    priority = ["T1", "T2", "T3", "T6", "T4", "T5", "T7"]
    
    for cat in priority:
        for kw in CATEGORY_MAP[cat]:
            # Use regex for short keywords (<= 3 chars) to ensure word boundaries
            # Vietnamese doesn't use word boundaries like spaces for everything, 
            # but for "AI", "VTV", etc., it works.
            if len(kw) <= 3:
                pattern = rf"\b{re.escape(kw.lower())}\b"
                if re.search(pattern, text_lower):
                    return cat
            else:
                if kw in text_lower:
                    return cat
            
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
        console.print("[dim]🧠 Pre-computing 7-Group Taxonomy Embeddings...[/dim]")
        descriptions = [CATEGORY_DESCRIPTIONS[c] for c in self.categories]
        self.category_embeddings = self.embedder.encode(descriptions)
        
    def classify(self, text, threshold=0.25):
        """
        Hybrid Classification:
        1. Check Keywords (Fast, High Precision).
        2. If None, use Semantic Similarity (High Recall).
        """
        # 1. Keyword Check
        cat_code = classify_by_keywords(text)
        if cat_code:
            return cat_code, "Keyword"
            
        # 2. Semantic Check (if model available)
        if self.embedder and self.category_embeddings is not None:
            text_emb = self.embedder.encode([text])
            sims = cosine_similarity(text_emb, self.category_embeddings)[0]
            best_idx = np.argmax(sims)
            best_score = sims[best_idx]
            
            if best_score > threshold:
                return self.categories[best_idx], "Semantic"
                
        return "Unclassified", "None"
