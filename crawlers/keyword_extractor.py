import re
from typing import List, Set, Dict
from collections import Counter
from crawlers.locations import get_known_locations
from crawlers.alias_normalizer import normalize_with_aliases
from crawlers.taxonomy_keywords import get_all_event_keywords

class KeywordExtractor:
    def __init__(self):
        self.known_locations = get_known_locations()
        self.taxonomy_keywords = get_all_event_keywords()
        # Common Vietnamese stopwords (minimal set for extraction)
        self.stopwords = {
            'và', 'của', 'là', 'có', 'trong', 'đã', 'ngày', 'theo', 'với', 
            'cho', 'người', 'những', 'tại', 'về', 'các', 'được', 'ra', 'khi',
            'mới', 'này', 'cho', 'nhiều'
        }

    def extract_keywords(self, text: str, max_keywords: int = 15) -> str:
        """
        Extract high-signal keywords from text.
        Returns a space-separated string of keywords.
        """
        if not text:
            return ""

        # 1. Alias Normalization (Phase 0)
        # This prepends canonical terms if informal ones are found
        text = normalize_with_aliases(text)
        
        # 2. Extract Locations (High Signal)
        found_locations = []
        text_lower = text.lower()
        for loc in self.known_locations:
            if len(loc) > 3 and loc.lower() in text_lower:
                found_locations.append(loc)

        # 3. Extract Taxonomy Keywords (High Signal)
        found_taxonomy = []
        for kw in self.taxonomy_keywords:
            if len(kw) > 3 and kw.lower() in text_lower:
                found_taxonomy.append(kw)

        # 4. Clean and Tokenize
        # Remove special characters, digits, etc.
        clean_text = re.sub(r'[^\w\s]', ' ', text_lower)
        clean_text = re.sub(r'\d+', ' ', clean_text)
        words = clean_text.split()

        # 5. Frequency Analysis
        filtered_words = [w for w in words if len(w) > 2 and w not in self.stopwords]
        word_counts = Counter(filtered_words)
        
        # Get most common topical words
        top_words = [w for w, c in word_counts.most_common(max_keywords)]

        # 6. Combine and Weight
        # Locations get triple weight, Taxonomy keywords get double
        keywords = found_locations * 2 + found_taxonomy * 2 + top_words
        
        # Deduplicate while preserving order (Locations first)
        seen = set()
        final_keywords = []
        for kw in keywords:
            kw_low = kw.lower()
            if kw_low not in seen:
                final_keywords.append(kw)
                seen.add(kw_low)
        
        return " ".join(final_keywords[:max_keywords])

    def batch_extract(self, texts: List[str]) -> List[str]:
        """Process a list of texts into keyword blobs."""
        return [self.extract_keywords(t) for t in texts]

if __name__ == "__main__":
    extractor = KeywordExtractor()
    sample = "Cơn bão số 3 đang gây mưa lớn tại thủ đô Hà Nội. Người dân Sài Gòn đang theo dõi tình hình bão Yagi."
    print(f"Original: {sample}")
    print(f"Keywords: {extractor.extract_keywords(sample)}")
