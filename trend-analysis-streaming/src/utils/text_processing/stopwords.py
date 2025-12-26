"""
Centralized Stopword Management for the Trend Discovery Pipeline.
Categorizes stopwords into 'Safe', 'Risky', and 'Journalistic' sets 
to allow different levels of filtering for different tasks (Clustering vs Sentiment vs Keywords).
"""

from typing import Set, List, Dict

# --- 1. SAFE STOPWORDS ---
# Pronouns, Demonstratives, Copulas, and basic conjunctions.
SAFE_STOPWORDS = {
    # Pronouns & Titles
    "ông", "bà", "anh", "chị", "em", "tôi", "chúng", "họ", "mình", "ta",
    "bác", "chú", "cô", "dì", "cậu", "mợ", "cụ", "hắn", "nó", "bọn",
    # Demonstratives
    "này", "đó", "kia", "ấy", "nọ", "đây", "kìa", "đấy",
    # Common Verbs (Neutral)
    "là", "có", "làm", "đi", "đến", "cho", "lấy", "về", "vào", "ra", "qua", 
    "lại", "theo", "gồm", "thuộc", "giữ",
    # Conjunctions & Prepositions
    "và", "với", "để", "của", "từ", "tại", "trong", "ngoài", "trên", "dưới",
    "bởi", "vì", "nên", "mà", "hoặc", "hay", "do", "như",
    # Adverbs
    "rất", "quá", "lắm", "hơn", "nhất", "đang", "sẽ", "đã", "vừa", "mới",
    "cũng", "còn", "chỉ", "ngay", "luôn", "thường", "như_vậy", "dường_như",
    # Quantifiers & Numbers
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười",
    "các", "những", "nhiều", "ít", "mọi", "tất_cả", "vài", "mấy", "chút", 
    "toàn_bộ", "đa_số", "phần_lớn",
    # Classifiers
    "cái", "con", "chiếc", "bài", "vụ", "trận", "đấu", "người", "việc", 
    "điều", "món", "tấm", "lá", "quả", "trái", "quyển", "cuốn",
    # High-frequency Unigrams
    "thì", "là", "rằng", "khi", "lúc", "nơi", "chỗ",
}

# --- 2. JOURNALISTIC STOPWORDS ---
# Words that appear heavily in news articles but carry little event-specific signal.
JOURNALISTIC_STOPWORDS = {
    "nêu", "cho_biết", "cho_hay", "theo", "nguồn", "tin", "cho_biết_thêm", 
    "nêu_rõ", "khẳng_định", "nhấn_mạnh", "phát_biểu", "ghi_nhận", 
    "truyền_thông", "báo_chí", "đại_diện", "phát_ngôn", "liên_quan",
    "hôm_nay", "hôm_qua", "ngày_mai", "vừa_qua", "thời_gian_qua", 
    "sắp_tới", "dự_kiến",
    # News Outlets
    "vnexpress", "vtv", "nld", "tuổi_trẻ", "thanh_niên", "dân_trí", "kenh14", 
    "zing", "vietnamnet", "tien_phong", "sggp", "soha", "vtc", "znews", "cafef",
    "theanh28", "tuoitre", "thanhnien", "baomoi", "dantri", "baotuoitre", 
    "baodantri", "vtv_go", "vtv24", "thanh_niên_online", "tuổi_trẻ_online",
    "lao_động", "pháp_luật"
}

# --- 3. RISKY STOPWORDS ---
# These are kept for sentiment/summarization.
RISKY_STOPWORDS = {
    # Negation
    "không", "chẳng", "chả", "chưa", "đừng", "chớ", "không_hề", "chưa_từng",
    # Contrast
    "nhưng", "tuy_nhiên", "song", "ngược_lại", "thay_vì", "dù", "mặc_dù",
    # Impact / Passive
    "bị", "được", "chịu", "mắc",
    # Spatio-Temporal Relations
    "trước", "sau", "dưới", "trên", "giữa"
}

# --- 4. NOISE KEYWORDS (Domain Specific / Pre-Filter) ---
# Used for pre-filtering broad non-event categories like lottery or weather reports.
NOISE_KEYWORDS = {
    # Lottery & Betting
    'xo so', 'xo so mb', 'xo so mn', 'xo so mt', 'xsmb', 'xsmn', 'xsmt', 'vietlott', 
    'so mien bac', 'so mien nam', 'so mien dong', 'so mb', 'so mn', 'so mt',
    'thống kề lô', 'thong ke lo', 'đề hôm nay', 'xspy', 'xshcm', 'xsbd',
    'bet', '88', 'bong88', 'fun88', 'new88', 's666', 'ee88', '188bet', '8xbet', 'w88',
    'bk8', 'livescore', 'socolive', 'xoilac',
    
    # Finance & Market Indicators (Generic)
    'gia vang', 'ti gia', 'lãi suất', 'lai suat', 'thuế thu nhập', 'thue thu nhap', 
    'vnindex', 'chung khoan', 'co phieu', 'giá bạc', 'gia bac', 'giá heo', 'gia heo',
    'crypto', 'bitcoin', 'eth', 'usdt',

    # Weather & Env Features (Generic)
    'weather', 'thoi tiet', 'nhiệt độ', 'nhiet do', 'nhiet do tphcm', 'nhiet do hcm',
    'nhiet do ha noi', 'nhiet do da nang',
    'mưa không', 'mua khong', 'có mưa không', 'dự báo thời tiết', 'du bao thoi tiet',
    'áp thấp nhiệt đới', 'ap thap nhiet doi', 'bão mặt trời', 'bao mat troi',
    'cúp điện', 'cup dien', 'lịch cúp điện',

    # Platforms & Generic Terms
    'code', 'wiki', 'spotify', 'youtube', 'netflix', 'twitch', 'discord', 'instagram', 'facebook', 'tiktok',
    'google', 'gemini', 'claude', 'meta', 'twitter', 'x.com', 'reddit', 'thread',
    'cloudflare', 'disney+', 'k+', 'vtv', 'fpt play', 'tv360', 'my tv', 'vieon',
    'live', 'online', 'stream', 'xem', 'truc tiep', 'ket qua', 'lich thi dau',
    'bxh', 'bang xep hang', 'kqbd',
    'time', 'date', 'doc', 'prep', 'test', 'demo', 'kq', 'cancel',
    'tết', 'nghỉ tết', 'lịch nghỉ',
    
    # Generic News Phrases (as source names often appear as trends)
    'vnexpress', 'tuoi tre', 'thanh nien', 'dan tri', 'kenh14', 'zing', 'bao moi', 
    'vietnamnet', 'vtv', 'tien phong', 'sggp', 'nld', 'nguoi lao dong', 'lao dong'
}

def get_stopwords(level: str = "aggressive", custom_list: List[str] = None) -> Set[str]:
    """
    Get a set of stopwords based on the desired strictness level.
    """
    level = level.lower()
    base_set = set()
    
    if level == "aggressive":
        base_set = SAFE_STOPWORDS | JOURNALISTIC_STOPWORDS | RISKY_STOPWORDS
    elif level == "balanced":
        base_set = SAFE_STOPWORDS | JOURNALISTIC_STOPWORDS
    else:  # safe
        base_set = SAFE_STOPWORDS
        
    if custom_list:
        base_set.update(custom_list)
        
    return base_set

def get_noise_keywords() -> Set[str]:
    """Returns the set of noise keywords for trend pre-filtering."""
    return NOISE_KEYWORDS
