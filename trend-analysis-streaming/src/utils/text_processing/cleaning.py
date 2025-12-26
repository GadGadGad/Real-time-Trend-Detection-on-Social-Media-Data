import re
import unicodedata

def clean_text(text):
    if not text: return ""
    patterns = [
        r'(?i)\b(cre|credit|via|nguồn)\s*[:.-]\s*.*$', 
        r'(?i)\b(cre|credit)\s+by\s*[:.-]?\s*.*$'
    ]
    cleaned = text
    for p in patterns: 
        cleaned = re.sub(p, '', cleaned)
    return re.sub(r'\s+', ' ', cleaned).strip()

def strip_news_source_noise(text):
    if not text: return ""
    patterns = [
        r'(?i)^Face:\s*[^:]+[-–—:]\s*',
        r'^[\(\[][^\]\)]+[\)\]]\s*[-–—:]\s*',
        r'(?i)^([A-ZÀ-Ỹ0-9.\s-]){2,30}\s*[:\-–—]\s*',
        r'(?i)^(VNEXPRESS|NLD|THANHNIEN|TUOITRE|VIETNAMNET|VTV|ZING|BAOMOI|DANTRI|REUTERS|AFP|TTXVN)\s*[:\-–—]\s*',
        r'^[\(\[][^\]\)]+[\)\]]\s*',
        r'(?i)^(Theo\s+(tin\s+từ\s+)?)?(Reuters|AFP|TTXVN|VNA|AP|BBC|CNN).+?[:\-–—]\s*',
        r'(?i)^\(REUTERS\)\s*[-–—:]?\s*',
        r'(?i)^\(AFP\)\s*[-–—:]?\s*',
        r'(?i)^\(TTXVN\)\s*[-–—:]?\s*',
    ]
    cleaned = text
    for p in patterns:
        cleaned = re.sub(p, '', cleaned, count=1)
    return cleaned.strip()

def normalize_text(text):
    if not text: return ""
    text = unicodedata.normalize('NFC', text.lower())
    text = re.sub(r'[^\w\s]', ' ', text)
    return re.sub(r'\s+', ' ', text).strip()