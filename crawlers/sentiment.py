import torch
from transformers import pipeline
from rich.console import Console

console = Console()

# Lazy Load Model
_sentiment_analyzer = None

def get_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        console.print("[bold yellow]🧠 Loading PhoBERT Sentiment Model...[/bold yellow]")
        try:
            device = 0 if torch.cuda.is_available() else -1
            _sentiment_analyzer = pipeline("sentiment-analysis", model="wonrax/phobert-base-vietnamese-sentiment", device=device)
        except Exception as e:
            console.print(f"[red]Failed to load sentiment model: {e}. using fallback.[/red]")
            _sentiment_analyzer = "fallback"
    return _sentiment_analyzer

def analyze_sentiment(text):
    """
    Analyze sentiment using PhoBERT.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    analyzer = get_analyzer()
    if analyzer == "fallback":
        return _fallback_sentiment(text)
        
    try:
        # Truncate text to 256 chars for speed/limit
        result = analyzer(text[:512])[0]
        label = result['label'] # 'POS', 'NEG', 'NEU' usually
        
        # Map labels
        if label in ['POS', 'POSITIVE']:
            return 'Positive'
        elif label in ['NEG', 'NEGATIVE']:
            return 'Negative'
        else:
            return 'Neutral'
    except:
        return _fallback_sentiment(text)

def batch_analyze_sentiment(texts):
    analyzer = get_analyzer()
    if analyzer == "fallback":
        return [_fallback_sentiment(t) for t in texts]
        
    # Batch processing is much faster
    results = []
    try:
        # Process in chunks of 32
        for i in range(0, len(texts), 32):
            batch = [t[:512] for t in texts[i:i+32]]
            batch_res = analyzer(batch)
            for res in batch_res:
                l = res['label']
                if l in ['POS', 'POSITIVE']: results.append('Positive')
                elif l in ['NEG', 'NEGATIVE']: results.append('Negative')
                else: results.append('Neutral')
    except Exception as e:
        console.print(f"[red]Batch sentiment error: {e}[/red]")
        return [_fallback_sentiment(t) for t in texts]
            
    return results

def _fallback_sentiment(text):
    """Dictionary fallback"""
    POSITIVE_WORDS = {'tuyệt vời', 'xuất sắc', 'hay', 'tốt', 'đẹp', 'yêu', 'thích', 'vui', 'hạnh phúc', 'ủng hộ', 'ngon', 'giỏi'}
    NEGATIVE_WORDS = {'tệ', 'kém', 'xấu', 'ghét', 'buồn', 'đau', 'khổ', 'thất vọng', 'phản đối', 'sai', 'nguy hiểm', 'chết', 'kinh khủng', 'sợ'}
    
    text = text.lower()
    pos = sum(1 for w in POSITIVE_WORDS if w in text)
    neg = sum(1 for w in NEGATIVE_WORDS if w in text)
    
    if pos > neg: return 'Positive'
    elif neg > pos: return 'Negative'
    else: return 'Neutral'
