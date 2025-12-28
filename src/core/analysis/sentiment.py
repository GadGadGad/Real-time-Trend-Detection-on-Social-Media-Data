import torch
import os
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from rich.console import Console

console = Console()

# --- Model Configuration ---
# Priority: Custom trained model > Default PhoBERT
DEFAULT_MODEL = "wonrax/phobert-base-vietnamese-sentiment"
CUSTOM_MODEL_PATH = None  # Will be set dynamically

# Try to find custom model in common locations
def _find_custom_model():
    """Look for trained sentiment classifier in common locations."""
    possible_paths = [
        "demo/models/models/sentiment-classifier-vietnamese-v1", # Local Demo Path
        "models/sentiment-classifier-vietnamese-v1",
        "../models/sentiment-classifier-vietnamese-v1",
        "/kaggle/input/sentiment-classifier/sentiment-classifier-vietnamese-v1",
    ]
    for path in possible_paths:
        if os.path.exists(path) and os.path.isdir(path):
            # Check for required files
            if os.path.exists(os.path.join(path, "config.json")):
                return path
    return None

# Lazy Load Model
_sentiment_analyzer = None
_using_custom_model = False

def get_analyzer(model_path=None):
    """
    Get or initialize the sentiment analyzer.
    
    Args:
        model_path: Optional path to custom trained model.
                   If None, will auto-detect or use default.
    """
    global _sentiment_analyzer, _using_custom_model, CUSTOM_MODEL_PATH
    
    if _sentiment_analyzer is None:
        device = 0 if torch.cuda.is_available() else -1
        
        # Determine which model to use
        if model_path:
            CUSTOM_MODEL_PATH = model_path
        else:
            CUSTOM_MODEL_PATH = _find_custom_model()
        
        if CUSTOM_MODEL_PATH and os.path.exists(CUSTOM_MODEL_PATH):
            # Use custom trained model
            console.print(f"[bold green]✅ Loading TRAINED Sentiment Classifier from {CUSTOM_MODEL_PATH}...[/bold green]")
            try:
                _sentiment_analyzer = pipeline(
                    "text-classification", 
                    model=CUSTOM_MODEL_PATH, 
                    tokenizer=CUSTOM_MODEL_PATH,
                    device=device
                )
                _using_custom_model = True
            except Exception as e:
                console.print(f"[red]Failed to load custom model: {e}. Falling back to default.[/red]")
                _sentiment_analyzer = None
        
        # Fallback to default PhoBERT
        if _sentiment_analyzer is None:
            console.print(f"[bold yellow]🧠 Loading Default PhoBERT Sentiment Model ({DEFAULT_MODEL})...[/bold yellow]")
            try:
                _sentiment_analyzer = pipeline("sentiment-analysis", model=DEFAULT_MODEL, device=device)
                _using_custom_model = False
            except Exception as e:
                console.print(f"[red]Failed to load sentiment model: {e}. Using fallback.[/red]")
                _sentiment_analyzer = "fallback"
                
    return _sentiment_analyzer

def analyze_sentiment(text, model_path=None):
    """
    Analyze sentiment using trained classifier or PhoBERT.
    Returns: 'Positive', 'Negative', or 'Neutral'
    """
    analyzer = get_analyzer(model_path)
    if analyzer == "fallback":
        return _fallback_sentiment(text)
        
    try:
        result = analyzer(text[:512])[0]
        label = result['label']
        return _normalize_label(label)
    except:
        return _fallback_sentiment(text)

def batch_analyze_sentiment(texts, model_path=None):
    """Batch sentiment analysis for efficiency."""
    analyzer = get_analyzer(model_path)
    if analyzer == "fallback":
        return [_fallback_sentiment(t) for t in texts]
        
    try:
        batch_res = analyzer([t[:512] for t in texts], batch_size=32, truncation=True)
        return [_normalize_label(res['label']) for res in batch_res]
    except Exception as e:
        console.print(f"[red]Batch sentiment error: {e}[/red]")
        return [_fallback_sentiment(t) for t in texts]

def _normalize_label(label):
    """Normalize label from various formats to Positive/Negative/Neutral."""
    label = label.upper()
    
    # Handle custom trained model labels (Negative=0, Neutral=1, Positive=2)
    if label in ['POSITIVE', 'POS', 'LABEL_2']:
        return 'Positive'
    elif label in ['NEGATIVE', 'NEG', 'LABEL_0']:
        return 'Negative'
    else:  # NEUTRAL, NEU, LABEL_1, or anything else
        return 'Neutral'

def clear_sentiment_analyzer():
    """Unload the sentiment model and free GPU memory"""
    global _sentiment_analyzer, _using_custom_model
    if _sentiment_analyzer and _sentiment_analyzer != "fallback":
        console.print("[yellow]🗑 Clearing Sentiment Model from memory...[/yellow]")
        del _sentiment_analyzer
        _sentiment_analyzer = None
        _using_custom_model = False
        import gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def is_using_custom_model():
    """Check if currently using the custom trained model."""
    return _using_custom_model

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

