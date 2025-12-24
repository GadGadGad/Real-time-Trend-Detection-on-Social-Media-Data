
"""
Word Segmentation Utility for Vietnamese.
Standardizes text for models like 'keepitreal/vietnamese-sbert' which require segmented input (e.g. "Hà_Nội").
Prioritizes VnCoreNLP (RDRSegmenter) for accuracy, falls back to Underthesea.
"""
import os
import sys
from rich.console import Console

console = Console()

# Singleton cache
_rdrsegmenter = None

def load_segmenter():
    global _rdrsegmenter
    if _rdrsegmenter:
        return _rdrsegmenter
        
    # Try loading py_vncorenlp first (Best for SBERT)
    try:
        import py_vncorenlp
        # Automatically download if needed (to current dir or handled by lib)
        # Note: Users might need to manage the model path manually in some envs
        # We assume standard setup or fallback
        save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../vncorenlp_models"))
        os.makedirs(save_dir, exist_ok=True)
        
        # Download if not exists (VnCoreNLP is distinct)
        if not os.path.exists(os.path.join(save_dir, "VnCoreNLP-1.2.jar")):
             console.print("[cyan]⬇️ Downloading VnCoreNLP model...[/cyan]")
             py_vncorenlp.download_model(save_dir=save_dir)
             
        _rdrsegmenter = py_vncorenlp.VnCoreNLP(save_dir=save_dir, annotators=["wseg"])
        console.print("[green]✅ Loaded VnCoreNLP Segmenter[/green]")
        return _rdrsegmenter
    except Exception as e:
        console.print(f"[dim yellow]⚠️ VnCoreNLP load failed ({e}). Falling back to Underthesea.[/dim yellow]")
        
    return None

def segment_text(text):
    """
    Segment Vietnamese text.
    Input: "Hà Nội là thủ đô"
    Output: "Hà_Nội là thủ_đô"
    """
    if not text: return ""
    
    # Method 1: VnCoreNLP
    segmenter = load_segmenter()
    if segmenter:
        try:
            # VnCoreNLP return list of sentences, each sentence list of words
            sentences = segmenter.word_segment(text)
            # Flatten
            return " ".join(sentences)
        except: pass
        
    # Method 2: Underthesea (Fallback)
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        console.print("[red]❌ No segmenter available. Please `pip install underthesea`[/red]")
        return text

def batch_segment_texts(texts):
    """Batch processing with progress bar."""
    from rich.progress import track
    if not texts: return []
    
    # Load once
    load_segmenter()
    
    results = []
    # If using vncorenlp, it's fast enough sequentially usually, but basic loop for now
    for t in track(texts, description="[cyan]Running Word Segmentation...[/cyan]"):
        results.append(segment_text(t))
    return results
