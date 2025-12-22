"""
Text Vectorization Methods for Trend Analysis.
Provides TF-IDF, BoW, and GloVe embeddings as alternatives to Sentence Transformers.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from rich.console import Console
import os

console = Console()

# GloVe cache
_glove_embeddings = None

# ProtonX client cache
_protonx_client = None


def get_tfidf_embeddings(texts: list, max_features: int = 5000) -> np.ndarray:
    """
    Create TF-IDF embeddings for texts.
    
    Args:
        texts: List of text strings
        max_features: Maximum vocabulary size
        
    Returns:
        numpy array of shape (n_texts, max_features)
    """
    console.print(f"[cyan]📊 Creating TF-IDF embeddings (max_features={max_features})...[/cyan]")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=2,
        max_df=0.95
    )
    
    embeddings = vectorizer.fit_transform(texts).toarray()
    console.print(f"[green]✅ TF-IDF: Shape {embeddings.shape}[/green]")
    
    return embeddings


def get_bow_embeddings(texts: list, max_features: int = 5000) -> np.ndarray:
    """
    Create Bag-of-Words embeddings for texts.
    
    Args:
        texts: List of text strings
        max_features: Maximum vocabulary size
        
    Returns:
        numpy array of shape (n_texts, max_features)
    """
    console.print(f"[cyan]📊 Creating BoW embeddings (max_features={max_features})...[/cyan]")
    
    vectorizer = CountVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95
    )
    
    embeddings = vectorizer.fit_transform(texts).toarray()
    
    # Normalize to unit vectors for cosine similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    embeddings = embeddings / norms
    
    console.print(f"[green]✅ BoW: Shape {embeddings.shape}[/green]")
    
    return embeddings


def load_glove_embeddings(glove_path: str = None, dim: int = 100) -> dict:
    """
    Load pre-trained GloVe embeddings.
    
    Args:
        glove_path: Path to GloVe file (e.g., glove.6B.100d.txt)
        dim: Embedding dimension
        
    Returns:
        Dictionary mapping words to vectors
    """
    global _glove_embeddings
    
    if _glove_embeddings is not None:
        return _glove_embeddings
    
    if glove_path is None:
        # Try common paths
        possible_paths = [
            f"glove.6B.{dim}d.txt",
            f"~/glove/glove.6B.{dim}d.txt",
            f"/kaggle/input/glove6b/glove.6B.{dim}d.txt",
        ]
        for p in possible_paths:
            expanded = os.path.expanduser(p)
            if os.path.exists(expanded):
                glove_path = expanded
                break
    
    if glove_path is None or not os.path.exists(glove_path):
        console.print("[yellow]⚠️ GloVe file not found. Using random embeddings.[/yellow]")
        return None
    
    console.print(f"[cyan]📥 Loading GloVe embeddings from {glove_path}...[/cyan]")
    
    embeddings = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            vector = np.asarray(values[1:], dtype='float32')
            embeddings[word] = vector
    
    _glove_embeddings = embeddings
    console.print(f"[green]✅ Loaded {len(embeddings)} GloVe vectors[/green]")
    
    return embeddings


def get_glove_embeddings(texts: list, glove_path: str = None, dim: int = 100) -> np.ndarray:
    """
    Create GloVe-based embeddings by averaging word vectors.
    
    Args:
        texts: List of text strings
        glove_path: Path to GloVe file
        dim: Embedding dimension
        
    Returns:
        numpy array of shape (n_texts, dim)
    """
    console.print(f"[cyan]📊 Creating GloVe embeddings (dim={dim})...[/cyan]")
    
    glove = load_glove_embeddings(glove_path, dim)
    
    embeddings = []
    oov_count = 0
    total_words = 0
    
    for text in texts:
        words = text.lower().split()
        total_words += len(words)
        
        word_vectors = []
        for word in words:
            if glove and word in glove:
                word_vectors.append(glove[word])
            else:
                oov_count += 1
        
        if word_vectors:
            # Average of word vectors
            embedding = np.mean(word_vectors, axis=0)
        else:
            # Zero vector for empty/OOV texts
            embedding = np.zeros(dim)
        
        embeddings.append(embedding)
    
    embeddings = np.array(embeddings)
    
    # Normalize
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings = embeddings / norms
    
    oov_rate = oov_count / total_words * 100 if total_words > 0 else 0
    console.print(f"[green]✅ GloVe: Shape {embeddings.shape}, OOV rate: {oov_rate:.1f}%[/green]")
    
    return embeddings


def get_protonx_embeddings(texts: list, api_key: str = None, batch_size: int = 5) -> np.ndarray:
    """
    Create embeddings using ProtonX Vietnamese Embedding API.
    
    Args:
        texts: List of text strings
        api_key: ProtonX API key (uses PROTONX_API_KEY env var if not provided)
        batch_size: Number of texts to send per API call (max 5-10 to avoid rate limits)
        
    Returns:
        numpy array of shape (n_texts, embedding_dim)
        
    Note: ProtonX has 10 requests/minute limit. This function adds delays between batches.
    """
    global _protonx_client
    import time
    
    # Get API key
    if api_key is None:
        api_key = os.environ.get("PROTONX_API_KEY")
    
    if not api_key:
        raise ValueError(
            "ProtonX API key required. Set PROTONX_API_KEY environment variable "
            "or pass api_key parameter. Get key from https://platform.protonx.io/"
        )
    
    console.print(f"[cyan]🌐 Creating ProtonX embeddings for {len(texts)} texts...[/cyan]")
    console.print(f"[dim]Rate limit: 10 req/min. Using batch_size={batch_size} with delays.[/dim]")
    
    # Set env var for protonx library
    os.environ["PROTONX_API_KEY"] = api_key
    
    try:
        from protonx import ProtonX
    except ImportError:
        console.print("[yellow]Installing protonx library...[/yellow]")
        import subprocess
        subprocess.run(["pip", "install", "protonx", "-q"], check=True)
        from protonx import ProtonX
    
    if _protonx_client is None:
        _protonx_client = ProtonX()
    
    client = _protonx_client
    embeddings = []
    
    # Process in batches with rate limiting
    from rich.progress import track
    total_batches = (len(texts) + batch_size - 1) // batch_size
    
    for batch_idx, i in enumerate(track(range(0, len(texts), batch_size), description="[cyan]ProtonX encoding...[/cyan]", total=total_batches)):
        batch = texts[i:i + batch_size]
        
        # Add delay between batches to respect rate limit (10 req/min = 1 req/6sec)
        if batch_idx > 0:
            time.sleep(7)  # 7 seconds between batches
        
        for text in batch:
            try:
                result = client.embeddings.create(text)
                if result and 'data' in result and len(result['data']) > 0:
                    emb = result['data'][0]['embedding']
                    embeddings.append(emb)
                else:
                    # Fallback to zero vector
                    console.print(f"[yellow]⚠️ Empty response for text, using zero vector[/yellow]")
                    embeddings.append([0.0] * 768)  # ProtonX uses 768-dim
            except Exception as e:
                error_str = str(e)
                if "429" in error_str:
                    console.print(f"[yellow]⚠️ Rate limited, waiting 60 seconds...[/yellow]")
                    time.sleep(60)
                    # Retry
                    try:
                        result = client.embeddings.create(text)
                        if result and 'data' in result:
                            embeddings.append(result['data'][0]['embedding'])
                        else:
                            embeddings.append([0.0] * 768)
                    except:
                        embeddings.append([0.0] * 768)
                else:
                    console.print(f"[red]❌ ProtonX error: {e}[/red]")
                    embeddings.append([0.0] * 768)
    
    embeddings = np.array(embeddings)
    console.print(f"[green]✅ ProtonX: Shape {embeddings.shape}[/green]")
    
    return embeddings


import hashlib
import pickle

def get_embeddings(texts: list, method: str = "sentence-transformer", 
                   model_name: str = None, device: str = None, 
                   existing_model=None, cache_dir="embeddings_cache", **kwargs) -> np.ndarray:
    """
    Get embeddings with optional disk caching.
    """
    if not texts: return np.array([])
    
    # Create unique hash for this request (texts + model + method)
    # Use full text list for reliable caching (hashing strings is fast enough)
    combined_text = "".join(texts) + str(len(texts)) + str(model_name)
    text_hash = hashlib.md5(combined_text.encode()).hexdigest()
    cache_filename = f"{method}_{model_name.replace('/', '_') if model_name else 'none'}_{text_hash}.npy"
    
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, cache_filename)
        if os.path.exists(cache_path):
            console.print(f"[green]📦 Loading cached embeddings from {cache_path}[/green]")
            return np.load(cache_path)

    method = method.lower()
    
    if method == "tfidf":
        return get_tfidf_embeddings(texts, max_features=kwargs.get('max_features', 5000))
    
    elif method == "bow":
        return get_bow_embeddings(texts, max_features=kwargs.get('max_features', 5000))
    
    elif method == "glove":
        return get_glove_embeddings(
            texts, 
            glove_path=kwargs.get('glove_path'),
            dim=kwargs.get('dim', 100)
        )
    
    elif method == "sentence-transformer":
        from sentence_transformers import SentenceTransformer
        import torch
        
        # Determine device
        if device is None:
            # If on Kaggle and we suspect memory pressure, default to CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            
        console.print(f"[cyan]🧠 Encoding with {model_name or 'existing model'} on {device}...[/cyan]")
        
        if existing_model:
            model = existing_model
        else:
            model = SentenceTransformer(model_name, device=device)
        
        # Enforce sequence length limit (default 256, strictly enforced)
        model.max_seq_length = kwargs.get('max_seq_length', 256)
            
        embeddings = model.encode(texts, show_progress_bar=True, device=device)
        
        # Save to cache if enabled
        if cache_dir:
            cache_path = os.path.join(cache_dir, cache_filename)
            np.save(cache_path, embeddings)
            console.print(f"[green]💾 Saved embeddings to {cache_path}[/green]")
            
        return embeddings
    
    elif method == "protonx":
        embeddings = get_protonx_embeddings(
            texts,
            api_key=kwargs.get('api_key'),
            batch_size=kwargs.get('batch_size', 32)
        )
        
        # Cache ProtonX embeddings too
        if cache_dir:
            cache_path = os.path.join(cache_dir, cache_filename)
            np.save(cache_path, embeddings)
            console.print(f"[green]💾 Saved embeddings to {cache_path}[/green]")
            
        return embeddings
    
    else:
        raise ValueError(f"Unknown embedding method: {method}. Available: tfidf, bow, glove, sentence-transformer, protonx")


# Test
if __name__ == "__main__":
    test_texts = [
        "Công Phượng ghi bàn thắng đẹp",
        "Bão Yagi đổ bộ Hà Nội gây thiệt hại",
        "Giá vàng tăng mạnh hôm nay",
    ]
    
    console.print("\n[bold]Testing embedding methods:[/bold]\n")
    
    for method in ["tfidf", "bow"]:
        emb = get_embeddings(test_texts, method=method)
        console.print(f"  {method}: {emb.shape}\n")
