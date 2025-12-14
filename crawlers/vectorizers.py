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


def get_embeddings(texts: list, method: str = "sentence-transformer", 
                   model_name: str = None, **kwargs) -> np.ndarray:
    """
    Get embeddings using specified method.
    
    Args:
        texts: List of text strings
        method: 'sentence-transformer', 'tfidf', 'bow', or 'glove'
        model_name: Model name for sentence-transformer
        **kwargs: Additional arguments (max_features, glove_path, etc.)
        
    Returns:
        numpy array of embeddings
    """
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
        console.print(f"[cyan]🧠 Loading Sentence Transformer: {model_name}...[/cyan]")
        model = SentenceTransformer(model_name)
        embeddings = model.encode(texts, show_progress_bar=True)
        console.print(f"[green]✅ Sentence Transformer: Shape {embeddings.shape}[/green]")
        return embeddings
    
    else:
        raise ValueError(f"Unknown embedding method: {method}")


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
