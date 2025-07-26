"""
Constants for the movie recommendation system.
"""

from sentence_transformers import SentenceTransformer

DATA_DIR = "imdb_data"

# Sentence Transformer Model Configuration
DEFAULT_MODEL = "all-MiniLM-L6-v2"
DEFAULT_MODEL_DIMENSION = 384  # Embedding dimension for all-MiniLM-L6-v2

# Global model instance - loaded once, reused everywhere
_text_model = None


def get_text_model():
    """Get the global text model instance, loading it only once."""
    global _text_model
    if _text_model is None:
        print(f"🔄 Loading sentence transformer model: {DEFAULT_MODEL} ...")
        _text_model = SentenceTransformer(DEFAULT_MODEL)
        print(f"✅ Model {DEFAULT_MODEL} loaded and cached")
    else:
        print(f"♻️  Reusing cached model: {DEFAULT_MODEL}")
    return _text_model
