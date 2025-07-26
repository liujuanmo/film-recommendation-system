"""
Constants for the movie recommendation system.
"""

from sentence_transformers import SentenceTransformer

# Sentence Transformer Model Configuration
DEFAULT_MODEL = 'all-MiniLM-L6-v2'
DEFAULT_MODEL_DIMENSION = 384  # Embedding dimension for all-MiniLM-L6-v2

# Global model instance - loaded once, reused everywhere
_text_model = None

def get_text_model(model_name=DEFAULT_MODEL):
    """Get the global text model instance, loading it only once."""
    global _text_model
    if _text_model is None:
        print(f"🔄 Loading sentence transformer model: {model_name} (first time)")
        _text_model = SentenceTransformer(model_name)
        print(f"✅ Model {model_name} loaded and cached")
    else:
        print(f"♻️  Reusing cached model: {model_name}")
    return _text_model