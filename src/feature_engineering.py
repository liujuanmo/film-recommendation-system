import numpy as np
from .constants import DEFAULT_MODEL, DEFAULT_MODEL_DIMENSION, get_text_model


class TextFeatureExtractor:
    def fit_transform(self, texts):
        """Fit the model and transform texts to embeddings."""
        print(f"🔄 Computing embeddings for {len(texts)} texts...")
        embeddings = get_text_model().encode(texts, show_progress_bar=True)
        return embeddings.astype(np.float32)

    def transform(self, texts):
        """Transform texts to embeddings."""
        embeddings = get_text_model().encode(texts, show_progress_bar=False)
        return embeddings.astype(np.float32)

    def get_embedding_dim(self):
        """Get the embedding dimension."""
        return get_text_model().get_sentence_embedding_dimension()


def build_embedding_dict(all_items, model_name=DEFAULT_MODEL):
    """
    Build semantic embeddings for a list of items (directors, actors, etc.)
    
    Args:
        all_items: List of items to embed
        model_name: Model name for sentence transformer
    """
    print(f"🔄 Computing semantic embeddings for {len(all_items)} items using {model_name}...")
    model = get_text_model(model_name)
    
    # Convert items to embedding texts
    texts = [str(item) for item in all_items]
    embeddings = model.encode(texts, show_progress_bar=True)
    
    return {item: emb.astype(np.float32) for item, emb in zip(all_items, embeddings)}


def get_mean_embedding(items, emb_dict, dim=None):
    """Get mean embedding for a list of items."""
    vecs = [emb_dict[i] for i in items if i in emb_dict]
    if not vecs:
        # If no embeddings found, return zero vector with appropriate dimension
        if dim is None:
            # Try to infer dimension from any existing embedding
            if emb_dict:
                dim = len(next(iter(emb_dict.values())))
            else:
                dim = DEFAULT_MODEL_DIMENSION  # default for all-MiniLM-L6-v2
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0).astype(np.float32)
        