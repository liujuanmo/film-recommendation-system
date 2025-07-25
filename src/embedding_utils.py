import numpy as np

def build_embedding_dict(all_items, dim=32, seed=42):
    np.random.seed(seed)
    return {item: np.random.randn(dim).astype(np.float32) for item in all_items}

def get_mean_embedding(items, emb_dict, dim=32):
    vecs = [emb_dict[i] for i in items if i in emb_dict]
    if not vecs:
        return np.zeros(dim, dtype=np.float32)
    return np.mean(vecs, axis=0) 