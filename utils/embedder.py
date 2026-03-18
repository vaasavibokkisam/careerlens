from sentence_transformers import SentenceTransformer
import numpy as np

# Load once and cache
_model = None

def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
    return _model


def get_embedding(text: str) -> list[float]:
    """Convert text to a normalised embedding vector."""
    model = _get_model()
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def get_embeddings_batch(texts: list[str]) -> list[list[float]]:
    """Batch encode a list of texts."""
    model = _get_model()
    vecs = model.encode(texts, normalize_embeddings=True, batch_size=32)
    return vecs.tolist()
