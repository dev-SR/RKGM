"""
Singleton embedder.

Primary: sentence-transformers all-MiniLM-L6-v2 (local, zero API cost).
Fallback: TF-IDF vectors (sklearn) when the model cannot be downloaded
          (e.g. offline environments, CI). Quality is lower but the
          pipeline remains functional end-to-end.

On the user's machine with internet access, sentence-transformers loads
automatically on first use and is cached by HuggingFace locally.
"""

import numpy as np
from core.config import EMBED_MODEL

# ── Primary: sentence-transformers ────────────────────────────────────────
try:
    from sentence_transformers import SentenceTransformer as _ST

    _ST_AVAILABLE = True
except ImportError:
    _ST_AVAILABLE = False

_st_model = None
_tfidf_model = None
_using_tfidf = False


def _get_st_model():
    global _st_model
    if _st_model is None:
        _st_model = _ST(EMBED_MODEL)
    return _st_model


# ── Fallback: TF-IDF ──────────────────────────────────────────────────────
def _tfidf_embed(texts: list[str]) -> np.ndarray:
    """
    TF-IDF sparse → dense L2-normalised vectors.
    Fits on first call, transforms on subsequent calls.
    Dimension = min(300, vocabulary size).
    """
    global _tfidf_model
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.preprocessing import normalize

    if _tfidf_model is None or not hasattr(_tfidf_model, "vocabulary_"):
        _tfidf_model = TfidfVectorizer(
            max_features=300,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        matrix = _tfidf_model.fit_transform(texts)
    else:
        try:
            matrix = _tfidf_model.transform(texts)
        except Exception:
            # Vocabulary mismatch on new call — refit
            _tfidf_model.fit(texts)
            matrix = _tfidf_model.transform(texts)

    dense = matrix.toarray().astype(np.float32)
    norms = np.linalg.norm(dense, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return dense / norms


def embed_texts(texts: list[str]) -> np.ndarray:
    """
    Batch-embed a list of texts → (N, dim) float32, L2-normalised.
    Always call with a full list (not one-by-one) for batch efficiency.
    """
    global _using_tfidf
    if not texts:
        return np.zeros((0, 300), dtype=np.float32)

    if _ST_AVAILABLE and not _using_tfidf:
        try:
            model = _get_st_model()
            return model.encode(
                texts,
                batch_size=64,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
        except Exception as e:
            print(
                f"[embedder] sentence-transformers unavailable ({e}), "
                "falling back to TF-IDF. Quality will be lower."
            )
            _using_tfidf = True

    return _tfidf_embed(texts)


def embed_single(text: str) -> np.ndarray:
    return embed_texts([text])[0]


def embedding_mode() -> str:
    """Return which backend is active (for display in UI)."""
    return "TF-IDF (fallback)" if _using_tfidf else "sentence-transformers"
