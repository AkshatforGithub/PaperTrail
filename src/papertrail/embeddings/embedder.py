from __future__ import annotations

from functools import lru_cache

from sentence_transformers import SentenceTransformer

from papertrail.config import get_settings

# BGE models work better on short search queries with this prefix.
# Passages (document chunks) are embedded without it.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@lru_cache
def _model() -> SentenceTransformer:
    # Loaded once per process; the first ever call downloads the weights (~130 MB).
    return SentenceTransformer(get_settings().embedding_model)


def embed_texts(texts: list[str], batch_size: int = 32) -> list[list[float]]:
    """Embed document chunks. Vectors are normalised, so cosine search works well."""
    if not texts:
        return []
    vectors = _model().encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    expected = get_settings().embedding_dim
    if vectors.shape[1] != expected:
        raise ValueError(
            f"Model produced {vectors.shape[1]}-dim vectors but EMBEDDING_DIM={expected}. "
            "Fix .env and re-create the chunks table."
        )
    return vectors.tolist()


def embed_query(query: str) -> list[float]:
    """Embed a user question for retrieval."""
    return embed_texts([QUERY_PREFIX + query])[0]
