"""Cross-encoder reranking: score (question, chunk) pairs jointly, keep the best k."""

from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from papertrail.retrieval.vector_search import RetrievedChunk

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model():
    from sentence_transformers import CrossEncoder

    return CrossEncoder(RERANK_MODEL)


def rerank(question: str, chunks: list[RetrievedChunk], k: int) -> list[RetrievedChunk]:
    if not chunks:
        return []
    scores = _model().predict([(question, c.content) for c in chunks])
    ranked = sorted(zip(chunks, scores, strict=True), key=lambda x: x[1], reverse=True)
    return [replace(c, score=float(s)) for c, s in ranked[:k]]
