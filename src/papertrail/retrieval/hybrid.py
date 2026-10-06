"""Hybrid retrieval: vector + keyword results merged with Reciprocal Rank Fusion (RRF)."""

from __future__ import annotations

from dataclasses import replace

import psycopg

from papertrail.config import get_settings
from papertrail.retrieval.keyword_search import keyword_search
from papertrail.retrieval.rerank import rerank
from papertrail.retrieval.vector_search import RetrievedChunk, vector_search

RRF_K = 60  # standard RRF constant
CANDIDATES = 50  # results taken from each retriever before fusing
RERANK_POOL = 30  # fused results handed to the cross-encoder


def rrf_fuse(
    rankings: list[list[RetrievedChunk]], k_rrf: int = RRF_K
) -> list[RetrievedChunk]:
    """score(chunk) = sum over rankings of 1 / (k_rrf + rank)."""
    scores: dict[int, float] = {}
    first_seen: dict[int, RetrievedChunk] = {}
    for ranking in rankings:
        for rank, chunk in enumerate(ranking, 1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (
                k_rrf + rank
            )
            first_seen.setdefault(chunk.chunk_id, chunk)
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [replace(first_seen[cid], score=scores[cid]) for cid in ordered]


def hybrid_search(
    question: str,
    k: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    k = settings.top_k if k is None else k
    if k < 1:
        raise ValueError("k must be >= 1")
    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(settings.database_url)
    try:
        conn.execute("SET hnsw.ef_search = 200")  # default 40 would cap results at 40
        vec = vector_search(question, k=CANDIDATES, conn=conn)
        kw = keyword_search(question, k=CANDIDATES, conn=conn)
    finally:
        if own_conn:
            conn.close()
    return rrf_fuse([vec, kw])[:k]


def hybrid_rerank_search(
    question: str,
    k: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    k = get_settings().top_k if k is None else k
    pool = hybrid_search(question, k=max(RERANK_POOL, k), conn=conn)
    return rerank(question, pool, k)
