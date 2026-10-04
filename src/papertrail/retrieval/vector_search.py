"""Baseline dense retrieval: embed the question, cosine-search pgvector."""

from __future__ import annotations

from dataclasses import dataclass

import psycopg

from papertrail.config import get_settings
from papertrail.embeddings.embedder import embed_query


@dataclass(frozen=True)
class RetrievedChunk:
    chunk_id: int
    arxiv_id: str
    title: str
    section: str | None
    chunk_index: int
    content: str
    score: float  # cosine similarity (1 - cosine distance); higher is better


# `<=>` is pgvector's cosine distance, matching the HNSW vector_cosine_ops index.
_SQL = """
SELECT c.id, c.arxiv_id, p.title, c.section, c.chunk_index, c.content,
       1 - (c.embedding <=> %(q)s::vector) AS score
FROM chunks c
JOIN papers p ON p.arxiv_id = c.arxiv_id
ORDER BY c.embedding <=> %(q)s::vector
LIMIT %(k)s
"""


def _to_pgvector(vec: list[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def vector_search(
    question: str,
    k: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    """Return the top-k chunks for a question, best first."""
    if not question.strip():
        raise ValueError("question must be non-empty")
    settings = get_settings()
    k = settings.top_k if k is None else k
    if k < 1:
        raise ValueError("k must be >= 1")

    q = _to_pgvector(embed_query(question))

    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(settings.database_url)
    try:
        rows = conn.execute(_SQL, {"q": q, "k": k}).fetchall()
    finally:
        if own_conn:
            conn.close()

    return [
        RetrievedChunk(
            chunk_id=r[0],
            arxiv_id=r[1],
            title=r[2],
            section=r[3],
            chunk_index=r[4],
            content=r[5],
            score=float(r[6]),
        )
        for r in rows
    ]
