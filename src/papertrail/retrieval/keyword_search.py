"""Keyword retrieval over the generated tsvector column (Postgres full-text search)."""

from __future__ import annotations

import re

import psycopg

from papertrail.config import get_settings
from papertrail.retrieval.vector_search import RetrievedChunk

STOPWORDS = {
    "a",
    "an",
    "the",
    "of",
    "in",
    "on",
    "at",
    "to",
    "for",
    "and",
    "or",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "what",
    "which",
    "who",
    "whom",
    "how",
    "why",
    "when",
    "where",
    "does",
    "do",
    "did",
    "this",
    "that",
    "these",
    "those",
    "with",
    "by",
    "from",
    "as",
    "it",
    "its",
    "can",
    "could",
    "would",
    "should",
    "about",
    "into",
    "than",
    "then",
}

_SQL = """
SELECT c.id, c.arxiv_id, p.title, c.section, c.chunk_index, c.content,
       ts_rank_cd(c.tsv, tsq) AS score
FROM chunks c
JOIN papers p ON p.arxiv_id = c.arxiv_id
CROSS JOIN to_tsquery('english', %(q)s) AS tsq
WHERE c.tsv @@ tsq
ORDER BY score DESC, c.id
LIMIT %(k)s
"""


def build_tsquery(question: str) -> str:
    """OR together the meaningful words, so one missing word does not kill the match."""
    terms = [
        t
        for t in re.findall(r"[a-z0-9]+", question.lower())
        if len(t) > 1 and t not in STOPWORDS
    ]
    return " | ".join(dict.fromkeys(terms))


def keyword_search(
    question: str,
    k: int | None = None,
    conn: psycopg.Connection | None = None,
) -> list[RetrievedChunk]:
    settings = get_settings()
    k = settings.top_k if k is None else k
    if k < 1:
        raise ValueError("k must be >= 1")
    query = build_tsquery(question)
    if not query:
        return []

    own_conn = conn is None
    if own_conn:
        conn = psycopg.connect(settings.database_url)
    try:
        rows = conn.execute(_SQL, {"q": query, "k": k}).fetchall()
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
