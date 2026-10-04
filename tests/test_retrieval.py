import psycopg
import pytest

from papertrail.config import get_settings
from papertrail.retrieval.vector_search import RetrievedChunk, vector_search


def _corpus_ready() -> bool:
    try:
        with psycopg.connect(get_settings().database_url, connect_timeout=2) as conn:
            return conn.execute("SELECT count(*) FROM chunks").fetchone()[0] > 0
    except psycopg.Error:
        return False


pytestmark = pytest.mark.skipif(
    not _corpus_ready(), reason="DB not running or no chunks ingested"
)


def test_returns_k_results_sorted_by_score():
    results = vector_search("How do transformers handle long context?", k=5)
    assert len(results) == 5
    assert all(isinstance(r, RetrievedChunk) for r in results)
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_default_k_comes_from_config():
    assert len(vector_search("attention mechanism")) == get_settings().top_k


def test_result_fields_populated():
    r = vector_search("attention mechanism", k=1)[0]
    assert r.title and r.arxiv_id and r.content
    assert -1.0 <= r.score <= 1.0


def test_chunk_text_retrieves_itself():
    """A chunk's own text as the query should bring that chunk back in the top 3."""
    with psycopg.connect(get_settings().database_url) as conn:
        cid, text = conn.execute(
            "SELECT id, content FROM chunks WHERE length(content) > 400 LIMIT 1"
        ).fetchone()
    assert cid in [r.chunk_id for r in vector_search(text, k=3)]


def test_invalid_inputs():
    with pytest.raises(ValueError):
        vector_search("   ")
    with pytest.raises(ValueError):
        vector_search("ok", k=0)
