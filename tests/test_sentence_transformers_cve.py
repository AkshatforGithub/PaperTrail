"""Regression tests for the sentence-transformers security bump.

The repository requires sentence-transformers>=5.6.0 to close
GHSA-jhr6-gm9c-rqjv / PYSEC-2026-4164. These tests pin that requirement (so an
old, vulnerable install fails) and guard the call sites that use the
SentenceTransformer / CrossEncoder public APIs.
"""

import numpy as np
from packaging.version import Version

import sentence_transformers

from papertrail.embeddings.embedder import embed_query, embed_texts
from papertrail.retrieval.rerank import rerank
from papertrail.retrieval.vector_search import RetrievedChunk

# Fixed in 5.6.0 per GHSA-jhr6-gm9c-rqjv / PYSEC-2026-4164.
_MIN_FIXED_VERSION = "5.6.0"


def test_sentence_transformers_at_or_above_fixed_version():
    assert Version(sentence_transformers.__version__) >= Version(_MIN_FIXED_VERSION)


class _FakeEmbedder:
    """Minimal stand-in for SentenceTransformer that records how encode() is called."""

    def __init__(self, dim: int = 384):
        self._dim = dim
        self.calls: list[dict] = []

    def encode(self, texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False):
        self.calls.append(
            {
                "texts": list(texts),
                "batch_size": batch_size,
                "normalize": normalize_embeddings,
                "progress": show_progress_bar,
            }
        )
        # Unit vectors, one non-zero entry per row.
        out = np.zeros((len(texts), self._dim), dtype=np.float32)
        for i in range(len(texts)):
            out[i, i % self._dim] = 1.0
        return out


def test_embed_texts_uses_current_encode_api(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr("papertrail.embeddings.embedder._model", lambda: fake)

    vectors = embed_texts(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
    call = fake.calls[-1]
    assert call["batch_size"] == 32
    assert call["normalize"] is True
    assert call["progress"] is False


def test_embed_query_prefixes_query(monkeypatch):
    fake = _FakeEmbedder()
    monkeypatch.setattr("papertrail.embeddings.embedder._model", lambda: fake)

    vec = embed_query("what is RAG?")

    assert len(vec) == 384
    assert fake.calls[0]["texts"][0].startswith("Represent this sentence")


class _FakeCrossEncoder:
    """Minimal stand-in for CrossEncoder; pair i scores i+1."""

    def predict(self, pairs, *args, **kwargs):
        return [float(i + 1) for i in range(len(pairs))]


def test_rerank_keeps_top_k_by_score(monkeypatch):
    monkeypatch.setattr("papertrail.retrieval.rerank._model", lambda: _FakeCrossEncoder())

    chunks = [
        RetrievedChunk(i, "p", "title", None, i, f"chunk {i}", 0.0)
        for i in range(1, 5)
    ]

    ranked = rerank("question", chunks, k=2)

    assert [c.chunk_id for c in ranked] == [4, 3]
    assert all(isinstance(c.score, float) for c in ranked)
