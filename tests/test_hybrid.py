from papertrail.retrieval.hybrid import rrf_fuse
from papertrail.retrieval.keyword_search import build_tsquery
from papertrail.retrieval.vector_search import RetrievedChunk


def _c(i: int) -> RetrievedChunk:
    return RetrievedChunk(i, "p", "t", None, i, "x", 0.0)


def test_rrf_prefers_chunks_found_by_both_retrievers():
    fused = rrf_fuse([[_c(1), _c(2), _c(3)], [_c(3), _c(1), _c(4)]])
    assert [c.chunk_id for c in fused][:2] == [1, 3]
    assert {c.chunk_id for c in fused} == {1, 2, 3, 4}


def test_rrf_scores_descend():
    fused = rrf_fuse([[_c(1), _c(2)], [_c(2), _c(3)]])
    scores = [c.score for c in fused]
    assert scores == sorted(scores, reverse=True)


def test_build_tsquery_drops_stopwords_and_dedupes():
    assert (
        build_tsquery("What is retrieval-augmented generation?")
        == "retrieval | augmented | generation"
    )
    assert build_tsquery("RAG rag RAG") == "rag"
    assert build_tsquery("what is the") == ""
