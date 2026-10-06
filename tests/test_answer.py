from papertrail.generation.answer import (
    ABSTAIN,
    analyze_answer,
    parse_citations,
    split_sentences,
)
from papertrail.retrieval.vector_search import RetrievedChunk


def _chunks(n: int) -> list[RetrievedChunk]:
    return [RetrievedChunk(i, "p", "t", None, i, "text", 0.0) for i in range(1, n + 1)]


def test_parse_citations_handles_both_styles():
    assert parse_citations("A claim [1][3] and another [2, 4].") == [1, 3, 2, 4]


def test_split_sentences_keeps_citation_with_its_sentence():
    expected = ["It rose [1].", "Then it fell [2]."]
    assert split_sentences("It rose. [1] Then it fell [2].") == expected
    assert split_sentences("It rose [1]. Then it fell [2].") == expected


def test_valid_answer_is_verified():
    a = analyze_answer("q", "Gacha halves diversity [1]. It collapses to the middle [2].", _chunks(2))
    assert a.cited == [1, 2] and a.verified


def test_invalid_and_missing_citations_are_flagged():
    a = analyze_answer("q", "One claim [3]. Another claim with no source.", _chunks(2))
    assert a.invalid_citations == [3]
    assert a.uncited_sentences == ["Another claim with no source."]
    assert not a.verified


def test_abstention_counts_as_verified():
    a = analyze_answer("q", ABSTAIN, _chunks(2))
    assert a.abstained and a.verified
