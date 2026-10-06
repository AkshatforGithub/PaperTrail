from papertrail.generation.answer import (
    analyze_answer,
    normalize_answer_text,
    split_sentences,
)
from papertrail.retrieval.vector_search import RetrievedChunk


def test_lenticular_citations_become_plain_brackets():
    raw = "It works【1】 and more【2†L3-L5】【3】."
    assert normalize_answer_text(raw) == "It works[1] and more[2][3]."


def test_unusual_spaces_and_hyphens_are_normalized():
    assert normalize_answer_text("by\u202fa factor of\u202f22.7 non\u2011zero") == "by a factor of 22.7 non-zero"


def test_abbreviations_do_not_split_sentences():
    parts = split_sentences("Models err (e.g., with flags vs. options) [1]. Next [2].")
    assert len(parts) == 2 and parts[0].endswith("[1].")


def test_lenticular_style_answer_counts_as_cited():
    chunks = [RetrievedChunk(i, "p", "t", None, i, "x", 0.0) for i in (1, 2)]
    a = analyze_answer("q", "One claim【1】. Another claim【2†L1-L2】.", chunks)
    assert a.cited == [1, 2] and a.uncited_sentences == [] and a.verified
