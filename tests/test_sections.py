from papertrail.ingestion.parse import Section
from papertrail.ingestion.pipeline import build_sections

ABSTRACT = (
    "We study how retrieval quality changes with chunk size across many different "
    "corpora and embedding models, and report consistent trends."
)


def test_headingless_body_is_kept():
    body = "Body sentence number one. " * 300
    parsed = [Section("Front matter", "Some Title A. Author " + ABSTRACT + " " + body)]
    out = build_sections(ABSTRACT, parsed)
    assert [s.title for s in out] == ["Abstract", "Body"]
    assert "Body sentence number one." in out[1].text
    assert "Some Title" not in out[1].text


def test_normal_front_matter_is_dropped():
    parsed = [
        Section("Front matter", "Some Title A. Author " + ABSTRACT),
        Section("Introduction", "Intro text here. " * 20),
    ]
    out = build_sections(ABSTRACT, parsed)
    assert [s.title for s in out] == ["Abstract", "Introduction"]


def test_parsed_abstract_duplicate_is_dropped():
    parsed = [Section("Abstract", ABSTRACT), Section("Method", "Method text. " * 20)]
    out = build_sections(ABSTRACT, parsed)
    assert [s.title for s in out] == ["Abstract", "Method"]


def test_unfindable_abstract_and_short_front_matter_is_dropped():
    parsed = [Section("Front matter", "Title and authors only")]
    assert [s.title for s in build_sections(ABSTRACT, parsed)] == ["Abstract"]
