from papertrail.ingestion.chunk import chunk_sections, split_text
from papertrail.ingestion.parse import Section

TEXT = " ".join(f"Sentence number {i} is right here." for i in range(40))


def test_chunks_respect_size():
    chunks = split_text(TEXT, size=200, overlap=70)
    assert len(chunks) > 1
    assert all(len(c) <= 200 for c in chunks)


def test_neighbouring_chunks_overlap():
    chunks = split_text(TEXT, size=200, overlap=70)
    last_sentence = chunks[0].rsplit(". ", 1)[-1]
    assert last_sentence in chunks[1]


def test_oversized_sentence_is_hard_split():
    chunks = split_text("x" * 500, size=200, overlap=20)
    assert all(len(c) <= 200 for c in chunks)
    assert sum(len(c) for c in chunks) == 500


def test_sections_are_labelled_and_tiny_ones_dropped():
    chunks = chunk_sections(
        [Section("Intro", TEXT), Section("Tiny", "too short")], size=200, overlap=70
    )
    assert chunks and all(c.section == "Intro" for c in chunks)


def test_empty_text():
    assert split_text("", size=200, overlap=20) == []