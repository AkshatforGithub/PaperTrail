from papertrail.ingestion.parse import dehyphenate


def test_keeps_real_compound_hyphen_seen_elsewhere():
    raw = "We use retrieval-augmented models. Retrieval-\naugmented generation helps."
    assert "Retrieval-augmented generation" in dehyphenate(raw)


def test_joins_soft_hyphenation_when_word_seen_elsewhere():
    raw = "The informa-\ntion is useful. More information follows."
    assert "The information is useful" in dehyphenate(raw)


def test_joins_unknown_soft_hyphenation():
    assert dehyphenate("a convo-\nlutional layer") == "a convolutional layer"


def test_keeps_hyphen_after_common_compound_prefix():
    assert dehyphenate("a multi-\nquery setup") == "a multi-query setup"


def test_no_change_without_line_break_hyphen():
    s = "state-of-the-art results"
    assert dehyphenate(s) == s
