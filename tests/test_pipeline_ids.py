from papertrail.ingestion.pipeline import read_ids


def test_read_ids_strips_versions_comments_and_duplicates(tmp_path):
    f = tmp_path / "ids.txt"
    f.write_text("2610.01984v2\n# comment\n\n2610.02001  # inline\n2610.01984\n")
    assert read_ids(f) == ["2610.01984", "2610.02001"]
