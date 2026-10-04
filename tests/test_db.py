import random

from pgvector import Vector

from papertrail.config import get_settings
from papertrail.db import repository as repo


def test_roundtrip_and_search():
    repo.init_db()
    dim = get_settings().embedding_dim
    vec = [random.random() for _ in range(dim)]

    with repo.connect() as conn:
        repo.upsert_paper(conn, repo.Paper(arxiv_id="test.0001", title="Test paper"))
        repo.replace_chunks(
            conn,
            "test.0001",
            [repo.Chunk(0, "Intro", "attention is all you need", vec)],
        )

        # vector search finds the chunk we just stored
        row = conn.execute(
            "SELECT arxiv_id FROM chunks ORDER BY embedding <=> %s LIMIT 1",
            (Vector(vec),),
        ).fetchone()
        assert row["arxiv_id"] == "test.0001"

        # keyword search works through the generated tsvector column
        hits = conn.execute(
            "SELECT count(*) AS n FROM chunks WHERE tsv @@ plainto_tsquery('english', 'attention')"
        ).fetchone()
        assert hits["n"] >= 1

        assert "test.0001" in repo.ingested_ids(conn)
        conn.rollback()  # leave no test data behind
