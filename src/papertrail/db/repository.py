from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import psycopg
from pgvector import Vector
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row

from papertrail.config import get_settings

SCHEMA_PATH = Path(__file__).parent / "schema.sql"


@dataclass
class Paper:
    arxiv_id: str
    title: str
    abstract: str = ""
    authors: list[str] | None = None
    categories: list[str] | None = None
    published: datetime | None = None
    pdf_url: str | None = None


@dataclass
class Chunk:
    chunk_index: int
    section: str | None
    content: str
    embedding: Sequence[float]


def connect(*, vector: bool = True) -> psycopg.Connection:
    """Open a connection. Rows come back as dicts.

    vector=False is only for init_db(), because the vector type
    doesn't exist until the extension has been created.
    """
    conn = psycopg.connect(get_settings().database_url, row_factory=dict_row)
    if vector:
        register_vector(conn)
    return conn


def init_db() -> None:
    """Create the extension, tables and indexes (safe to run repeatedly)."""
    dim = get_settings().embedding_dim
    sql = SCHEMA_PATH.read_text().replace("__EMBEDDING_DIM__", str(dim))
    with connect(vector=False) as conn:
        conn.execute(sql)


def upsert_paper(conn: psycopg.Connection, paper: Paper) -> None:
    params = asdict(paper)
    params["authors"] = params["authors"] or []
    params["categories"] = params["categories"] or []
    conn.execute(
        """
        INSERT INTO papers (arxiv_id, title, abstract, authors, categories, published, pdf_url)
        VALUES (%(arxiv_id)s, %(title)s, %(abstract)s, %(authors)s,
                %(categories)s, %(published)s, %(pdf_url)s)
        ON CONFLICT (arxiv_id) DO UPDATE SET
            title = EXCLUDED.title,
            abstract = EXCLUDED.abstract,
            authors = EXCLUDED.authors,
            categories = EXCLUDED.categories,
            published = EXCLUDED.published,
            pdf_url = EXCLUDED.pdf_url
        """,
        params,
    )


def replace_chunks(conn: psycopg.Connection, arxiv_id: str, chunks: list[Chunk]) -> None:
    """Delete a paper's existing chunks and insert the new ones.

    Replacing (not appending) makes re-ingestion with different chunk
    settings safe to repeat.
    """
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE arxiv_id = %s", (arxiv_id,))
        cur.executemany(
            """
            INSERT INTO chunks (arxiv_id, chunk_index, section, content, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            [
                (arxiv_id, c.chunk_index, c.section, c.content, Vector(c.embedding))
                for c in chunks
            ],
        )


def ingested_ids(conn: psycopg.Connection) -> set[str]:
    """IDs of papers that already have chunks, so ingestion can skip them."""
    rows = conn.execute("SELECT DISTINCT arxiv_id FROM chunks").fetchall()
    return {r["arxiv_id"] for r in rows}


def stats(conn: psycopg.Connection) -> dict[str, int]:
    papers = conn.execute("SELECT count(*) AS n FROM papers").fetchone()["n"]
    chunks = conn.execute("SELECT count(*) AS n FROM chunks").fetchone()["n"]
    return {"papers": papers, "chunks": chunks}


if __name__ == "__main__":
    init_db()
    print("Database initialised.")