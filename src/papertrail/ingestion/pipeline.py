from __future__ import annotations

import argparse
import re
from pathlib import Path

from papertrail.db import repository as repo
from papertrail.embeddings.embedder import embed_texts
from papertrail.ingestion.chunk import chunk_sections
from papertrail.ingestion.fetch import (
    DEFAULT_QUERY,
    download_pdf,
    fetch_papers,
    fetch_papers_by_ids,
)
from papertrail.ingestion.parse import Section, parse_pdf


def read_ids(path: Path) -> list[str]:
    """Read arXiv IDs from a text file: one per line, '#' comments allowed."""
    ids: list[str] = []
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            ids.append(re.sub(r"v\d+$", "", line))
    return list(dict.fromkeys(ids))  # de-duplicate, keep order


FRONT_MATTER_KEEP_CHARS = 4000  # longer than this and it must contain real body text


def _strip_abstract(front_text: str, abstract: str) -> str:
    """Remove title/authors/abstract from the start of front matter, keep the rest."""
    tail = abstract[-60:]
    idx = front_text.find(tail) if tail else -1
    if idx != -1:
        return front_text[idx + len(tail) :].strip()
    # Abstract not found verbatim: short front matter is just title/authors.
    return front_text if len(front_text) > FRONT_MATTER_KEEP_CHARS else ""


def build_sections(abstract: str, parsed: list[Section]) -> list[Section]:
    """Abstract (from arXiv metadata) first, then the parsed sections.

    Parsed duplicates of the abstract are dropped. Text before the first detected
    heading ("Front matter") keeps whatever follows the abstract as a "Body" section,
    so papers whose headings were not detected are not thrown away.
    """
    sections = [Section("Abstract", abstract)]
    for s in parsed:
        if s.title == "Abstract":
            continue
        if s.title == "Front matter":
            body = _strip_abstract(s.text, abstract)
            if body:
                sections.append(Section("Body", body))
            continue
        sections.append(s)
    return sections


def ingest_paper(paper: repo.Paper) -> int:
    """Download, parse, chunk, embed and store one paper. Returns the chunk count."""
    parsed = parse_pdf(download_pdf(paper))
    if not parsed:
        raise ValueError("no text extracted from PDF")

    sections = build_sections(paper.abstract, parsed)

    chunks = chunk_sections(sections)
    vectors = embed_texts([c.content for c in chunks])
    rows = [
        repo.Chunk(i, c.section, c.content, v)
        for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
    ]
    with repo.connect() as conn:
        repo.upsert_paper(conn, paper)
        repo.replace_chunks(conn, paper.arxiv_id, rows)
    return len(rows)


def run(
    query: str,
    max_results: int,
    reingest: bool = False,
    ids_file: str | None = None,
) -> None:
    repo.init_db()
    if ids_file:
        papers = fetch_papers_by_ids(read_ids(Path(ids_file)))
    else:
        papers = fetch_papers(query, max_results)
    with repo.connect() as conn:
        done = set() if reingest else repo.ingested_ids(conn)
    todo = [p for p in papers if p.arxiv_id not in done]
    print(f"Fetched {len(papers)} papers; {len(todo)} to ingest")

    failed = 0
    for i, paper in enumerate(todo, 1):
        try:
            n = ingest_paper(paper)
            print(
                f"[{i}/{len(todo)}] {paper.arxiv_id}: {n} chunks | {paper.title[:60]}"
            )
        except Exception as exc:  # noqa: BLE001 - keep going if one paper fails
            failed += 1
            print(f"[{i}/{len(todo)}] {paper.arxiv_id}: FAILED ({exc})")

    with repo.connect() as conn:
        print(f"Done. {failed} failed. Database now holds: {repo.stats(conn)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into PaperTrail")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument(
        "--ids-file", help="ingest exactly these arXiv IDs (overrides --query)"
    )
    parser.add_argument(
        "--reingest", action="store_true", help="re-chunk papers already stored"
    )
    args = parser.parse_args()
    run(args.query, args.max_results, args.reingest, args.ids_file)
