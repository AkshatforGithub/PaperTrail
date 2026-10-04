from __future__ import annotations

import argparse

from papertrail.db import repository as repo
from papertrail.embeddings.embedder import embed_texts
from papertrail.ingestion.chunk import chunk_sections
from papertrail.ingestion.fetch import DEFAULT_QUERY, download_pdf, fetch_papers
from papertrail.ingestion.parse import Section, parse_pdf


def ingest_paper(paper: repo.Paper) -> int:
    """Download, parse, chunk, embed and store one paper. Returns the chunk count."""
    parsed = parse_pdf(download_pdf(paper))
    if not parsed:
        raise ValueError("no text extracted from PDF")

    # The abstract comes from arXiv metadata, so drop the parsed duplicates.
    sections = [Section("Abstract", paper.abstract)]
    sections += [s for s in parsed if s.title not in ("Abstract", "Front matter")]

    chunks = chunk_sections(sections)
    vectors = embed_texts([c.content for c in chunks])
    rows = [
        repo.Chunk(i, c.section, c.content, v)
        for i, (c, v) in enumerate(zip(chunks, vectors))
    ]
    with repo.connect() as conn:
        repo.upsert_paper(conn, paper)
        repo.replace_chunks(conn, paper.arxiv_id, rows)
    return len(rows)


def run(query: str, max_results: int, reingest: bool = False) -> None:
    repo.init_db()
    papers = fetch_papers(query, max_results)
    with repo.connect() as conn:
        done = set() if reingest else repo.ingested_ids(conn)
    todo = [p for p in papers if p.arxiv_id not in done]
    print(f"Fetched {len(papers)} papers; {len(todo)} to ingest")

    failed = 0
    for i, paper in enumerate(todo, 1):
        try:
            n = ingest_paper(paper)
            print(f"[{i}/{len(todo)}] {paper.arxiv_id}: {n} chunks | {paper.title[:60]}")
        except Exception as exc:  # keep going if one paper fails
            failed += 1
            print(f"[{i}/{len(todo)}] {paper.arxiv_id}: FAILED ({exc})")

    with repo.connect() as conn:
        print(f"Done. {failed} failed. Database now holds: {repo.stats(conn)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest arXiv papers into PaperTrail")
    parser.add_argument("--query", default=DEFAULT_QUERY, help="arXiv search query")
    parser.add_argument("--max-results", type=int, default=20)
    parser.add_argument("--reingest", action="store_true", help="re-chunk papers already stored")
    args = parser.parse_args()
    run(args.query, args.max_results, args.reingest)