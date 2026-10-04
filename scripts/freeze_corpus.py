"""Freeze the corpus: existing DB papers + newest cs.CL papers, up to N.

  python scripts/freeze_corpus.py --size 100
Writes evaluation/corpus.txt (one arXiv ID per line, no version suffix).
"""

import argparse
import re
from pathlib import Path

import arxiv
import psycopg

from papertrail.config import get_settings

OUT = Path("evaluation/corpus.txt")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100)
    parser.add_argument("--query", default="cat:cs.CL")
    args = parser.parse_args()

    with psycopg.connect(get_settings().database_url) as conn:
        ids = [
            r[0] for r in conn.execute("SELECT arxiv_id FROM papers ORDER BY arxiv_id")
        ]
    print(f"{len(ids)} papers already in DB")

    client = arxiv.Client(page_size=100, delay_seconds=3, num_retries=3)
    search = arxiv.Search(
        query=args.query,
        max_results=args.size,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )
    for result in client.results(search):
        short = re.sub(r"v\d+$", "", result.get_short_id())
        if short not in ids:
            ids.append(short)
        if len(ids) >= args.size:
            break

    OUT.write_text("\n".join(ids[: args.size]) + "\n")
    print(f"Wrote {min(len(ids), args.size)} IDs to {OUT}")


if __name__ == "__main__":
    main()
