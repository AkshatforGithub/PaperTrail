"""Validate evaluation/questions.jsonl against the frozen corpus and the database."""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import psycopg

from papertrail.config import get_settings

QUESTIONS = Path("evaluation/questions.jsonl")
CORPUS = Path("evaluation/corpus.txt")
TYPES = {"numeric", "acronym", "method", "comparison", "definition"}
FIELDS = ("id", "question", "arxiv_id", "evidence", "type")

SQL = r"""
SELECT count(*) FROM chunks
WHERE arxiv_id = %s
  AND position(%s in lower(regexp_replace(content, '\s+', ' ', 'g'))) > 0
"""


def main() -> None:
    corpus = {ln.strip() for ln in CORPUS.read_text().splitlines() if ln.strip()}
    questions = [
        json.loads(ln) for ln in QUESTIONS.read_text().splitlines() if ln.strip()
    ]
    issues: list[str] = []
    seen_ids: set[str] = set()
    seen_q: set[str] = set()

    with psycopg.connect(get_settings().database_url) as conn:
        for q in questions:
            qid = q.get("id", "?")
            missing = [f for f in FIELDS if not q.get(f)]
            if missing:
                issues.append(f"{qid}: missing {missing}")
                continue
            if qid in seen_ids:
                issues.append(f"{qid}: duplicate id")
            seen_ids.add(qid)
            if q["question"].lower() in seen_q:
                issues.append(f"{qid}: duplicate question")
            seen_q.add(q["question"].lower())
            if q["arxiv_id"] not in corpus:
                issues.append(f"{qid}: {q['arxiv_id']} not in corpus.txt")
            if q["type"] not in TYPES:
                issues.append(f"{qid}: bad type {q['type']!r}")
            if not 3 <= len(q["evidence"].split()) <= 15:
                issues.append(f"{qid}: evidence should be 3-15 words")
            ev = re.sub(r"\s+", " ", q["evidence"]).strip().lower()
            n = conn.execute(SQL, (q["arxiv_id"], ev)).fetchone()[0]
            if n == 0:
                issues.append(
                    f"{qid}: evidence not found in any chunk of {q['arxiv_id']}"
                )
            elif n > 2:
                issues.append(f"{qid}: evidence appears in {n} chunks (too generic?)")

    print(
        f"{len(questions)} questions across {len({q.get('arxiv_id') for q in questions})} papers"
    )
    print("by type:", dict(Counter(q.get("type") for q in questions)))
    if issues:
        print(f"\n{len(issues)} issue(s):")
        for i in issues:
            print("  -", i)
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
