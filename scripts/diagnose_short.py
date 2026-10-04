"""Explain why some papers have few chunks.

python scripts/diagnose_short.py            # papers with < 20 chunks
python scripts/diagnose_short.py --below 30
"""

import argparse

import psycopg
import pymupdf

from papertrail.config import get_settings
from papertrail.ingestion.fetch import PDF_DIR
from papertrail.ingestion.parse import parse_pdf

parser = argparse.ArgumentParser()
parser.add_argument("--below", type=int, default=20)
args = parser.parse_args()

with psycopg.connect(get_settings().database_url) as conn:
    rows = conn.execute(
        """SELECT arxiv_id, count(*), sum(length(content))
           FROM chunks GROUP BY arxiv_id HAVING count(*) < %s ORDER BY 2""",
        (args.below,),
    ).fetchall()

print(
    f"{'arxiv_id':<12}{'chunks':>7}{'pages':>7}{'raw_chars':>10}{'stored':>8}{'kept%':>7}  last section / verdict"
)
for arxiv_id, n_chunks, stored in rows:
    path = PDF_DIR / f"{arxiv_id.replace('/', '_')}.pdf"
    with pymupdf.open(path) as doc:
        pages = len(doc)
        raw = sum(len(p.get_text()) for p in doc)
    sections = parse_pdf(path)
    last = sections[-1].title if sections else "-"
    kept = 100 * stored / raw if raw else 0
    if raw / max(pages, 1) < 800:
        verdict = "SCANNED/IMAGE-HEAVY"
    elif pages >= 7 and kept < 40:
        verdict = "LIKELY TRUNCATED"
    else:
        verdict = "ok (genuinely short)"
    print(
        f"{arxiv_id:<12}{n_chunks:>7}{pages:>7}{raw:>10}{stored:>8}{kept:>6.0f}%  {last} | {verdict}"
    )
