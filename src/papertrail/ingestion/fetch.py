from __future__ import annotations

import re
import time
from pathlib import Path

import arxiv
import httpx

from papertrail.db.repository import Paper

DATA_DIR = Path(__file__).resolve().parents[3] / "data"
PDF_DIR = DATA_DIR / "raw_pdfs"
HEADERS = {"User-Agent": "PaperTrail/0.1 (personal research project)"}
DEFAULT_QUERY = "cat:cs.CL"

_VERSION = re.compile(r"v\d+$")


def fetch_papers(query: str = DEFAULT_QUERY, max_results: int = 20) -> list[Paper]:
    """Fetch paper metadata from arXiv, newest first."""
    client = arxiv.Client(
        page_size=min(max_results, 100), delay_seconds=3.0, num_retries=3
    )
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.SubmittedDate,
        sort_order=arxiv.SortOrder.Descending,
    )
    papers = []
    for r in client.results(search):
        pdf_url = (r.pdf_url or "").replace("http://", "https://") or None
        papers.append(
            Paper(
                arxiv_id=_VERSION.sub("", r.get_short_id()),  # drop the version suffix
                title=" ".join(r.title.split()),
                abstract=" ".join(r.summary.split()),
                authors=[a.name for a in r.authors],
                categories=list(r.categories),
                published=r.published,
                pdf_url=pdf_url,
            )
        )
    return papers


def download_pdf(paper: Paper) -> Path:
    """Download a paper's PDF into data/raw_pdfs (skipped if already there)."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    path = PDF_DIR / f"{paper.arxiv_id.replace('/', '_')}.pdf"
    if path.exists():
        return path
    if not paper.pdf_url:
        raise ValueError("no PDF URL")
    resp = httpx.get(paper.pdf_url, headers=HEADERS, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    if not resp.content.startswith(b"%PDF"):
        raise ValueError("response was not a PDF")
    path.write_bytes(resp.content)
    time.sleep(3)  # be polite to arXiv's servers
    return path