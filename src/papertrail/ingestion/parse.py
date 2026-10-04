from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

KNOWN_HEADINGS = {
    "abstract", "introduction", "related work", "background", "method",
    "methods", "methodology", "approach", "experiments", "experimental setup",
    "results", "discussion", "conclusion", "conclusions", "limitations",
    "acknowledgements", "acknowledgments", "references", "bibliography",
}
STOP_HEADINGS = {"references", "bibliography"}
NUMBERED = re.compile(r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z0-9 ,:;\-&()/']{2,80})$")


@dataclass
class Section:
    title: str
    text: str


def _is_heading(line: str) -> bool:
    if len(line) > 80:
        return False
    return line.lower() in KNOWN_HEADINGS or bool(NUMBERED.match(line))


def _heading_title(line: str) -> str:
    m = NUMBERED.match(line)
    return m.group(2).strip() if m else line.title()

def _extract_lines(path: Path) -> list[str]:
    with pymupdf.open(path) as doc:
        raw = "\n".join(page.get_text("text") for page in doc)
    raw = raw.replace("\x00", "") 
    raw = re.sub(r"(\w)-\n(\w)", r"\1\2", raw)  # rejoin words hyphenated across lines
    lines = [ln.strip() for ln in raw.splitlines()]
    # drop blank lines and bare page numbers
    return [ln for ln in lines if ln and not re.fullmatch(r"\d{1,3}", ln)]


def _make_section(title: str, buf: list[str]) -> Section | None:
    text = re.sub(r"\s+", " ", " ".join(buf)).strip()
    return Section(title, text) if text else None


def parse_pdf(path: Path) -> list[Section]:
    """Split a paper into titled sections, stopping at the reference list.

    Heading detection is heuristic (numbered headings and common names), so
    some papers will come out with fewer or odd sections. That is fine here:
    the section is just a label stored alongside each chunk.
    """
    sections: list[Section] = []
    title, buf = "Front matter", []
    for line in _extract_lines(path):
        if _is_heading(line):
            if (s := _make_section(title, buf)):
                sections.append(s)
            title, buf = _heading_title(line), []
            if title.lower() in STOP_HEADINGS:
                return sections
        else:
            buf.append(line)
    if (s := _make_section(title, buf)):
        sections.append(s)
    return sections