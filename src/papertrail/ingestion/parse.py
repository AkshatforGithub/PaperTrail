from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf

KNOWN_HEADINGS = {
    "abstract",
    "introduction",
    "related work",
    "background",
    "method",
    "methods",
    "methodology",
    "approach",
    "experiments",
    "experimental setup",
    "results",
    "discussion",
    "conclusion",
    "conclusions",
    "limitations",
    "acknowledgements",
    "acknowledgments",
    "references",
    "bibliography",
}
STOP_HEADINGS = {"references", "bibliography"}
NUMBERED = re.compile(
    r"^(\d{1,2}(?:\.\d{1,2}){0,2})\.?\s+([A-Z][A-Za-z0-9 ,:;\-&()/']{2,80})$"
)

# When a line break falls right after one of these, it is almost always a real
# compound ("multi-query", "long-context"), not a soft hyphenation.
COMPOUND_PREFIXES = {
    "multi",
    "non",
    "self",
    "cross",
    "long",
    "short",
    "end",
    "fine",
    "pre",
    "post",
    "zero",
    "few",
    "low",
    "high",
    "large",
    "small",
    "open",
    "state",
    "human",
    "data",
    "task",
    "domain",
    "language",
    "context",
    "query",
    "instruction",
    "chain",
    "rule",
    "model",
    "text",
    "speech",
    "vision",
    "semi",
    "retrieval",
    "llm",
    "in",
    "out",
    "co",
}

_LINE_BREAK_HYPHEN = re.compile(r"(\w+)-\n(\w+)")
_HYPHENATED_TOKEN = re.compile(r"\b[A-Za-z]+-[A-Za-z]+\b")
_WORD = re.compile(r"\b[A-Za-z]+\b")


@dataclass
class Section:
    title: str
    text: str


def dehyphenate(raw: str) -> str:
    """Rejoin words split across lines without destroying real hyphens.

    "informa-\\ntion"        -> "information"      (soft hyphenation)
    "retrieval-\\naugmented" -> "retrieval-augmented" (real compound)
    """
    hyphenated = {m.lower() for m in _HYPHENATED_TOKEN.findall(raw)}
    words = {w.lower() for w in _WORD.findall(raw)}

    def repl(m: re.Match[str]) -> str:
        a, b = m.group(1), m.group(2)
        if f"{a}-{b}".lower() in hyphenated:  # the paper writes it hyphenated elsewhere
            return f"{a}-{b}"
        if f"{a}{b}".lower() in words:  # the paper writes it joined elsewhere
            return a + b
        if a.lower() in COMPOUND_PREFIXES:
            return f"{a}-{b}"
        return a + b

    return _LINE_BREAK_HYPHEN.sub(repl, raw)


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
    raw = raw.replace("\x00", "")  # Postgres text columns reject NUL bytes
    raw = re.sub("\u00ad\n?", "", raw)  # explicit soft hyphens
    raw = raw.replace("\u2010", "-").replace("\u2011", "-")  # unicode hyphens
    raw = dehyphenate(raw)
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
            if s := _make_section(title, buf):
                sections.append(s)
            title, buf = _heading_title(line), []
            if title.lower() in STOP_HEADINGS:
                return sections
        else:
            buf.append(line)
    if s := _make_section(title, buf):
        sections.append(s)
    return sections
