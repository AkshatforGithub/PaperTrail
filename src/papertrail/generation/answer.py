"""Answer a question from retrieved chunks, with numbered citations that are checked."""

from __future__ import annotations

import re
from dataclasses import dataclass

from papertrail.generation.llm import chat
from papertrail.retrieval.registry import DEFAULT_RETRIEVER, RETRIEVERS
from papertrail.retrieval.vector_search import RetrievedChunk

ABSTAIN = "I can't find this in the indexed papers."

SYSTEM = (
    "You answer questions about arXiv computer-science papers using ONLY the numbered "
    "sources provided. Rules: (1) Every sentence of your answer must cite the source "
    "number(s) that support it, written in plain ASCII square brackets like [1] or [2][3] "
    "(never 【1】 or any other bracket style) and placed before the final period. "
    "(2) Use only facts stated in the sources; never add outside knowledge. "
    "(3) Be concise: at most 4 sentences. (4) If the sources do not contain the answer, "
    f"reply exactly: {ABSTAIN} (with no citations)."
)

CITE_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_CITES_AFTER_PERIOD = re.compile(r"([.!?])\s*((?:\[\d+(?:\s*,\s*\d+)*\]\s*)+)")
_ABBREV = re.compile(r"\b(e\.g|i\.e|et al|vs|cf|approx|Figs?|Eqs?|Sec)\.", re.IGNORECASE)
_DOT = "\u2024"  # stand-in for a period that must not end a sentence


@dataclass
class Answer:
    question: str
    text: str
    sources: list[RetrievedChunk]
    cited: list[int]  # 1-based source numbers that were actually cited
    invalid_citations: list[int]  # cited numbers that do not exist
    uncited_sentences: list[str]
    abstained: bool

    @property
    def verified(self) -> bool:
        """Structural check only: every sentence cites a real source."""
        if self.abstained:
            return True
        return bool(self.cited) and not self.invalid_citations and not self.uncited_sentences


def normalize_answer_text(text: str) -> str:
    """Clean model quirks: odd bracket styles for citations, unusual spaces and hyphens."""
    text = text.replace("\uff3b", "[").replace("\uff3d", "]")
    text = re.sub(r"[【\[]\s*(\d+)\s*†[^】\]]*[】\]]", r"[\1]", text)  # 【1†L2-L4】 -> [1]
    text = re.sub(r"【\s*(\d+(?:\s*,\s*\d+)*)\s*】", r"[\1]", text)  # 【1】 -> [1]
    for odd in ("\u202f", "\u00a0", "\u2009", "\u200a"):
        text = text.replace(odd, " ")
    text = text.replace("\u2011", "-")
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    text = _ABBREV.sub(lambda m: m.group(1) + _DOT, text)  # "vs." must not end a sentence
    # "It rose. [1] Then it fell." -> "It rose [1]. Then it fell." so each citation
    # stays with the sentence it belongs to.
    text = _CITES_AFTER_PERIOD.sub(lambda m: f" {m.group(2).strip()}{m.group(1)} ", text)
    parts = [p.strip() for p in _SENTENCE_SPLIT.split(text) if p.strip()]
    return [p.replace(_DOT, ".") for p in parts]


def parse_citations(sentence: str) -> list[int]:
    return [int(n) for group in CITE_RE.findall(sentence) for n in re.findall(r"\d+", group)]


def is_abstention(text: str) -> bool:
    head = text.replace("\u2019", "'").lower()[:80]
    return "can't find" in head or "cannot find" in head


def analyze_answer(question: str, text: str, sources: list[RetrievedChunk]) -> Answer:
    text = normalize_answer_text(text)
    abstained = is_abstention(text)
    sentences = [] if abstained else split_sentences(text)
    cited = sorted({n for s in sentences for n in parse_citations(s)})
    return Answer(
        question=question,
        text=text,
        sources=sources,
        cited=[n for n in cited if 1 <= n <= len(sources)],
        invalid_citations=[n for n in cited if not 1 <= n <= len(sources)],
        uncited_sentences=[s for s in sentences if not parse_citations(s)],
        abstained=abstained,
    )


def format_context(chunks: list[RetrievedChunk]) -> str:
    return "\n\n".join(
        f"[{i}] ({c.title}; arXiv:{c.arxiv_id}; {c.section or 'n/a'})\n{c.content}"
        for i, c in enumerate(chunks, 1)
    )


def answer_question(
    question: str,
    k: int = 5,
    retriever: str = DEFAULT_RETRIEVER,
    model: str | None = None,
) -> Answer:
    chunks = RETRIEVERS[retriever](question, k=k)
    if not chunks:
        return analyze_answer(question, ABSTAIN, [])
    user = f"Sources:\n\n{format_context(chunks)}\n\nQuestion: {question}"
    text = chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}],
        model=model,
    )
    return analyze_answer(question, text, chunks)
