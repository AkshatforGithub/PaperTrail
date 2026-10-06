"""Faithfulness: is each cited sentence actually supported by the sources it cites?

An LLM judge answers yes/no per sentence. It is an approximation, so spot-check the
flagged sentences by hand, and prefer a judge model different from the generator.
"""

from __future__ import annotations

import json
import re

from papertrail.generation.answer import (
    CITE_RE,
    Answer,
    parse_citations,
    split_sentences,
)
from papertrail.generation.llm import chat
from papertrail.retrieval.vector_search import RetrievedChunk

JUDGE_SYSTEM = "You check whether a claim is supported by source text. Reply with JSON only."
JUDGE_PROMPT = """Sources:

{sources}

Claim: {claim}

Is every factual statement in the claim directly supported by the sources above? \
Ignore citation markers. Reply {{"supported": true}} or {{"supported": false}}."""


def judge_claim(claim: str, sources: list[tuple[int, RetrievedChunk]], model: str | None) -> bool:
    text = "\n\n".join(f"[{i}] {c.content}" for i, c in sources)
    raw = chat(
        [
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_PROMPT.format(sources=text, claim=claim)},
        ],
        model=model,
        max_tokens=600,
    )
    match = re.search(r"\{.*?\}", raw, re.DOTALL)
    try:
        return bool(match and json.loads(match.group(0)).get("supported") is True)
    except json.JSONDecodeError:
        return False  # unparseable verdicts count as unsupported


def score_answer(answer: Answer, judge_model: str | None = None) -> dict:
    sentences = [] if answer.abstained else split_sentences(answer.text)
    result = {
        "abstained": answer.abstained,
        "total_sentences": len(sentences),
        "cited_sentences": 0,
        "supported_sentences": 0,
        "uncited_sentences": len(answer.uncited_sentences),
        "invalid_citations": len(answer.invalid_citations),
        "unsupported": [],
    }
    for sentence in sentences:
        idxs = sorted({i for i in parse_citations(sentence) if 1 <= i <= len(answer.sources)})
        if not idxs:
            continue
        result["cited_sentences"] += 1
        claim = CITE_RE.sub("", sentence).strip()
        if judge_claim(claim, [(i, answer.sources[i - 1]) for i in idxs], judge_model):
            result["supported_sentences"] += 1
        else:
            result["unsupported"].append(sentence)
    cited = result["cited_sentences"]
    result["faithfulness"] = result["supported_sentences"] / cited if cited else None
    return result
