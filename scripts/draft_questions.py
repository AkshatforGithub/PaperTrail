"""Draft candidate eval questions with an LLM (Groq), pre-checked against the source chunk.

  python scripts/draft_questions.py --papers 75 --per-paper 2 --seed 42
Writes evaluation/candidates.jsonl for review with scripts/review_questions.py.
"""

import argparse
import json
import random
import re
import time
from collections import Counter
from pathlib import Path

import httpx
import psycopg

from papertrail.config import get_settings

CORPUS = Path("evaluation/corpus.txt")
OUT = Path("evaluation/candidates.jsonl")
SKIP_PAPERS = {"2610.01378"}  # parsed poorly; see diagnose_short.py
SKIP_SECTIONS = [
    "Abstract",
    "Acknowledgements",
    "Acknowledgments",
    "References",
    "Bibliography",
]
TYPES = {"numeric", "acronym", "method", "comparison", "definition"}
BANNED = (
    "this paper",
    "the paper",
    "the authors",
    "the passage",
    "the text",
    "this work",
    "this study",
)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

SYSTEM = (
    "You write evaluation questions for a retrieval system over arXiv papers. "
    "You reply with JSON only."
)

PROMPT = '''Passage from the paper "{title}" (section: {section}):

"""{passage}"""

Write ONE question that can be answered from this passage alone.

Rules:
1. Self-contained: someone who has not seen the paper must be able to tell what the question is about. Name the specific method, benchmark, dataset or system exactly as the passage does. Never say "this paper", "the authors", "the passage" or "the text".
2. Paraphrase. Do not copy runs of 5 or more consecutive words from the passage, except proper names.
3. Ask about one concrete fact: a number, a finding, a method detail, a definition or a comparison.
4. "evidence" must be an exact contiguous span of 4 to 12 words copied character for character from the passage, which states the answer.
5. If the passage has no concrete, answerable fact (boilerplate, bare lists of citations, acknowledgements), reply {{"skip": true}}.

Reply with JSON only: {{"question": "...", "evidence": "...", "type": "numeric|acronym|method|comparison|definition"}}'''


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def shares_5gram(question: str, passage: str) -> bool:
    q, p = words(question), words(passage)
    grams = {tuple(p[i : i + 5]) for i in range(len(p) - 4)}
    return any(tuple(q[i : i + 5]) in grams for i in range(len(q) - 4))


def alpha_ratio(s: str) -> float:
    return sum(c.isalpha() for c in s) / max(len(s), 1)


def sample_chunks(conn, ids, n_papers, per_paper, rng):
    rows = conn.execute(
        """
        WITH n AS (SELECT arxiv_id, count(*) AS n FROM chunks GROUP BY arxiv_id)
        SELECT c.arxiv_id, p.title, c.chunk_index, c.section, c.content
        FROM chunks c
        JOIN papers p ON p.arxiv_id = c.arxiv_id
        JOIN n ON n.arxiv_id = c.arxiv_id
        WHERE c.arxiv_id = ANY(%s) AND n.n >= 20 AND length(c.content) >= 500
          AND coalesce(c.section, '') <> ALL(%s)
        """,
        (ids, SKIP_SECTIONS),
    ).fetchall()

    by_paper: dict[str, list[tuple]] = {}
    for arxiv_id, title, idx, section, content in rows:
        if arxiv_id in SKIP_PAPERS or alpha_ratio(content) < 0.7:
            continue
        by_paper.setdefault(arxiv_id, []).append((title, idx, section, content))

    papers = sorted(by_paper)
    rng.shuffle(papers)
    picked = []
    for arxiv_id in papers[:n_papers]:
        pool = by_paper[arxiv_id]
        rng.shuffle(pool)
        chosen: list[tuple] = []
        for item in pool:
            if all(
                abs(item[1] - c[1]) > 1 for c in chosen
            ):  # avoid overlapping neighbours
                chosen.append(item)
            if len(chosen) == per_paper:
                break
        picked += [(arxiv_id, *c) for c in chosen]
    rng.shuffle(picked)  # mix papers so a partial run still spans the corpus
    return picked


def ask(client: httpx.Client, key: str, model: str, prompt: str) -> str:
    for attempt in range(6):
        resp = client.post(
            GROQ_URL,
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "temperature": 0.2,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": prompt},
                ],
            },
            timeout=60,
        )
        if resp.status_code == 429:
            time.sleep(float(resp.headers.get("retry-after", 5 * (attempt + 1))))
            continue
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]
    raise RuntimeError("rate-limited too many times")


def parse_json(text: str):
    m = re.search(r"\{.*\}", text, re.DOTALL)
    return json.loads(m.group(0)) if m else None


def check(data: dict, passage: str):
    """Return (question, evidence, type, None) if valid, else (None, None, None, reason)."""
    q = str(data.get("question", "")).strip()
    ev = str(data.get("evidence", "")).strip()
    t = str(data.get("type", "")).strip().lower()
    reason = None
    if not q.endswith("?") or not 25 <= len(q) <= 250:
        reason = "bad question format"
    elif any(b in q.lower() for b in BANNED):
        reason = "refers to 'the paper'"
    elif not 3 <= len(ev.split()) <= 15:
        reason = "bad evidence length"
    elif norm(ev) not in norm(passage):
        reason = "evidence not verbatim"
    elif shares_5gram(q, passage):
        reason = "copies passage wording"
    elif t not in TYPES:
        reason = "bad type"
    return (None, None, None, reason) if reason else (q, ev, t, None)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Groq model name (or set LLM_MODEL in .env)")
    parser.add_argument("--papers", type=int, default=75)
    parser.add_argument("--per-paper", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sleep", type=float, default=1.0)
    args = parser.parse_args()

    settings = get_settings()
    model = args.model or settings.llm_model
    if not settings.groq_api_key or not model:
        parser.error("set GROQ_API_KEY and LLM_MODEL in .env (or pass --model)")

    ids = [ln.strip() for ln in CORPUS.read_text().splitlines() if ln.strip()]
    with psycopg.connect(settings.database_url) as conn:
        picked = sample_chunks(
            conn, ids, args.papers, args.per_paper, random.Random(args.seed)
        )
    print(
        f"Sampled {len(picked)} chunks from {len({p[0] for p in picked})} papers; model={model}"
    )

    kept = 0
    reasons: Counter[str] = Counter()
    with httpx.Client() as client, OUT.open("w") as out:
        for n, (arxiv_id, title, idx, section, content) in enumerate(picked, 1):
            prompt = PROMPT.format(title=title, section=section or "-", passage=content)
            try:
                data = parse_json(ask(client, settings.groq_api_key, model, prompt))
            except (httpx.HTTPError, ValueError, KeyError, RuntimeError) as exc:
                reasons[f"error: {type(exc).__name__}"] += 1
                continue
            if not isinstance(data, dict):
                reasons["no json"] += 1
            elif data.get("skip"):
                reasons["model skipped"] += 1
            else:
                q, ev, t, why = check(data, content)
                if why:
                    reasons[why] += 1
                else:
                    kept += 1
                    out.write(
                        json.dumps(
                            {
                                "cand": f"{arxiv_id}:{idx}",
                                "arxiv_id": arxiv_id,
                                "title": title,
                                "section": section,
                                "question": q,
                                "evidence": ev,
                                "type": t,
                                "_context": content,
                            }
                        )
                        + "\n"
                    )
                    out.flush()
            print(f"[{n}/{len(picked)}] kept {kept}", end="\r")
            time.sleep(args.sleep)

    print(f"\nKept {kept} candidates -> {OUT}")
    for why, count in reasons.most_common():
        print(f"  rejected ({why}): {count}")


if __name__ == "__main__":
    main()
