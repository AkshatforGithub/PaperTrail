"""Sanity-check the faithfulness judge with deliberately wrong claims.

  python evaluation/judge_sanity.py --name smoke2 [--judge-model MODEL] [--max 15]
Real claims should be judged supported; corrupted claims (every number x3) should not.
"""

import argparse
import json
import re
from pathlib import Path

from papertrail.generation.answer import (
    CITE_RE,
    is_abstention,
    normalize_answer_text,
    parse_citations,
    split_sentences,
)
from papertrail.generation.faithfulness import judge_claim
from papertrail.retrieval.registry import RETRIEVERS


def corrupt(claim: str) -> str:
    return re.sub(r"\d+(?:\.\d+)?", lambda m: f"{float(m.group()) * 3:g}", claim)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name", required=True, help="faithfulness run to take answers from"
    )
    parser.add_argument("--judge-model")
    parser.add_argument("--max", type=int, default=15, help="max sentences to test")
    args = parser.parse_args()

    data = json.loads(
        Path(f"evaluation/results/faithfulness_{args.name}.json").read_text()
    )
    retrieve = RETRIEVERS[data["retriever"]]
    real_ok = bad_ok = n = 0
    leaks: list[str] = []

    for row in data["rows"]:
        if n >= args.max:
            break
        answer = normalize_answer_text(row["answer"])
        if is_abstention(answer):
            continue
        sources = retrieve(
            row["question"], k=5
        )  # deterministic: same sources as the answer saw
        for sentence in split_sentences(answer):
            if n >= args.max:
                break
            idxs = sorted(
                {i for i in parse_citations(sentence) if 1 <= i <= len(sources)}
            )
            claim = CITE_RE.sub("", sentence).strip()
            if not idxs or not re.search(r"\d", claim):
                continue
            cited = [(i, sources[i - 1]) for i in idxs]
            n += 1
            real_ok += judge_claim(claim, cited, args.judge_model)
            if judge_claim(corrupt(claim), cited, args.judge_model):
                bad_ok += 1
                leaks.append(corrupt(claim))
            print(f"tested {n}/{args.max}", end="\r")

    if n == 0:
        print(
            "\nNo sentences with numbers found; use a run with more answers (--n bigger)."
        )
        return
    print(f"\nclaims tested (sentences containing numbers): {n}")
    print(f"real claims judged supported:      {real_ok}/{n}  (want most)")
    print(f"corrupted claims judged supported: {bad_ok}/{n}  (want about 0)")
    for c in leaks[:3]:
        print("  judge accepted a WRONG claim:", c[:150])
    usable = real_ok / n >= 0.7 and bad_ok / n <= 0.2
    print(
        "verdict (rule of thumb):",
        "judge looks usable" if usable else "judge is unreliable",
    )


if __name__ == "__main__":
    main()
