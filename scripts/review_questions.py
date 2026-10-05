"""Review drafted questions and build evaluation/questions.jsonl (the hand-verified set).

  python scripts/review_questions.py --target 75
Keys: y accept | n reject | e edit question | t change type | s skip for now | q quit
Safe to quit and resume: accepted questions and rejections are saved as you go.
"""

import argparse
import json
import textwrap
from pathlib import Path

CANDS = Path("evaluation/candidates.jsonl")
OUT = Path("evaluation/questions.jsonl")
REJ = Path("evaluation/rejected.txt")
TYPES = ["numeric", "acronym", "method", "comparison", "definition"]
BOLD, RESET = "\033[1;33m", "\033[0m"

CHECKLIST = """For every question, ask yourself:
  1. Is the answer really in the passage?
  2. Can the question be understood without the paper title?
  3. Could a DIFFERENT paper also answer it? (reject if so)
  4. Is it phrased differently from the passage? (edit with 'e' if it just mirrors it)
"""


def load(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text().splitlines() if ln.strip()]


def show(c: dict, done: int, target: int) -> None:
    ctx, ev = c["_context"], c["evidence"]
    i = ctx.lower().find(ev.lower())
    if i >= 0:
        ctx = ctx[:i] + BOLD + ctx[i : i + len(ev)] + RESET + ctx[i + len(ev) :]
    print("\n" + "=" * 90)
    print(
        f"[{done}/{target} accepted]  {c['title'][:70]}  (arXiv:{c['arxiv_id']}, {c['section']})"
    )
    print(f"\nQUESTION: {c['question']}")
    print(f"EVIDENCE: {ev}")
    print(f"TYPE:     {c['type']}\n")
    print(textwrap.fill(ctx, 100, replace_whitespace=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=75)
    args = parser.parse_args()

    accepted = load(OUT)
    taken = {(q["arxiv_id"], q["evidence"]) for q in accepted}
    rejected = set(REJ.read_text().split()) if REJ.exists() else set()
    print(CHECKLIST)

    for c in load(CANDS):
        if len(accepted) >= args.target:
            break
        if c["cand"] in rejected or (c["arxiv_id"], c["evidence"]) in taken:
            continue
        show(c, len(accepted), args.target)
        while True:
            try:
                key = input("\n[y/n/e/t/s/q] > ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                key = "q"
            if key == "t":
                new = input(f"type {TYPES} > ").strip().lower()
                if new in TYPES:
                    c["type"] = new
                continue
            if key == "e":
                new = input("new question > ").strip()
                if not new.endswith("?"):
                    print("must end with '?'")
                    continue
                c["question"] = new
                key = "y"
            if key == "y":
                rec = {
                    "id": f"q{len(accepted) + 1:03d}",
                    "question": c["question"],
                    "arxiv_id": c["arxiv_id"],
                    "evidence": c["evidence"],
                    "type": c["type"],
                }
                with OUT.open("a") as f:
                    f.write(json.dumps(rec) + "\n")
                accepted.append(rec)
            elif key == "n":
                with REJ.open("a") as f:
                    f.write(c["cand"] + "\n")
            elif key == "q":
                print(f"\nSaved {len(accepted)} questions to {OUT}")
                return
            elif key != "s":
                continue
            break

    print(f"\n{len(accepted)} questions in {OUT}")


if __name__ == "__main__":
    main()
