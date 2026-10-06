"""Measure generation quality on a sample of test questions.

python evaluation/run_faithfulness.py --name v1 --n 30
python evaluation/run_faithfulness.py --name v1 --judge-model <a different Groq model>
"""

import argparse
import json
import random
import time
from pathlib import Path

from run_eval import QUESTIONS, assign_splits  # same folder, so it is on sys.path

from papertrail.generation.answer import answer_question
from papertrail.generation.faithfulness import score_answer
from papertrail.retrieval.registry import DEFAULT_RETRIEVER, RETRIEVERS

RESULTS = Path("evaluation/results")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument(
        "--retriever", choices=sorted(RETRIEVERS), default=DEFAULT_RETRIEVER
    )
    parser.add_argument("--model", help="generator model (default LLM_MODEL)")
    parser.add_argument(
        "--judge-model", help="judge model (default: same as generator)"
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    out_path = RESULTS / f"faithfulness_{args.name}.json"
    if out_path.exists() and not args.force:
        parser.error(f"{out_path} exists; choose another --name or pass --force")

    questions = [
        json.loads(ln) for ln in QUESTIONS.read_text().splitlines() if ln.strip()
    ]
    splits = assign_splits(questions)
    pool = sorted(
        (q for q in questions if splits[q["id"]] == "test"), key=lambda q: q["id"]
    )
    sample = random.Random(5).sample(pool, min(args.n, len(pool)))

    rows = []
    for i, q in enumerate(sample, 1):
        ans = answer_question(q["question"], retriever=args.retriever, model=args.model)
        score = score_answer(ans, judge_model=args.judge_model)
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "answer": ans.text,
                "verified": ans.verified,
                "score": score,
            }
        )
        print(
            f"[{i}/{len(sample)}] {q['id']} faithfulness={score['faithfulness']}",
            end="\r",
        )
        time.sleep(0.5)

    answered = [r for r in rows if not r["score"]["abstained"]]
    cited = sum(r["score"]["cited_sentences"] for r in answered)
    supported = sum(r["score"]["supported_sentences"] for r in answered)
    total = sum(r["score"]["total_sentences"] for r in answered)
    uncited = sum(r["score"]["uncited_sentences"] for r in answered)
    fully = sum(
        1
        for r in answered
        if r["verified"]
        and r["score"]["supported_sentences"] == r["score"]["cited_sentences"]
    )
    summary = {
        "questions": len(rows),
        "abstain_rate": 1 - len(answered) / len(rows),
        "faithfulness": supported / cited if cited else None,
        "uncited_sentence_rate": uncited / total if total else None,
        "fully_verified_answers": fully / len(answered) if answered else None,
    }
    print("\n" + json.dumps(summary, indent=2))
    print("\nUnsupported sentences to spot-check by hand:")
    for r in rows:
        for s in r["score"]["unsupported"][:2]:
            print(f"  {r['id']}: {s[:140]}")

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "name": args.name,
                "retriever": args.retriever,
                "summary": summary,
                "rows": rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
