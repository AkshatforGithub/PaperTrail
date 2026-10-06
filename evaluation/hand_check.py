"""Write flagged and sampled passed answers with their sources to a text file.

    python evaluation/hand_check.py --name v1 [--passed 10]
"""
import argparse
import json
import random
from pathlib import Path

from papertrail.retrieval.registry import RETRIEVERS

TERMS = {"q024": ["scaling factor", "low-rank", "lora", "sliding"]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--passed", type=int, default=10)
    args = ap.parse_args()

    d = json.loads(
        Path(f"evaluation/results/faithfulness_{args.name}.json").read_text()
    )
    retrieve = RETRIEVERS[d["retriever"]]
    rows = [r for r in d["rows"] if not r["score"]["abstained"]]
    flagged = [r for r in rows if r["score"]["unsupported"]]
    passed = [r for r in rows if not r["score"]["unsupported"]]
    random.Random(13).shuffle(passed)
    chosen = [("FLAGGED", r) for r in flagged] + [
        ("PASSED", r) for r in passed[: args.passed]
    ]

    out = []
    for label, r in chosen:
        sources = retrieve(r["question"], k=5)
        out.append("=" * 100)
        out.append(f"{label} {r['id']}: {r['question']}")
        out.append(f"ANSWER: {r['answer']}")
        if label == "FLAGGED":
            out.append(f"JUDGE FLAGGED: {r['score']['unsupported']}")
        for term in TERMS.get(r["id"], []):
            hits = [i for i, s in enumerate(sources, 1) if term in s.content.lower()]
            out.append(f"  term '{term}' found in sources: {hits or 'none'}")
        for i, s in enumerate(sources, 1):
            out.append(f"--- [{i}] {s.arxiv_id} | {s.section}")
            out.append(s.content)
        out.append("")

    path = Path(f"evaluation/results/handcheck_{args.name}.txt")
    path.write_text("\n".join(out))
    print(f"{len(flagged)} flagged, {min(args.passed, len(passed))} passed -> {path}")


if __name__ == "__main__":
    main()
