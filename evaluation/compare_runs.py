"""Compare saved runs side by side.

python evaluation/compare_runs.py                      # every run, oldest first
python evaluation/compare_runs.py baseline hybrid ...  # chosen runs (first = reference)
python evaluation/compare_runs.py --md                 # markdown table for the README
"""

import argparse
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
HEADERS = [
    "Run",
    "Retriever",
    "Chunk",
    "Dev R@5",
    "Test R@5",
    "Test MRR@10",
    "Paper R@5",
    "dTest R@5",
    "ms/query",
]


def load_runs() -> list[dict]:
    runs = []
    for path in sorted(RESULTS.glob("*.json")):
        data = json.loads(path.read_text())
        if "summary" in data and "per_question" in data:  # skips faithfulness files
            runs.append(data)
    return sorted(runs, key=lambda r: r["created"])


def get(run: dict, level: str, subset: str, key: str) -> float | None:
    s = run["summary"][level].get(subset)
    return s[key] if s else None


def pct(x: float | None) -> str:
    return "n/a" if x is None else f"{x * 100:.1f}%"


def hit5(rank: int | None) -> bool:
    return rank is not None and rank <= 5


def table_rows(runs: list[dict], ref: dict) -> list[list[str]]:
    ref_test = get(ref, "chunk", "test", "recall@5")
    rows = []
    for r in runs:
        test5 = get(r, "chunk", "test", "recall@5")
        delta = (
            "-"
            if r is ref or None in (test5, ref_test)
            else f"{(test5 - ref_test) * 100:+.1f} pts"
        )
        latency = sum(q["latency_ms"] for q in r["per_question"]) / len(
            r["per_question"]
        )
        mrr = get(r, "chunk", "test", "mrr@10")
        rows.append(
            [
                r["name"],
                r["retriever"],
                str(r["config"]["chunk_size"]),
                pct(get(r, "chunk", "dev", "recall@5")),
                pct(test5),
                "n/a" if mrr is None else f"{mrr:.3f}",
                pct(get(r, "paper", "all", "recall@5")),
                delta,
                f"{latency:.0f}",
            ]
        )
    return rows


def print_plain(rows: list[list[str]]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(HEADERS, *rows, strict=True)]
    for line in [HEADERS, *rows]:
        print("  ".join(str(x).ljust(w) for x, w in zip(line, widths, strict=True)))


def print_markdown(rows: list[list[str]]) -> None:
    print("| " + " | ".join(HEADERS) + " |")
    print("|" + "---|" * len(HEADERS))
    for row in rows:
        print("| " + " | ".join(row) + " |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("names", nargs="*")
    parser.add_argument("--md", action="store_true")
    args = parser.parse_args()

    runs = load_runs()
    if args.names:
        by_name = {r["name"]: r for r in runs}
        missing = [n for n in args.names if n not in by_name]
        if missing:
            parser.error(f"unknown runs: {missing}; available: {sorted(by_name)}")
        runs = [by_name[n] for n in args.names]
    if not runs:
        parser.error("no saved runs in evaluation/results")
    ref = (
        runs[0]
        if args.names
        else next((r for r in runs if r["name"] == "baseline"), runs[0])
    )

    rows = table_rows(runs, ref)
    (print_markdown if args.md else print_plain)(rows)

    ref_ranks = {q["id"]: q["chunk_rank"] for q in ref["per_question"]}
    print(
        f"\nReference: {ref['name']}. Per-question changes at chunk-level recall@5 (all questions):"
    )
    for r in runs:
        if r is ref:
            continue
        fixed = broken = 0
        for q in r["per_question"]:
            was, now = hit5(ref_ranks.get(q["id"])), hit5(q["chunk_rank"])
            fixed += (not was) and now
            broken += was and (not now)
        print(f"  {r['name']}: {fixed} questions fixed, {broken} questions broken")
    n_test = get(ref, "chunk", "test", "n")
    print(
        f"\nTest set is n={n_test}: judge a change by fixed/broken counts, not by a few points."
    )
    print("Tune on dev, then report the test numbers.")


if __name__ == "__main__":
    main()
