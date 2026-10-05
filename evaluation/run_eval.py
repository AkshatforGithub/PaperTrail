"""Run the retrieval evaluation and save the results.

  python evaluation/run_eval.py --retriever vector --name baseline

A retrieved chunk counts as a CHUNK hit if it is from the gold paper AND contains the
gold evidence phrase; it counts as a PAPER hit if it is from the gold paper.
Metrics: recall@k (k = 1, 3, 5, 10) and MRR@10. Results go to evaluation/results/<name>.json.
Questions are split into a fixed dev set (tune on this) and test set (report this).
"""

import argparse
import json
import random
import re
import subprocess
import time
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import psycopg

from papertrail.config import get_settings
from papertrail.retrieval.vector_search import vector_search

QUESTIONS = Path("evaluation/questions.jsonl")
RESULTS = Path("evaluation/results")
KS = (1, 3, 5, 10)
MAX_K = max(KS)
DEV_SIZE = 20
SPLIT_SEED = 13

# Phase 6 adds more retrievers here: every entry is f(question, k, conn) -> list[chunk].
RETRIEVERS = {"vector": vector_search}

TRANS = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u02bc": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
        "\u00ad": "-",
        "\u00a0": " ",
    }
)


def fold(s: str) -> str:
    return re.sub(r"\s+", " ", s.translate(TRANS)).strip().lower()


def assign_splits(questions: list[dict]) -> dict[str, str]:
    ids = sorted(q["id"] for q in questions)
    random.Random(SPLIT_SEED).shuffle(ids)
    dev = set(ids[:DEV_SIZE])
    return {q["id"]: ("dev" if q["id"] in dev else "test") for q in questions}


def summarize(rows: list[dict], key: str) -> dict | None:
    n = len(rows)
    if n == 0:
        return None
    out = {"n": n}
    for k in KS:
        out[f"recall@{k}"] = sum(1 for r in rows if r[key] and r[key] <= k) / n
    out["mrr@10"] = sum(1 / r[key] for r in rows if r[key]) / n
    return out


def by_subset(rows: list[dict], key: str) -> dict:
    subsets = {
        "all": rows,
        "dev": [r for r in rows if r["split"] == "dev"],
        "test": [r for r in rows if r["split"] == "test"],
    }
    return {name: summarize(group, key) for name, group in subsets.items()}


def by_type(rows: list[dict], key: str) -> dict:
    groups = defaultdict(list)
    for r in rows:
        groups[r["type"]].append(r)
    return {t: summarize(g, key) for t, g in sorted(groups.items())}


def print_table(title: str, table: dict) -> None:
    print(f"\n{title}")
    header = (
        f"{'':<12}{'n':>4}"
        + "".join(f"{'R@' + str(k):>8}" for k in KS)
        + f"{'MRR@10':>9}"
    )
    print(header)
    for name, s in table.items():
        if not s:
            continue
        cells = "".join(f"{s[f'recall@{k}'] * 100:>7.1f}%" for k in KS)
        print(f"{name:<12}{s['n']:>4}{cells}{s['mrr@10']:>9.3f}")


def git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--retriever", choices=sorted(RETRIEVERS), default="vector")
    parser.add_argument("--name", required=True, help="run name, e.g. baseline")
    parser.add_argument(
        "--force", action="store_true", help="overwrite an existing run"
    )
    args = parser.parse_args()

    out_path = RESULTS / f"{args.name}.json"
    if out_path.exists() and not args.force:
        parser.error(f"{out_path} exists; choose another --name or pass --force")

    questions = [
        json.loads(ln) for ln in QUESTIONS.read_text().splitlines() if ln.strip()
    ]
    splits = assign_splits(questions)
    retrieve = RETRIEVERS[args.retriever]
    settings = get_settings()

    rows = []
    with psycopg.connect(settings.database_url) as conn:
        retrieve("warm up", k=1, conn=conn)  # load the model before timing
        for q in questions:
            start = time.perf_counter()
            results = retrieve(q["question"], k=MAX_K, conn=conn)
            latency_ms = (time.perf_counter() - start) * 1000
            ev = fold(q["evidence"])
            paper_rank = chunk_rank = None
            for rank, r in enumerate(results, 1):
                if r.arxiv_id != q["arxiv_id"]:
                    continue
                if paper_rank is None:
                    paper_rank = rank
                if chunk_rank is None and ev in fold(r.content):
                    chunk_rank = rank
            rows.append(
                {
                    "id": q["id"],
                    "type": q["type"],
                    "split": splits[q["id"]],
                    "arxiv_id": q["arxiv_id"],
                    "question": q["question"],
                    "paper_rank": paper_rank,
                    "chunk_rank": chunk_rank,
                    "latency_ms": round(latency_ms, 1),
                }
            )

    chunk_table = by_subset(rows, "chunk_rank")
    paper_table = by_subset(rows, "paper_rank")
    type_table = by_type(rows, "chunk_rank")
    print(f"Run '{args.name}' | retriever={args.retriever} | {len(rows)} questions")
    print_table(
        "CHUNK-level (right paper AND evidence phrase in the chunk)", chunk_table
    )
    print_table("PAPER-level (any chunk from the right paper)", paper_table)
    print_table("CHUNK-level by question type", type_table)
    mean_latency = sum(r["latency_ms"] for r in rows) / len(rows)
    print(f"\nmean query latency: {mean_latency:.0f} ms")

    misses = [r for r in rows if r["chunk_rank"] is None]
    print(
        f"\n{len(misses)} questions missed the top {MAX_K} at chunk level (first 10):"
    )
    for r in misses[:10]:
        print(
            f"  {r['id']} [{r['type']}] paper_rank={r['paper_rank']} :: {r['question'][:90]}"
        )

    RESULTS.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {
                "name": args.name,
                "retriever": args.retriever,
                "created": datetime.now(UTC).isoformat(timespec="seconds"),
                "git_commit": git_commit(),
                "config": {
                    "embedding_model": settings.embedding_model,
                    "chunk_size": settings.chunk_size,
                    "chunk_overlap": settings.chunk_overlap,
                },
                "split": {"seed": SPLIT_SEED, "dev_size": DEV_SIZE},
                "summary": {"chunk": chunk_table, "paper": paper_table},
                "by_type_chunk": type_table,
                "per_question": rows,
            },
            indent=2,
        )
    )
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
