"""Try questions against the corpus.

python scripts/search.py "How does LoRA reduce trainable parameters?" -k 5
python scripts/search.py            # interactive mode
"""

import argparse
import textwrap

from papertrail.retrieval.vector_search import vector_search


def show(question: str, k: int | None) -> None:
    results = vector_search(question, k=k)
    print(f"\nQ: {question}\n" + "-" * 80)
    for rank, r in enumerate(results, 1):
        print(f"{rank}. [{r.score:.3f}] {r.title}  (arXiv:{r.arxiv_id})")
        print(f"   section: {r.section or '-'}  chunk #{r.chunk_index}")
        print(textwrap.indent(textwrap.fill(r.content[:400], 90), "   "))
        print()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="?")
    parser.add_argument(
        "-k", type=int, default=None, help="defaults to TOP_K in config"
    )
    args = parser.parse_args()

    if args.question:
        show(args.question, args.k)
        return
    while True:
        try:
            q = input("question> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if q:
            show(q, args.k)


if __name__ == "__main__":
    main()
