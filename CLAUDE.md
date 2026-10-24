# PaperTrail - CLAUDE.md

## Rules

- You have permission to run the test suite: `python3 -m pytest tests/ -v --tb=short`. The Postgres container (`docker compose up -d`) is running with chunks ingested, so all 36 tests should run with none skipped.

## Project overview

PaperTrail is a citation-verified question answering system over arXiv cs.CL papers. Retrieval is hybrid: pgvector dense search plus Postgres full-text search, fused with RRF and then reranked by a cross-encoder. An LLM (Groq, OpenAI-compatible API) writes the answer with mandatory numbered citations, and a judge model can verify them.

## Stack

- Python, FastAPI, psycopg 3 with raw SQL (no ORM)
- Postgres 16 + pgvector via docker-compose (HNSW cosine index, GIN index on the tsv column)
- Embeddings: BAAI/bge-small-en-v1.5 (384-dim). Reranker: cross-encoder/ms-marco-MiniLM-L-6-v2

## Layout

- src/papertrail/: app code (api, retrieval, db, ingestion, generation)
- evaluation/: frozen eval set, run_eval.py, results/
- scripts/: one-off pipeline scripts
- tests/: pytest suite

## Commands

- Start DB: `docker compose up -d`
- Run tests: `python3 -m pytest tests/ -v --tb=short`
- Run API: `uvicorn papertrail.api.main:app --reload`

## Secrets

- Never print, paste, log, or commit the Groq API key or .env contents.
- .env stays untracked. Before any push, confirm it is gitignored.

## Conventions

- Match existing patterns: dataclasses, pydantic models, type hints, `from __future__ import annotations`.
- Keep the citation format strict: [1], [2][3]. Changing it breaks answer parsing.
- Chunking (800 chars / 100 overlap), the embedding model, and the default retriever are validated settings. Do not change them without re-running the evaluation.
- API errors use FastAPI HTTPException with proper status codes (422, 502, 503).

## Working style

- Change only the files the task needs. No drive-by refactors.
- Add or update tests with any new behavior, and run the full suite before saying a task is done.
- Do not commit or push unless asked.
- Explain what changed and why in plain terms. The owner must be able to explain every design choice in an interview.
- Keep responses short and avoid dumping large files into the chat.

## Next Steps (one by one)

1. Build POST /ask to return the answer with numbered citations, cited sentences, judge-supported flags per sentence, and retrieved chunks with rerank scores. Keep GET /health.
2. Add guardrails: per-IP rate limit (slowapi), max question length, daily LLM-call cap, provider spend limit.
3. Build the frontend as a single static HTML/JS page served from FastAPI: search box, numbered citations, collapsible "how retrieved" panel (vector scores, keyword matches, rerank scores), 3-4 example questions, headline-numbers banner.
4. Containerize with docker-compose: app + pgvector, named volume, .env not committed.
5. Deploy on an Azure Ubuntu VM (B2s): ports 22/80/443 only, key-only SSH, Caddy for HTTPS, restart unless-stopped.
6. Harden and verify: external rate-limit test, no key in logs, pg_dump backup, Langfuse trace per request.
