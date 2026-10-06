"""FastAPI app.  Run:  uvicorn papertrail.api.main:app --reload"""

from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
import psycopg
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from papertrail.generation.answer import answer_question
from papertrail.generation.faithfulness import score_answer
from papertrail.retrieval.registry import DEFAULT_RETRIEVER, RETRIEVERS


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:  # load the embedding model once at startup
        RETRIEVERS[DEFAULT_RETRIEVER]("warm up", k=1)
    except Exception:  # noqa: BLE001 - the API should still start if the DB is down
        pass
    yield


app = FastAPI(title="PaperTrail", version="0.1.0", lifespan=lifespan)


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    k: int = Field(5, ge=1, le=10)
    retriever: str = DEFAULT_RETRIEVER
    verify: bool = False  # also run the LLM faithfulness judge (slower)


class SourceOut(BaseModel):
    n: int
    arxiv_id: str
    title: str
    section: str | None
    chunk_index: int
    score: float
    snippet: str


class AskResponse(BaseModel):
    answer: str
    abstained: bool
    verified: bool  # every sentence cites a real source
    cited: list[int]
    invalid_citations: list[int]
    uncited_sentences: list[str]
    faithfulness: float | None = None  # only when verify=true
    sources: list[SourceOut]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "default_retriever": DEFAULT_RETRIEVER}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    if req.retriever not in RETRIEVERS:
        raise HTTPException(422, f"retriever must be one of {sorted(RETRIEVERS)}")
    try:
        result = answer_question(req.question, k=req.k, retriever=req.retriever)
        faithfulness = score_answer(result)["faithfulness"] if req.verify else None
    except psycopg.OperationalError as exc:
        raise HTTPException(503, f"database unavailable: {exc}") from exc
    except (httpx.HTTPError, RuntimeError) as exc:
        raise HTTPException(502, f"LLM request failed: {exc}") from exc
    return AskResponse(
        answer=result.text,
        abstained=result.abstained,
        verified=result.verified,
        cited=result.cited,
        invalid_citations=result.invalid_citations,
        uncited_sentences=result.uncited_sentences,
        faithfulness=faithfulness,
        sources=[
            SourceOut(
                n=i, arxiv_id=c.arxiv_id, title=c.title, section=c.section,
                chunk_index=c.chunk_index, score=c.score, snippet=c.content[:300],
            )
            for i, c in enumerate(result.sources, 1)
        ],
    )
