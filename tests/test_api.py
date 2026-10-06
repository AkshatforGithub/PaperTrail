from fastapi.testclient import TestClient

from papertrail.api import main
from papertrail.generation.answer import analyze_answer
from papertrail.retrieval.vector_search import RetrievedChunk

client = TestClient(main.app)  # no 'with', so the startup warm-up is skipped


def test_health():
    assert client.get("/health").json()["status"] == "ok"


def test_ask_returns_answer_and_sources(monkeypatch):
    chunk = RetrievedChunk(1, "2610.00001", "A Title", "Method", 3, "some chunk text", 0.5)
    monkeypatch.setattr(
        main, "answer_question",
        lambda q, k, retriever: analyze_answer(q, "It works [1].", [chunk]),
    )
    r = client.post("/ask", json={"question": "Does it work?"})
    body = r.json()
    assert r.status_code == 200
    assert body["verified"] and body["cited"] == [1]
    assert body["sources"][0]["arxiv_id"] == "2610.00001"


def test_unknown_retriever_is_rejected():
    r = client.post("/ask", json={"question": "anything", "retriever": "nope"})
    assert r.status_code == 422
