from fastapi.testclient import TestClient

from app.main import app


class FakeEmbeddingModel:
    def encode(self, questions, normalize_embeddings=True):
        return [[0.1, 0.2, 0.3]]


class FakeCollection:
    def query(self, **kwargs):
        return {
            "documents": [["A neural network learns patterns from data."]],
            "metadatas": [[{"source": "notes.pdf", "page": 3, "page_end": 5}]],
            "distances": [[0.2]],
        }


class FakeOllamaClient:
    def chat(self, **kwargs):
        return {"message": {"content": "It learns patterns [notes.pdf, page 3]."}}


def test_query_happy_path(monkeypatch):
    app.state.embedding_model = FakeEmbeddingModel()
    app.state.collection = FakeCollection()
    app.state.rag_config = {
        "k": 6,
        "relevance_distance_threshold": 0.62,
        "llm_name": "llama3.2",
    }
    app.state.ollama_client = FakeOllamaClient()

    client = TestClient(app)
    response = client.post("/query", json={"question": "What does it learn?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "It learns patterns [notes.pdf, page 3].",
        "sources": ["notes.pdf, pages 3-5"],
    }


def test_query_rejects_empty_question():
    client = TestClient(app)
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
