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


class IrrelevantCollection:
    def query(self, **kwargs):
        return {
            "documents": [["Unrelated document text."]],
            "metadatas": [[{"source": "unrelated.pdf", "page": 1}]],
            "distances": [[0.9]],
        }


class FailingOllamaClient:
    def chat(self, **kwargs):
        raise AssertionError("Ollama must not be called without relevant evidence")


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


def test_query_refuses_without_relevant_evidence():
    app.state.embedding_model = FakeEmbeddingModel()
    app.state.collection = IrrelevantCollection()
    app.state.rag_config = {
        "k": 6,
        "relevance_distance_threshold": 0.62,
        "llm_name": "llama3.2",
    }
    app.state.ollama_client = FailingOllamaClient()

    client = TestClient(app)
    response = client.post("/query", json={"question": "What is the capital of France?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "I don't know based on the documents.",
        "sources": [],
    }


def test_query_rejects_empty_question():
    client = TestClient(app)
    response = client.post("/query", json={"question": ""})

    assert response.status_code == 422
