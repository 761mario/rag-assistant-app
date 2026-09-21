from fastapi import APIRouter, Request

from app.schemas.query import QueryRequest, QueryResponse
from app.services.generation import generate_answer
from app.services.retrieval import retrieve

router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/query", response_model=QueryResponse)
def query(payload: QueryRequest, request: Request) -> QueryResponse:
    retrieved = retrieve(
        question=payload.question,
        embedding_model=request.app.state.embedding_model,
        collection=request.app.state.collection,
        rag_config=request.app.state.rag_config,
    )
    if not retrieved:
        return QueryResponse(
            answer="I don't know based on the documents.",
            sources=[],
        )

    answer = generate_answer(
        question=payload.question,
        retrieved=retrieved,
        ollama_client=request.app.state.ollama_client,
        model_name=request.app.state.rag_config.get("llm_name", "llama3.2"),
    )
    return QueryResponse(
        answer=answer,
        sources=[item["source_label"] for item in retrieved],
    )
