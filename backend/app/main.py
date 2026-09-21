from contextlib import asynccontextmanager
from pathlib import Path

import chromadb
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from app.api.routes.query import router as query_router
from app.core.config import get_settings
from app.services.generation import create_ollama_client
from app.utils.logging_config import configure_logging

configure_logging()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    vector_store = Path(settings.vector_store_path)
    config = settings.rag_config

    app.state.embedding_model = SentenceTransformer(config["embedding_model"])
    client = chromadb.PersistentClient(path=str(vector_store))
    app.state.collection = client.get_collection(config["collection_name"])
    app.state.ollama_client = create_ollama_client(settings.ollama_base_url)
    app.state.rag_config = config
    yield


settings = get_settings()
app = FastAPI(title="RAG Document Assistant API", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(query_router)
