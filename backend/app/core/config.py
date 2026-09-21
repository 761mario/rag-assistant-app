import json
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


BACKEND_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_DIR = BACKEND_DIR / "data" / "vector_store"


class Settings(BaseSettings):
    frontend_origin: str = "http://localhost:3000"
    ollama_base_url: str = "http://localhost:11434"
    llm_model: str | None = None

    model_config = SettingsConfigDict(
        env_file=BACKEND_DIR / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def vector_store_path(self) -> str:
        return str(VECTOR_STORE_DIR)

    @property
    def rag_config(self) -> dict:
        config_path = VECTOR_STORE_DIR / "rag_config.json"
        with config_path.open(encoding="utf-8") as file:
            config = json.load(file)
        if self.llm_model:
            config["llm_name"] = self.llm_model
        return config


@lru_cache
def get_settings() -> Settings:
    return Settings()
