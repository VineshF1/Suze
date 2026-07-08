"""Application configuration using pydantic-settings."""

import json
import os
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())


class Settings(BaseSettings):
    # NVIDIA NIM config
    nvidia_api_key: str = os.getenv("NVIDIA_API_KEY", "")
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"

    # Models
    embed_model: str = "nvidia/nv-embed-qa-4"
    llm_model: str = "meta/llama-3.1-8b-instruct"

    # ChromaDB
    chroma_persist_dir: str = str(Path(__file__).parent.parent / "chroma_data")
    collection_name: str = "suze_docs"

    # Chunking
    chunk_size: int = 500
    chunk_overlap: int = 50

    # Retrieval
    top_k_initial: int = 20
    top_k_final: int = 5

    # Agent
    max_retries: int = 3
    temperature: float = 0.2
    enable_critique: bool = False

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    cors_origins: list[str] = ["http://localhost:5173", "http://localhost:3000", "https://suze-frontend.vercel.app"]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, v):
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [x.strip() for x in v.split(",") if x.strip()]
        return v


settings = Settings()
