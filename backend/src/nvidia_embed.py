"""NVIDIA NIM embedding function for ChromaDB."""

import os
import re
from openai import OpenAI
from chromadb.api.types import Documents, Embeddings

from src.config import settings


class NvidiaEmbeddingFunction:
    """ChromaDB-compatible embedding function using NVIDIA NIM API."""

    @staticmethod
    def name() -> str:
        return "nvidia_nim"

    def __init__(self):
        self.model = settings.embed_model
        api_key = settings.nvidia_api_key
        if not api_key or api_key in ("***", "your_key_here", "nvapi-YOUR_KEY_HERE"):
            self._client = None
        else:
            self._client = OpenAI(api_key=api_key, base_url=settings.nvidia_base_url)

    def __call__(self, input: Documents) -> Embeddings:
        if self._client is None:
            msg = "NVIDIA_API_KEY not set — cannot use NvidiaEmbeddingFunction. "
            msg += "Either set the env var or use ChromaDB's built-in ONNX embedding."
            raise RuntimeError(msg)
        input = [re.sub(r"\s+", " ", t).strip() for t in input]
        resp = self._client.embeddings.create(model=self.model, input=input)
        return [d.embedding for d in resp.data]

    def embed_query(self, *args, **kwargs) -> list:
        text = args[0] if args else (kwargs.get("query") or kwargs.get("input", ""))
        if isinstance(text, list):
            return self.__call__(text)
        return self.__call__([text])
