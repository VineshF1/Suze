"""ChromaDB vector store wrapper."""

import os
import tempfile
from pathlib import Path
from typing import Optional

import chromadb
from chromadb.api.models.Collection import Collection

from chromadb.errors import NotFoundError

from src.config import settings
from src.ingestion.chunker import DocumentChunk
from src.nvidia_embed import NvidiaEmbeddingFunction


class VectorStore:
    """ChromaDB wrapper with NVIDIA NIM embedding support."""

    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or settings.chroma_persist_dir
        self._client = self._make_client()
        self._collection: Optional[Collection] = None

    def _make_client(self):
        """Create ChromaDB client with persistence."""
        os.makedirs(self.persist_dir, exist_ok=True)
        return chromadb.PersistentClient(path=self.persist_dir)

    @property
    def collection(self) -> Collection:
        if self._collection is None:
            self._collection = self.get_or_create_collection()
        return self._collection

    def get_or_create_collection(self) -> Collection:
        """Get existing collection or create new one with NVIDIA embedding if configured."""
        ef = None
        if settings.nvidia_api_key and settings.nvidia_api_key not in ("***", "your_key_here", "nvapi-YOUR_KEY_HERE"):
            ef = NvidiaEmbeddingFunction()

        name = settings.collection_name
        kwargs = {"name": name}
        if ef is not None:
            kwargs["embedding_function"] = ef
        try:
            return self._client.get_collection(**kwargs)
        except NotFoundError:
            return self._client.create_collection(**kwargs)

    def add_chunks(self, chunks: list[DocumentChunk]) -> None:
        """Add document chunks to the collection."""
        if not chunks:
            return
        col = self.collection
        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]
        col.add(documents=texts, metadatas=metadatas, ids=ids)

    def query(
        self,
        query_text: str,
        n_results: int = 10,
        where: Optional[dict] = None,
        where_document: Optional[dict] = None,
    ) -> dict:
        """Query the collection with optional metadata filters."""
        col = self.collection
        kwargs = {
            "query_texts": [query_text],
            "n_results": n_results,
        }
        if where:
            kwargs["where"] = where
        if where_document:
            kwargs["where_document"] = where_document
        return col.query(**kwargs)

    def get_all_chunks(self) -> list[dict]:
        """Get all chunks from the collection."""
        col = self.collection
        data = col.get()
        result = []
        for i in range(len(data["ids"])):
            result.append({
                "id": data["ids"][i],
                "text": data["documents"][i] if data["documents"] else "",
                "metadata": data["metadatas"][i] if data["metadatas"] else {},
            })
        return result

    def delete_collection(self) -> None:
        """Delete the collection."""
        try:
            self._client.delete_collection(settings.collection_name)
            self._collection = None
        except NotFoundError:
            pass

    def count(self) -> int:
        """Return number of chunks in collection."""
        return self.collection.count()
