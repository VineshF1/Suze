"""Document ingestion pipeline — extract, chunk, embed, store."""

import os
import tempfile
from pathlib import Path
from typing import Optional

from src.config import settings
from src.nvidia_embed import NvidiaEmbeddingFunction
from src.ingestion.chunker import Document, DocumentChunk, TextChunker
from src.retrieval.vector_store import VectorStore


class IngestionPipeline:
    """Full ingestion pipeline: read → chunk → embed → store."""

    def __init__(self, vector_store: VectorStore, chunker: Optional[TextChunker] = None):
        self.vs = vector_store
        self.chunker = chunker or TextChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    def ingest_text(self, content: str, source: str = "manual", metadata: Optional[dict] = None) -> int:
        """Ingest a plain text document."""
        doc = Document(content=content, source=source, doc_type="text", metadata=metadata or {})
        return self._ingest(doc)

    def ingest_file(self, filepath: str, metadata: Optional[dict] = None) -> int:
        """Ingest a file (PDF, MD, TXT) by path."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {filepath}")

        ext = path.suffix.lower()
        content, doc_type = self._extract(path, ext)
        doc = Document(content=content, source=path.name, doc_type=doc_type, metadata=metadata or {})
        return self._ingest(doc)

    def _extract(self, path: Path, ext: str) -> tuple[str, str]:
        """Extract text from file based on extension."""
        if ext == ".pdf":
            return self._extract_pdf(path), "pdf"
        elif ext in (".md", ".markdown"):
            return path.read_text(encoding="utf-8", errors="replace"), "markdown"
        elif ext == ".txt":
            return path.read_text(encoding="utf-8", errors="replace"), "text"
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    def _extract_pdf(self, path: Path) -> str:
        """Extract text from PDF using PyMuPDF."""
        try:
            import fitz  # PyMuPDF
        except ImportError:
            raise RuntimeError("PyMuPDF (fitz) is required for PDF extraction. Install: pip install pymupdf")
        text_parts = []
        with fitz.open(str(path)) as doc:
            for page in doc:
                text_parts.append(page.get_text())
        return "\n\n".join(text_parts)

    def _ingest(self, document: Document) -> int:
        """Chunk, embed, and store a document. Returns chunk count."""
        chunks = self.chunker.chunk(document)
        if not chunks:
            return 0
        self.vs.add_chunks(chunks)
        return len(chunks)
