"""Document chunking with metadata extraction."""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


@dataclass
class DocumentChunk:
    text: str
    metadata: dict = field(default_factory=dict)
    chunk_id: str = ""


@dataclass
class Document:
    content: str
    source: str = ""
    doc_type: str = "text"  # pdf, markdown, text
    metadata: dict = field(default_factory=dict)


class TextChunker:
    """Semantic-aware chunker that splits on boundaries, not fixed tokens."""

    def __init__(self, chunk_size: int = 500, overlap: int = 50):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, document: Document) -> list[DocumentChunk]:
        """Split document into chunks with metadata."""
        raw = document.content
        chunks = self._split_text(raw)
        result = []
        now_ts = datetime.now(timezone.utc).timestamp()  # Unix timestamp float
        for i, text in enumerate(chunks):
            meta = dict(document.metadata)
            meta.update({
                "source": document.source,
                "doc_type": document.doc_type,
                "chunk_index": i,
                "date_created": now_ts,
            })
            result.append(DocumentChunk(text=text, metadata=meta, chunk_id=f"{document.source}#chunk{i}"))
        return result

    def _split_text(self, text: str) -> list[str]:
        """Split text into chunks at sentence/paragraph boundaries."""
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) <= self.chunk_size:
            return [text] if text else []

        # Try paragraph breaks first
        paragraphs = re.split(r"\n\s*\n", text)
        if len(paragraphs) > 1:
            return self._merge_chunks(paragraphs)

        # Try sentence boundaries
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 1:
            return self._merge_chunks(sentences)

        # Fallback: char-level sliding
        return self._sliding_chunks(text)

    def _merge_chunks(self, segments: list[str]) -> list[str]:
        """Merge segments into chunks respecting size and overlap."""
        chunks = []
        current = ""
        for seg in segments:
            if not seg.strip():
                continue
            if len(current) + len(seg) + 1 <= self.chunk_size:
                current = (current + " " + seg).strip()
            else:
                if current:
                    chunks.append(current)
                current = seg
        if current:
            chunks.append(current)

        # Apply overlap by merging boundary words
        if self.overlap > 0 and len(chunks) > 1:
            overlapped = []
            for i, c in enumerate(chunks):
                if i > 0:
                    prev_end = chunks[i - 1][-self.overlap:] if len(chunks[i - 1]) > self.overlap else chunks[i - 1]
                    c = prev_end + " " + c
                overlapped.append(c)
            chunks = overlapped

        return chunks

    def _sliding_chunks(self, text: str) -> list[str]:
        """Character-level sliding window fallback."""
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + self.chunk_size, len(text))
            chunks.append(text[start:end])
            if end >= len(text):
                break
            start = end - self.overlap
            if start < 0:
                start = 0
        return chunks
