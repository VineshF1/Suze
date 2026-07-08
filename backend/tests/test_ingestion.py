"""Tests for the ingestion pipeline: chunker + pipeline."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.ingestion.chunker import TextChunker, Document, DocumentChunk
from src.ingestion.pipeline import IngestionPipeline


class TestTextChunker:
    def test_empty_text(self, chunker):
        chunks = chunker.chunk(Document(content=""))
        assert len(chunks) == 0

    def test_short_text_single_chunk(self, chunker):
        doc = Document(content="Short text.")
        chunks = chunker.chunk(doc)
        assert len(chunks) == 1
        assert "Short text." in chunks[0].text

    def test_long_text_multiple_chunks(self):
        chunker = TextChunker(chunk_size=100, overlap=10)
        text = "Word. " * 50
        doc = Document(content=text, source="test.txt")
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2
        for c in chunks:
            assert "source" in c.metadata
            assert c.metadata["source"] == "test.txt"

    def test_chunk_metadata(self, chunker, sample_document):
        chunks = chunker.chunk(sample_document)
        for c in chunks:
            assert "source" in c.metadata
            assert "doc_type" in c.metadata
            assert "chunk_index" in c.metadata
            assert "date_created" in c.metadata
            assert c.metadata["source"] == "ai_overview_2025.txt"

    def test_chunk_ids_unique(self, chunker, sample_document):
        chunks = chunker.chunk(sample_document)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))

    def test_paragraph_splitting(self):
        chunker = TextChunker(chunk_size=60, overlap=0)
        text = "First paragraph about artificial intelligence.\n\nSecond paragraph about machine learning and deep learning.\n\nThird paragraph about natural language processing."
        doc = Document(content=text)
        chunks = chunker.chunk(doc)
        assert len(chunks) >= 2


class TestIngestionPipeline:
    def test_ingest_text(self, vector_store):
        pipeline = IngestionPipeline(vector_store)
        count = pipeline.ingest_text("Hello world, this is a test document.", source="test.txt")
        assert count == 1
        assert vector_store.count() == 1

    def test_ingest_empty_text(self, vector_store):
        pipeline = IngestionPipeline(vector_store)
        count = pipeline.ingest_text("", source="empty.txt")
        assert count == 0

    def test_ingest_multiple_chunks(self, vector_store):
        pipeline = IngestionPipeline(vector_store)
        long_text = "Sentence one. " * 100
        count = pipeline.ingest_text(long_text, source="long.txt")
        assert count >= 2
        assert vector_store.count() >= 2

    def test_ingest_no_api_key_fallback(self, tmp_persist_dir):
        """Ingestion should work with ChromaDB's built-in ONNX embedding when no API key set."""
        # ChromaDB uses built-in ONNX all-MiniLM-L6-v2 when no embedding function provided
        from src.retrieval.vector_store import VectorStore
        vs = VectorStore(persist_dir=tmp_persist_dir)
        pipeline = IngestionPipeline(vs)
        count = pipeline.ingest_text("Test content without API key.", source="test.txt")
        assert count == 1
