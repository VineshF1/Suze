"""Test fixtures and configuration for all test suites."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from httpx import ASGITransport, AsyncClient

from src.main import app
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.staged_filter import StagedFilter
from src.ingestion.pipeline import IngestionPipeline
from src.ingestion.chunker import TextChunker, Document
from src.agent.graph import build_rag_graph


# ── Backend unit test fixtures ──────────────────────────────

@pytest.fixture
def tmp_persist_dir():
    """Temporary directory for ChromaDB persistence."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
        yield d


@pytest.fixture
def vector_store(tmp_persist_dir):
    # Prevent tests from making live NVIDIA API calls via embedding function
    import src.config as cfg
    cfg.settings.nvidia_api_key = ""
    return VectorStore(persist_dir=tmp_persist_dir)


@pytest.fixture
def hybrid_search(vector_store):
    return HybridSearch(vector_store)


@pytest.fixture
def staged_filter(hybrid_search):
    return StagedFilter(hybrid_search)


@pytest.fixture
def chunker():
    return TextChunker(chunk_size=200, overlap=20)


@pytest.fixture
def sample_document():
    return Document(
        content=(
            "This is a sample document about artificial intelligence. "
            "AI is transforming the way we work and live. "
            "Machine learning models can now generate text and images. "
            "Deep learning has revolutionized natural language processing. "
            "Transformers are the backbone of modern LLMs. "
            "This document covers advanced AI topics for the year 2025. "
            "It was created for the engineering department. "
            "Access level is general. "
            "The author is Dr. Smith. "
            "Tags include AI, machine-learning, NLP."
        ),
        source="ai_overview_2025.txt",
        doc_type="text",
        metadata={
            "access_level": "general",
            "department": "engineering",
            "author": "Dr. Smith",
            "tags": ["AI", "machine-learning", "NLP"],
        },
    )


@pytest.fixture
def populated_vs(vector_store, sample_document, chunker):
    chunks = chunker.chunk(sample_document)
    vector_store.add_chunks(chunks)
    return vector_store


# ── API test fixtures ──────────────────────────────────────

@pytest.fixture
def app_with_state():
    """Set up app state manually (httpx ASGITransport doesn't trigger lifespan)."""
    import src.config as cfg
    cfg.settings.nvidia_api_key = ""
    vs = VectorStore(persist_dir=tempfile.mkdtemp(prefix="suze_test_"))
    hybrid = HybridSearch(vs)
    sf = StagedFilter(hybrid)
    pipeline = IngestionPipeline(vs)
    graph = build_rag_graph(sf, enable_critique=False)
    tmp_dir = tempfile.mkdtemp(prefix="suze_uploads_test_")

    app.state.vs = vs
    app.state.pipeline = pipeline
    app.state.graph = graph
    app.state.temp_dir = tmp_dir

    yield app

    del app.state.vs
    del app.state.pipeline
    del app.state.graph
    del app.state.temp_dir


@pytest.fixture
async def client(app_with_state):
    transport = ASGITransport(app=app_with_state)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
