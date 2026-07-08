"""Tests for the FastAPI API layer."""

import sys
from pathlib import Path

# Add backend/src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["service"] == "suze-rag"


@pytest.mark.asyncio
async def test_ingest_text(client):
    resp = await client.post(
        "/api/v1/ingest/text",
        json={"content": "Test document about artificial intelligence.", "source": "test.txt"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_ingested"] >= 1
    assert data["total_chunks"] >= 1


@pytest.mark.asyncio
async def test_ingest_empty_text(client):
    resp = await client.post(
        "/api/v1/ingest/text",
        json={"content": "", "source": "empty.txt"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["chunks_ingested"] == 0


@pytest.mark.asyncio
async def test_chat_endpoint(client):
    """Test chat endpoint — requires mock due to API key dependency."""
    # Ingest some content first
    await client.post(
        "/api/v1/ingest/text",
        json={
            "content": "The capital of France is Paris. Paris is known for the Eiffel Tower.",
            "source": "geography.txt",
            "metadata": {"access_level": "general", "tags": ["geography"]},
        },
    )

    # Mock the LLM calls in the graph nodes
    from unittest.mock import patch

    with patch("src.agent.nodes._call_llm") as mock_llm:
        def side_effect(prompt, **kwargs):
            if "Evaluate these criteria" in prompt:
                return '{"passed": true, "score": 0.95, "feedback": "Well grounded."}'
            elif "Reformulated query:" in prompt:
                return "Reformulated query"
            else:
                return "The capital of France is Paris, known for the Eiffel Tower."

        mock_llm.side_effect = side_effect

        resp = await client.post(
            "/api/v1/chat",
            json={"query": "What is the capital of France?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is the capital of France?"
        assert data["answer"] is not None


@pytest.mark.asyncio
async def test_chat_with_filters(client):
    from unittest.mock import patch

    with patch("src.agent.nodes._call_llm") as mock_llm:
        def side_effect(prompt, **kwargs):
            if "Evaluate these criteria" in prompt:
                return '{"passed": true, "score": 0.95, "feedback": "Well grounded."}'
            else:
                return "Paris is the capital."

        mock_llm.side_effect = side_effect

        resp = await client.post(
            "/api/v1/chat",
            json={
                "query": "What is the capital?",
                "access_level": "general",
                "tags": ["geography"],
            },
        )
        assert resp.status_code == 200


@pytest.mark.asyncio
async def test_reset(client):
    resp = await client.delete("/api/v1/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert "reset" in data["message"]


@pytest.mark.asyncio
async def test_ingest_file_unsupported_type(client):
    """Uploading unsupported file type should fail."""
    resp = await client.post(
        "/api/v1/ingest/file",
        files={"file": ("test.exe", b"fake content", "application/x-msdownload")},
    )
    assert resp.status_code == 400
