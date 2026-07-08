"""Tests for the agent layer: state, nodes, graph with mocks."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import pytest
from unittest.mock import patch, MagicMock

from src.agent.state import GraphState
from src.agent.graph import build_rag_graph, router_after_critique
from src.retrieval.staged_filter import StagedFilter
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.ingestion.chunker import Document, TextChunker


@pytest.fixture
def empty_staged_filter(vector_store):
    hs = HybridSearch(vector_store)
    return StagedFilter(hs)


class TestGraphState:
    def test_initial_state(self):
        state = GraphState(query="What is AI?")
        assert state.query == "What is AI?"
        assert state.current_query == "What is AI?"  # Auto-initialized
        assert state.retry_count == 0

    def test_custom_max_retries(self):
        state = GraphState(query="test", max_retries=5)
        assert state.max_retries == 5


class TestRouter:
    def test_route_to_finalize_when_passed(self):
        state = GraphState(query="test", critique_passed=True)
        assert router_after_critique(state) == "finalize"

    def test_route_to_reformulate_when_failed_with_retries(self):
        state = GraphState(query="test", critique_passed=False, retry_count=0, max_retries=3)
        assert router_after_critique(state) == "reformulate"

    def test_route_to_fallback_when_exhausted(self):
        state = GraphState(query="test", critique_passed=False, retry_count=3, max_retries=3)
        assert router_after_critique(state) == "fallback"


class TestGraphBuilding:
    def test_graph_compiles(self, empty_staged_filter):
        """The graph should compile without errors."""
        graph = build_rag_graph(empty_staged_filter)
        assert graph is not None

    def test_graph_success_path(self, empty_staged_filter):
        """Test success path: retrieve → generate → critique → finalize."""
        graph = build_rag_graph(empty_staged_filter)

        # Mock the LLM calls inside nodes
        with patch("src.agent.nodes._call_llm") as mock_llm:
            # generate_node returns a grounded answer
            # critique_node returns passed=True
            # We need to handle 2 calls: generate then critique

            def side_effect(prompt, **kwargs):
                if "Evaluate these criteria" in prompt:
                    return json.dumps({"passed": True, "score": 0.95, "feedback": "Well grounded."})
                elif "Reformulated query:" in prompt:
                    return "Reformulated test query"
                else:
                    return "AI is artificial intelligence."

            mock_llm.side_effect = side_effect

            state = GraphState(query="What is AI?")
            result = graph.invoke(state, {"configurable": {"thread_id": "test-1"}})

            # Should have a final_response
            assert result.get("final_response") is not None
            # Should have trace entries
            assert len(result.get("trace", [])) > 0

    def test_graph_fallback_path(self, empty_staged_filter):
        """Test fallback path: all retries exhausted → fallback."""
        # The _staged_filter module var is set inside build_rag_graph()
        graph = build_rag_graph(empty_staged_filter)

        with patch("src.agent.nodes._call_llm") as mock_llm:
            def side_effect(prompt, **kwargs):
                if "Evaluate these criteria" in prompt:
                    return json.dumps({"passed": False, "score": 0.2, "feedback": "Not grounded."})
                elif "Reformulated query:" in prompt:
                    return "Reformulated: What is AI?"
                else:
                    return "Some answer about AI."

            mock_llm.side_effect = side_effect

            # Set max_retries to 1 for fast test
            state = GraphState(query="What is AI?", max_retries=1)
            result = graph.invoke(state, {"configurable": {"thread_id": "test-2"}})

            # After exhausting retries, should get fallback
            assert result.get("final_response") is not None
            trace = result.get("trace", [])
            node_names = [t["node"] for t in trace]
            assert "fallback" in node_names
