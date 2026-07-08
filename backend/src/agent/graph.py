"""LangGraph wiring for the self-correction RAG loop."""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from src.agent.state import GraphState
from src.agent.nodes import (
    retrieve_node,
    generate_node,
    critique_node,
    reformulate_node,
    fallback_node,
    finalize_node,
)
from src.retrieval.staged_filter import StagedFilter


def build_rag_graph(staged_filter: StagedFilter, enable_critique: bool = True) -> StateGraph:
    """Build and compile the LangGraph for the agentic RAG loop.

    When enable_critique=False, the graph skips the critique/reformulate/retry
    cycle and goes directly from generate → finalize for maximum speed.
    """
    import src.agent.nodes as nodes
    nodes._staged_filter = staged_filter  # noqa

    builder = StateGraph(GraphState)

    builder.add_node("retrieve", retrieve_node)
    builder.add_node("generate", generate_node)
    builder.add_node("finalize", finalize_node)

    builder.set_entry_point("retrieve")
    builder.add_edge("retrieve", "generate")

    if enable_critique:
        builder.add_node("critique", critique_node)
        builder.add_node("reformulate", reformulate_node)
        builder.add_node("fallback", fallback_node)

        builder.add_edge("generate", "critique")
        builder.add_conditional_edges(
            "critique",
            router_after_critique,
            {"finalize": "finalize", "reformulate": "reformulate", "fallback": "fallback"},
        )
        builder.add_edge("reformulate", "retrieve")
        builder.add_edge("fallback", END)
    else:
        # Fast path: skip critique loop
        builder.add_edge("generate", "finalize")

    builder.add_edge("finalize", END)

    app = builder.compile(checkpointer=MemorySaver())
    return app


def router_after_critique(state: GraphState) -> str:
    if state.critique_passed:
        return "finalize"
    if state.retry_count < state.max_retries:
        return "reformulate"
    return "fallback"
