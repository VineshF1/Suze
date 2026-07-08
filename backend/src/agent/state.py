"""LangGraph state schema for the self-correction loop."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class GraphState:
    """State maintained across the agentic RAG loop."""
    query: str                                   # Original user question (never changes)
    current_query: str = ""                      # May be reformulated
    reformulation_history: list[str] = field(default_factory=list)
    documents: list[dict] = field(default_factory=list)  # Retrieved chunks
    generation: Optional[str] = None              # Draft answer
    critique_passed: Optional[bool] = None
    critique_score: Optional[float] = None
    critique_feedback: Optional[str] = None
    retry_count: int = 0
    max_retries: int = 3
    final_response: Optional[str] = None
    trace: list[dict] = field(default_factory=list)
    # Filter state for staged retrieval
    pre_filter: Optional[dict] = None
    tags_filter: Optional[list[str]] = None
    author_filter: Optional[str] = None
    date_from: Optional[str] = None
    date_to: Optional[str] = None

    def __post_init__(self):
        if not self.current_query:
            self.current_query = self.query
