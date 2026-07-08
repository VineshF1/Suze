"""LangGraph nodes for the self-correction RAG loop."""

import json
import re

from openai import OpenAI

from src.config import settings
from src.agent.state import GraphState
from src.retrieval.staged_filter import StagedFilter

# Injected by build_rag_graph()
_staged_filter: StagedFilter = None  # type: ignore


def _get_llm() -> OpenAI:
    return OpenAI(api_key=settings.nvidia_api_key, base_url=settings.nvidia_base_url)


def _call_llm(prompt: str, temperature: float = 0.2, max_tokens: int = 1024) -> str:
    client = _get_llm()
    resp = client.chat.completions.create(
        model=settings.llm_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content or ""


# ─── Nodes ────────────────────────────────────────────────────

def retrieve_node(state: GraphState) -> dict:
    """Stage 1-2-3: Retrieve documents using staged filter."""
    sf = _staged_filter
    docs = sf.retrieve(
        query=state.current_query,
        date_from=state.date_from,
        date_to=state.date_to,
        tags=state.tags_filter,
        author=state.author_filter,
        top_k=5,
    )
    trace_entry = {"node": "retrieve", "query": state.current_query, "doc_count": len(docs)}
    return {"documents": docs, "trace": state.trace + [trace_entry]}


def generate_node(state: GraphState) -> dict:
    """Generate answer grounded in retrieved documents."""
    if not state.documents:
        return {"generation": "I don't have enough information to answer that question."}

    context = "\n\n".join([
        f"[Source: {d['metadata'].get('source', 'unknown')}] {d['text']}"
        for d in state.documents
    ])

    prompt = (
        "You are a precise AI assistant. Answer the user's question using ONLY the context below. "
        "If the context doesn't contain the answer, say 'I don't have enough information.' "
        "Do not include source citations in your answer.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {state.current_query}\n\n"
        "Answer:"
    )

    answer = _call_llm(prompt, temperature=settings.temperature)
    trace_entry = {"node": "generate", "answer_length": len(answer)}
    return {"generation": answer, "trace": state.trace + [trace_entry]}


def critique_node(state: GraphState) -> dict:
    """Evaluate the generation for grounding, faithfulness, relevance."""
    if not state.generation or not state.documents:
        return {
            "critique_passed": False,
            "critique_score": 0.0,
            "critique_feedback": "No generation or documents to evaluate.",
            "trace": state.trace + [{"node": "critique", "error": "empty state"}],
        }

    context = "\n\n".join([d["text"] for d in state.documents])

    prompt = (
        "You are a strict factuality evaluator. Determine if the ANSWER is fully grounded in the CONTEXT.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{state.generation}\n\n"
        "Evaluate these criteria:\n"
        "1. GROUNDING: Is every claim in the answer supported by the context?\n"
        "2. FAITHFULNESS: Does the answer avoid introducing external knowledge?\n"
        "3. RELEVANCE: Does the answer directly address the original question?\n\n"
        'Return a JSON object with these fields:\n'
        '{\n'
        '  "passed": true/false,\n'
        '  "score": 0.0-1.0,\n'
        '  "feedback": "explanation of what is missing or wrong"\n'
        '}\n\n'
        "JSON:"
    )

    raw = _call_llm(prompt, temperature=0.1, max_tokens=512)
    # Strip markdown code fences if present
    raw = re.sub(r'^```(?:json)?\s*', '', raw.strip())
    raw = re.sub(r'\s*```$', '', raw)

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = {"passed": False, "score": 0.0, "feedback": "Failed to parse critique JSON."}

    trace_entry = {"node": "critique", "result": result}
    return {
        "critique_passed": result.get("passed", False),
        "critique_score": result.get("score", 0.0),
        "critique_feedback": result.get("feedback", ""),
        "trace": state.trace + [trace_entry],
    }


def reformulate_node(state: GraphState) -> dict:
    """Reformulate the query based on critic feedback."""
    feedback = state.critique_feedback or "The context was insufficient."
    prompt = (
        "The previous retrieval did not find sufficient information to answer the user's question. "
        f"Feedback: {feedback}\n\n"
        f"Original question: {state.query}\n\n"
        "Rewrite the question to be more specific or to target the missing information. "
        "Output ONLY the reformulated query, nothing else.\n\n"
        "Reformulated query:"
    )

    new_query = _call_llm(prompt, temperature=0.3, max_tokens=256).strip()
    history = list(state.reformulation_history) + [new_query]

    trace_entry = {"node": "reformulate", "new_query": new_query}
    return {
        "current_query": new_query,
        "retry_count": state.retry_count + 1,
        "reformulation_history": history,
        "trace": state.trace + [trace_entry],
    }


def fallback_node(state: GraphState) -> dict:
    """Safe fallback after exhausting retries."""
    msg = "I don't have enough information to answer that question accurately."
    trace_entry = {
        "node": "fallback",
        "reason": f"Exhausted {state.max_retries} retries",
    }
    return {"final_response": msg, "trace": state.trace + [trace_entry]}


def finalize_node(state: GraphState) -> dict:
    """Copy validated generation to final_response."""
    return {"final_response": state.generation}
