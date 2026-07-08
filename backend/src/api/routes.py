"""FastAPI routes for the Suze RAG assistant."""

import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from langgraph.graph.state import CompiledStateGraph

from src.api.models import (
    ChatRequest,
    ChatResponse,
    HealthResponse,
    IngestResponse,
    IngestTextRequest,
)
from src.agent.state import GraphState
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.vector_store import VectorStore

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Main chat endpoint — runs the full agentic RAG loop."""
    graph: CompiledStateGraph = request.app.state.graph

    initial_state = GraphState(
        query=req.query,
        date_from=req.date_from,
        date_to=req.date_to,
        tags_filter=req.tags,
    )
    # Add access_level to pre-filter via metadata
    if req.access_level:
        initial_state.pre_filter = {"access_level": req.access_level}
    if req.doc_type:
        if initial_state.pre_filter:
            initial_state.pre_filter["doc_type"] = req.doc_type
        else:
            initial_state.pre_filter = {"doc_type": req.doc_type}

    config = {"configurable": {"thread_id": str(uuid.uuid4())}}
    result = graph.invoke(initial_state, config=config)

    answer = result.get("final_response") or result.get("generation") or "No answer generated."
    sources = [
        {
            "text": d["text"][:300],
            "source": d["metadata"].get("source", "unknown"),
            "score": d.get("score", 0),
        }
        for d in (result.get("documents") or [])
    ]
    trace = result.get("trace", [])

    return ChatResponse(query=req.query, answer=answer, sources=sources, trace=trace)


@router.post("/ingest/text", response_model=IngestResponse)
async def ingest_text(req: IngestTextRequest, request: Request):
    """Ingest raw text."""
    pipeline: IngestionPipeline = request.app.state.pipeline
    vs: VectorStore = request.app.state.vs
    chunks = pipeline.ingest_text(content=req.content, source=req.source, metadata=req.metadata)
    return IngestResponse(
        chunks_ingested=chunks,
        total_chunks=vs.count(),
        message=f"Ingested {chunks} chunks from text.",
    )


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...), request: Request = None):
    """Upload and ingest a file (PDF, MD, TXT)."""
    pipeline: IngestionPipeline = request.app.state.pipeline
    vs: VectorStore = request.app.state.vs

    ext = Path(file.filename or "upload").suffix.lower()
    if ext not in (".pdf", ".md", ".markdown", ".txt"):
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {ext}")

    # Save to temp, ingest, clean up
    tmp = Path(request.app.state.temp_dir) / f"{uuid.uuid4()}{ext}"
    tmp.parent.mkdir(parents=True, exist_ok=True)
    content = await file.read()
    tmp.write_bytes(content)

    try:
        chunks = pipeline.ingest_file(str(tmp))
    except Exception as e:
        tmp.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=str(e))

    tmp.unlink(missing_ok=True)
    return IngestResponse(
        chunks_ingested=chunks,
        total_chunks=vs.count(),
        message=f"Ingested {chunks} chunks from {file.filename}.",
    )


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    """Health check."""
    vs: VectorStore = request.app.state.vs
    return HealthResponse(chunks_count=vs.count())


@router.delete("/reset")
async def reset(request: Request):
    """Reset the vector store (delete all chunks)."""
    vs: VectorStore = request.app.state.vs
    vs.delete_collection()
    return {"message": "Vector store reset."}
