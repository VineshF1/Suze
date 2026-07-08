"""FastAPI application entry point for Suze RAG assistant."""

import tempfile
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.retrieval.vector_store import VectorStore
from src.retrieval.hybrid_search import HybridSearch
from src.retrieval.staged_filter import StagedFilter
from src.ingestion.pipeline import IngestionPipeline
from src.agent.graph import build_rag_graph
from src.api.routes import router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — init and teardown."""
    # ── Init ──────────────────────────────────────────────
    vs = VectorStore()
    hybrid = HybridSearch(vs)
    sf = StagedFilter(hybrid)
    pipeline = IngestionPipeline(vs)
    graph = build_rag_graph(sf, enable_critique=settings.enable_critique)
    tmp_dir = tempfile.mkdtemp(prefix="suze_uploads_")

    app.state.vs = vs
    app.state.pipeline = pipeline
    app.state.graph = graph
    app.state.temp_dir = tmp_dir

    yield

    # ChromaDB background threads hold file handles; let it be.


app = FastAPI(
    title="Suze — Secure Agentic RAG Assistant",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.main:app", host=settings.host, port=settings.port, reload=True)
