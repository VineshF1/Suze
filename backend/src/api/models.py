"""Pydantic request/response models for the API."""

from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class ChatRequest(BaseModel):
    query: str
    date_from: Optional[str] = None
    date_to: Optional[str] = None
    access_level: Optional[str] = None
    doc_type: Optional[str] = None
    tags: Optional[list[str]] = None


class ChatResponse(BaseModel):
    query: str
    answer: str
    sources: list[dict] = []
    trace: list[dict] = []


class IngestTextRequest(BaseModel):
    content: str
    source: str = "manual"
    metadata: dict = {}


class IngestResponse(BaseModel):
    chunks_ingested: int
    total_chunks: int
    message: str = ""


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "suze-rag"
    chunks_count: int = 0
