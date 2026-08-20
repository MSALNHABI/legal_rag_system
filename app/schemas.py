from typing import Any
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str
    indexed_chunks: int
    collection_name: str


class RetrieveRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=20)
    semantic_k: int = Field(default=20, ge=1, le=50)
    keyword_k: int = Field(default=20, ge=1, le=50)


class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1)
    conversation_id: str | None = None
    top_k: int = Field(default=10, ge=1, le=20)
    semantic_k: int = Field(default=20, ge=1, le=50)
    keyword_k: int = Field(default=20, ge=1, le=50)


class CitationOut(BaseModel):
    label: str
    chunk_id: str
    law_id: str
    law_name_ar: str
    law_name_en: str
    article_number: int
    article_heading_ar: str
    source_file: str
    text_preview: str


class RetrievedChunkOut(BaseModel):
    rank: int
    score: float
    retrieval_method: str
    chunk_id: str
    law_id: str
    law_name_ar: str
    law_name_en: str
    article_number: int
    article_heading_ar: str
    text_preview: str
    semantic_rank: int | None = None
    keyword_rank: int | None = None
    matched_by: list[str] = []


class RetrieveResponse(BaseModel):
    query: str
    top_k: int
    retrieved_chunks: list[RetrievedChunkOut]
    latency_ms: float


class ChatResponse(BaseModel):
    conversation_id: str
    query: str
    answer: str
    citations: list[CitationOut]
    retrieved_chunks: list[RetrievedChunkOut]
    grounded: bool
    model: str
    latency_ms: float


class ErrorResponse(BaseModel):
    error: str
    details: Any | None = None