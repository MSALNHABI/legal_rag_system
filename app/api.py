from pathlib import Path
import json
from time import perf_counter

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app.retriever import LegalRetriever
from app.generator import LegalAnswerGenerator
from app.memory import ConversationMemory
from app.logging_config import write_query_log
from app.schemas import (
    HealthResponse,
    RetrieveRequest,
    RetrieveResponse,
    ChatRequest,
    ChatResponse,
    RetrievedChunkOut,
    CitationOut,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVALUATION_RESULTS_PATH = PROJECT_ROOT / "data" / "evaluation" / "retrieval_evaluation_results.json"


api = FastAPI(
    title="Saudi Legal RAG API",
    description="RAG API for Saudi Labor Law and Social Insurance Law.",
    version="1.0.0",
)

app = api


api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


retriever = LegalRetriever()
generator = LegalAnswerGenerator()
memory = ConversationMemory()


def serialize_retrieved_chunk(chunk: dict) -> RetrievedChunkOut:
    return RetrievedChunkOut(
        rank=chunk["rank"],
        score=float(chunk["score"]),
        retrieval_method=chunk["retrieval_method"],
        chunk_id=chunk["chunk_id"],
        law_id=chunk["law_id"],
        law_name_ar=chunk["law_name_ar"],
        law_name_en=chunk["law_name_en"],
        article_number=int(chunk["article_number"]),
        article_heading_ar=chunk["article_heading_ar"],
        text_preview=chunk["text"][:700],
        semantic_rank=chunk.get("semantic_rank"),
        keyword_rank=chunk.get("keyword_rank"),
        matched_by=chunk.get("matched_by", []),
    )


def serialize_citation(citation: dict) -> CitationOut:
    return CitationOut(
        label=citation["label"],
        chunk_id=citation["chunk_id"],
        law_id=citation["law_id"],
        law_name_ar=citation["law_name_ar"],
        law_name_en=citation["law_name_en"],
        article_number=int(citation["article_number"]),
        article_heading_ar=citation["article_heading_ar"],
        source_file=citation["source_file"],
        text_preview=citation["text_preview"],
    )


@api.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        indexed_chunks=len(retriever.chunks),
        collection_name=retriever.collection_name,
    )


@api.get("/metrics/retrieval")
def retrieval_metrics() -> dict:
    if not EVALUATION_RESULTS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Retrieval evaluation results not found. "
                "Run scripts/evaluate_retrieval.py first."
            ),
        )

    return json.loads(EVALUATION_RESULTS_PATH.read_text(encoding="utf-8"))


@api.post("/retrieve", response_model=RetrieveResponse)
def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    start_time = perf_counter()

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    retrieved_chunks = retriever.hybrid_search(
        query=query,
        top_k=request.top_k,
        semantic_k=request.semantic_k,
        keyword_k=request.keyword_k,
    )

    latency_ms = round((perf_counter() - start_time) * 1000, 2)

    write_query_log(
        {
            "endpoint": "/retrieve",
            "query": query,
            "top_k": request.top_k,
            "latency_ms": latency_ms,
            "retrieved_chunk_ids": [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
        }
    )

    return RetrieveResponse(
        query=query,
        top_k=request.top_k,
        retrieved_chunks=[
            serialize_retrieved_chunk(chunk)
            for chunk in retrieved_chunks
        ],
        latency_ms=latency_ms,
    )


@api.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    start_time = perf_counter()

    query = request.query.strip()

    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    conversation_id = memory.get_or_create_conversation(
        request.conversation_id
    )

    chat_history = memory.get_history(conversation_id)

    retrieved_chunks = retriever.hybrid_search(
        query=query,
        top_k=request.top_k,
        semantic_k=request.semantic_k,
        keyword_k=request.keyword_k,
    )

    generation_result = generator.generate_answer(
        question=query,
        retrieved_chunks=retrieved_chunks,
        chat_history=chat_history,
    )

    answer = generation_result["answer"]

    memory.add_message(
        conversation_id=conversation_id,
        role="user",
        content=query,
    )

    memory.add_message(
        conversation_id=conversation_id,
        role="assistant",
        content=answer,
    )

    latency_ms = round((perf_counter() - start_time) * 1000, 2)

    write_query_log(
        {
            "endpoint": "/chat",
            "conversation_id": conversation_id,
            "query": query,
            "answer_preview": answer[:300],
            "grounded": generation_result["grounded"],
            "model": generation_result["model"],
            "top_k": request.top_k,
            "latency_ms": latency_ms,
            "retrieved_chunk_ids": [
                chunk["chunk_id"]
                for chunk in retrieved_chunks
            ],
            "citation_chunk_ids": [
                citation["chunk_id"]
                for citation in generation_result["citations"]
            ],
        }
    )

    return ChatResponse(
        conversation_id=conversation_id,
        query=query,
        answer=answer,
        citations=[
            serialize_citation(citation)
            for citation in generation_result["citations"]
        ],
        retrieved_chunks=[
            serialize_retrieved_chunk(chunk)
            for chunk in retrieved_chunks
        ],
        grounded=generation_result["grounded"],
        model=generation_result["model"],
        latency_ms=latency_ms,
    )