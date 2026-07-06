"""
Structured logging middleware for FastAPI.
Logs every HTTP request and provides a dedicated RAG audit logger
that captures query, retrieval metadata, and LLM response.
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request


# --- JSON Formatter ---

class JSONFormatter(logging.Formatter):
    """Format log records as single-line JSON."""

    def format(self, record):
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        # Merge extra fields from the log record
        if hasattr(record, "extra_data"):
            log_entry.update(record.extra_data)
        return json.dumps(log_entry, default=str)


# --- Setup Loggers ---

def setup_loggers():
    """Initialize the HTTP request logger and RAG audit logger."""
    os.makedirs("logs", exist_ok=True)

    # HTTP request logger
    http_logger = logging.getLogger("rag.http")
    http_logger.setLevel(logging.INFO)
    if not http_logger.handlers:
        handler = logging.FileHandler("logs/http_requests.jsonl", encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        http_logger.addHandler(handler)

    # RAG audit logger (queries, sources, answers)
    audit_logger = logging.getLogger("rag.audit")
    audit_logger.setLevel(logging.INFO)
    if not audit_logger.handlers:
        handler = logging.FileHandler("logs/rag_audit.jsonl", encoding="utf-8")
        handler.setFormatter(JSONFormatter())
        audit_logger.addHandler(handler)

    return http_logger, audit_logger


http_logger, audit_logger = setup_loggers()


# --- FastAPI Middleware ---

class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """
    Logs method, path, status code, and latency for every HTTP request.
    Skips noisy /health polls.
    """

    async def dispatch(self, request: Request, call_next):
        start = time.time()
        response = await call_next(request)
        elapsed_ms = round((time.time() - start) * 1000, 2)

        # Skip logging /health to avoid log flood from UI polling
        path = request.url.path
        if path == "/health":
            return response

        record = http_logger.makeRecord(
            name="rag.http",
            level=logging.INFO,
            fn="",
            lno=0,
            msg=f"{request.method} {path} {response.status_code}",
            args=(),
            exc_info=None,
        )
        record.extra_data = {
            "method": request.method,
            "path": path,
            "status_code": response.status_code,
            "latency_ms": elapsed_ms,
            "client": request.client.host if request.client else "unknown",
        }
        http_logger.handle(record)

        return response


# --- RAG Audit Logger ---

def log_rag_query(
    query: str,
    sources: list[dict],
    reranker_scores: list[float],
    answer: str,
    latency_ms: float,
    retrieval_k: int,
    reranked_k: int,
):
    """
    Log a structured audit entry for a RAG query.

    Args:
        query: The user's question.
        sources: List of dicts with 'file', 'page', 'section', 'score'.
        reranker_scores: List of cross-encoder scores for the final chunks.
        answer: The LLM-generated answer.
        latency_ms: Total end-to-end latency in milliseconds.
        retrieval_k: Number of chunks retrieved from hybrid search.
        reranked_k: Number of chunks after reranking.
    """
    record = audit_logger.makeRecord(
        name="rag.audit",
        level=logging.INFO,
        fn="",
        lno=0,
        msg=f"RAG query: {query[:80]}...",
        args=(),
        exc_info=None,
    )
    record.extra_data = {
        "event": "rag_query",
        "query": query,
        "retrieval_k": retrieval_k,
        "reranked_k": reranked_k,
        "sources": sources,
        "reranker_scores": reranker_scores,
        "answer_preview": answer[:300],
        "answer_length": len(answer),
        "latency_ms": latency_ms,
    }
    audit_logger.handle(record)
