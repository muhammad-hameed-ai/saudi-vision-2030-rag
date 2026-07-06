"""
Saudi Vision 2030 RAG API — Production Grade (V2.1)

Architecture:
  Hybrid Retrieval (Dense + Sparse BM25) → Cross-Encoder Reranker → LLM Generation
  Features: Async Non-Blocking Endpoints, Dynamic Schema Validation, Structured Logging
"""

import os
import asyncio
import json
import time
import math
import ollama
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Local module imports
from src.retriever import HybridRetriever, QdrantUnavailableError
from src.reranker import Reranker
from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query

# ---------------------------------------------------------------------------
# Environment & Global State
# ---------------------------------------------------------------------------
os.environ["OLLAMA_HOST"] = "http://127.0.0.1:11434"

retriever = HybridRetriever()
reranker = Reranker()

startup_time = None
request_count = 0
feedback_log = []
SYSTEM_STATS = {
    "queries_served_this_session": 0,
    "latency_history": [],
}

# Pipeline Configuration
RETRIEVAL_K = 20    # Retrieve 20 candidates from hybrid search
RERANK_TOP_K = 5    # Rerank down to top 5 for LLM context


# ---------------------------------------------------------------------------
# Lifespan (Startup / Shutdown)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan: Eagerly load models to memory to ensure fast initial queries."""
    global startup_time

    print("Loading retriever models...")
    try:
        retriever._get_dense_model()
        retriever._get_sparse_model()
        healthy = retriever.health_check()
        status = "CONNECTED" if healthy else "UNREACHABLE"
        print(f"Qdrant status: {status}")
    except Exception as e:
        print(f"WARNING: Qdrant connection failed during startup: {e}")

    startup_time = datetime.now(timezone.utc).isoformat()
    print("API ready.")
    yield


# ---------------------------------------------------------------------------
# Application Setup & Middleware
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Saudi Vision 2030 RAG API",
    description="Production-grade asynchronous RAG pipeline with Hybrid Search and Cross-Encoder Reranking.",
    version="2.1.0",
    lifespan=lifespan,
)

app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Helper Functions
# ---------------------------------------------------------------------------
def _require_qdrant():
    """Fail-closed guard: raises 503 if Qdrant is unreachable."""
    if not retriever.health_check():
        raise HTTPException(
            status_code=503,
            detail="Vector database unavailable. Cannot process query.",
        )

def _clean_source_path(raw: str) -> str:
    """Strips local file paths for clean citation formatting."""
    return raw.replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")


# ---------------------------------------------------------------------------
# Pydantic Schemas (Dynamic & Indestructible)
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    """
    Validates incoming chat requests. Automatically maps 'message', 'query', 
    or 'question' from the frontend to prevent 422 Unprocessable Content errors.
    """
    question: str = Field(default="")
    k: int = Field(default=5, ge=1, le=10, description="Final chunks to inject into LLM")

    @model_validator(mode='before')
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            # Absorb multiple naming conventions from the UI
            text = data.get('message') or data.get('query') or data.get('question')
            if not text:
                raise ValueError("Payload must contain 'message', 'query', or 'question'")
            data['question'] = text
        return data

class SourceDoc(BaseModel):
    source: str
    page: int
    section: str
    preview: str
    score: float

class AskResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceDoc]
    retrieval_chunks: int
    reranked_chunks: int
    latency_ms: float
    model: str
    timestamp: str

class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int
    comment: Optional[str] = ""


# ---------------------------------------------------------------------------
# RAG Endpoints (Fully Asynchronous)
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def process_rag_chat(request: ChatRequest):
    """
    Dashboard chat endpoint. Uses asyncio.to_thread to prevent CPU-heavy 
    generation and vector calculations from blocking other users.
    """
    _require_qdrant()
    start_time = time.time()
    top_k = request.k

    try:
        # Stage 1: Async Hybrid retrieval
        candidates = await asyncio.to_thread(
            retriever.retrieve, request.question, k=RETRIEVAL_K
        )

        # Stage 2: Async Cross-encoder reranking
        reranked = await asyncio.to_thread(
            reranker.rerank, request.question, candidates, top_k=top_k
        )
    except QdrantUnavailableError:
        raise HTTPException(
            status_code=503,
            detail="Vector database offline. Cannot execute retrieval.",
        )

    # Stage 3: Context Conditioning
    context = "\n\n".join([c.content for c in reranked])
    system_prompt = (
        "You are an elite policy analyst for Saudi Vision 2030.\n"
        "Use ONLY the following CONTEXT to answer the user's question.\n"
        "If the CONTEXT does not contain information to answer the question, or if the user asks about an unrelated topic (e.g., foreign countries, general trivia), you MUST reply exactly with: 'I cannot find this information in the provided Saudi Vision 2030 policy documents.' Do not hallucinate.\n\n"
        f"CONTEXT:\n{context}"
    )

    # Stage 4: Async LLM Generation
    response = await asyncio.to_thread(
        ollama.chat,
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ],
        options={"num_ctx": 2048, "num_predict": 300, "num_thread": 8},
    )
    ai_answer = response["message"]["content"].strip()

    # Metrics computation
    elapsed_time = time.time() - start_time
    elapsed_ms = round(elapsed_time * 1000, 2)
    SYSTEM_STATS["queries_served_this_session"] += 1
    SYSTEM_STATS["latency_history"].append(elapsed_time)

    source_citations = []
    for chunk in reranked:
        normalized_score = 1 / (1 + math.exp(-chunk.score))
        source_citations.append({
            "file": _clean_source_path(chunk.source),
            "page": chunk.page,
            "section": chunk.section,
            "score": normalized_score,
        })

    # Stage 5: Async Audit Logging
    await asyncio.to_thread(
        log_rag_query,
        query=request.question,
        sources=source_citations,
        reranker_scores=[c.score for c in reranked],
        answer=ai_answer,
        latency_ms=elapsed_ms,
        retrieval_k=len(candidates),
        reranked_k=len(reranked),
    )

    return {
        "answer": ai_answer,
        "citations": source_citations,
        "metrics": {
            "latency_seconds": round(elapsed_time, 3),
            "retrieval_depth_k": RETRIEVAL_K,
            "reranked_k": len(reranked),
        },
    }

@app.post("/ask", response_model=AskResponse)
async def ask(request: ChatRequest):
    """
    Standard programmatic RAG endpoint. Mirrors the async architecture of /api/chat.
    """
    global request_count
    _require_qdrant()

    if len(request.question) > 500:
        raise HTTPException(status_code=422, detail="Question too long. Maximum 500 characters.")

    t0 = time.time()
    
    try:
        candidates = await asyncio.to_thread(retriever.retrieve, request.question, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker.rerank, request.question, candidates, top_k=request.k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Vector database unavailable.")

    context_text = "\n\n".join([c.content for c in reranked])
    system_prompt = (
        "You are an elite policy analyst for Saudi Vision 2030.\n"
        "Use ONLY the following CONTEXT to answer the user's question.\n"
        "If the CONTEXT does not contain information to answer the question, or if the user asks about an unrelated topic (e.g., foreign countries, general trivia), you MUST reply exactly with: 'I cannot find this information in the provided Saudi Vision 2030 policy documents.' Do not hallucinate.\n\n"
        f"CONTEXT:\n{context_text}"
    )

    response = await asyncio.to_thread(
        ollama.chat,
        model="llama3.2:1b",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": request.question}
        ],
        options={"num_ctx": 2048, "num_predict": 300, "num_thread": 8},
    )
    answer = response["message"]["content"].strip()

    latency_ms = round((time.time() - t0) * 1000, 2)
    request_count += 1

    sources = []
    for c in reranked:
        normalized_score = 1 / (1 + math.exp(-c.score))
        sources.append(SourceDoc(
            source=_clean_source_path(c.source),
            page=c.page,
            section=c.section,
            preview=c.content[:150].strip(),
            score=normalized_score
        ))

    await asyncio.to_thread(
        log_rag_query,
        query=request.question,
        sources=[s.model_dump() for s in sources],
        reranker_scores=[c.score for c in reranked],
        answer=answer,
        latency_ms=latency_ms,
        retrieval_k=len(candidates),
        reranked_k=len(reranked),
    )

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_chunks=len(candidates),
        reranked_chunks=len(reranked),
        latency_ms=latency_ms,
        model="llama3.2:1b",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Utility & UI Routes
# ---------------------------------------------------------------------------
@app.get("/")
def read_index():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "index.html",
    )
    return FileResponse(path)

@app.get("/health")
def health():
    healthy = retriever.health_check()
    return {
        "status": "healthy" if healthy else "degraded",
        "qdrant": "connected" if healthy else "unreachable",
        "model": "llama3.2:1b",
        "vector_store": "saudi_vision_2030",
        "architecture": "hybrid_search + cross_encoder_reranker",
        "requests_served": request_count + SYSTEM_STATS["queries_served_this_session"],
        "uptime_since": startup_time,
    }

@app.get("/api/pipeline-info")
def get_pipeline_info():
    coll_info = retriever.get_collection_info()
    return {
        "corpus_summary": {
            "documents": 48,
            "pages": 2184,
            "chunks": coll_info.get("points_count", 0),
            "dimensions": 384,
        },
        "configuration": {
            "chunking_strategy": "Structure-Aware Recursive Splitting",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "sparse_model": "Qdrant/bm25",
            "vector_database": "Qdrant (Hybrid: Dense + Sparse)",
            "distance_metric": "Cosine + RRF Fusion",
            "reranker_model": reranker.model_name,
            "retrieval_k": RETRIEVAL_K,
            "reranked_k": RERANK_TOP_K,
            "llm_backbone": "llama3.2:1b (Ollama Engine)",
        },
    }

@app.get("/api/analytics")
def get_analytics_dashboard():
    avg_latency = 0.0
    if SYSTEM_STATS["latency_history"]:
        avg_latency = sum(SYSTEM_STATS["latency_history"]) / len(SYSTEM_STATS["latency_history"])

    return {
        "session_metrics": {
            "queries_served": SYSTEM_STATS["queries_served_this_session"],
            "average_latency_seconds": round(avg_latency, 3),
        },
        "architecture": {
            "retrieval": "Hybrid (Dense + BM25 Sparse)",
            "fusion": "Reciprocal Rank Fusion (RRF)",
            "reranker": reranker.model_name,
        },
    }

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    if request.rating not in [1, -1]:
        raise HTTPException(status_code=422, detail="Rating must be 1 or -1.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }
    feedback_log.append(entry)

    os.makedirs("data/feedback", exist_ok=True)
    path = "data/feedback/feedback_log.json"
    
    # Offload file I/O
    def write_feedback():
        existing = []
        if os.path.exists(path):
            with open(path) as f:
                existing = json.load(f)
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
            
    await asyncio.to_thread(write_feedback)
    return {"status": "recorded"}