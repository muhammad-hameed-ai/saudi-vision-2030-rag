"""
Saudi Vision 2030 Policy Intelligence Hub — Production Grade API (V2.2)

Architecture:
  Hybrid Retrieval (Dense + Sparse BM25) → Cross-Encoder Reranker → LLM Generation
  Features: Async Non-Blocking Endpoints, Dynamic Schema Validation, Structured Logging, Auto-Healing Guards
"""

import os
import asyncio
import json
import uuid
import time
import math
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List

import ollama
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator

# Local module imports
from src.retriever import HybridRetriever, QdrantUnavailableError
from src.reranker import Reranker
from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query
from src.hyde_retriever import generate_hypothesis
from src.memory import save_message, get_session_history, summarize_history

# ---------------------------------------------------------------------------
# Environment & Global State Configuration
# ---------------------------------------------------------------------------
# Normalize Ollama binding address to prevent Python HTTP client route errors
client_host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
if "0.0.0.0" in client_host:
    client_host = "http://127.0.0.1:11434"
os.environ["OLLAMA_HOST"] = client_host
os.environ["USE_HYDE"] = os.getenv("USE_HYDE", "true")

# Thread-safe global singletons
retriever = HybridRetriever()
reranker = Reranker()

startup_time: Optional[str] = None
request_count: int = 0
feedback_log: List[dict] = []

SYSTEM_STATS = {
    "queries_served_this_session": 0,
    "latency_history": [],
}

# Pipeline Consts
RETRIEVAL_K = 10    # Hybrid engine candidate pooling size
RERANK_TOP_K = 5    # Compressed context size passed to the LLM backbone
HEALTH_CHECK_TTL = 5.0  
MAX_LATENCY_HISTORY = 100
MAX_FEEDBACK_LOG = 500

# Thread-safe health state cache
_cached_health = {"healthy": False, "checked_at": 0.0}


# ---------------------------------------------------------------------------
# Lifespan Hook (Model Preloading & Warm-up)
# ---------------------------------------------------------------------------
async def warmup_llm():
    """Preloads the LLM backbone into VRAM asynchronously post-server initialization."""
    print("[INIT] Initiating asynchronous LLM warm-up sequence...")
    await asyncio.sleep(2)  
    try:
        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.AsyncClient(host=host, timeout=60.0)
        await client.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": "ping"}],
            options={"num_predict": 1}
        )
        print("[INIT] LLM engine warm-up verified. Weights loaded successfully.")
    except Exception as e:
        print(f"[WARN] Non-fatal: LLM model warm-up skipped (Engine offline/sleeping): {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handles context initialization, safe eager-loading, and environment mapping."""
    global startup_time

    print("[INIT] Eager-loading vector and sparse model tokenizers...")
    try:
        retriever._get_dense_model()
        retriever._get_sparse_model()
        healthy = retriever.health_check()
        print(f"[INIT] Vector Store status: {'CONNECTED' if healthy else 'UNREACHABLE'}")

        _cached_health["healthy"] = healthy
        _cached_health["checked_at"] = time.time()

        if healthy:
            collections = retriever._get_client().get_collections().collections
            if not any(c.name == "saudi_vision_2030" for c in collections):
                print("[WARN] Target collection 'saudi_vision_2030' was not found in remote Qdrant Cluster.")
    except Exception as e:
        print(f"[ERROR] Failures detected during lifespan boot sequence: {e}")

    startup_time = datetime.now(timezone.utc).isoformat()
    asyncio.create_task(warmup_llm())
    yield
    print("[SHUTDOWN] Terminating server context loops.")


# ---------------------------------------------------------------------------
# Application Instance & Middleware Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Saudi Vision 2030 Policy Intelligence Hub API",
    description="Enterprise-grade production asynchronous RAG engine utilizing Hybrid Architecture.",
    version="2.2.0",
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
# Internal Core Helper Methods
# ---------------------------------------------------------------------------
def _require_qdrant():
    """
    Self-healing fail-closed cluster safety mechanism. Uses an adaptive TTL cache 
    when performance is stable, and triggers immediate connection verification 
    upon unexpected platform dropouts.
    """
    now = time.time()
    if _cached_health["healthy"] and (now - _cached_health["checked_at"] < HEALTH_CHECK_TTL):
        return
    
    try:
        retriever._client = None  # Flush potentially corrupted sockets
        is_healthy = retriever.health_check()
    except Exception:
        is_healthy = False

    _cached_health["healthy"] = is_healthy
    _cached_health["checked_at"] = now

    if not is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Upstream vector infrastructure is down. Service temporarily degraded.",
        )

def _clean_source_path(raw_path: str) -> str:
    """Strips local system file architectures for presentation-tier cleanliness."""
    return raw_path.replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")

async def _build_memory_string(session_id: str) -> str:
    """Safely handles context generation out-of-thread to ensure responsive main execution."""
    memory_context = await asyncio.to_thread(get_session_history, session_id, 4)
    memory_str = ""
    if memory_context.get("summary"):
        memory_str += f"Summary of past conversation: {memory_context['summary']}\n"
    for m in memory_context.get("messages", []):
        memory_str += f"{m['role'].upper()}: {m['content']}\n"
    return memory_str


# ---------------------------------------------------------------------------
# Robust Pydantic Validation Tier
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(default="")
    k: int = Field(default=5, ge=1, le=10)

    @model_validator(mode='before')
    @classmethod
    def normalize_input(cls, data: Any) -> Any:
        if isinstance(data, dict):
            text = data.get('message') or data.get('query') or data.get('question')
            if not text or not str(text).strip():
                raise ValueError("Payload structural mutation failed: 'message', 'query', or 'question' must be present.")
            data['question'] = str(text).strip()
        return data

class SourceDoc(BaseModel):
    source: str
    page: int
    section: str
    preview: str
    score: float

class AskResponse(BaseModel):
    session_id: str
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
# Asynchronous RAG Processing Routes
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def process_rag_chat(request: ChatRequest):
    """
    Main asynchronous interactive dashboard route. Offloads dense computation matrix 
    to managed system worker pools via asyncio to ensure non-blocking scalability.
    """
    _require_qdrant()
    start_time = time.time()
    top_k = request.k

    try:
        # Step 1: Query Conditioning (HyDE)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(request.question) if use_hyde else request.question
        
        # Step 2: Dense + Sparse Candidate Extraction
        candidates = await asyncio.to_thread(retriever.retrieve, search_query, k=RETRIEVAL_K)

        # Step 3: Deep Cross-Encoder Re-scoring
        reranked = await asyncio.to_thread(reranker.rerank, request.question, candidates, top_k=top_k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Vector search engine timed out during document pooling.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal collection failure: {str(e)}")

    # Step 4: System Memory Injection
    memory_str = await _build_memory_string(request.session_id)
    context = "\n\n".join([c.content for c in reranked])
    
    system_prompt = (
        "You are an elite policy analyst for Saudi Vision 2030.\n"
        "Use ONLY the following CONTEXT and MEMORY to answer the user's question.\n"
        "If the CONTEXT does not contain information to answer the question, or if the user asks about an unrelated topic (e.g., foreign countries, general trivia), you MUST reply exactly with: 'I cannot find this information in the provided Saudi Vision 2030 policy documents.' Do not hallucinate.\n\n"
        f"MEMORY:\n{memory_str}\n\n"
        f"CONTEXT:\n{context}"
    )

    # Step 5: Asynchronous LLM Text Synthesis Loop
    try:
        client = ollama.AsyncClient(host=os.environ["OLLAMA_HOST"], timeout=120.0)
        response = await client.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            options={
                "num_ctx": 2048,
                "temperature": 0.2,
                "top_k": 40,
                "repeat_penalty": 1.1,
            },
        )
        ai_answer = response["message"]["content"].strip()
        
        # Fixed state memory bug: Corrected execution parameters to bind cleanly to 'ai_answer'
        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
        asyncio.create_task(summarize_history(request.session_id))
        
    except Exception as e:
        err_str = str(e).lower()
        if "connect" in err_str or "timeout" in err_str:
            return JSONResponse(
                status_code=503,
                content={"error": "LLM infrastructure endpoint timeout. Please verify backend status."}
            )
        raise HTTPException(status_code=504, detail=f"LLM compilation context failed: {str(e)[:150]}")

    # Step 6: Telemetry Parsing
    elapsed_time = time.time() - start_time
    elapsed_ms = round(elapsed_time * 1000, 2)
    
    SYSTEM_STATS["queries_served_this_session"] += 1
    SYSTEM_STATS["latency_history"].append(elapsed_time)
    if len(SYSTEM_STATS["latency_history"]) > MAX_LATENCY_HISTORY:
        SYSTEM_STATS["latency_history"].pop(0)

    source_citations = []
    for chunk in reranked:
        # Convert raw Cross-Encoder scores to standard logistic probability metric
        normalized_score = 1 / (1 + math.exp(-chunk.score))
        source_citations.append({
            "file": _clean_source_path(chunk.source),
            "page": chunk.page,
            "section": chunk.section,
            "score": round(normalized_score, 4),
        })

    # Step 7: System Logging
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
        "session_id": request.session_id,
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
    """Programmatic standard validation endpoint mirroring the async core pipeline."""
    global request_count
    _require_qdrant()

    if len(request.question) > 500:
        raise HTTPException(status_code=422, detail="Query validation breach: Input string exceeds 500 characters maximum.")

    t0 = time.time()
    try:
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(request.question) if use_hyde else request.question
        candidates = await asyncio.to_thread(retriever.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker.rerank, request.question, candidates, top_k=request.k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Remote database engine cluster rejected operation request.")

    memory_str = await _build_memory_string(request.session_id)
    context_text = "\n\n".join([c.content for c in reranked])
    system_prompt = (
        "You are an elite policy analyst for Saudi Vision 2030.\n"
        "Use ONLY the following CONTEXT and MEMORY to answer the user's question.\n"
        "If the CONTEXT does not contain information to answer the question, or if the user asks about an unrelated topic (e.g., foreign countries, general trivia), you MUST reply exactly with: 'I cannot find this information in the provided Saudi Vision 2030 policy documents.' Do not hallucinate.\n\n"
        f"MEMORY:\n{memory_str}\n\n"
        f"CONTEXT:\n{context_text}"
    )

    try:
        client = ollama.AsyncClient(host=os.environ["OLLAMA_HOST"], timeout=120.0)
        response = await client.chat(
            model="llama3.2:1b",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            options={
                "num_ctx": 2048,
                "temperature": 0.2,
                "top_k": 40,
                "repeat_penalty": 1.1,
            },
        )
        ai_answer = response["message"]["content"].strip()
        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
        asyncio.create_task(summarize_history(request.session_id))
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"Downstream programmatic synthesis failed: {str(e)}")

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
            score=round(normalized_score, 4)
        ))

    await asyncio.to_thread(
        log_rag_query,
        query=request.question,
        sources=[s.model_dump() for s in sources],
        reranker_scores=[c.score for c in reranked],
        answer=ai_answer,
        latency_ms=latency_ms,
        retrieval_k=len(candidates),
        reranked_k=len(reranked),
    )

    return AskResponse(
        session_id=request.session_id,
        question=request.question,
        answer=ai_answer,
        sources=sources,
        retrieval_chunks=len(candidates),
        reranked_chunks=len(reranked),
        latency_ms=latency_ms,
        model="llama3.2:1b",
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Telemetry Analytics & Metadata Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_index():
    """Serves the static production UI matrix directly from root context."""
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "index.html")
    return FileResponse(path)

@app.get("/health")
def health():
    """Telemetry route tracking total processing matrix loops across instances."""
    try:
        healthy = retriever.health_check()
    except Exception:
        healthy = False
        
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
    """Returns static and dynamic pipeline state parameters directly to frontend charts."""
    coll_info = {}
    try:
        coll_info = retriever.get_collection_info()
    except Exception:
        coll_info = {"points_count": "Unknown (Database Disconnected)"}

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
    """Calculates active query processing performance curves across user operations."""
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
    """Persists user evaluation metrics asynchronously without locking pipeline loops."""
    if request.rating not in [1, -1]:
        raise HTTPException(status_code=422, detail="Dynamic compliance exception: Rating constraint boundaries are 1 or -1.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }
    
    feedback_log.append(entry)
    if len(feedback_log) > MAX_FEEDBACK_LOG:
        feedback_log.pop(0)

    os.makedirs("data/feedback", exist_ok=True)
    path = "data/feedback/feedback_log.json"
    
    def write_feedback():
        existing = []
        if os.path.exists(path):
            try:
                with open(path) as f:
                    existing = json.load(f)
            except Exception:
                existing = []
        existing.append(entry)
        with open(path, "w") as f:
            json.dump(existing, f, indent=2)
            
    await asyncio.to_thread(write_feedback)
    return {"status": "recorded"}