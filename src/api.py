"""
Saudi Vision 2030 Policy Intelligence Hub — Production Grade API (V2.3 Cloud)

Architecture:
  Hybrid Retrieval (Dense + Sparse BM25) → Cross-Encoder Reranker → Groq Cloud LLM
  Features: Lazy-loaded Heavy Models, Async Non-Blocking Endpoints, Dynamic Schema Validation
"""

import os
import re
import asyncio
import json
import uuid
import time
import math
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from groq import AsyncGroq, APIError

# Local module imports
from src.retriever import HybridRetriever, QdrantUnavailableError
from src.reranker import Reranker
from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query
from src.hyde_retriever import generate_hypothesis
from src.memory import save_message, get_session_history, summarize_history

# ---------------------------------------------------------------------------
# Environment & Path Configuration
# ---------------------------------------------------------------------------
os.environ["USE_HYDE"] = os.getenv("USE_HYDE", "true")

BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent

# Search for index.html at root, src/, or static/
INDEX_PATH = None
for candidate in [
    PROJECT_ROOT / "index.html",
    BASE_DIR / "index.html",
    PROJECT_ROOT / "static" / "index.html",
]:
    if candidate.exists():
        INDEX_PATH = candidate
        break

# ---------------------------------------------------------------------------
# Lazy-Loaded Global Singletons
# ---------------------------------------------------------------------------
_retriever_instance: Optional[HybridRetriever] = None
_reranker_instance: Optional[Reranker] = None

def get_retriever() -> HybridRetriever:
    """Lazy initializer for HybridRetriever to prevent blocking Uvicorn startup."""
    global _retriever_instance
    if _retriever_instance is None:
        print("[INIT] Lazy loading HybridRetriever models (FastEmbed + Qdrant)...")
        _retriever_instance = HybridRetriever()
    return _retriever_instance

def get_reranker() -> Reranker:
    """Lazy initializer for Reranker model."""
    global _reranker_instance
    if _reranker_instance is None:
        print("[INIT] Lazy loading Cross-Encoder Reranker model...")
        _reranker_instance = Reranker()
    return _reranker_instance


startup_time: Optional[str] = None
request_count: int = 0
feedback_log: List[dict] = []

SYSTEM_STATS = {
    "queries_served_this_session": 0,
    "latency_history": [],
}

# Pipeline Consts
RETRIEVAL_K = 10        # Hybrid engine candidate pooling size
RERANK_TOP_K = 5        # Compressed context size passed to LLM
HEALTH_CHECK_TTL = 5.0  
MAX_LATENCY_HISTORY = 100
MAX_FEEDBACK_LOG = 500

# Health state cache
_cached_health = {"healthy": True, "checked_at": 0.0}

# Cloud LLM Settings
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT_TEMPLATE = """You are a strictly non-conversational Data Extraction Engine for Saudi Vision 2030.

Core Mandate: Your output must be purely factual. Never use introductory phrases (e.g., "According to the documents," "Here is the information").

Extraction Rules:
1. Verbatim-First: If a direct match (line or bullet point) exists, return it verbatim.
2. Controlled Synthesis: If the answer is dispersed across a paragraph, extract the relevant data points and present them in a compact bulleted list. You are permitted to perform minimal rephrasing for clarity, but you are forbidden from generating sentences, summaries, or explanatory fluff.
3. Strict Scope Enforcement: You must ONLY answer based on the provided context. If the context does not explicitly contain the answer, you must return exactly: "I cannot find this information in the provided Saudi Vision 2030 policy documents."
4. No Conversational Fillers: Never write "I can help with that," "It is important to note," or any other AI-typical conversational filler.
5. Negative Constraint: If a user asks a question about an unrelated topic (e.g., foreign countries or general math), immediately trigger the hard fallback: "I cannot find this information in the provided Saudi Vision 2030 policy documents."

Goal: Provide only the facts. If the information isn't there, admit it instantly without explanation.

MEMORY (prior conversation):
{memory}

CONTEXT (your ONLY source of truth):
{context}"""


def optimize_search_query(user_query: str) -> str:
    """Normalizes typos and expands keywords standard to Saudi Vision 2030 docs."""
    query = user_query.lower().strip()

    query = re.sub(r'\b(min|mian)\b', 'main', query)
    query = re.sub(r'\b(there|their)\b', 'the', query)
    query = re.sub(r'\bpopullation\b', 'population', query)
    query = re.sub(r'\b(forieng|forign)\b', 'foreign', query)
    query = re.sub(r'\bsaudiarab\b', 'saudi arabia', query)
    query = re.sub(r'\bvison\b', 'vision', query)
    query = re.sub(r'\b2030s?\b', '2030', query)

    query = re.sub(
        r'\b(targets?|goals?|objectives?|aims?|purpos(e|es)?)\b', 
        'strategic objectives pillars targets goals', 
        query
    )
    query = re.sub(
        r'\b(projects?|initiatives?|programs?)\b', 
        'vision realization programs VRP initiatives projects', 
        query
    )

    if "oil" in query or "economy" in query:
        query += " non-oil GDP diversification revenue"

    return query


# ---------------------------------------------------------------------------
# Lifespan Hook (Non-blocking Fast Startup)
# ---------------------------------------------------------------------------
async def warmup_llm():
    """Asynchronously pings Groq API without holding up port binding."""
    await asyncio.sleep(1)
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            client = AsyncGroq(api_key=api_key)
            await client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[{"role": "user", "content": "ping"}],
                max_tokens=2
            )
            print("[INIT] Groq cloud inference engine reachable.")
    except Exception as e:
        print(f"[WARN] Non-fatal: Groq warmup ping skipped/failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast non-blocking startup lifecycle."""
    global startup_time
    startup_time = datetime.now(timezone.utc).isoformat()
    print("[INIT] FastAPI engine active. Port listening ready.")
    asyncio.create_task(warmup_llm())
    yield
    print("[SHUTDOWN] Terminating server context loops.")


# ---------------------------------------------------------------------------
# Application Instance & Middleware Setup
# ---------------------------------------------------------------------------
app = FastAPI(
    title="Saudi Vision 2030 Policy Intelligence Hub API",
    description="Enterprise-grade production asynchronous RAG engine utilizing Hybrid Architecture.",
    version="2.3.0",
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
    """Self-healing vector store verification."""
    now = time.time()
    if _cached_health["healthy"] and (now - _cached_health["checked_at"] < HEALTH_CHECK_TTL):
        return
    
    retriever_obj = get_retriever()
    try:
        is_healthy = retriever_obj.health_check()
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
    return raw_path.replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")

async def _build_memory_string(session_id: str) -> str:
    memory_context = await asyncio.to_thread(get_session_history, session_id, 4)
    memory_str = ""
    if memory_context.get("summary"):
        memory_str += f"Summary of past conversation: {memory_context['summary']}\n"
    for m in memory_context.get("messages", []):
        memory_str += f"{m['role'].upper()}: {m['content']}\n"
    return memory_str


# ---------------------------------------------------------------------------
# Pydantic Schemas
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
                raise ValueError("Payload must contain 'message', 'query', or 'question'.")
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
# API Routes
# ---------------------------------------------------------------------------
@app.post("/api/chat")
async def process_rag_chat(request: ChatRequest):
    """Main interactive chat endpoint."""
    _require_qdrant()
    start_time = time.time()
    top_k = request.k

    retriever_obj = get_retriever()
    reranker_obj = get_reranker()

    try:
        optimized_query = optimize_search_query(request.question)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(optimized_query) if use_hyde else optimized_query
        
        candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker_obj.rerank, request.question, candidates, top_k=top_k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Vector search engine timed out.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal retrieval failure: {str(e)}")

    memory_str = await _build_memory_string(request.session_id)
    context = "\n\n".join([c.content for c in reranked])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context)

    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"), timeout=30.0)
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            max_tokens=2048,
            temperature=0.2,
            top_p=1.0,
        )
        ai_answer = response.choices[0].message.content.strip()
        
        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
        asyncio.create_task(summarize_history(request.session_id))
        
    except APIError as e:
        return JSONResponse(
            status_code=503,
            content={"error": f"Groq cloud endpoints unreachable: {str(e)}"}
        )
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"LLM generation failed: {str(e)[:150]}")

    elapsed_time = time.time() - start_time
    elapsed_ms = round(elapsed_time * 1000, 2)
    
    SYSTEM_STATS["queries_served_this_session"] += 1
    SYSTEM_STATS["latency_history"].append(elapsed_time)
    if len(SYSTEM_STATS["latency_history"]) > MAX_LATENCY_HISTORY:
        SYSTEM_STATS["latency_history"].pop(0)

    source_citations = []
    for chunk in reranked:
        normalized_score = 1 / (1 + math.exp(-chunk.score))
        source_citations.append({
            "file": _clean_source_path(chunk.source),
            "page": chunk.page,
            "section": chunk.section,
            "score": round(normalized_score, 4),
        })

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
    """Programmatic standard validation endpoint."""
    global request_count
    _require_qdrant()

    if len(request.question) > 500:
        raise HTTPException(status_code=422, detail="Query exceeds maximum allowed limit of 500 characters.")

    t0 = time.time()
    retriever_obj = get_retriever()
    reranker_obj = get_reranker()

    try:
        optimized_query = optimize_search_query(request.question)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(optimized_query) if use_hyde else optimized_query
        candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker_obj.rerank, request.question, candidates, top_k=request.k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Remote vector store unavailable.")

    memory_str = await _build_memory_string(request.session_id)
    context_text = "\n\n".join([c.content for c in reranked])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context_text)

    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"), timeout=30.0)
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
            ],
            max_tokens=2048,
            temperature=0.2,
            top_p=1.0,
        )
        ai_answer = response.choices[0].message.content.strip()
        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
        asyncio.create_task(summarize_history(request.session_id))
    except Exception as e:
        raise HTTPException(status_code=504, detail=f"LLM compilation failed: {str(e)}")

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
        model=GROQ_MODEL,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Static UI & Health Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_index():
    """Serves index.html UI directly at the root path."""
    if INDEX_PATH and INDEX_PATH.exists():
        return FileResponse(INDEX_PATH)
    return JSONResponse(
        status_code=404,
        content={"error": "index.html static frontend page not found on server."}
    )

@app.get("/health")
def health():
    """Instant health check endpoint for cloud uptime probes."""
    return {
        "status": "ok",
        "model": f"{GROQ_MODEL} (Groq Cloud)",
        "vector_store": "saudi_vision_2030",
        "requests_served": request_count + SYSTEM_STATS["queries_served_this_session"],
        "uptime_since": startup_time,
    }

@app.get("/api/pipeline-info")
def get_pipeline_info():
    """Pipeline metadata information."""
    coll_info = {}
    try:
        retriever_obj = get_retriever()
        coll_info = retriever_obj.get_collection_info()
    except Exception:
        coll_info = {"points_count": "Unknown (Database Disconnected)"}

    reranker_name = get_reranker().model_name if _reranker_instance else "Cross-Encoder (Lazy Loaded)"

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
            "reranker_model": reranker_name,
            "retrieval_k": RETRIEVAL_K,
            "reranked_k": RERANK_TOP_K,
            "llm_backbone": f"{GROQ_MODEL} (Groq API)",
        },
    }

@app.get("/api/analytics")
def get_analytics_dashboard():
    """Performance metrics endpoint."""
    avg_latency = 0.0
    if SYSTEM_STATS["latency_history"]:
        avg_latency = sum(SYSTEM_STATS["latency_history"]) / len(SYSTEM_STATS["latency_history"])

    reranker_name = get_reranker().model_name if _reranker_instance else "Cross-Encoder (Lazy Loaded)"

    return {
        "session_metrics": {
            "queries_served": SYSTEM_STATS["queries_served_this_session"],
            "average_latency_seconds": round(avg_latency, 3),
        },
        "architecture": {
            "retrieval": "Hybrid (Dense + BM25 Sparse)",
            "fusion": "Reciprocal Rank Fusion (RRF)",
            "reranker": reranker_name,
        },
    }

@app.post("/feedback")
async def feedback(request: FeedbackRequest):
    """Saves user evaluation feedback."""
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