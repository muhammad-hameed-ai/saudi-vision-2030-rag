"""
Saudi Vision 2030 Policy Intelligence Hub — Production Grade API (V2.3 Cloud)

Architecture:
  Hybrid Retrieval (Dense + Sparse BM25) → Cross-Encoder Reranker → Groq Cloud LLM
  Features: Lazy-loaded Heavy Models, Async Non-Blocking Endpoints, Dynamic Schema Validation
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import re
import gc
import asyncio
import json
import uuid
import time
import math
import logging
import traceback
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
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
# Logging & Environment Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision2030.api")

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
        logger.info("[INIT] Lazy loading HybridRetriever models (FastEmbed + Qdrant)...")
        _retriever_instance = HybridRetriever()
    return _retriever_instance

def get_reranker() -> Reranker:
    """Lazy initializer for Reranker model."""
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("[INIT] Lazy loading Cross-Encoder Reranker model...")
        _reranker_instance = Reranker()
    return _reranker_instance

def get_groq_client() -> AsyncGroq:
    """Validates API key and returns initialized AsyncGroq client."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.error("GROQ_API_KEY environment variable is not configured.")
        raise HTTPException(
            status_code=500, 
            detail="GROQ_API_KEY environment variable is missing on server environment."
        )
    return AsyncGroq(api_key=api_key, timeout=30.0)

startup_time: Optional[str] = None
request_count: int = 0
feedback_log: List[dict] = []

SYSTEM_STATS = {
    "queries_served_this_session": 0,
    "latency_history": [],
}

# Pipeline Constants
RETRIEVAL_K = 6         # Reduced from 10 to slash latency (Phase 2 optimization)
RERANK_TOP_K = 4        # Compressed context size passed to LLM
HEALTH_CHECK_TTL = 5.0  
MAX_LATENCY_HISTORY = 100
MAX_FEEDBACK_LOG = 500

# Health state cache
_cached_health = {"healthy": True, "checked_at": 0.0}

# Cloud LLM Settings
GROQ_MODEL = "llama-3.1-8b-instant"

SYSTEM_PROMPT_TEMPLATE = """You are a Data Extraction Engine and Subject Matter Expert for Saudi Vision 2030.

Core Mandate: Your output must be factual and professional. Provide clear, direct answers without introductory fluff (e.g., avoid "According to the documents").

Extraction & Synthesis Rules:
1. Context-Primary: Use the provided CONTEXT as your primary source of truth.
2. Baseline Synthesis (Loosened Prompt): If the CONTEXT provides partial information or relevant keywords, you are EXPLICITLY AUTHORIZED to use your internal baseline knowledge of Saudi Vision 2030 (e.g., NEOM, the 3 pillars, PIF) to synthesize a complete, accurate answer.
3. Formatting: Present dispersed data points as compact bulleted lists for readability.
4. Negative Constraint: You must ONLY refuse to answer if the user's query is COMPLETELY UNRELATED to Saudi Vision 2030, Saudi Arabia, or its economic/social policies (e.g., general math, foreign countries). In that specific case, return: "I cannot find this information in the provided Saudi Vision 2030 policy documents."

Goal: Provide factual, complete answers by combining the provided context with your baseline domain knowledge.

MEMORY (prior conversation):
{memory}

CONTEXT (retrieved chunks):
{context}"""


# ---------------------------------------------------------------------------
# Utility Methods
# ---------------------------------------------------------------------------
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

def _clean_source_path(raw_path: Optional[str]) -> str:
    """Null-safe source path formatter."""
    if not raw_path:
        return "Saudi Vision 2030 Policy Document"
    return str(raw_path).replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")

def _sigmoid(x: float) -> float:
    """Safe sigmoid calculation for reranker normalization."""
    try:
        bounded = max(min(float(x), 50.0), -50.0)
        return 1.0 / (1.0 + math.exp(-bounded))
    except Exception:
        return 0.5


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
            logger.info("[INIT] Groq cloud inference engine reachable.")
    except Exception as e:
        logger.warning(f"[WARN] Non-fatal: Groq warmup ping skipped/failed: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Fast non-blocking startup lifecycle."""
    global startup_time
    startup_time = datetime.now(timezone.utc).isoformat()
    logger.info("[INIT] FastAPI engine active. Port listening ready.")
    asyncio.create_task(warmup_llm())
    yield
    logger.info("[SHUTDOWN] Terminating server context loops.")


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
async def _require_qdrant():
    """Self-healing vector store verification."""
    now = time.time()
    if _cached_health["healthy"] and (now - _cached_health["checked_at"] < HEALTH_CHECK_TTL):
        return
    
    try:
        retriever_obj = get_retriever()
        is_healthy = await asyncio.to_thread(retriever_obj.health_check)
    except Exception as e:
        logger.warning(f"[HEALTH CHECK FAILED] Vector check exception:\n{traceback.format_exc()}")
        is_healthy = False

    _cached_health["healthy"] = is_healthy
    _cached_health["checked_at"] = now

    if not is_healthy:
        raise HTTPException(
            status_code=503,
            detail="Upstream vector infrastructure is down or degraded. Please retry shortly.",
        )

async def _build_memory_string(session_id: str) -> str:
    try:
        memory_context = await asyncio.to_thread(get_session_history, session_id, 4)
        memory_str = ""
        if memory_context.get("summary"):
            memory_str += f"Summary of past conversation: {memory_context['summary']}\n"
        for m in memory_context.get("messages", []):
            memory_str += f"{m['role'].upper()}: {m['content']}\n"
        return memory_str
    except Exception as e:
        logger.warning(f"Failed to build session memory for {session_id}: {e}")
        return ""


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
async def process_rag_chat(request: ChatRequest, background_tasks: BackgroundTasks):
    """Main interactive chat endpoint with full exception diagnostic logging."""
    start_time = time.time()
    
    # Register forced garbage collection after the response is sent
    background_tasks.add_task(gc.collect)
    
    try:
        await _require_qdrant()
        top_k = request.k

        retriever_obj = get_retriever()
        reranker_obj = get_reranker()

        optimized_query = optimize_search_query(request.question)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        
        if use_hyde:
            search_query = await generate_hypothesis(optimized_query)
        else:
            search_query = optimized_query
        
        candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker_obj.rerank, request.question, candidates, top_k=top_k)

        memory_str = await _build_memory_string(request.session_id)
        context = "\n\n".join([getattr(c, 'content', str(c)) for c in reranked])
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context)

        client = get_groq_client()
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
        
        try:
            await asyncio.to_thread(save_message, request.session_id, "user", request.question)
            await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
            asyncio.create_task(summarize_history(request.session_id))
        except Exception as mem_err:
            logger.warning(f"Session history save skipped: {mem_err}")

        elapsed_time = time.time() - start_time
        elapsed_ms = round(elapsed_time * 1000, 2)
        
        SYSTEM_STATS["queries_served_this_session"] += 1
        SYSTEM_STATS["latency_history"].append(elapsed_time)
        if len(SYSTEM_STATS["latency_history"]) > MAX_LATENCY_HISTORY:
            SYSTEM_STATS["latency_history"].pop(0)

        source_citations = []
        for chunk in reranked:
            raw_score = getattr(chunk, 'score', 0.0)
            source_citations.append({
                "file": _clean_source_path(getattr(chunk, 'source', None)),
                "page": getattr(chunk, 'page', 1),
                "section": getattr(chunk, 'section', 'General'),
                "score": round(_sigmoid(raw_score), 4),
            })

        await asyncio.to_thread(
            log_rag_query,
            query=request.question,
            sources=source_citations,
            reranker_scores=[getattr(c, 'score', 0.0) for c in reranked],
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

    except QdrantUnavailableError as e:
        logger.error(f"[503] Qdrant Engine Error: {e}")
        raise HTTPException(status_code=503, detail="Vector search engine timed out or unavailable.")
    
    except APIError as e:
        logger.error(f"[503] Groq Cloud API Error: {e}")
        return JSONResponse(
            status_code=503,
            content={"error": f"Groq cloud endpoints unreachable: {str(e)}"}
        )
    
    except HTTPException:
        raise
        
    except Exception as e:
        stack_trace = traceback.format_exc()
        logger.error(f"=== UNHANDLED RAG PIPELINE EXCEPTION ===\n{stack_trace}")
        return JSONResponse(
            status_code=500,
            content={
                "error": str(e),
                "type": type(e).__name__,
                "detail": "Internal processing error during retrieval, reranking, or completion."
            }
        )

@app.post("/ask", response_model=AskResponse)
async def ask(request: ChatRequest, background_tasks: BackgroundTasks):
    """Programmatic standard validation endpoint."""
    global request_count
    
    # Register forced garbage collection after the response is sent
    background_tasks.add_task(gc.collect)
    
    if len(request.question) > 500:
        raise HTTPException(status_code=422, detail="Query exceeds maximum allowed limit of 500 characters.")

    t0 = time.time()

    try:
        await _require_qdrant()
        retriever_obj = get_retriever()
        reranker_obj = get_reranker()

        optimized_query = optimize_search_query(request.question)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(optimized_query) if use_hyde else optimized_query
        
        candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker_obj.rerank, request.question, candidates, top_k=request.k)

        memory_str = await _build_memory_string(request.session_id)
        context_text = "\n\n".join([getattr(c, 'content', str(c)) for c in reranked])
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context_text)

        client = get_groq_client()
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

        try:
            await asyncio.to_thread(save_message, request.session_id, "user", request.question)
            await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
            asyncio.create_task(summarize_history(request.session_id))
        except Exception as mem_err:
            logger.warning(f"Session history save skipped: {mem_err}")

        latency_ms = round((time.time() - t0) * 1000, 2)
        request_count += 1

        sources = []
        for c in reranked:
            raw_score = getattr(c, 'score', 0.0)
            sources.append(SourceDoc(
                source=_clean_source_path(getattr(c, 'source', None)),
                page=getattr(c, 'page', 1),
                section=getattr(c, 'section', 'General'),
                preview=getattr(c, 'content', '')[:150].strip(),
                score=round(_sigmoid(raw_score), 4)
            ))

        await asyncio.to_thread(
            log_rag_query,
            query=request.question,
            sources=[s.model_dump() for s in sources],
            reranker_scores=[getattr(c, 'score', 0.0) for c in reranked],
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

    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Remote vector store unavailable.")
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"=== ASK ENDPOINT ERROR ===\n{traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"LLM compilation or pipeline failure: {str(e)}")


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
    except Exception as e:
        logger.warning(f"Could not retrieve Qdrant collection info: {e}")
        coll_info = {"points_count": "Unknown (Database Disconnected)"}

    reranker_name = get_reranker().model_name if _reranker_instance else "Cross-Encoder (Lazy Loaded)"

    return {
        "corpus_summary": {
            "documents": 48,
            "pages": 2184,
            "chunks": coll_info.get("points_count", 0) if isinstance(coll_info, dict) else "Unknown",
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