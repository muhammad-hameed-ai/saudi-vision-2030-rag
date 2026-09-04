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
import logging
import traceback
import secrets
from pathlib import Path
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List
from collections import deque

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks, UploadFile, File, Header
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from groq import AsyncGroq, APIError
from cachetools import TTLCache

# Local module imports
from src.retriever import HybridRetriever, QdrantUnavailableError
from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query
from src.hyde_retriever import generate_hypothesis
from src.memory import save_message, get_session_history, summarize_history
from src.groq_client import get_client as get_shared_groq_client, close_client as close_groq_client

# ---------------------------------------------------------------------------
# Logging & Environment Setup
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vision2030.api")

import sys
from dotenv import load_dotenv
load_dotenv()

# Strict Environment Validation (Security Requirement)
if not os.environ.get("GROQ_API_KEY"):
    logger.critical("FATAL: Missing GROQ_API_KEY. Server booting halted.")
    sys.exit(1)

if not (os.environ.get("QDRANT_URL") or os.environ.get("QDRANT_CLOUD_URL")):
    logger.critical("FATAL: Missing QDRANT_URL or QDRANT_CLOUD_URL. Server booting halted.")
    sys.exit(1)

if not (os.environ.get("QDRANT_API_KEY") or os.environ.get("QDRANT_CLOUD_API_KEY")):
    logger.critical("FATAL: Missing QDRANT_API_KEY or QDRANT_CLOUD_API_KEY. Server booting halted.")
    sys.exit(1)

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
def get_retriever() -> HybridRetriever:
    """Lazy initializer for HybridRetriever to prevent blocking Uvicorn startup."""
    global _retriever_instance
    if _retriever_instance is None:
        logger.info("[INIT] Lazy loading HybridRetriever models (FastEmbed + Qdrant)...")
        _retriever_instance = HybridRetriever()
    return _retriever_instance

def get_groq_client() -> AsyncGroq:
    """Returns the process-wide Groq client, so connections are reused between requests."""
    client = get_shared_groq_client()
    if client is None:
        logger.error("GROQ_API_KEY environment variable is not configured.")
        raise HTTPException(
            status_code=500,
            detail="GROQ_API_KEY environment variable is missing on server environment.",
        )
    return client

# Resolved once at import. Falling back to a fresh random token per request would
# silently lock the endpoint with no way to diagnose it, so we log the state.
ADMIN_PASSPHRASE = os.getenv("ADMIN_PASSPHRASE")
if not ADMIN_PASSPHRASE:
    ADMIN_PASSPHRASE = secrets.token_hex(16)
    logger.warning(
        "ADMIN_PASSPHRASE is not configured. Document deletion is disabled: "
        "the endpoint now requires an unguessable token that is never issued. "
        "Set ADMIN_PASSPHRASE in the environment to enable it."
    )

startup_time: Optional[str] = None
request_count: int = 0
feedback_log: List[dict] = []

# Pipeline Constants
RETRIEVAL_K = 6         # Reduced from 10 to slash latency (Phase 2 optimization)
RERANK_TOP_K = 4        # Compressed context size passed to LLM
# The health probe costs a Qdrant round trip; 5s re-checked on nearly every request.
HEALTH_CHECK_TTL = 30.0
HYDE_MAX_WORDS = 8      # above this a query embeds well enough on its own
MAX_UPLOAD_BYTES = 8 * 1024 * 1024   # the whole file is buffered in RAM (512MB box)
MAX_LATENCY_HISTORY = 100
MAX_FEEDBACK_LOG = 500

# Health state cache
_cached_health = {"healthy": True, "checked_at": 0.0}

# Global In-Memory Cache for Streaming
RAG_CACHE = TTLCache(maxsize=100, ttl=3600)
app_telemetry_logs = deque(maxlen=20)

_rag_pipeline_lock = None

def get_pipeline_lock() -> asyncio.Lock:
    global _rag_pipeline_lock
    if _rag_pipeline_lock is None:
        _rag_pipeline_lock = asyncio.Lock()
    return _rag_pipeline_lock

# Cloud LLM Settings
GROQ_MODEL = "allam-2-7b"

# allam-2-7b exposes a 4096-token window shared between prompt and completion.
# At k=10 the retrieved context alone reached ~2000 tokens, which together with a
# 2048-token response allowance left nothing for the system prompt or conversation
# memory. Everything below is budgeted against this window.
MODEL_CONTEXT_TOKENS = 4096
MAX_RESPONSE_TOKENS = 1024      # ~750 words: ample here, and half the worst-case latency
CONTEXT_SAFETY_TOKENS = 256     # headroom for tokenizer drift vs. the estimate below
CHARS_PER_TOKEN = 4             # conservative for mixed English/Arabic policy prose
MAX_MEMORY_CHARS = 2000

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

def _assemble_context(chunks: List[Any], memory_str: str) -> str:
    """
    Packs the highest-scoring chunks into whatever the context window has left.

    Chunks arrive sorted by true similarity, so filling from the front and stopping at
    the budget drops the weakest evidence first. Without this, a large k silently
    overflows the model window and the request either errors or is truncated server-side.
    """
    overhead = len(SYSTEM_PROMPT_TEMPLATE) + len(memory_str)
    budget = max(
        0,
        (MODEL_CONTEXT_TOKENS - MAX_RESPONSE_TOKENS - CONTEXT_SAFETY_TOKENS) * CHARS_PER_TOKEN
        - overhead,
    )
    parts, used = [], 0
    for chunk in chunks:
        text = getattr(chunk, "content", str(chunk))
        if used + len(text) > budget:
            remaining = budget - used
            if remaining > 200:   # a smaller fragment is noise, not evidence
                parts.append(text[:remaining])
            break
        parts.append(text)
        used += len(text) + 2
    return ("\n\n").join(parts)


def _clean_source_path(raw_path: Optional[str]) -> str:
    """Null-safe source path formatter."""
    if not raw_path:
        return "Saudi Vision 2030 Policy Document"
    return str(raw_path).replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")

def _clamp_score(x: float) -> float:
    """
    Normalizes a retriever score for display.

    The retriever now returns a true cosine similarity already bounded to [0, 1].
    The previous sigmoid existed to squash cross-encoder logits; applying it to a
    cosine compresses every value into a narrow band around 0.6 and destroys the
    signal, so this is a straight clamp.
    """
    try:
        return max(0.0, min(1.0, float(x)))
    except (TypeError, ValueError):
        return 0.0


# ---------------------------------------------------------------------------
# Lifespan Hook (Non-blocking Fast Startup)
# ---------------------------------------------------------------------------
async def warmup_llm():
    """Asynchronously pings Groq API without holding up port binding."""
    await asyncio.sleep(1)
    try:
        client = get_shared_groq_client()
        if client is not None:
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
    await close_groq_client()
    logger.info("[SHUTDOWN] Terminating server context loops.")


from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# ---------------------------------------------------------------------------
# Application Instance & Middleware Setup
# ---------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)

app = FastAPI(
    title="Saudi Vision 2030 Policy Intelligence Hub API",
    description="Enterprise-grade production asynchronous RAG engine utilizing Hybrid Architecture.",
    version="2.3.0",
    lifespan=lifespan,
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(StructuredLoggingMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:8000", 
        "http://127.0.0.1:8000", 
        "https://saudi-vision-2030-rag-3.onrender.com",
        "https://muhammad-hameed-ai.github.io"
    ],
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

async def _maybe_hyde(query: str) -> str:
    """
    Expands the query via HyDE only when it is short enough to be ambiguous.

    HyDE costs a full Groq round trip (roughly 0.5-1.5s) on every request. It earns
    that on terse queries with little to embed, but a long, specific question already
    carries more signal than a generated hypothesis, so we skip it and answer faster.
    """
    if os.environ.get("USE_HYDE", "false").lower() != "true":
        return query
    if len(query.split()) > HYDE_MAX_WORDS:
        return query
    return await generate_hypothesis(query)


async def _build_memory_string(session_id: str) -> str:
    try:
        memory_context = await asyncio.to_thread(get_session_history, session_id, 4)
        memory_str = ""
        if memory_context.get("summary"):
            memory_str += f"Summary of past conversation: {memory_context['summary']}\n"
        for m in memory_context.get("messages", []):
            memory_str += f"{m['role'].upper()}: {m['content']}\n"
        # Keep the most recent exchanges: an unbounded history would crowd out the
        # retrieved evidence it is supposed to accompany.
        if len(memory_str) > MAX_MEMORY_CHARS:
            memory_str = "...\n" + memory_str[-MAX_MEMORY_CHARS:]
        return memory_str
    except Exception as e:
        logger.warning(f"Failed to build session memory for {session_id}: {e}")
        return ""


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------
class ChatRequest(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str = Field(default="", max_length=1000)
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
    # Bounded so an unauthenticated caller cannot write arbitrarily large payloads.
    question: str = Field(max_length=1000)
    answer: str = Field(max_length=8000)
    rating: int
    comment: Optional[str] = Field(default="", max_length=2000)


# ---------------------------------------------------------------------------
# API Routes
# ---------------------------------------------------------------------------
async def generate_rag_stream(request: ChatRequest):
    """Async generator to stream RAG tokens and metadata via SSE."""
    global request_count
    start_time = time.time()
    query = request.question
    request_count += 1

    # Memory is loaded before the cache check deliberately: once a session has
    # history the answer is specific to that conversation, so a shared cached reply
    # would be wrong. k selects different evidence, so it belongs in the key too.
    memory_str = await _build_memory_string(request.session_id)
    cacheable = not memory_str
    query_key = (request.k, query.strip().lower())

    if cacheable and query_key in RAG_CACHE:
        cached = RAG_CACHE[query_key]
        yield f"data: {json.dumps({'type': 'metadata', 'sources': cached['sources'], 'cached': True})}\n\n"
        yield f"data: {json.dumps({'token': cached['response']})}\n\n"
        elapsed = round(time.time() - start_time, 2)
        yield f"data: {json.dumps({'type': 'telemetry', 'generation_time': elapsed, 'retrieval_k': request.k, 'relevance_score': cached.get('relevance_score')})}\n\n"
        # A cache hit is still a conversation turn. Skipping the write here leaves
        # holes in the history and breaks follow-ups that refer back to it.
        try:
            await asyncio.to_thread(save_message, request.session_id, 'user', query)
            await asyncio.to_thread(save_message, request.session_id, 'assistant', cached['response'])
        except Exception as mem_err:
            logger.warning(f"Session history save skipped: {mem_err}")
        yield "data: [DONE]\n\n"
        return

    try:
        await _require_qdrant()
        top_k = request.k

        retriever_obj = get_retriever()

        optimized_query = optimize_search_query(query)
        search_query = await _maybe_hyde(optimized_query)

        # Fetch at least as many candidates as the caller asked to keep. Pinning this
        # to RETRIEVAL_K silently capped the UI's depth slider at 6.
        fetch_k = max(RETRIEVAL_K, top_k)
        lock = get_pipeline_lock()
        async with lock:
            candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=fetch_k)
            # Bypass memory-heavy ONNX cross-encoder to prevent Render 512MB OOM crash
            reranked = candidates[:top_k]

        source_citations = []
        for chunk in reranked:
            raw_score = getattr(chunk, 'score', 0.0)
            source_citations.append({
                "file": _clean_source_path(getattr(chunk, 'source', None)),
                "page": getattr(chunk, 'page', 1),
                "section": getattr(chunk, 'section', 'General'),
                "score": round(_clamp_score(raw_score), 4),
            })

        # Render's proxy buffers small chunks. Padding forces immediate flush to UI.
        yield ":" + " " * 2048 + "\n\n"
        yield f"data: {json.dumps({'type': 'metadata', 'sources': source_citations, 'cached': False})}\n\n"

        context = _assemble_context(reranked, memory_str)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context)

        client = get_groq_client()
        stream = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query}
            ],
            stream=True,
            temperature=0.2,
            timeout=30.0,
            max_tokens=MAX_RESPONSE_TOKENS,
        )

        full_response = ""
        async for chunk in stream:
            token = chunk.choices[0].delta.content
            if token:
                full_response += token
                yield f"data: {json.dumps({'token': token})}\n\n"

        # Background history save
        try:
            await asyncio.to_thread(save_message, request.session_id, "user", query)
            await asyncio.to_thread(save_message, request.session_id, "assistant", full_response)
            asyncio.create_task(summarize_history(request.session_id))
        except Exception as mem_err:
            logger.warning(f"Session history save skipped: {mem_err}")

        elapsed = round(time.time() - start_time, 2)
        
        avg_relevance = 0.0
        if source_citations:
            avg_relevance = sum(c["score"] for c in source_citations) / len(source_citations)
            
        if cacheable:
            RAG_CACHE[query_key] = {
                "sources": source_citations,
                "response": full_response,
                "relevance_score": round(avg_relevance * 100, 2),
            }

        app_telemetry_logs.append({
            "latency": elapsed,
            "relevance_score": round(avg_relevance * 100, 2),
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        yield f"data: {json.dumps({'type': 'telemetry', 'generation_time': elapsed, 'retrieval_k': request.k, 'relevance_score': round(avg_relevance * 100, 2)})}\n\n"
        yield "data: [DONE]\n\n"

    except Exception as e:
        logger.error(f"Stream generation error:\n{traceback.format_exc()}")
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"

def _require_admin(token: Optional[str]) -> None:
    """
    Shared gate for every route that mutates the vector store or writes to disk.

    Ingest previously had no gate at all while delete did, even though both write to
    the same collection -- an unauthenticated writer can poison retrieval, exhaust the
    free-tier quota, or drive the container out of memory.
    """
    if not token or not secrets.compare_digest(token, ADMIN_PASSPHRASE):
        raise HTTPException(status_code=403, detail="Forbidden. Invalid administrative passcode.")


@app.post("/api/chat")
@limiter.limit("10/minute")
async def process_rag_chat(request: Request, payload: ChatRequest, background_tasks: BackgroundTasks):
    """Main interactive chat endpoint using Server-Sent Events (SSE)."""
    # Register forced garbage collection after the response is sent
    background_tasks.add_task(gc.collect)
    
    return StreamingResponse(
        generate_rag_stream(payload),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/ask", response_model=AskResponse)
@limiter.limit("10/minute")
async def ask(request: Request, payload: ChatRequest, background_tasks: BackgroundTasks):
    """Programmatic standard validation endpoint."""
    global request_count
    
    # Register forced garbage collection after the response is sent
    background_tasks.add_task(gc.collect)
    
    t0 = time.time()

    try:
        await _require_qdrant()
        retriever_obj = get_retriever()

        optimized_query = optimize_search_query(payload.question)
        search_query = await _maybe_hyde(optimized_query)

        fetch_k = max(RETRIEVAL_K, payload.k)
        candidates = await asyncio.to_thread(retriever_obj.retrieve, search_query, k=fetch_k)
        # Bypass memory-heavy ONNX cross-encoder to prevent Render 512MB OOM crash
        reranked = candidates[:payload.k]

        memory_str = await _build_memory_string(payload.session_id)
        context_text = _assemble_context(reranked, memory_str)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context_text)

        client = get_groq_client()
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": payload.question}
            ],
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.2,
            top_p=1.0,
            timeout=30.0,
        )
        ai_answer = response.choices[0].message.content.strip()

        try:
            await asyncio.to_thread(save_message, payload.session_id, "user", payload.question)
            await asyncio.to_thread(save_message, payload.session_id, "assistant", ai_answer)
            asyncio.create_task(summarize_history(payload.session_id))
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
                score=round(_clamp_score(raw_score), 4)
            ))

        await asyncio.to_thread(
            log_rag_query,
            query=payload.question,
            sources=[s.model_dump() for s in sources],
            reranker_scores=[getattr(c, 'score', 0.0) for c in reranked],
            answer=ai_answer,
            latency_ms=latency_ms,
            retrieval_k=len(candidates),
            reranked_k=len(reranked),
        )

        return AskResponse(
            session_id=payload.session_id,
            question=payload.question,
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
        return FileResponse(
            INDEX_PATH,
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0",
            }
        )
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
        "requests_served": request_count,
        "uptime_since": startup_time,
    }

@app.get("/api/pipeline-info")
def get_pipeline_info():
    """Pipeline metadata information."""
    coll_info = {}
    try:
        retriever_obj = get_retriever()
        coll_info = retriever_obj.get_telemetry_stats()
    except Exception as e:
        logger.warning(f"Could not retrieve Qdrant collection info: {e}")
        coll_info = {"points_count": None, "unique_sources": None, "available": False}

    return {
        "corpus_summary": {
            "documents": coll_info.get("unique_sources"),
            "chunks": coll_info.get("points_count"),
            "dimensions": 384,
            "available": coll_info.get("available", True),
        },
        "configuration": {
            "document_loader": "PyMuPDFLoader",
            "chunking_strategy": "Structure-Aware Recursive Splitting",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "sparse_model": "Qdrant/bm25",
            "vector_database": "Qdrant (Hybrid: Dense + Sparse)",
            "distance_metric": "Cosine + RRF Fusion",
            "reranker_model": "Disabled (bypassed to stay within the 512MB memory limit)",
            "retrieval_k": RETRIEVAL_K,
            "reranked_k": RERANK_TOP_K,
            "llm_backbone": f"{GROQ_MODEL} (Groq API)",
        },
    }

@app.get("/api/documents")
@limiter.limit("30/minute")
def get_documents(request: Request):
    """Returns the list of documents and their chunk counts."""
    try:
        retriever_obj = get_retriever()
        registry = retriever_obj.get_document_registry()
        return {"documents": registry}
    except Exception as e:
        logger.error(f"Error fetching document registry: {e}")
        return JSONResponse(status_code=500, content={"error": "Failed to fetch document registry"})

@app.delete("/api/documents")
def delete_document_endpoint(filename: str, x_admin_access_token: Optional[str] = Header(None)):
    """Deletes all chunks associated with a specific document."""
    _require_admin(x_admin_access_token)

    if not filename:
        return JSONResponse(status_code=400, content={"error": "Filename parameter is required."})
    try:
        retriever_obj = get_retriever()
        success = retriever_obj.delete_document(filename)
        retriever_obj._vision_sources = None
        if success:
            return {"status": "success", "message": f"Document '{filename}' purged successfully."}
        else:
            return JSONResponse(status_code=500, content={"error": f"Failed to purge document '{filename}'."})
    except Exception as e:
        logger.error(f"Error deleting document '{filename}': {e}")
        return JSONResponse(status_code=500, content={"error": f"Exception occurred while purging '{filename}'."})

@app.get("/api/documents/auth")
def verify_document_auth(x_admin_access_token: Optional[str] = Header(None)):
    """Pre-flight check to verify admin passcode without deleting anything."""
    _require_admin(x_admin_access_token)
    return {"status": "success"}

import io
import fitz

@app.post("/api/ingest/stream")
@limiter.limit("5/minute")
async def ingest_pdf_stream(
    request: Request,
    file: UploadFile = File(...),
    x_admin_access_token: Optional[str] = Header(None),
):
    # Writes to the same collection that DELETE guards, so it takes the same gate.
    _require_admin(x_admin_access_token)

    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF documents are supported.")

    # Read with a ceiling: the whole file is buffered in RAM, so an unbounded upload
    # is an out-of-memory kill on a 512MB instance.
    pdf_bytes = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(pdf_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)}MB limit for in-memory indexing. "
                   "Index large documents locally instead.",
        )
    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Strip any directory component a client may have supplied.
    filename = os.path.basename(file.filename)

    async def event_generator():
        try:
            # Step 1: In-memory Parsing
            yield f"data: {json.dumps({'stage': 'parsing', 'progress': 15, 'message': 'Extracting text streams in RAM...'})}\n\n"
            await asyncio.sleep(0.01) # Yield to event loop
            
            pdf_stream = io.BytesIO(pdf_bytes)
            doc = fitz.open(stream=pdf_stream, filetype="pdf")
            
            chunks = []
            total_pages = len(doc)
            
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if text.strip():
                    page_chunks = [text[i:i+800] for i in range(0, len(text), 700)]
                    for idx, chunk_text in enumerate(page_chunks):
                        chunks.append({
                            "text": chunk_text,
                            "metadata": {
                                "source": filename,
                                "page": page_num + 1,
                                "section": "General",
                                "chunk_id": f"{filename}_p{page_num+1}_c{idx}"
                            }
                        })
            doc.close()
            pdf_stream.close()

            if not chunks:
                # A scanned/image-only PDF yields no text. Reporting success would add
                # the file to the registry with nothing retrievable behind it.
                yield f"data: {json.dumps({'stage': 'error', 'message': 'No extractable text found. This PDF is likely scanned images and needs OCR before indexing.'})}\n\n"
                return

            # Step 2: Vectorization
            yield f"data: {json.dumps({'stage': 'embedding', 'progress': 50, 'message': f'Generated {len(chunks)} chunks. Vectorizing via FastEmbed...'})}\n\n"
            await asyncio.sleep(0.01)

            # Batch process vectors to avoid RAM spikes
            batch_size = 64
            retriever_obj = get_retriever()
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                await retriever_obj.upsert_in_memory_chunks(batch)
                progress = 50 + int((i / len(chunks)) * 40)
                yield f"data: {json.dumps({'stage': 'indexing', 'progress': progress, 'message': f'Indexed {i + len(batch)}/{len(chunks)} chunks in Qdrant...'})}\n\n"
                await asyncio.sleep(0.01)

            # The booster caches the resolved Vision 2030 source list; a new upload
            # can change it, so drop it and let the next query re-resolve.
            retriever_obj._vision_sources = None

            # Step 3: Atomic Registry Update
            yield f"data: {json.dumps({'stage': 'complete', 'progress': 100, 'message': f'Successfully added {filename} to Qdrant Cloud!'})}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'stage': 'error', 'message': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/analytics")
def get_analytics_dashboard():
    """Performance metrics endpoint."""
    # Returns [] until real queries have been served. Inventing a baseline here would
    # render as genuine telemetry in the dashboard.
    return list(app_telemetry_logs)

@app.post("/feedback")
@limiter.limit("20/minute")
async def feedback(request: Request, payload: FeedbackRequest):
    """Saves user evaluation feedback."""
    if payload.rating not in [1, -1]:
        raise HTTPException(status_code=422, detail="Rating must be 1 or -1.")

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "question": payload.question,
        "answer": payload.answer,
        "rating": payload.rating,
        "comment": payload.comment,
    }
    
    feedback_log.append(entry)
    if len(feedback_log) > MAX_FEEDBACK_LOG:
        feedback_log.pop(0)

    # Anchored to the project root rather than the process CWD.
    feedback_dir = PROJECT_ROOT / "data" / "feedback"
    os.makedirs(feedback_dir, exist_ok=True)
    path = str(feedback_dir / "feedback_log.jsonl")
    
    def write_feedback():
        # Append-only JSONL: the previous version re-read and re-serialised the entire
        # file on every submission, which is O(n) per write and unbounded on disk.
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + chr(10))
            
    await asyncio.to_thread(write_feedback)
    return {"status": "recorded"}