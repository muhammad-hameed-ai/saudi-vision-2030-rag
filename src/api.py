"""
Saudi Vision 2030 Policy Intelligence Hub — Production Grade API (V2.3 Cloud)

Architecture:
  Hybrid Retrieval (Dense + Sparse BM25) → Cross-Encoder Reranker → Groq Cloud LLM
  Features: Async Non-Blocking Endpoints, Dynamic Schema Validation, Structured Logging
"""

import os
import re
import asyncio
import json
import uuid
import time
import math
from datetime import datetime, timezone
from contextlib import asynccontextmanager
from typing import Optional, Any, List

from fastapi import FastAPI, HTTPException
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
# Environment & Global State Configuration
# ---------------------------------------------------------------------------
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

# Cloud LLM Settings
GROQ_MODEL = "llama-3.1-8b-instant"

# Smart Intent System Prompt (shared across all endpoints)
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
    """
    Normalizes typos and expands keywords standard to Saudi Vision 2030 docs
    to maximize Qdrant hybrid search recall.
    """
    query = user_query.lower().strip()

    # Structural Typo & Phonetic Normalization
    query = re.sub(r'\b(min|mian)\b', 'main', query)
    query = re.sub(r'\b(there|their)\b', 'the', query)
    query = re.sub(r'\bpopullation\b', 'population', query)
    query = re.sub(r'\b(forieng|forign)\b', 'foreign', query)
    query = re.sub(r'\bsaudiarab\b', 'saudi arabia', query)
    query = re.sub(r'\bvison\b', 'vision', query)
    query = re.sub(r'\b2030s?\b', '2030', query)

    # Synonym & Concept Cluster Mapping (Plural & Boundary Resilient)
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
# Lifespan Hook (Model Preloading & Warm-up)
# ---------------------------------------------------------------------------
async def warmup_llm():
    """Preloads the LLM backbone via the Groq Cloud API asynchronously."""
    print("[INIT] Initiating asynchronous Groq API warm-up sequence...")
    await asyncio.sleep(2)  
    try:
        client = AsyncGroq(api_key=os.environ.get("GROQ_API_KEY"))
        await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=2
        )
        print("[INIT] Groq cloud inference engine reachable. Warm-up successful.")
    except Exception as e:
        print(f"[WARN] Non-fatal: Groq connection failed during warmup: {e}")

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
    """
    Self-healing fail-closed cluster safety mechanism.
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
    Main asynchronous interactive dashboard route utilizing the Groq SDK.
    """
    _require_qdrant()
    start_time = time.time()
    top_k = request.k

    try:
        # Step 1: Query Optimization (typo tolerance + keyword expansion)
        optimized_query = optimize_search_query(request.question)

        # Step 2: Query Conditioning (HyDE)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(optimized_query) if use_hyde else optimized_query
        
        # Step 3: Dense + Sparse Candidate Extraction
        candidates = await asyncio.to_thread(retriever.retrieve, search_query, k=RETRIEVAL_K)

        # Step 4: Deep Cross-Encoder Re-scoring
        reranked = await asyncio.to_thread(reranker.rerank, request.question, candidates, top_k=top_k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Vector search engine timed out during document pooling.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal collection failure: {str(e)}")

    # Step 5: System Memory Injection & Prompt Assembly
    memory_str = await _build_memory_string(request.session_id)
    context = "\n\n".join([c.content for c in reranked])
    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory=memory_str, context=context)

    # Step 5: Asynchronous Groq LLM Text Synthesis Loop
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
    """Programmatic standard validation endpoint."""
    global request_count
    _require_qdrant()

    if len(request.question) > 500:
        raise HTTPException(status_code=422, detail="Query validation breach: Input string exceeds 500 characters maximum.")

    t0 = time.time()
    try:
        optimized_query = optimize_search_query(request.question)
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(optimized_query) if use_hyde else optimized_query
        candidates = await asyncio.to_thread(retriever.retrieve, search_query, k=RETRIEVAL_K)
        reranked = await asyncio.to_thread(reranker.rerank, request.question, candidates, top_k=request.k)
    except QdrantUnavailableError:
        raise HTTPException(status_code=503, detail="Remote database engine cluster rejected operation request.")

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
        model=GROQ_MODEL,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ---------------------------------------------------------------------------
# Telemetry Analytics & Metadata Core Endpoints
# ---------------------------------------------------------------------------
@app.get("/")
def read_index():
    """Serves the static production UI matrix directly from root context."""
    # Fixed path resolution: looks in the same directory as app.py
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "index.html")
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
        "model": f"{GROQ_MODEL} (Groq Cloud)",
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
            "llm_backbone": f"{GROQ_MODEL} (Groq API)",
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