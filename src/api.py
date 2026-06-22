import os
import json
import time
import ollama
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

app = FastAPI(
    title="Saudi Vision 2030 RAG API",
    description=(
        "A production-grade Retrieval Augmented Generation system "
        "built on 48 official Saudi Vision 2030 policy documents. "
        "Ask any question about Vision 2030 goals, programs, and strategies."
    ),
    version="1.0.0",
    contact={
        "name": "Muhammad Hameed",
        "url": "https://github.com/muhammad-hameed-ai/rag-project",
    },
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

vector_store = None
startup_time = None
request_count = 0
feedback_log = []


@app.on_event("startup")
async def load_resources():
    global vector_store, startup_time
    print("Loading embedding model and connecting to Qdrant...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )
    vector_store = QdrantVectorStore.from_existing_collection(
        embedding=embeddings,
        collection_name="saudi_vision_2030",
        url=os.getenv("QDRANT_URL", "http://localhost:6333"),
    )
    startup_time = datetime.utcnow().isoformat()
    print("API ready.")


class AskRequest(BaseModel):
    question: str
    k: Optional[int] = 3


class SourceDoc(BaseModel):
    source: str
    preview: str


class AskResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDoc]
    retrieval_chunks: int
    latency_ms: float
    model: str
    timestamp: str


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int
    comment: Optional[str] = ""


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "llama3.2:1b",
        "vector_store": "saudi_vision_2030",
        "total_chunks": 5852,
        "requests_served": request_count,
        "uptime_since": startup_time,
    }


@app.get("/info")
def info():
    return {
        "project": "Saudi Vision 2030 RAG System",
        "description": (
            "Production RAG pipeline over 48 official Saudi Vision 2030 "
            "policy documents totalling 2,184 pages."
        ),
        "corpus": {
            "documents": 48,
            "pages": 2184,
            "chunks": 5852,
            "chunking_strategy": "recursive character splitting (1000/200 overlap)",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "vector_dimensions": 384,
            "vector_db": "Qdrant",
            "distance_metric": "cosine",
        },
        "generation": {
            "model": "llama3.2:1b via Ollama",
            "context_window": 2048,
        },
        "evaluation_baseline": {
            "faithfulness": 0.10,
            "answer_relevancy": 0.40,
            "context_precision": 0.35,
            "note": "Baseline scores using llama3.2:1b as judge",
        },
        "github": "https://github.com/muhammad-hameed-ai/rag-project",
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    global request_count

    if not request.question.strip():
        raise HTTPException(
            status_code=422,
            detail="Question cannot be empty."
        )

    if len(request.question) > 500:
        raise HTTPException(
            status_code=422,
            detail="Question too long. Maximum 500 characters."
        )

    k = max(1, min(request.k, 5))
    t0 = time.time()

    results = vector_store.similarity_search(request.question, k=k)

    context_text = "\n\n".join([doc.page_content for doc in results])
    prompt = (
        "You are an expert analyst on Saudi Vision 2030 policy documents.\n"
        "Answer the question using ONLY the context provided below.\n"
        "If the answer is not in the context, say: "
        "I cannot find this information in the provided documents.\n"
        "Be concise and specific.\n\n"
        "CONTEXT:\n" + context_text
        + "\n\nQUESTION:\n" + request.question
        + "\n\nANSWER:"
    )

    response = ollama.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_ctx": 2048, "num_predict": 400},
    )
    answer = response["message"]["content"]

    latency_ms = round((time.time() - t0) * 1000, 2)
    request_count += 1

    sources = []
    seen = set()
    for doc in results:
        src = doc.metadata.get("source", "unknown")
        src_clean = src.replace("data\\raw_pdfs\\", "").replace(
            "data/raw_pdfs/", ""
        )
        if src_clean not in seen:
            seen.add(src_clean)
            sources.append(
                SourceDoc(
                    source=src_clean,
                    preview=doc.page_content[:150].strip()
                )
            )

    return AskResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        retrieval_chunks=len(results),
        latency_ms=latency_ms,
        model="llama3.2:1b",
        timestamp=datetime.utcnow().isoformat(),
    )


@app.post("/feedback")
def feedback(request: FeedbackRequest):
    if request.rating not in [1, -1]:
        raise HTTPException(
            status_code=422,
            detail="Rating must be 1 (positive) or -1 (negative)."
        )

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": request.question,
        "answer": request.answer,
        "rating": request.rating,
        "comment": request.comment,
    }
    feedback_log.append(entry)

    os.makedirs("data/feedback", exist_ok=True)
    path = "data/feedback/feedback_log.json"
    existing = []
    if os.path.exists(path):
        with open(path) as f:
            existing = json.load(f)
    existing.append(entry)
    with open(path, "w") as f:
        json.dump(existing, f, indent=2)

    return {
        "status": "recorded",
        "total_feedback": len(existing)
    }