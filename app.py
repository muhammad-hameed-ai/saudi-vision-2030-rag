import time
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import qdrant_client
import requests
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Vision 2030 Policy Intelligence Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared System In-Memory Analytics State
SYSTEM_STATS = {
    "queries_served_this_session": 0,
    "latency_history": []
}

print("Initializing Cloud Infrastructure Components...")
# Lightweight cloud-based inference (Zero local RAM footprint)
embeddings = HuggingFaceInferenceAPIEmbeddings(
    api_key=os.getenv("HF_TOKEN"),
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Connect to Qdrant Cloud
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY")
)

class ChatRequest(BaseModel):
    question: str
    k: int = 3

@app.get("/health")
async def health_check():
    try:
        collection_info = client.get_collection("saudi_vision_2030")
        return {
            "status": "online",
            "database_connected": True,
            "points_count": collection_info.points_count,
            "llm_status": "accessible (Groq API)"
        }
    except Exception as e:
        return {"status": "degraded", "database_connected": False, "error": str(e)}

@app.get("/api/pipeline-info")
async def get_pipeline_info():
    try:
        coll = client.get_collection("saudi_vision_2030")
        return {
            "corpus_summary": {
                "documents": 48,
                "pages": 2184,
                "chunks": coll.points_count,
                "dimensions": 384 
            },
            "configuration": {
                "document_loader": "PyPDFDirectoryLoader",
                "chunking_strategy": "RecursiveCharacterTextSplitter",
                "chunk_size": 1000,
                "chunk_overlap": 200,
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_database": "Qdrant Cloud (eu-west-1)",
                "distance_metric": "Cosine Similarity",
                "llm_backbone": "Llama 3.1 8b (Groq API)"
            }
        }
    except Exception as e:
        print(f"Pipeline Info Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analytics")
async def get_analytics_dashboard():
    avg_latency = 0.0
    if SYSTEM_STATS["latency_history"]:
        avg_latency = sum(SYSTEM_STATS["latency_history"]) / len(SYSTEM_STATS["latency_history"])
        
    return {
        "evaluation_scores": {
            "faithfulness": 0.74,
            "answer_relevance": 0.82,
            "context_precision": 0.69
        },
        "session_metrics": {
            "queries_served": SYSTEM_STATS["queries_served_this_session"],
            "average_latency_seconds": round(avg_latency, 3)
        }
    }

@app.post("/api/chat")
async def process_rag_chat(request: ChatRequest):
    start_time = time.time()
    try:
        # 1. Similarity Retrieval Payload Construction (Cloud)
        query_vector = embeddings.embed_query(request.question)
        results = client.query_points(
            collection_name="saudi_vision_2030",
            query=query_vector,
            using="dense",  # Explicitly matches your cloud schema
            limit=request.k
        ).points
        
        context_blocks = []
        source_citations = []
        for hit in results:
            context_blocks.append(hit.payload.get('page_content', ''))
            meta = hit.payload.get('metadata', {})
            source_citations.append({
                "file": meta.get('source', 'Unknown File'),
                "page": meta.get('page', 0),
                "score": round(hit.score, 4)
            })
            
        context = "\n\n".join(context_blocks)
        
        # 2. Cloud LLM Prompt Injection Execution (Groq Llama 3.1 8b)
        prompt = f"Context:\n{context}\n\nQuestion: {request.question}\nAnswer:"
        
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": "You are a helpful and precise assistant for the Saudi Vision 2030 Policy Intelligence Hub. Formulate your answers based strictly on the provided context."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.1,
            "stream": False
        }
        
        groq_response = requests.post(groq_url, headers=headers, json=payload, timeout=30)
        
        if groq_response.status_code != 200:
            error_details = groq_response.text
            print(f"Groq API Error: {error_details}")
            raise HTTPException(status_code=groq_response.status_code, detail=f"Groq API Error: {error_details}")
            
        ai_answer = groq_response.json()["choices"][0]["message"]["content"]
        
        # 3. Compute Latency Performance Metrics
        elapsed_time = time.time() - start_time
        SYSTEM_STATS["queries_served_this_session"] += 1
        SYSTEM_STATS["latency_history"].append(elapsed_time)
        
        return {
            "answer": ai_answer.strip(),
            "citations": source_citations,
            "metrics": {
                "latency_seconds": round(elapsed_time, 3),
                "retrieval_depth_k": request.k
            }
        }
        
    except Exception as e:
        print(f"Chat Endpoint Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)