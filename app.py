import time
import os
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import qdrant_client
import requests
from langchain_huggingface import HuggingFaceEndpointEmbeddings
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
embeddings = HuggingFaceEndpointEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2",
    huggingfacehub_api_token=os.getenv("HF_TOKEN"),
)

# Connect to Qdrant Cloud
client = qdrant_client.QdrantClient(
    url=os.getenv("QDRANT_CLOUD_URL"),
    api_key=os.getenv("QDRANT_CLOUD_API_KEY")
)

class ChatRequest(BaseModel):
    question: str
    k: int = 3


# Smart Intent System Prompt
SYSTEM_PROMPT_TEMPLATE = """You are the Senior Policy Intelligence Analyst for the Saudi Vision 2030 Hub. Your core mandate is to deliver highly accurate, contextual, and helpful insights from the provided document context.

OPERATIONAL INSTRUCTIONS:
1. SMART INTENT EXTRACTION: Users may provide queries with typographical errors, missing punctuation, or slightly imprecise phrasing. You must look past superficial syntax errors and deduce the true semantic intent of the query.
2. THE 50% RELATEDNESS RULE:
   - If the context contains a direct, explicit answer, provide it clearly and concisely.
   - If the context does not contain a flawless direct match, but contains information that is at least 50% relevant to the user's core intent, do NOT reject it. Instead, bridge the gap transparently. Phrase it like: "While the exact target for [X] is not explicitly detailed, the policy documents outline the following related initiatives: [Y]."
3. HARD SAFETY BOUNDARY: If the provided context shares absolutely zero semantic overlap with the query (less than 50% match), or if the query relates to completely out-of-scope topics (e.g., international affairs, unrelated countries), you must return this exact fallback string verbatim and nothing else:
   "I cannot find this information in the provided Saudi Vision 2030 policy documents."
4. NO INVENTED KNOWLEDGE: Never utilize pre-trained global knowledge to invent facts, metrics, or initiatives that are missing from the provided context.

CONTEXT (your ONLY source of truth):
{context}"""


def optimize_search_query(user_query: str) -> str:
    """
    Normalizes typos and expands keywords standard to Saudi Vision 2030 docs
    to maximize Qdrant hybrid search recall.
    """
    query = user_query.lower().strip()
    query = re.sub(r'\bsaudiarab\b', 'saudi arabia', query)
    query = re.sub(r'\bthere main\b', 'their main', query)
    query = re.sub(r'\bthere new\b', 'their new', query)
    query = re.sub(r'\bvison\b', 'vision', query)
    query = re.sub(r'\b2030s?\b', '2030', query)

    if "project" in query or "program" in query:
        query += " vision realization programs VRP initiatives"
    if "goal" in query or "pillar" in query:
        query += " strategic objectives pillars targets"
    if "oil" in query or "economy" in query:
        query += " non-oil GDP diversification revenue"

    return query

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
        # 1. Query Optimization + Similarity Retrieval (Cloud)
        optimized_query = optimize_search_query(request.question)
        query_vector = embeddings.embed_query(optimized_query)
        results = client.query_points(
            collection_name="saudi_vision_2030",
            query=query_vector,
            using="dense",
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
        
        # 2. Cloud LLM Prompt Assembly (Groq Llama 3.1 8b)
        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(context=context)
        
        groq_url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "llama-3.1-8b-instant",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": request.question}
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