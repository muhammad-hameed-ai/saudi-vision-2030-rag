"""
Saudi Vision 2030 Policy Intelligence Hub — Core RAG Pipeline (V2.2)

Features: 
  - Cloud-Native Qdrant Cluster Synchronization
  - Serverless Low-RAM Embedding Inference (No Torch/Transformers)
  - Auto-fallback for Local/Production Environment Tiers
"""

import os
import ollama
from typing import Tuple, List
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient

def get_vector_store() -> QdrantVectorStore:
    """
    Initializes a production-grade secure connection to the Qdrant Vector database.
    Dynamically falls back to localhost if cloud environment variables are missing.
    """
    # 1. Serverless Low-RAM Embeddings Config (Fixes Render Free-Tier RAM limits)
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.getenv("HF_TOKEN"),
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        # Fixes the deprecated Hugging Face domain DNS resolution failure (Errno -5)
        api_url="https://router.huggingface.co/hf-inference/models"
    )

    # 2. Dynamic Connection Routing
    cloud_url = os.getenv("QDRANT_CLOUD_URL")
    cloud_key = os.getenv("QDRANT_CLOUD_API_KEY")

    if cloud_url and cloud_key:
        print("[INFO] Establishing high-performance connection to Qdrant Cloud Cluster...")
        client = QdrantClient(
            url=cloud_url,
            api_key=cloud_key,
            timeout=60.0
        )
    else:
        print("[WARN] Cloud variables missing. Routing to local fallback (http://localhost:6333)...")
        client = QdrantClient(url="http://localhost:6333")

    return QdrantVectorStore(
        client=client,
        collection_name="saudi_vision_2030",
        embedding=embeddings,
    )


def retrieve_context(store: QdrantVectorStore, query: str, k: int = 3) -> Tuple[str, List[str]]:
    """Retrieves document chunks matching the query semantic profile along with citations."""
    try:
        results = store.similarity_search(query, k=k)
    except Exception as e:
        print(f"[ERROR] Failed to extract vectors from database cluster: {e}")
        return "", []

    context_parts = []
    sources = []
    
    for doc in results:
        context_parts.append(doc.page_content)
        # Clean up absolute file path formatting for presentation-tier scannability
        source = doc.metadata.get('source', 'unknown')
        clean_source = source.replace("data\\raw_pdfs\\", "").replace("data/raw_pdfs/", "")
        if clean_source not in sources:
            sources.append(clean_source)
            
    return "\n\n".join(context_parts), sources


def build_prompt(context: str, question: str) -> str:
    """Enforces rigid guardrails to eliminate hallucination vectors during synthesis."""
    return (
        "You are an expert analyst on Saudi Vision 2030 policy documents.\n"
        "Answer the question using ONLY the context provided below.\n"
        "If the answer is not in the context, say exactly: "
        "'I cannot find this information in the provided documents.'\n"
        "Do not make up information that is not inside the context data.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION:\n{question}\n\n"
        "ANSWER:"
    )


def generate_answer(prompt: str) -> str:
    """Dispatches structural contextual prompts to the LLM core generation loop."""
    # Route the client connection through host runtime parameters dynamically
    host = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
    if "0.0.0.0" in host:
        host = "http://127.0.0.1:11434"

    try:
        response = ollama.chat(
            model='llama3.2:1b',
            messages=[{'role': 'user', 'content': prompt}],
            options={
                'num_ctx': 2048,
                'temperature': 0.2,
                'num_predict': 512,
            }
        )
        return response['message']['content'].strip()
    except Exception as e:
        return f"[ERROR] Generation loop blocked by backend failure: {str(e)}"


def ask(store: QdrantVectorStore, question: str) -> str:
    """Execution wrapper displaying analytics pipelines on-screen."""
    print(f"\nQuestion: {question}")
    print("-" * 70)
    
    context, sources = retrieve_context(store, question, k=3)
    
    if not context.strip():
        print("Answer:\nSystem degradation: Zero matching context arrays returned.")
        print("-" * 70)
        return ""
        
    prompt = build_prompt(context, question)
    answer = generate_answer(prompt)
    
    print(f"Answer:\n{answer}")
    print(f"\nSources used:")
    for s in sources:
        print(f"  - {s}")
    print("-" * 70)
    return answer


if __name__ == "__main__":
    print("Initializing Cloud-Ready RAG Pipeline Framework...")
    vector_store = get_vector_store()
    print("Pipeline compilation successful. Running active diagnostics...\n")

    test_questions = [
        "What are the main economic goals of Saudi Vision 2030?",
        "How does Vision 2030 plan to reduce dependence on oil?",
        "What role does the private sector play in Vision 2030?"
    ]

    for q in test_questions:
        ask(vector_store, q)