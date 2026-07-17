# Saudi Vision 2030 RAG System

A production-grade Retrieval Augmented Generation (RAG) 
system built on 48 official Saudi Vision 2030 policy 
documents totalling 2,184 pages.

# 🚀 Release V2.1: Production-Grade RAG Architecture & UI Overhaul

This major release transforms the local pipeline into a fully asynchronous, enterprise-grade Retrieval-Augmented Generation (RAG) system. It features a complete rewrite of the frontend interaction engine and introduces mathematical normalizations and strict state management to the backend.

## 🧠 Backend & MLOps Upgrades (`src/api.py`)
* **Asynchronous Threading:** Refactored FastAPI endpoints to use `async def` and `asyncio.to_thread()`, preventing CPU-heavy local Llama generation from blocking the main server thread.
* **Hybrid Search & Reranking:** Successfully implemented dual-vector retrieval (Dense + BM25 Sparse) with Reciprocal Rank Fusion (RRF) and a Cross-Encoder reranker (`ms-marco-MiniLM-L-6-v2`) for pinpoint context accuracy.
* **Sigmoid Math Normalization:** Fixed logit leakage from the Cross-Encoder. Applied a Sigmoid function (`1 / (1 + math.exp(-score))`) to normalize raw mathematical logits into clean 0.0 - 1.0 probability percentages.
* **Strict Role Segregation:** Eliminated prompt bleeding and LLM hallucinations by isolating System instructions from User queries within the Ollama message array payload.
* **Dynamic Schema Validation:** Added indestructible Pydantic `@model_validator` handlers to catch and normalize UI payload discrepancies automatically (fixing 422 HTTP errors).

## 🖥️ Frontend & UX Architecture (`index.html`)
* **Client-Side Session State Machine:** Implemented a unified JavaScript array matrix to cache, swap, and manage multiple chat histories dynamically without refreshing the page.
* **Auto-Topic Extraction:** Sessions automatically rename themselves based on the user's first generated question rather than relying on generic timestamps.
* **Flexbox Holy Grail Sidebar:** Refactored layout constraints to ensure static top/bottom panels with a dynamic, scrolling middle container. Integrated real-time search filtering for the session list.
* **Zero-Freeze Editing & Abort Controllers:** Integrated native `AbortController` APIs. Users can now click the Edit (Pencil) icon to instantly terminate an active LLM generation, clear the DOM, and recycle the prompt back into the textarea without duplicating message bubbles.
* **Chart.js Telemetry Engine:** Replaced static text metrics with live, high-fidelity visualizers, including a Server Health Pulse, Latency History bar charts, and System Accuracy donut gauges.
* **Markdown Rendering:** Integrated `marked.js` and `DOMPurify` for secure, beautiful rich-text rendering of LLM outputs (bolding, lists, code blocks).

---
## Live API
https://uninstall-blast-halved.ngrok-free.dev/docs

## What it does
Ask any question about Saudi Vision 2030 goals, programs,
and strategies. The system retrieves the most relevant
document chunks and generates a grounded answer.

## Architecture
PDFs -> Chunking -> Embeddings -> Qdrant -> RAG Chain -> FastAPI

## Technical Stack
- 48 PDFs, 2,184 pages, 5,852 chunks
- Chunking: Recursive character splitting (1000/200 overlap)
- Embeddings: sentence-transformers/all-MiniLM-L6-v2 (384-dim)
- Vector DB: Qdrant (cosine similarity)
- LLM: llama3.2:1b via Ollama (local, zero API cost)
- API: FastAPI with auto-generated Swagger docs
- Versioning: Git + DVC for code and data

## API Endpoints
- GET  /health   - System health check
- GET  /info     - Project metadata and corpus stats  
- POST /ask      - Submit a question, get a RAG answer
- POST /feedback - Submit rating on answer quality

## Evaluation Baseline
- Faithfulness:       0.10
- Answer Relevancy:   0.40
- Context Precision:  0.35
(Baseline using llama3.2:1b as judge - improvements ongoing)

## 🏗️ Architecture Diagram
[React Dashboard] ---> [FastAPI / ngrok] ---> [LangChain RAG Chain]
                                                    |
                                                    v
[MLflow & Evidently] <--- [Metrics] <--- [Ollama (llama3.2:1b)] <---> [Qdrant Vector DB]

## 📊 DVC Experiment Results
We compared 5 different chunking strategies using DVC to find the optimal retrieval balance.

| Experiment | Chunk Size | Overlap | Total Chunks |
|------------|------------|---------|--------------|
| Exp 1      | 500        | 100     | 10,105       |
| Exp 2      | 800        | 150     | 7,083        |
| **Exp 3 (Winner)** | **1000** | **200** | **5,852** |
| Exp 4      | 1200       | 250     | 5,049        |
| Exp 5      | 1500       | 300     | 4,208        |

## 📈 Custom Evaluation Scores
* **Faithfulness:** 0.10
* **Answer Relevancy:** 0.40
* **Context Precision:** 0.35

## 🚢 Kubernetes Deployment
The API is designed to scale automatically using a Horizontal Pod Autoscaler (HPA).
```bash
# Start cluster
minikube start --driver=docker

# Apply all configurations
kubectl apply -f k8s/

# Port forward to local machine
kubectl port-forward service/rag-api-service 8080:80

## Kubernetes Deployment (Verified)

This project features a fully tested local Kubernetes deployment using Minikube, demonstrating horizontal scaling, zero-downtime rollouts, and autoscaling.

### 1. Start Cluster & Apply Infrastructure
```bash
minikube start --driver=docker --memory=2048 --cpus=2
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/service.yaml
kubectl apply -f k8s/deployment.yaml
## Author
Muhammad Hameed
github.com/muhammad-hameed-ai