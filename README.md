# Saudi Vision 2030 RAG System

A production-grade Retrieval Augmented Generation (RAG) 
system built on 48 official Saudi Vision 2030 policy 
documents totalling 2,184 pages.

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
## Author
Muhammad Hameed
github.com/muhammad-hameed-ai