# Saudi Vision 2030 RAG Pipeline
![CI Status](https://github.com/muhammad-hameed-ai/saudi-vision-2030-rag/actions/workflows/pipeline.yml/badge.svg)

## Description
A localized Retrieval-Augmented Generation (RAG) system designed to query structural timelines, metrics, and policies related to the Saudi Vision 2030 initiative. The system operates entirely on local hardware, processing user queries to fetch precise, policy-specific data without relying on external cloud LLM APIs.

## Architecture
```text
[Vision 2030 PDFs] 
       │
   (Ingest)
       │
   [Chunks] ──(Embed)──> [(Qdrant Vector Store)]
                                │
   [User Query] ──(HyDE)──> (Retrieve)
                                │
                            [Rerank]
                                │
                             (LLM)
                                │
                           [Answer]

Metric,Baseline,After HyDE,Change
Faithfulness,0.10,0.52,+0.42
Answer Relevancy,0.40,0.42,+0.02
Context Precision,0.35,0.34,-0.01

Experiment,Chunk Size,Overlap,Faithfulness,Relevancy,Precision
exp-baseline,500,50,0.10,0.40,0.35
exp-chunk-A,500,100,0.12,0.41,0.35
exp-chunk-B,1000,100,0.35,0.41,0.34
exp-chunk-C,1500,200,0.45,0.38,0.30
exp-final,1000,200,0.52,0.42,0.34

Technical Stack
Python 3.10

FastAPI (Backend API)

Qdrant (Local Vector Database)

Ollama (Local LLM Inference)

Llama 3.2:1b (Generative Model)

sentence-transformers/all-MiniLM-L6-v2 (Dense Embedding)

cross-encoder/ms-marco-MiniLM-L-6-v2 (Reranker)

DVC (Data Version Control)

GitHub Actions (CI/CD Quality Gates)

How to Reproduce
git clone [https://github.com/muhammad-hameed-ai/saudi-vision-2030-rag.git](https://github.com/muhammad-hameed-ai/saudi-vision-2030-rag.git)
cd saudi-vision-2030-rag
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --host 127.0.0.1 --port 8000

Key Engineering Decisions
Why HyDE (Hypothetical Document Embeddings): The baseline relevancy was struggling because user queries were short (e.g., "What year is its net zero goal?"). Generating a hypothetical answer first drastically improved vector space matching, jumping faithfulness from 0.10 to 0.52.

Why 1000/200 Chunking: Smaller chunks (500) fractured the policy context too much, while larger chunks (1500) introduced noise and lowered precision. 1000 tokens with a 200 token overlap provided the best balance for capturing complete policy paragraphs.

Why Llama 3.2:1b: Strict hardware constraints required a model that could fit into limited VRAM while leaving room for the embedding and reranking models. It is fast enough to prevent 504 Gateway Timeout errors during the multi-step HyDE pipeline.

Known Limitations
Small Model Faithfulness: Despite HyDE improvements, a 1B parameter model still occasionally hallucinates specific statistical figures.

Evaluation Bias: The evaluation pipeline uses the same small local model as the "judge" for faithfulness and relevancy, which may inflate or skew the perceived accuracy.

Deployment Readiness: The system is heavily optimized for local Windows environments. Containerization and Kubernetes orchestration have not been tested or verified for production scale.

Author
Muhammad Hameed

GitHub: github.com/muhammad-hameed-ai