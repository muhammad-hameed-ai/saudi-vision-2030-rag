# 🇸🇦 Vision 2030: Enterprise Policy Intelligence Pipeline

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Qdrant](https://img.shields.io/badge/Qdrant-Cloud-FF5252?style=for-the-badge&logo=qdrant)
![Groq](https://img.shields.io/badge/Llama_3.2-Groq-F56600?style=for-the-badge)
![Vanilla JS](https://img.shields.io/badge/Vanilla_JS-F7DF1E?style=for-the-badge&logo=javascript&logoColor=black)

An autonomous, memory-safe Retrieval-Augmented Generation (RAG) microservice designed to ingest, index, and analyze complex policy documents. Engineered with a strict zero-disk footprint for ephemeral cloud environments, featuring real-time MLOps telemetry, deferred execution lifecycle management, and a high-performance ASGI backend.

## 🌐 Live Demo
**https://saudi-vision-2030-rag-3.onrender.com**

> **Note:** No setup required. Opens in any browser. The application is hosted on Render's free tier and may take ~30 seconds to wake up from a cold start on the first visit. Ask anything about Saudi Vision 2030 policy.

---

## 🏛️ System Architecture

This pipeline is built on a highly optimized, asynchronous architecture designed to bypass the traditional I/O bottlenecks of ephemeral cloud deployments.

### 1. Zero-Disk In-Memory Ingestion Engine
Traditional RAG pipelines rely on physical disk storage to parse PDFs, which fails on ephemeral servers that wipe data upon restart. This system utilizes a **Zero-Disk RAM Pipeline**:
* `PyMuPDF` (`fitz`) intercepts the `io.BytesIO` multipart stream directly in memory.
* Text is chunked and vectorized via `FastEmbed` (`sentence-transformers/all-MiniLM-L6-v2`) entirely within RAM.
* Real-time progress is streamed back to the client via **Server-Sent Events (SSE)**.

### 2. $O(1)$ MLOps Telemetry & Vector Storage
* **Database:** Qdrant Cloud (Hybrid Dense + Sparse).
* **Telemetry Sync:** Utilizing Qdrant's `KEYWORD` Facet API to achieve $O(1)$ time complexity for real-time document counting and chunk aggregation, bypassing full-table scans.
* **Distance Metric:** Cosine Similarity + Reciprocal Rank Fusion (RRF).

### 3. Asynchronous Inference Backbone
* **LLM:** `Llama-3.1-8b-instant` served via the Groq API for ultra-low latency generation.
* **Context Anchoring:** The backend dynamically generates precise citation tooltips mapping the LLM's response to the exact source document and page number.

---

## 🛠️ Technology Matrix

| Layer | Technologies Used | Purpose |
| :--- | :--- | :--- |
| **Frontend UI** | HTML5, CSS3, Vanilla JS, Chart.js | DOM manipulation, Deferred Execution (Undo Queue), Analytics |
| **Backend API** | Python, FastAPI, Uvicorn | ASGI routing, Multipart stream parsing, SSE generation |
| **AI / ML Ops** | Groq API, FastEmbed, PyMuPDF | Inference, Vector embedding, Structure-aware chunking |
| **Data Eng.** | Qdrant Cloud | Permanent vector storage, exact-match keyword filtering |

---

## 🚀 Key Engineering Features

- **Document Control Directory (CRUD):** A complete lifecycle management console allowing users to inspect active vectors and permanently purge documents.
- **Gmail-Style Deferred Execution:** Deletions trigger an optimistic UI update and a 10-second client-side "Undo Queue", preventing expensive accidental database re-indexing with zero network overhead.
- **Persistent Local Sessions:** Chat history and telemetry arrays are managed via a robust JavaScript `SessionManager` class writing to `localStorage`, entirely bypassing backend state management.

---

## 💻 Local Development Setup

To run this pipeline locally for development or architectural testing:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/muhammad-hameed-ai/saudi-vision-2030-rag.git
   cd saudi-vision-2030-rag
   ```

2. **Establish the virtual environment:**
   ```bash
   python -m venv venv
   source venv/Scripts/activate  # Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Configure Environment Variables:**
   Create a `.env` file in the root directory:
   ```env
   GROQ_API_KEY=your_groq_key
   QDRANT_URL=your_qdrant_cluster_url
   QDRANT_API_KEY=your_qdrant_api_key
   ```

5. **Initialize the ASGI Server:**
   ```bash
   uvicorn src.api:app --reload
   ```
