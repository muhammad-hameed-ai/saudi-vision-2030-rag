# 🇸🇦 Saudi Vision 2030: Policy Intelligence Pipeline (RAG)

[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)](https://fastapi.tiangolo.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-EF4B4B?style=for-the-badge&logo=qdrant)](https://qdrant.tech/)
[![Groq](https://img.shields.io/badge/Groq-Llama_3.2-F55036?style=for-the-badge)](https://groq.com/)
[![Render](https://img.shields.io/badge/Render-Deployed-46E3B7?style=for-the-badge)](https://render.com/)

A production-grade Retrieval-Augmented Generation service for Saudi policy documents, engineered to run inside a 512 MB / 0.1 CPU envelope. Hybrid dense + sparse retrieval, true cosine relevance scoring, context-window budgeting, and streaming inference — with every reported metric measured rather than asserted.

**Live Application:** [saudi-vision-2030-rag-3.onrender.com](https://saudi-vision-2030-rag-3.onrender.com)

---

## 🛑 The Engineering Problem

Running a hybrid RAG pipeline on Render's free tier means fitting embedding inference, vector search, conversation memory, and LLM streaming into 512 MB of RAM and 0.1 CPU, on a container that sleeps after 15 minutes and has no persistent disk.

Most of the interesting decisions in this codebase exist because of that ceiling. The sections below document what broke, why, and what the fix measured.

### Memory: Bypassing the Cross-Encoder
Loading FastEmbed's dense and sparse models alongside an ONNX cross-encoder reranker peaked near 700 MB and the OS killed the process mid-request. The reranker is now bypassed and retrieval candidates pass directly to the LLM, holding steady usage near 300 MB. The module is retained for a future paid tier; `/api/pipeline-info` reports it as disabled rather than claiming it is active.

### Relevance: RRF Scores are not Similarity
Qdrant's `FusionQuery` returns a Reciprocal Rank Fusion score — `sum(1/(60+rank))` — which depends only on rank position. Position one scored `0.6225` whether the passage was perfect or irrelevant, so every displayed confidence figure was a rank artefact.

Dense vectors are now requested on the same round trip (`with_vectors`) and true cosine similarity is computed locally, at no additional query cost. Ordering is a separate question from scoring, and the two use different signals deliberately. Results stay in RRF fusion order; an earlier revision re-sorted them by cosine, which measured worse at every depth because RRF fuses the dense and sparse rankings while cosine sees only the dense one:

| Ordering | P@1 | P@3 | P@5 |
|:---|:---:|:---:|:---:|
| **RRF Fusion (current)** | **0.320** | **0.367** | **0.370** |
| Cosine | 0.260 | 0.293 | 0.330 |

Cosine remains the displayed confidence figure because it is a real similarity measure. The RRF score is a rank artefact and would be meaningless there.

| Query Context | Before (RRF) | After (Cosine) |
|:---|:---:|:---:|
| "PIF's role" *(highly relevant)* | ~60% | **86.3%** |
| "NEOM renewables" *(relevant)* | ~60% | **67.6%** |
| "capital of France" *(off-topic)* | ~60% | **50.7%** |

### Context: Budgeting a 4096-Token Window
`allam-2-7b` exposes a 4096-token window shared between prompt and completion. At `k=10` the retrieved context alone reached ~2,013 tokens while `max_tokens` reserved another 2,048 — 4,061 of 4,096, leaving nothing for the system prompt or conversation memory.
`_assemble_context()` now packs chunks against a computed budget and memory is bounded.

| Metric at k=10 | Before | After |
|:---|:---:|:---:|
| **Time to first token (TTFT)** | 34,805 ms | **1,119 ms** |
| **Total response time** | 35,957 ms | **2,247 ms** |

### Retrieval: A Keyword Index Cannot Substring-Match
The policy booster over-weights the core Vision 2030 documents to counteract term-frequency dominance from long financial circulars — two bond and sukuk prospectuses alone are 34% of the corpus. `metadata.source` must carry a keyword index because the document registry facets on it and deletion filters it by exact value. 

Qdrant permits one index per field, so `MatchText` returned 400 on every query, and a later `MatchAny(["vision2030"])` silently matched zero documents. The working implementation resolves the actual filenames once via `facet()`, caches them, and matches those exactly. 

### Cold Starts: An Adaptive Stream Timeout
Free-tier containers sleep after 15 minutes. Measured cold start is ~52 s, so the original flat 45 s frontend timeout guaranteed the first query after sleep always failed — and reported it as a backend crash. The first request of a page session now gets 120 s; subsequent requests stay at 45 s.

---

## 🏗️ System Architecture

### Retrieval Pipeline
1. **Query Normalisation:** Regex typo correction and domain keyword expansion.
2. **Conditional HyDE:** Queries of 8 words or fewer are expanded into a hypothetical answer before embedding. Longer queries carry enough signal on their own and skip the round trip, saving ~2 s.
3. **Hybrid Search:** A 384-dim dense vector (`all-MiniLM-L6-v2`) and a BM25 sparse vector drive three parallel prefetches, fused by Reciprocal Rank Fusion.
4. **Cosine Scoring:** Dense vectors returned alongside results are scored locally for display. 
5. **Context Budgeting:** The highest-scoring chunks are packed into the tokens remaining after the system prompt, conversation memory, and the reserved response allowance.
6. **Streaming Synthesis:** Tokens relay to the browser over SSE as Groq produces them.

### Measured Performance
*Production, with requests spaced to avoid upstream rate limiting.*

| Scenario | Retrieval | First token (TTFT) | Total |
|:---|:---:|:---:|:---:|
| Short query (HyDE runs) | 3,192 ms | 3,724 ms | 6,170 ms |
| Long query (HyDE skipped) | 2,988 ms | 3,407 ms | 4,105 ms |
| Deep retrieval, k=10 | 3,769 ms | 4,338 ms | 6,383 ms |
| Cache hit | — | — | 222 ms |

*Note: The same retrieval takes 235 ms locally (184 ms Qdrant round trip + 37 ms dense embedding). The entire difference in production is ONNX inference bound by the 0.1 CPU envelope.*

### Retrieval Quality
*Measured against the deployed pipeline (generation by `allam-2-7b`, judged by `openai/gpt-oss-120b`).*

The most significant achievement of this RAG architecture is a **+420% improvement in Faithfulness** (from a baseline of 0.10 up to 0.52) by utilizing Hybrid Retrieval, conditional HyDE query expansion, and careful context budgeting.

| Metric | Value | What it measures |
|:---|:---:|:---|
| **Faithfulness** | `0.520` | Are the answer's claims supported by the retrieved context? |
| **Answer Relevancy** | `0.740` | Does the answer address the question? |
| **Context Precision** | `0.368` | What share of retrieved chunks are useful? (set metric) |
| **Precision@1** | `0.310` | How good is the top-ranked chunk? (rank metric) |

---

## 📊 Corpus Composition
**49 policy PDFs, 5,941 chunks, 384 dimensions.** Chunked at 1,000 characters with 200-character overlap, with section headers detected structurally and carried into metadata.

| Document | Chunks | Share |
|:---|---:|---:|
| International sukuk offering circular 2025 | 1,112 | 17.7% |
| International bond offering circular 2025 | 996 | 15.9% |
| Vision 2030 annual report 2025 | 390 | 6.2% |
| PIF annual report 2024 | 314 | 5.0% |
| National strategy for data & AI | 277 | 4.4% |
| *44 further documents* | *2,852* | *50.8%* |

---

## 🛡️ Security Architecture

| Layer | Mechanism | Detail |
|:---|:---|:---|
| **Boot Validation** | Fail-fast check | `sys.exit(1)` if API keys are missing. |
| **Origin Protection** | Strict CORS | Production and local development origins only. |
| **Rate Limiting** | SlowAPI (per IP) | 10/min chat, 5/min ingest, 30/min registry. |
| **Payload Hardening** | Pydantic V2 | Bounded lengths on every free-text schema field. |
| **Write Authorisation** | `secrets.compare_digest` | Destructive routes (`DELETE`, `POST /ingest`) strictly require `ADMIN_PASSWORD`. |
| **Fail-Closed Default** | Random fallback | If `ADMIN_PASSWORD` is unset, write routes are disabled. |
| **Upload Bounds** | 8 MB ceiling | Files are buffered in RAM; prevents memory exhaustion. |
| **XSS Prevention** | DOMPurify | All model output is sanitised before `innerHTML` injection. |

---

## 💻 Tech Stack & Deployment
* **Backend:** FastAPI 0.141.1 on Python 3.14 (pinned via `.python-version`)
* **Vector Database:** Qdrant Cloud (dense + sparse BM25 vectors)
* **Embeddings:** FastEmbed `all-MiniLM-L6-v2` & `Qdrant/bm25` (local, no API calls)
* **LLM:** Groq API, `allam-2-7b` (4096-token window)
* **Document Parsing:** PyMuPDF 1.28.2
* **Frontend:** Vanilla JavaScript, Tailwind CSS, Chart.js, DOMPurify
* **Deployment:** Render (native Python runtime)

*Every direct dependency is pinned to an exact version. Nothing was pinned previously, meaning a breaking library release could take the service down with no code change.*

---

## 🔌 API Reference

### `POST /api/chat`
Streams policy analysis over Server-Sent Events. Rate limit 10/min.
```json
{
  "session_id": "session-1756449021",
  "question": "What are the primary targets for non-oil GDP growth?",
  "k": 5
}
```

### `DELETE /api/documents?filename=<path>`
Purges a document's vectors. Requires `X-Admin-Access-Token`.

### `POST /api/ingest/stream`
Uploads and indexes a PDF entirely in memory, streaming progress. Requires `X-Admin-Access-Token`. Rate limit 5/min, 8 MB maximum.

### `GET /api/documents/auth`
Pre-flight passcode verification for the UI Admin Modal.

---

## 🚀 Local Setup (For Judges / Auditors)

```bash
git clone https://github.com/muhammad-hameed-ai/saudi-vision-2030-rag.git
cd saudi-vision-2030-rag
```

Create a `.env` file in the project root:

```env
GROQ_API_KEY=<your Groq API key>
QDRANT_CLOUD_URL=<your Qdrant cluster URL>
QDRANT_CLOUD_API_KEY=<your Qdrant API key>
ADMIN_PASSWORD=<a long secure passcode>
```

Activate virtual environment and run:

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
uvicorn src.api:app --reload --port 8000
```
The application will be served at `http://localhost:8000`.

---
*Distributed under the MIT License. Developed for the Alibaba Cloud AI Hackathon 2026.*
