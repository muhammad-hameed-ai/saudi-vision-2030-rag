# Handoff Brief — Saudi Vision 2030 Policy Intelligence Hub

Paste everything below the line into the Gemini agent. It contains the current state,
what changed and why, and the two remaining tasks with exact paths and commands.

---

## CONTEXT: You are picking up a live, working project

Repository: `muhammad-hameed-ai/saudi-vision-2030-rag` (public), branch `main`
Live service: https://saudi-vision-2030-rag-3.onrender.com
Current HEAD: `3444896`
Local path: `c:\Users\hameed\saudi-vision-2030-rag`

This is a **portfolio / showcase project**. The objective is a polished, defensively
engineered RAG application that holds up under technical scrutiny while running inside
Render's free tier (512 MB RAM, 0.1 CPU). Every visible metric must be live and
defensible — placeholder numbers were deliberately removed and must not return.

**The system currently works.** Do not refactor for its own sake. Two specific tasks
remain, defined at the end.

---

## CURRENT POSITION

| Aspect | State |
| :--- | :--- |
| Deployment | Live, healthy, auto-deploys from `main` |
| Corpus | 49 documents / 5,941 chunks / 384 dims in Qdrant Cloud |
| LLM | `allam-2-7b` via Groq (**4096-token context window**) |
| Embeddings | FastEmbed `all-MiniLM-L6-v2` + `Qdrant/bm25`, both local |
| Runtime | Python 3.14 pinned, all dependencies pinned exactly |
| Security | Both write endpoints gated; `ADMIN_PASSPHRASE` set in Render |
| Telemetry | Real measurements only; no hardcoded metrics anywhere |
| Tests | 8/8 production endpoint checks passing |

Measured production latency: 4.1–6.4 s per query, 222 ms on cache hit, ~52 s cold start.

---

## WHAT CHANGED RECENTLY (three commits)

### `40720e1` — retrieval scoring, write endpoints, context budget

**Errors found and fixed:**

1. **Relevance scores were rank artefacts.** Qdrant's `FusionQuery` returns an RRF
   score — `sum(1/(60+rank))` — which depends only on rank position. Every query
   returned the same values (0.6225, 0.6225, 0.5826…). A sigmoid meant for
   cross-encoder logits then compressed them further.
   *Fix:* request dense vectors on the same round trip (`with_vectors`), compute true
   cosine locally, re-sort by it. Verified: relevant queries now score 86%, off-topic
   50%.

2. **Context window silently overflowed.** At `k=10` the prompt reached ~2,013 tokens
   while `max_tokens` reserved 2,048 — 4,061 of 4,096, leaving nothing for memory.
   *Fix:* `_assemble_context()` packs to a computed budget. Time to first token at
   `k=10` went from 34,805 ms to 1,119 ms.

3. **The policy booster matched nothing.** `MatchText` needs a text index;
   `metadata.source` carries a keyword index. A later `MatchAny(["vision2030"])`
   stopped the 400 error but matched **zero** documents.
   *Fix:* resolve real filenames via `facet()`, cache, match exactly.

4. **`/api/ingest/stream` had no authentication** while `DELETE` did, despite writing
   to the same collection. Anyone could write to the vector store.

5. **`ADMIN_PASSPHRASE` defaulted to a hardcoded `3331604`** in a public repository.

6. **Conversation memory was dead** — the frontend never sent `session_id`.

7. **Cache crossed conversations** — keyed on query text alone, ignoring `k` and
   session history.

8. **Retrieval depth capped at 6** regardless of the requested `k`.

9. **Fabricated telemetry** — faithfulness 94.2% / relevance 89.5% / hallucinations
   2.3% were a hardcoded array; the server-load chart was `Math.random()`; the
   latency bars were a static list.

10. Minor: SQLite held a connection across a network call; paths were CWD-relative;
    ingest never wrote `section` and reported success on text-free PDFs;
    `requests_served` never incremented; CORS had a `"*"` negating the allowlist.

### `692d00d` — reproducibility

- Pinned Python to 3.14 (`.python-version`) and every dependency exactly. Nothing was
  pinned before, so any redeploy could break with no code change.
- Rewrote `src/create_embeddings.py`: it pointed at `localhost:6333` with no API key
  and called `delete_collection` **unconditionally** — running it against the cloud
  would have destroyed the corpus. Now cloud-aware, `--recreate` required to delete,
  creates payload indexes, `--dry-run` supported, idempotent.
- Raised the CI quality-gate thresholds from 0.05 (unfailable) and added a staleness
  warning.
- Added `OPERATIONS.md`.

### `4ad3555` — documentation and hygiene

- **Rewrote `README.md`.** It advertised Llama 3.2 / `llama-3.1-8b-instant` in the
  badge, diagram and stack list while the service runs `allam-2-7b`. It also
  **published `ADMIN_PASSPHRASE=3331604`** as the documented setup value — that is
  where the passcode leaked from. Endpoint signatures and response shapes were wrong.
- **Deduplicated the corpus.** `neom-sr-annual-report-en-2023 (1).pdf` and
  `ntp_en_annual_report_2025 (1).pdf` were indexed twice. 6,281 → 5,941 points,
  51 → 49 documents.
- **Untracked 267 MB of generated artifacts** — `qdrant_storage/` (a local Qdrant
  database, 256 MB), `mlartifacts/`, `mlflow.db`, Evidently reports. Files remain on
  disk; only tracking was removed. Tracked tree is now 0.3 MB.

### WHAT WAS REMOVED

| Removed | Reason |
| :--- | :--- |
| `src/rag_pipeline.py` | Dead code, zero importers, imports not in `requirements.txt` |
| `get_reranker()` and its singleton | Reranker is bypassed for memory; calls were dead |
| `_sigmoid()` | Wrong transform for cosine values |
| `SYSTEM_STATS` | Nothing read it |
| Hardcoded `94.2 / 89.5 / 2.3` chart, `Math.random()` pulse, static latency bars | Fabricated metrics |
| `accuracyChart` binding | Orphaned after its canvas was removed |
| CORS `"*"` | Negated the allowlist |
| `ENV QDRANT_URL` in Dockerfile | Landmine if `QDRANT_CLOUD_URL` were unset |
| 340 duplicate chunks | Two documents indexed twice |

---

## ⚠️ GUARDRAILS — read before touching anything

These are non-obvious constraints. Violating them breaks production.

1. **NEVER create a TEXT index on `metadata.source`.** Qdrant allows one index per
   field. `metadata.source` must stay **keyword**, because `get_document_registry()`
   facets on it and `delete_document()` filters it with `MatchValue`. Creating a text
   index replaces the keyword index and instantly breaks the document registry page
   and deletion. This was verified empirically on the live cluster.

2. **Never run `create_embeddings.py --recreate` casually.** It drops the entire
   collection. Always run `--dry-run` first. There is no undo — `data/raw_pdfs/` is
   DVC-tracked and not in git.

3. **Do not unpin `requirements.txt` or `.python-version`.** The pins are the exact
   versions proven working by a successful Render build.

4. **Do not mix embedding implementations.** Production uses FastEmbed. Vectors from
   langchain `HuggingFaceEmbeddings` diverge to cosine ~0.91 on chunks near the
   256-token limit. A collection must be built wholly by one implementation.

5. **Do not reintroduce placeholder metrics.** Any displayed number must come from a
   real measurement or be absent.

6. **The context budget is sized against `allam-2-7b`'s 4096-token window.** If the
   model changes, `MODEL_CONTEXT_TOKENS` in `src/api.py` must change with it.

7. **Verify claims against the live system before asserting them.** Several past
   fixes were reported as complete when they were not — a `MatchAny` filter that
   matched zero documents, and a "metrics removed" claim while two fabricated charts
   remained. Run the check, paste the output.

---

## TASK 1 — COMPLETE

`src/evaluate_rag.py` was rewritten to drive the deployed pipeline: retrieval through
`HybridRetriever` against the cloud cluster, prompts from `SYSTEM_PROMPT_TEMPLATE` and
`_assemble_context` unmodified, generation with `allam-2-7b`, and judging by
`openai/gpt-oss-120b` (120B judging a 7B generator).

Three defects surfaced while getting it working, all fixed:
* `llama-3.3-70b-versatile` does not exist on this account and 404s.
* The judge resolver probed for *a response*, not a *usable* one. `gpt-oss-120b`
  answers a trivial probe but returns empty content on real scoring calls, so the
  resolver would have selected a judge that failed every question.
* A 300-token judge budget truncated the reasoning models mid-deliberation
  (`finish_reason="length"`, empty content). Raised to 1000.

First real measurement, commit `835f70d`:

| metric | value | note |
| :--- | ---: | :--- |
| faithfulness | 0.708 | |
| answer_relevancy | 0.740 | |
| context_precision | 0.368 | set metric: all retrieved chunks |
| precision_at_1 | 0.310 | rank metric: the top chunk |

These replace 0.52 / 0.42 / 0.34, which measured a local Ollama pipeline that was
never deployed.

**Ranking correction.** `retrieve()` had been re-sorting fused results by cosine
similarity. Measured with chunks and judgements held fixed and only order varied, RRF
ordering wins at every depth (p@1 0.320 vs 0.260, p@5 0.370 vs 0.330) because RRF
fuses dense and sparse while cosine sees only dense. RRF order is restored; the cosine
score is retained for display only. Do not re-sort by score.

**What did not help.** An ablation over the booster legs and query expansion moved
precision less than judge noise (0.367-0.409 across four configurations). Only 1-3
chunks in 8 are judged useful per question, which is corpus coverage rather than a
retrieval defect. The lever for further quality is more Vision 2030 policy content,
not code.

## TASK 2 — RESOLVED, DO NOT ATTEMPT

An earlier version of this brief asked for `chunk_size` to be reduced from 1000 to 700
followed by a full re-index, on the grounds that chunks were being truncated at the
model's 256-token ceiling. **That diagnosis was wrong and the task has been withdrawn.**

What was actually true, measured against the live cluster:

* The corpus was indexed at a 256-token window and the mean chunk is 197 tokens, so
  nothing was being truncated. The corpus is intact.
* FastEmbed defaults to a **128-token** window. That is what produced the apparent
  "truncation" — the measuring tool was narrower than the corpus, not the reverse.
* At 128, FastEmbed did not reproduce the stored vectors: mean cosine 0.949, worst
  0.876. At 256 it reproduces them exactly: 1.0000 mean, 1.0000 minimum.
* Short query vectors are bit-identical at either width, so no re-index was needed.

The fix is `HybridRetriever.EMBED_MAX_TOKENS = 256`, applied in `src/retriever.py` and
reused by `src/create_embeddings.py`. This also corrected two real defects: documents
uploaded through the web console were getting vectors inconsistent with the corpus, and
questions longer than ~630 characters were being cut before embedding even though the
API accepts 1000.

**Do not reduce `chunk_size` and do not re-index.** Doing so would rebuild the corpus
for no benefit and take the service down while it ran.

## WHAT IS NOT LEFT TO DO

Do not redo these — they are complete and verified:

- Security: both write endpoints gated, upload capped at 8 MB, rate limits in place
- Retrieval scoring: real cosine, verified discriminating across queries
- Context budgeting: verified fitting at every `k` with and without memory
- Booster: resolves and matches the three Vision 2030 documents
- Session memory: `session_id` wired end to end, verified in SQLite
- Cache: keyed on `(k, query)`, bypassed when a session has history
- Dependency and Python pinning
- Index rebuild path
- README accuracy, corpus deduplication, repository hygiene

---

## REFERENCE

- `OPERATIONS.md` — free-tier limits, what resets on spin-down, failure modes, runbook
- `README.md` — architecture and API
- Health checks:
  ```bash
  curl https://saudi-vision-2030-rag-3.onrender.com/health
  curl https://saudi-vision-2030-rag-3.onrender.com/api/pipeline-info
  ```
- Render does **not** use the Dockerfile. It uses the native Python runtime, reads
  `.python-version`, and runs `pip install -r requirements.txt`.
- GitHub Actions writes a score-history commit back to `main` after pushes touching
  `src/**`. Expect to `git fetch` and merge before pushing.
