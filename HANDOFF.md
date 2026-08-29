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

## TASK 1 (PRIMARY) — Make the evaluation measure the real pipeline

**File:** `src/evaluate_rag.py`

**The problem.** This script does not evaluate the deployed system. It currently:

- connects to `url="http://localhost:6333"` — a local Qdrant, not the cloud cluster
- embeds with `HuggingFaceEmbeddings`, not FastEmbed
- calls `store.similarity_search()` — dense only, with **no sparse retrieval, no RRF
  fusion, no policy booster, and no cosine rescoring**
- generates answers with **Ollama `llama3.2:1b`**, not Groq `allam-2-7b`
- judges those answers with the same 1B model

The committed scores in `data/evaluation/evaluation_scores.json` (faithfulness 0.52,
answer_relevancy 0.42, context_precision 0.34) therefore describe a completely
different system. They have never measured production. The CI gate
(`src/ci_quality_gate.py`) reads these numbers and now warns that they are stale.

**What to build.** Rewrite the script so it exercises the production path:

```python
from src.retriever import HybridRetriever   # same retriever the API uses
from groq import AsyncGroq                  # same LLM as production
```

Requirements:

- Retrieve with `HybridRetriever.retrieve()` so sparse, RRF, the booster and cosine
  rescoring are all exercised.
- Resolve Qdrant from `QDRANT_CLOUD_URL` / `QDRANT_CLOUD_API_KEY` (see
  `src.create_embeddings.resolve_target` for the pattern to copy).
- Generate answers with `allam-2-7b` through Groq, reusing the real
  `SYSTEM_PROMPT_TEMPLATE` and `_assemble_context` from `src/api.py` so the prompt
  matches production exactly.
- Use a **stronger judge than the generator.** A 1B model scoring its own output is
  not a meaningful measurement. Use a larger Groq model for scoring, and state in the
  output which model judged.
- Keep the output schema identical so `ci_quality_gate.py` keeps working:
  `{"faithfulness": float, "answer_relevancy": float, "context_precision": float,
  "per_question": [...]}` written to `data/evaluation/evaluation_scores.json`.
- Record the evaluated commit SHA and the judge model in the output.
- Remove the `ollama`, `langchain_huggingface` and `langchain_qdrant` imports. They
  are not in `requirements.txt` and only work on this machine by accident.

**Run it:**

```bash
cd c:\Users\hameed\saudi-vision-2030-rag
python -m src.evaluate_rag
```

**Then:**

```bash
python -m src.ci_quality_gate     # must pass; staleness warning should disappear
```

**Expect the numbers to change.** They may go up or down. Report the honest result —
do not tune thresholds to make them pass. If they fall below the floors in
`ci_quality_gate.py` (0.40 / 0.30 / 0.20), say so and explain why rather than
adjusting the floors.

**Verification to paste back:** the before/after scores, the judge model used, and
the `ci_quality_gate` output.

---

## TASK 2 (SECONDARY, OPTIONAL) — Fix embedding truncation

**Files:** `params.yaml`, then a full re-index.

**The problem.** `chunk_size: 1000` characters is roughly 250 tokens, against
`all-MiniLM-L6-v2`'s **256-token ceiling**. The longest chunks are silently truncated
during embedding, so their final portion is not represented in the vector at all.

Measured against the live corpus by re-embedding stored chunks and comparing:

| Chunk length | Self-similarity |
| :--- | ---: |
| ≤ 100 tokens | 1.0000 |
| ~210 tokens | 0.96–0.98 |
| ~240–248 tokens | 0.88–0.96 |

**Only attempt this if Task 1 is complete**, so the change can be measured.

**Procedure:**

1. Edit `params.yaml`: `chunk_size: 1000` → `700`, keep `chunk_overlap: 200`.
2. Rebuild chunks: `dvc repro` (requires `data/raw_pdfs/` present locally —
   `dvc pull` first if missing).
3. Dry run: `python -m src.create_embeddings --dry-run` — confirm the target is the
   **cloud** URL and note the new chunk count.
4. Full rebuild: `python -m src.create_embeddings --recreate` (it will prompt for the
   collection name).
5. Verify the collection: point count, `status: green`, and both payload indexes
   present as **keyword**.
6. Re-run Task 1's evaluation and compare.
7. Update the corpus figures in `README.md` and `OPERATIONS.md`.

**Risk:** this destroys and rebuilds the live collection. The service returns 503
while it runs. Do not start it without confirming `data/raw_pdfs/` is present.

---

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
