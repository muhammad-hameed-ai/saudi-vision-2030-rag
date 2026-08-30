# Operations & Runtime Limits

What this deployment can and cannot do on its current free-tier footprint, and what
to check when something looks wrong. Figures below were measured against production
on 2026-08-29, not estimated.

## Footprint

| Layer | Plan | Hard limits |
| --- | --- | --- |
| Render web service | Free | 0.1 CPU, 512 MB RAM, **no persistent disk**, spins down after ~15 min idle |
| Qdrant Cloud | Free | 1 GB cluster; **suspends after ~7 days with no traffic** |
| Groq API | Free | Per-minute request and token limits; models can be deprecated without notice |

## What resets, and when

Render free instances have **no persistent disk**. Every spin-down, redeploy, or
restart wipes the container filesystem. Three stores live there:

| Store | Path | Consequence when it resets |
| --- | --- | --- |
| Conversation memory | `data/sessions.db` | Multi-turn context is lost; follow-up questions stop resolving their referent |
| User feedback | `data/feedback/feedback_log.jsonl` | Thumbs up/down submissions are lost |
| Request + RAG audit logs | `logs/*.jsonl` | Log history is lost |

Nothing here is a bug to fix in code — it is the plan. Conversation memory works
correctly **within a warm container** and is expected to reset. Chat history shown in
the sidebar is stored separately in the browser's `localStorage`, so what the *user*
sees survives; only the server-side context the model receives is lost.

Making any of this durable requires either a paid Render instance with a disk, or
moving the store off-box (Postgres, Redis, or a Qdrant collection).

## Measured performance (production, un-throttled)

| Scenario | Retrieval | First token | Total |
| --- | --- | --- | --- |
| Short query (HyDE runs) | 3,192 ms | 3,724 ms | 6,170 ms |
| Long query (HyDE skipped) | 2,988 ms | 3,407 ms | 4,105 ms |
| Deep retrieval, k=10 | 3,769 ms | 4,338 ms | 6,383 ms |
| Cache hit | — | — | **222 ms** |

Retrieval is ~3 s in production against ~235 ms locally. The whole difference is
FastEmbed ONNX inference on 0.1 CPU; only a paid instance changes it.

**Cold start is ~52 s.** The frontend therefore allows 120 s for the first request of
a page session and 45 s afterwards. A first query that appears to hang is the
container waking, not a crash.

**Rapid consecutive queries degrade** (measured 5 s climbing to 28 s over six
back-to-back requests). That is Groq's free-tier rate limiting upstream, not this
service. It recovers on its own.

## Known upstream failure modes

These have each occurred at least once:

1. **Groq deprecates the model.** `llama-3.1-8b-instant` was withdrawn and every
   request returned 404. Fix: change `GROQ_MODEL` in `src/api.py`. Currently
   `allam-2-7b`, which has a **4096-token context window** — the prompt budget in
   `_assemble_context` is sized against that number and must be revisited if the
   model changes.
2. **Qdrant suspends the cluster** after ~7 days of inactivity, and every query
   returns 503. Fix: resume it from the Qdrant Cloud dashboard.
3. **A dependency or Python release breaks the build.** Mitigated: `requirements.txt`
   pins exact versions and `.python-version` pins Python to 3.14. Do not unpin
   without redeploying deliberately and watching the build.

## Required environment variables

Set on Render under Environment (not Secret Files):

| Variable | Required | Notes |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Server refuses to boot without it |
| `QDRANT_CLOUD_URL` | yes | `QDRANT_URL` also accepted |
| `QDRANT_CLOUD_API_KEY` | yes | `QDRANT_API_KEY` also accepted |
| `ADMIN_PASSPHRASE` | **yes in practice** | Without it, document upload and deletion are disabled — the gate falls back to an unguessable random token and logs a warning at startup |
| `USE_HYDE` | no | Defaults to `true`; only applies to queries of 8 words or fewer |

## Rebuilding the vector index

The one operation with no undo. `data/raw_pdfs/` is DVC-tracked and **not** in git, so
the corpus PDFs must be present locally first.

```bash
dvc repro                                  # PDFs -> chunks (data/processed_data/)
python -m src.create_embeddings --dry-run  # confirm target and chunk count
python -m src.create_embeddings            # idempotent upsert, keeps existing points
```

`--recreate` drops the collection and rebuilds from scratch. It prompts for the
collection name before deleting anything. Use it only when the corpus must be rebuilt
wholesale — for example after changing `chunk_size`, or if the cluster is lost.

Embeddings come from FastEmbed with the window pinned to 256 tokens, matching the
query path and the existing corpus. Do not index with the default 128-token window --
those vectors disagree with everything already stored.

## Embedding window

`all-MiniLM-L6-v2` was trained with a 256-token window and the corpus was indexed at
that width. The mean chunk is 197 tokens, so chunks fit and nothing is being lost.

FastEmbed, however, defaults to a **128-token** window. At that width it does not
reproduce the stored vectors (mean cosine 0.949, worst 0.876) and it truncates long
questions before they are embedded. `HybridRetriever.EMBED_MAX_TOKENS = 256` pins it
to the corpus width, where FastEmbed reproduces stored vectors exactly (cosine 1.0000)
and short query vectors are bit-identical — so no re-index was required.

If the embedding model is ever changed, check its trained window and set
`EMBED_MAX_TOKENS` to match before indexing anything.

## Health checks

Run the diagnostic first -- it covers every failure mode listed above in one pass:

```bash
python -m scripts.doctor --live
```

It verifies the configured Groq model still exists, the Qdrant cluster is awake, both
payload indexes are still `keyword`, the embedding window matches the corpus, and the
deployed write endpoints are gated.

Individual endpoints:

```bash
curl https://saudi-vision-2030-rag-3.onrender.com/health          # liveness + uptime
curl https://saudi-vision-2030-rag-3.onrender.com/api/pipeline-info  # corpus + config
```

`pipeline-info` reports `available: false` with null counts when Qdrant is
unreachable — it does not substitute placeholder numbers. A rising `uptime_since`
between calls means the container restarted.
