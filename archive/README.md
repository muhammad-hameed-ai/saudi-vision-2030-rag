# Archive — superseded scripts

Nothing in this directory is part of the running system. None of it is imported by
`src/`, referenced by `dvc.yaml`, invoked by CI, or mentioned in the documentation.
It is kept for provenance rather than reuse, and moved out of the project root so the
root reflects what actually runs.

Verified unreferenced before archiving. Full history is preserved — these were moved
with `git mv`, so `git log --follow archive/<file>` still works.

## What is here, and what replaced it

| Files | What they were | Superseded by |
| :--- | :--- | :--- |
| `patch_api.py`, `patch_api_memory.py`, `patch_eval.py`, `fix_ask_route.py`, `update_pipeline.py` | One-off scripts that rewrote `src/api.py` by regex during earlier development | Direct edits to `src/api.py` |
| `write_k8s.py`, `write_k8s_v2.py`, `write_k8s_manifests.py`, `write_hpa.py`, `write_dvc.py`, `write_files.py` | Generators that emitted the `k8s/` manifests and DVC config | The generated files themselves, committed under `k8s/` and `dvc.yaml`. The Kubernetes manifests were never deployed — the service runs on Render |
| `write_monitoring.py`, `write_monitoring_v2.py`, `monitor_drift.py` | Monitoring scaffolding from the MLflow/Evidently phase | `src/monitor_drift.py` |
| `test_llm.py`, `test_ollama.py`, `test_query.py`, `test_rag.py`, `test_api_memory.py` | Ad-hoc probes from the Ollama era, before the move to Groq | `scripts/doctor.py` for diagnostics, `src/evaluate_rag.py` for quality |
| `run_eval_comparison.py` | Compared two evaluation runs by hand | `data/evaluation/scores_history.json`, appended by the CI quality gate |
| `raw_response.txt` | A captured API response used while debugging | — |

## If you need one of these again

Prefer the replacement in the right-hand column. The Ollama-era probes in particular
will not run: the project no longer uses Ollama, and `ollama`, `langchain_huggingface`
and `langchain_qdrant` are deliberately absent from `requirements.txt`.
