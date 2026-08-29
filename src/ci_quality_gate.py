"""
Quality gate over the committed evaluation baseline.

Scope, stated plainly: this does NOT run the evaluation. It reads the scores committed
in data/evaluation/evaluation_scores.json, appends them to the history ledger, and
fails if they fall below the thresholds below. Re-running the evaluation needs a live
Groq key and a reachable Qdrant cluster, which CI does not have, so `dvc repro` /
`python -m src.evaluate_rag` must be run locally and the refreshed scores committed.

Because of that, a stale scores file will happily pass forever while the retrieval code
underneath it changes. The staleness check warns when that has happened. It warns
rather than fails, so an unrelated commit does not block the pipeline -- but a warning
here means the reported numbers no longer describe the code that is running.
"""

import json
import os
import subprocess
import sys
from datetime import datetime, timezone

SCORES_FILE = os.path.join("data", "evaluation", "evaluation_scores.json")
HISTORY_FILE = os.path.join("data", "evaluation", "scores_history.json")

# Set below the measured baseline (faithfulness 0.52 / relevancy 0.42 / precision 0.34
# as of 2026-08-29) so a genuine collapse fails the build while normal variance does
# not. The previous threshold of 0.05 could not fail for any output a working pipeline
# would produce.
MIN_FAITHFULNESS = 0.40
MIN_ANSWER_RELEVANCY = 0.30
MIN_CONTEXT_PRECISION = 0.20

# Changes here invalidate a previously measured score.
PIPELINE_PATHS = (
    "src/retriever.py",
    "src/api.py",
    "src/hyde_retriever.py",
    "src/create_embeddings.py",
    "params.yaml",
)


def _git(*args):
    try:
        out = subprocess.run(["git", *args], capture_output=True, text=True, timeout=30)
        return out.stdout.strip() if out.returncode == 0 else ""
    except Exception:
        return ""


def check_staleness():
    """Warn when retrieval code moved more recently than the committed scores."""
    scores_commit = _git("log", "-1", "--format=%ct", "--", SCORES_FILE)
    if not scores_commit:
        return
    stale = []
    for path in PIPELINE_PATHS:
        path_commit = _git("log", "-1", "--format=%ct", "--", path)
        if path_commit and int(path_commit) > int(scores_commit):
            stale.append(path)
    if stale:
        print(
            "WARNING: these changed after the evaluation scores were last updated, so "
            "the reported metrics no longer describe the current pipeline:"
        )
        for path in stale:
            print(f"           - {path}")
        print(
            "         Re-run `python -m src.evaluate_rag` locally and commit the "
            "refreshed data/evaluation/evaluation_scores.json."
        )


def main():
    if not os.path.exists(SCORES_FILE):
        print(f"ERROR: {SCORES_FILE} not found.", file=sys.stderr)
        return 1

    with open(SCORES_FILE, encoding="utf-8") as f:
        scores = json.load(f)

    faithfulness = scores.get("faithfulness", 0.0)
    relevancy = scores.get("answer_relevancy", 0.0)
    precision = scores.get("context_precision", 0.0)

    entry = {
        "timestamp": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "commit": os.getenv("GITHUB_SHA", "local-dev")[:7],
        "faithfulness": round(faithfulness, 2),
        "answer_relevancy": round(relevancy, 2),
        "context_precision": round(precision, 2),
        "hyde_enabled": os.getenv("USE_HYDE", "true").lower() == "true",
    }

    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, encoding="utf-8") as f:
                loaded = json.load(f)
            if isinstance(loaded, list):
                history = loaded
        except json.JSONDecodeError:
            history = []

    history.append(entry)
    os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
    with open(HISTORY_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print(f"Appended entry for commit {entry['commit']} to the history ledger.")

    check_staleness()

    failures = []
    for label, value, floor in (
        ("faithfulness", faithfulness, MIN_FAITHFULNESS),
        ("answer_relevancy", relevancy, MIN_ANSWER_RELEVANCY),
        ("context_precision", precision, MIN_CONTEXT_PRECISION),
    ):
        status = "ok" if value >= floor else "BELOW FLOOR"
        print(f"  {label:<18} {value:.3f}  (floor {floor:.2f})  {status}")
        if value < floor:
            failures.append(f"{label} {value:.3f} < {floor:.2f}")

    if failures:
        print("QUALITY GATE FAILED: " + "; ".join(failures))
        return 1

    print("QUALITY GATE PASSED (against the committed baseline).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
