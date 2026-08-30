"""
Pre-flight diagnostic for every failure mode this project has actually hit.

Replaces the ad-hoc test_groq.py / test_qdrant.py scratch scripts with one check that
covers the real recurring problems recorded in OPERATIONS.md:

  * Groq withdrew the configured model and every request returned 404
  * Qdrant suspended the free cluster after inactivity and every query returned 503
  * ADMIN_PASSPHRASE unset, leaving the write endpoints permanently unusable
  * a TEXT index on metadata.source silently breaking the registry and deletion
  * FastEmbed's window drifting away from the width the corpus was indexed at

Run it before blaming the code:

    python -m scripts.doctor              # local configuration and upstreams
    python -m scripts.doctor --live       # also probe the deployed service
"""

import argparse
import asyncio
import os
import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv  # noqa: E402

LIVE_URL = "https://saudi-vision-2030-rag-3.onrender.com"
COLLECTION = "saudi_vision_2030"
EXPECTED_INDEX_TYPE = "keyword"
INDEXED_FIELDS = ("metadata.source", "metadata.section")

results = []


def report(ok, label, detail="", warn_only=False):
    """warn_only marks a gap that does not break the checked component."""
    if not warn_only:
        results.append(ok)
    mark = "PASS" if ok else ("WARN" if warn_only else "FAIL")
    print(f"  [{mark}] {label}" + (f"  -- {detail}" if detail else ""))


def check_env():
    print("\nConfiguration")
    for name, required in (("GROQ_API_KEY", True),
                           ("QDRANT_CLOUD_URL", True),
                           ("QDRANT_CLOUD_API_KEY", True)):
        present = bool(os.getenv(name) or os.getenv(name.replace("_CLOUD", "")))
        report(present or not required, name,
               "set" if present else "MISSING - the server will refuse to boot")
    # Only needed for write operations. Read-only local work does not require it, and
    # the deployment sets it independently, so its absence here is not a failure.
    admin = bool(os.getenv("ADMIN_PASSPHRASE"))
    report(admin, "ADMIN_PASSPHRASE",
           "set" if admin else "not set locally - upload and delete unavailable here; "
                               "set it in Render for the deployment",
           warn_only=not admin)


async def check_groq():
    print("\nGroq")
    from src.api import GROQ_MODEL
    from src.groq_client import get_client
    client = get_client()
    if client is None:
        report(False, "client", "no API key")
        return
    try:
        models = {m.id for m in (await client.models.list()).data}
    except Exception as e:
        report(False, "reachable", str(e)[:70])
        return
    report(True, "reachable", f"{len(models)} models visible")
    # The single most common outage in this project's history.
    report(GROQ_MODEL in models, f"configured model {GROQ_MODEL}",
           "available" if GROQ_MODEL in models
           else "NOT AVAILABLE - update GROQ_MODEL in src/api.py and re-check the "
                "context budget against the replacement's window")
    from src.evaluate_rag import JUDGE_CANDIDATES
    usable = [m for m in JUDGE_CANDIDATES if m in models]
    report(bool(usable), "judge model for evaluation",
           f"{usable[0]} available" if usable else "none of the candidates exist")
    await client.close()


def check_qdrant():
    print("\nQdrant")
    from src.retriever import HybridRetriever
    r = HybridRetriever()
    try:
        client = r._get_client()
        info = client.get_collection(COLLECTION)
    except Exception as e:
        report(False, "cluster reachable",
               f"{str(e)[:60]} - a free cluster suspends after ~7 idle days; "
               f"resume it from the Qdrant dashboard")
        return
    report(str(info.status).lower().endswith("green"), "collection status", str(info.status))
    report(info.points_count > 0, "points", f"{info.points_count:,}")

    schema = info.payload_schema or {}
    for field in INDEXED_FIELDS:
        entry = schema.get(field)
        actual = str(getattr(entry, "data_type", "")).split(".")[-1].lower().strip("'\"")
        ok = actual == EXPECTED_INDEX_TYPE
        report(ok, f"index {field}",
               f"{actual or 'MISSING'}" + ("" if ok else
               " - must be keyword; a text index breaks the document registry and delete"))

    window = r.EMBED_MAX_TOKENS
    try:
        tok = r._get_dense_model().model.tokenizer.truncation["max_length"]
        report(tok == window, "embedding window", f"{tok} tokens")
    except Exception as e:
        report(False, "embedding window", str(e)[:60])


def check_live():
    print("\nDeployed service")

    def probe(path, method="GET"):
        req = urllib.request.Request(LIVE_URL + path, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status
        except urllib.error.HTTPError as e:
            return e.code
        except Exception:
            return None

    status = probe("/health")
    report(status == 200, "health", f"HTTP {status}"
           if status else "unreachable (a cold start takes ~52s; retry once)")
    code = probe("/api/documents?filename=x", "DELETE")
    report(code == 403, "delete endpoint gated", f"HTTP {code}")


async def main():
    parser = argparse.ArgumentParser(description="Diagnose configuration and upstreams.")
    parser.add_argument("--live", action="store_true", help="Also probe the deployed service.")
    args = parser.parse_args()

    load_dotenv(PROJECT_ROOT / ".env")
    print("Saudi Vision 2030 RAG - diagnostic")

    check_env()
    await check_groq()
    check_qdrant()
    if args.live:
        check_live()

    failed = results.count(False)
    print(f"\n{len(results) - failed}/{len(results)} checks passed")
    if failed:
        print("See OPERATIONS.md for the runbook covering each failure above.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
