"""
Evaluates the RAG pipeline that is actually deployed.

The previous version of this script measured a different system entirely: it read from
a local Qdrant at http://localhost:6333, embedded with langchain HuggingFaceEmbeddings,
called plain similarity_search (no sparse retrieval, no RRF fusion, no policy booster,
no cosine rescoring), generated with Ollama llama3.2:1b, and judged those answers with
that same 1B model. None of that is production, so the scores it produced never
described the service.

This version drives the real path:

  * retrieval  -> src.retriever.HybridRetriever, the same object the API uses
  * prompt     -> src.api.SYSTEM_PROMPT_TEMPLATE and _assemble_context, unmodified
  * generation -> Groq, the same model the API serves with
  * judging    -> a deliberately LARGER Groq model than the generator; a model scoring
                  its own output is not a measurement

Output schema is unchanged so src/ci_quality_gate.py keeps working.

Usage:
    python -m src.evaluate_rag --dry-run   # retrieval + prompts only, no LLM calls
    python -m src.evaluate_rag             # full evaluation
    python -m src.evaluate_rag --judge-model llama-3.3-70b-versatile
"""

import argparse
import asyncio
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = PROJECT_ROOT / "data" / "evaluation" / "evaluation_scores.json"

# The judge must be stronger than the generator or the score is meaningless. Candidates
# are tried in order and the first reachable one is used, so a single deprecation does
# not break evaluation the way it once broke the service.
JUDGE_CANDIDATES = [
    "llama-3.3-70b-versatile",
    "openai/gpt-oss-120b",
    "llama-3.1-70b-versatile",
]

EVAL_QUESTIONS = [
    "What are the main economic goals of Saudi Vision 2030?",
    "What role does the private sector play in Vision 2030?",
    "What is the Public Investment Fund and what is its role?",
    "What are the Vision Realization Programs?",
    "How does Vision 2030 aim to develop the entertainment sector?",
]


def _git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=20)
        return out.stdout.strip() if out.returncode == 0 else "unknown"
    except Exception:
        return "unknown"


def _parse_score(text: str) -> float:
    """
    Extracts the first number in [0,1] from a judge reply.

    Judges add commentary despite instructions not to, so a bare float() on the whole
    reply throws and silently degrades every score to a 0.5 default.
    """
    match = re.search(r"\d*\.?\d+", text or "")
    if not match:
        return None
    try:
        return max(0.0, min(1.0, float(match.group(0))))
    except ValueError:
        return None


class Judge:
    """Scores generated answers with a model larger than the one under test."""

    def __init__(self, client, model: str):
        self.client = client
        self.model = model
        self.failures = 0

    async def _score(self, prompt: str) -> float:
        try:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=12,
                temperature=0.0,
            )
            value = _parse_score(response.choices[0].message.content)
        except Exception as e:
            print(f"      judge call failed: {e}")
            value = None
        if value is None:
            # Counted and reported rather than silently folded into the average as a
            # neutral 0.5, which is how the old script hid judge failures.
            self.failures += 1
            return None
        return value

    async def faithfulness(self, answer: str, context: str) -> float:
        return await self._score(
            "Rate how faithful the ANSWER is to the CONTEXT, from 0.0 to 1.0.\n"
            "1.0 means every claim in the answer is supported by the context.\n"
            "0.0 means the answer asserts things the context does not support.\n"
            "Reply with only a number.\n\n"
            f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\nSCORE:"
        )

    async def relevancy(self, question: str, answer: str) -> float:
        return await self._score(
            "Rate how well the ANSWER addresses the QUESTION, from 0.0 to 1.0.\n"
            "1.0 means it directly and completely answers it.\n"
            "0.0 means it is off-topic or refuses without cause.\n"
            "Reply with only a number.\n\n"
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\nSCORE:"
        )

    async def context_precision(self, question: str, chunks) -> float:
        scores = []
        for chunk in chunks:
            value = await self._score(
                "Rate how useful this CONTEXT CHUNK is for answering the QUESTION, "
                "from 0.0 to 1.0.\nReply with only a number.\n\n"
                f"QUESTION:\n{question}\n\nCONTEXT CHUNK:\n{chunk[:1500]}\n\nSCORE:"
            )
            if value is not None:
                scores.append(value)
        return sum(scores) / len(scores) if scores else None


async def resolve_judge(client, preferred: str = None) -> str:
    """Returns the first candidate judge model that answers, so a deprecation is survivable."""
    for model in ([preferred] if preferred else []) + JUDGE_CANDIDATES:
        try:
            await client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": "0.5"}], max_tokens=3)
            return model
        except Exception as e:
            print(f"  judge candidate unavailable: {model} ({str(e)[:70]})")
    return None


async def run(args) -> int:
    load_dotenv()

    # Imported here so --dry-run still works when the API module's boot checks would
    # otherwise exit the process.
    from src.retriever import HybridRetriever
    from src.api import (SYSTEM_PROMPT_TEMPLATE, _assemble_context,
                         optimize_search_query, GROQ_MODEL,
                         RETRIEVAL_K, MAX_RESPONSE_TOKENS)

    retriever = HybridRetriever()
    print(f"Retriever  : {retriever.qdrant_url}")
    print(f"Generator  : {GROQ_MODEL}")

    client = judge = None
    if not args.dry_run:
        from groq import AsyncGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("ERROR: GROQ_API_KEY is not set.", file=sys.stderr)
            return 1
        client = AsyncGroq(api_key=api_key, timeout=90.0)
        judge_model = await resolve_judge(client, args.judge_model)
        if not judge_model:
            print("ERROR: no judge model reachable. Evaluation aborted rather than "
                  "reporting scores from an unverified judge.", file=sys.stderr)
            return 1
        judge = Judge(client, judge_model)
        print(f"Judge      : {judge_model}")
        if judge_model == GROQ_MODEL:
            print("WARNING: judge and generator are the same model; scores are not "
                  "independent.")
    print(f"Questions  : {len(EVAL_QUESTIONS)}   k={args.k}\n")

    faith_scores, relev_scores, prec_scores, log = [], [], [], []

    for i, question in enumerate(EVAL_QUESTIONS, 1):
        print(f"[{i}/{len(EVAL_QUESTIONS)}] {question}")

        # Exactly the production retrieval path.
        search_query = optimize_search_query(question)
        chunks = await asyncio.to_thread(
            retriever.retrieve, search_query, k=max(RETRIEVAL_K, args.k))
        chunks = chunks[:args.k]
        context = _assemble_context(chunks, "")
        texts = [getattr(c, "content", "") for c in chunks]
        sources = [getattr(c, "source", "?").split(os.sep)[-1] for c in chunks]
        top_score = round(getattr(chunks[0], "score", 0.0), 4) if chunks else 0.0

        print(f"      retrieved {len(chunks)} chunks, top cosine {top_score}, "
              f"{len(context)} context chars")
        print(f"      sources: {', '.join(s[:32] for s in sources[:3])}")

        if args.dry_run:
            log.append({"question": question, "retrieved": len(chunks),
                        "top_score": top_score, "context_chars": len(context),
                        "sources": sources})
            continue

        system_prompt = SYSTEM_PROMPT_TEMPLATE.format(memory="", context=context)
        completion = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "system", "content": system_prompt},
                      {"role": "user", "content": question}],
            max_tokens=MAX_RESPONSE_TOKENS,
            temperature=0.2,
        )
        answer = completion.choices[0].message.content.strip()

        faith = await judge.faithfulness(answer, context)
        relev = await judge.relevancy(question, answer)
        prec = await judge.context_precision(question, texts)

        for value, bucket in ((faith, faith_scores), (relev, relev_scores),
                              (prec, prec_scores)):
            if value is not None:
                bucket.append(value)

        def fmt(v):
            return f"{v:.2f}" if v is not None else "n/a"
        print(f"      faithfulness {fmt(faith)} | relevancy {fmt(relev)} "
              f"| precision {fmt(prec)}\n")

        log.append({
            "question": question,
            "answer": answer,
            "faithfulness": faith,
            "answer_relevancy": relev,
            "context_precision": prec,
            "sources": sources,
            "top_cosine": top_score,
        })

    if args.dry_run:
        print("\n--dry-run: retrieval and prompt assembly verified. "
              "No LLM calls made, no scores written.")
        return 0

    def mean(xs):
        return round(sum(xs) / len(xs), 4) if xs else 0.0

    results = {
        "faithfulness": mean(faith_scores),
        "answer_relevancy": mean(relev_scores),
        "context_precision": mean(prec_scores),
        # Provenance: without these, a score cannot be attributed to a pipeline.
        "evaluated_commit": _git_sha(),
        "generator_model": GROQ_MODEL,
        "judge_model": judge.model,
        "retrieval_k": args.k,
        "judge_failures": judge.failures,
        "evaluated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "per_question": log,
    }

    print("=" * 62)
    print(f"  faithfulness       {results['faithfulness']:.4f}")
    print(f"  answer_relevancy   {results['answer_relevancy']:.4f}")
    print(f"  context_precision  {results['context_precision']:.4f}")
    print(f"  generator {GROQ_MODEL}   judge {judge.model}   commit {results['evaluated_commit']}")
    if judge.failures:
        print(f"  WARNING: {judge.failures} judge call(s) failed and were excluded "
              f"rather than defaulted to 0.5")
    print("=" * 62)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nWritten to {OUTPUT_PATH.relative_to(PROJECT_ROOT)}")
    print("Next: python -m src.ci_quality_gate")
    return 0


def main():
    parser = argparse.ArgumentParser(description="Evaluate the deployed RAG pipeline.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Exercise retrieval and prompt assembly without calling any LLM.")
    parser.add_argument("--judge-model", default=None,
                        help="Override the judge model. Must differ from the generator.")
    parser.add_argument("--k", type=int, default=5, help="Retrieval depth (default 5).")
    return asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
