"""
HyDE (Hypothetical Document Embedding) Retriever — Cloud Edition

Generates a short hypothetical answer via Groq Cloud API, then uses it
as the search query for denser semantic matching against the vector store.
"""

import os
import asyncio

from src.groq_client import get_client


async def generate_hypothesis(query: str) -> str:
    """
    Generate a concise hypothetical document snippet via Groq Cloud LLM.
    Falls back to the original query if Groq is unavailable.
    """
    client = get_client()
    if client is None:
        print("[HyDE] GROQ_API_KEY not set - skipping hypothesis generation.")
        return query

    prompt = (
        "Write a concise, 1-paragraph hypothetical answer to this policy question "
        "as if it were extracted directly from an official Saudi Vision 2030 document. "
        "Keep it extremely brief (under 60 words).\n"
        f"Question: {query}"
    )

    try:
        response = await client.chat.completions.create(
            model="allam-2-7b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=100,
            temperature=0.3,
            timeout=15.0,
        )
        hypothesis = response.choices[0].message.content.strip()
        print(f"[HyDE] Generated hypothesis ({len(hypothesis)} chars)")
        return hypothesis
    except Exception as e:
        print(f"[HyDE] Groq hypothesis generation failed, falling back to raw query: {e}")
        return query