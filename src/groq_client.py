"""
One shared Groq client for the whole process.

Every caller previously constructed its own `AsyncGroq` on each request -- the chat
route, the HyDE expansion, and the session summariser. Each instance carries its own
httpx connection pool, and none were ever closed, so a busy container accumulated
pools it could not reclaim and paid a fresh TLS handshake on every single call.

Reusing one client keeps connections alive between requests, which removes that
handshake from the hot path and stops the leak. Per-call timeouts still vary, so
callers pass `timeout=` to the request rather than baking it into the client.
"""

import logging
import os
from typing import Optional

from groq import AsyncGroq

logger = logging.getLogger("vision2030.groq")

_client: Optional[AsyncGroq] = None

# Generous ceiling; individual calls pass their own tighter timeout.
DEFAULT_TIMEOUT = 60.0


def get_client() -> Optional[AsyncGroq]:
    """
    Returns the shared client, or None when no API key is configured.

    Returning None rather than raising keeps the optional features that use this
    (HyDE, session summarisation) non-fatal when the key is absent.
    """
    global _client
    if _client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            return None
        _client = AsyncGroq(api_key=api_key, timeout=DEFAULT_TIMEOUT, max_retries=1)
        logger.info("[Groq] Shared async client initialised.")
    return _client


async def close_client() -> None:
    """Releases the connection pool on shutdown."""
    global _client
    if _client is not None:
        try:
            await _client.close()
            logger.info("[Groq] Shared async client closed.")
        except Exception as e:
            logger.warning(f"[Groq] Error closing client: {e}")
        finally:
            _client = None
