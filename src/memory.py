"""
Session Memory Manager — Cloud Edition

Stores conversation history in SQLite and optionally summarizes long
sessions via Groq Cloud API (no local Ollama dependency).
"""

import os
import sqlite3
import asyncio
import json
from pathlib import Path
from datetime import datetime, timezone

# Anchored to the project root rather than the process CWD: uvicorn is not always
# started from the repository root, and a relative path would silently create a
# second, empty database somewhere else.
DB_PATH = str(Path(__file__).resolve().parent.parent / "data" / "sessions.db")

_initialized = False


def _connect() -> sqlite3.Connection:
    """
    Opens a connection with a write timeout.

    Every request handler opens its own connection from a worker thread, so
    concurrent turns contend for the same file. Without a timeout SQLite raises
    'database is locked' immediately instead of waiting for the writer to finish.
    """
    return sqlite3.connect(DB_PATH, timeout=10.0)


def init_db():
    """Initializes the SQLite database and creates the messages table."""
    global _initialized
    if _initialized:
        return
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary TEXT
        )
    """)
    conn.commit()
    conn.close()
    _initialized = True


def save_message(session_id: str, role: str, content: str):
    """Saves a message to the SQLite database."""
    init_db()
    conn = _connect()
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()


def get_session_history(session_id: str, limit: int = 4) -> dict:
    """Retrieves the last N messages for a session, plus any existing summary."""
    init_db()
    conn = _connect()
    cursor = conn.cursor()

    # Check for the latest summary if any exists
    cursor.execute(
        "SELECT summary FROM messages WHERE session_id = ? AND summary IS NOT NULL ORDER BY id DESC LIMIT 1",
        (session_id,)
    )
    summary_row = cursor.fetchone()
    summary = summary_row[0] if summary_row else None

    # Fetch the last N messages
    cursor.execute(
        "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
        (session_id, limit)
    )
    rows = cursor.fetchall()
    conn.close()

    # Reverse to restore chronological order
    messages = [{"role": r, "content": c} for r, c in reversed(rows)]
    return {"summary": summary, "messages": messages}


async def summarize_history(session_id: str):
    """
    Summarizes history if session has more than 8 messages.
    Uses Groq Cloud API instead of local Ollama.
    Silently skips if Groq is unavailable (non-critical feature).
    """
    from src.groq_client import get_client
    if get_client() is None:
        return

    init_db()
    conn = _connect()
    cursor = conn.cursor()

    # Count total messages in this session
    cursor.execute("SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,))
    count = cursor.fetchone()[0]

    if count <= 8:
        conn.close()
        return

    # Fetch all messages to build a comprehensive summary
    cursor.execute("SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = cursor.fetchall()

    conn.close()

    history_text = "\n".join([f"{role.upper()}: {content}" for role, content in rows])

    prompt = (
        "Summarize the following conversation history briefly and concisely, focusing "
        "only on key topics discussed. Keep the summary under 150 words.\n\n"
        f"Conversation:\n{history_text}"
    )

    try:
        response = await get_client().chat.completions.create(
            model="allam-2-7b",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=200,
            temperature=0.3,
            timeout=15.0,
        )
        summary_text = response.choices[0].message.content.strip()

        # Reopen only once the network call has returned.
        conn = _connect()
        cursor = conn.cursor()

        # Save summary to the most recent message row
        cursor.execute(
            "SELECT id FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT 1",
            (session_id,)
        )
        latest_id = cursor.fetchone()[0]

        cursor.execute(
            "UPDATE messages SET summary = ? WHERE id = ?",
            (summary_text, latest_id)
        )
        conn.commit()
        conn.close()
        print(f"[Memory] Summarized session {session_id} ({len(summary_text)} chars)")
    except Exception as e:
        print(f"[Memory] Summary skipped (non-critical): {e}")


def clear_session(session_id: str):
    """Deletes all messages for a session."""
    init_db()
    conn = _connect()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()