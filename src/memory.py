import os
import sqlite3
import asyncio
import json
import ollama
from datetime import datetime, timezone

DB_PATH = "data/sessions.db"

def init_db():
    """Initializes the SQLite database and creates the messages table."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
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

def save_message(session_id: str, role: str, content: str):
    """Saves a message to the SQLite database."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    timestamp = datetime.now(timezone.utc).isoformat()
    cursor.execute(
        "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
        (session_id, role, content, timestamp)
    )
    conn.commit()
    conn.close()

def get_session_history(session_id: str, limit: int = 4) -> list:
    """Retrieves the last N messages for a session, plus any existing summary."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
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
    """Summarizes history if session has more than 8 messages, saving summary to the DB."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
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
    
    history_text = "\n".join([f"{role.upper()}: {content}" for role, content in rows])
    
    prompt = (
        "Summarize the following conversation history briefly and concisely, focusing "
        "only on key topics discussed. Keep the summary under 150 words.\n\n"
        f"Conversation:\n{history_text}"
    )
    
    try:
        client = ollama.AsyncClient(host="http://localhost:11434")
        response = await client.chat(
            model="llama3.2:1b",
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200, "temperature": 0.3}
        )
        summary_text = response["message"]["content"].strip()
        
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
        print(f"[Memory] Successfully summarized history for session {session_id}.")
    except Exception as e:
        print(f"[Memory] Failed to summarize history: {e}")
    finally:
        conn.close()

def clear_session(session_id: str):
    """Deletes all messages for a session."""
    init_db()
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
    conn.commit()
    conn.close()

async def test_memory():
    session_id = "test_session_123"
    clear_session(session_id)
    
    print("Testing saving messages...")
    save_message(session_id, "user", "Hello, I am asking about Saudi Vision 2030.")
    save_message(session_id, "assistant", "I am an analyst for Saudi Vision 2030. How can I help?")
    save_message(session_id, "user", "What are the goals of the green initiative?")
    save_message(session_id, "assistant", "To plant 10 billion trees and reduce emissions.")
    
    history = get_session_history(session_id, limit=2)
    print("\nSession history retrieved (limit 2):")
    print(json.dumps(history, indent=2))
    
    # Save more messages to trigger a summary run
    print("\nAdding more messages to trigger summary mechanism...")
    save_message(session_id, "user", "Who manages the PIF?")
    save_message(session_id, "assistant", "The Public Investment Fund is chaired by the Crown Prince.")
    save_message(session_id, "user", "What is the 2030 asset target?")
    save_message(session_id, "assistant", "The target was revised up to 2.67 trillion dollars.")
    save_message(session_id, "user", "Tell me about tourism targets.")
    save_message(session_id, "assistant", "The kingdom aims to attract 150 million visitors annually by 2030.")
    
    print("\nRunning summary process...")
    await summarize_history(session_id)
    
    history_after = get_session_history(session_id, limit=2)
    print("\nSession history retrieved after summary:")
    print(json.dumps(history_after, indent=2))

if __name__ == "__main__":
    asyncio.run(test_memory())