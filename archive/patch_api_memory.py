import os
import re

with open(r"src\api.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Imports
if "import uuid" not in content:
    content = content.replace(
        "import json\n", 
        "import json\nimport uuid\nfrom src.memory import save_message, get_session_history, summarize_history\n"
    )

# 2. ChatRequest Update
if "session_id: str =" not in content:
    content = content.replace(
        'question: str = Field(default="")',
        'session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))\n    question: str = Field(default="")'
    )

# 3. Memory Helper Function
memory_helper = """
async def _build_memory_string(session_id: str) -> str:
    memory_context = await asyncio.to_thread(get_session_history, session_id, 4)
    memory_str = ""
    if memory_context["summary"]:
        memory_str += f"Summary of past conversation: {memory_context['summary']}\\n"
    for m in memory_context["messages"]:
        memory_str += f"{m['role'].upper()}: {m['content']}\\n"
    return memory_str
"""
if "_build_memory_string" not in content:
    content = content.replace("@app.post(\"/api/chat\")", memory_helper + "\n@app.post(\"/api/chat\")")

# 4. Inject Memory String generation
if "memory_str =" not in content:
    content = content.replace(
        'context = "\\n\\n".join([c.content for c in reranked])',
        'memory_str = await _build_memory_string(request.session_id)\n    context = "\\n\\n".join([c.content for c in reranked])'
    )
    content = content.replace(
        'context_text = "\\n\\n".join([c.content for c in reranked])',
        'memory_str = await _build_memory_string(request.session_id)\n    context_text = "\\n\\n".join([c.content for c in reranked])'
    )

# 5. Update Prompts
content = content.replace(
    "Use ONLY the following CONTEXT to answer",
    "Use ONLY the following CONTEXT and MEMORY to answer"
)
if 'f"MEMORY:\\n{memory_str}\\n\\n"' not in content:
    content = re.sub(
        r'f"CONTEXT:\\n\{context\}"',
        r'f"MEMORY:\\n{memory_str}\\n\\n"\n        f"CONTEXT:\\n{context}"',
        content
    )
    content = re.sub(
        r'f"CONTEXT:\\n\{context_text\}"',
        r'f"MEMORY:\\n{memory_str}\\n\\n"\n        f"CONTEXT:\\n{context_text}"',
        content
    )

# 6. Save Messages & Summarize Task
save_logic_chat = """        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", ai_answer)
        asyncio.create_task(summarize_history(request.session_id))"""
if "save_message" not in content:
    content = content.replace(
        'ai_answer = response["message"]["content"].strip()',
        'ai_answer = response["message"]["content"].strip()\n' + save_logic_chat
    )
    
save_logic_ask = """        await asyncio.to_thread(save_message, request.session_id, "user", request.question)
        await asyncio.to_thread(save_message, request.session_id, "assistant", answer)
        asyncio.create_task(summarize_history(request.session_id))"""
if "save_message" in content and content.count("save_message") < 4:
    content = content.replace(
        'answer = response["message"]["content"].strip()',
        'answer = response["message"]["content"].strip()\n' + save_logic_ask
    )

# 7. Add session_id to Responses
if '"session_id": request.session_id' not in content:
    content = content.replace(
        '"answer": ai_answer,',
        '"session_id": request.session_id,\n        "answer": ai_answer,'
    )
if 'session_id=request.session_id' not in content:
    content = content.replace(
        'question=request.question,',
        'session_id=request.session_id,\n        question=request.question,'
    )
if 'class AskResponse(BaseModel):' in content and 'session_id: str' not in content.split('class AskResponse(BaseModel):')[1][:100]:
    content = content.replace(
        'class AskResponse(BaseModel):',
        'class AskResponse(BaseModel):\n    session_id: str'
    )

with open(r"src\api.py", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: src\\api.py has been updated with SQLite persistent memory integration.")