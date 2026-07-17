with open(r"src\api.py", "r", encoding="utf-8") as f:
    content = f.read()

# Fix the /ask route specifically
if 'memory_str = await _build_memory_string(request.session_id)' not in content.split('def ask(request: ChatRequest):')[1]:
    # Replace the context_text definition inside the ask function
    old_ask_context = 'context_text = "\\n\\n".join([c.content for c in reranked])'
    new_ask_context = 'memory_str = await _build_memory_string(request.session_id)\n    context_text = "\\n\\n".join([c.content for c in reranked])'
    
    # We only want to replace it after the "def ask(" definition
    parts = content.split('def ask(request: ChatRequest):')
    parts[1] = parts[1].replace(old_ask_context, new_ask_context, 1)
    
    content = 'def ask(request: ChatRequest):'.join(parts)

with open(r"src\api.py", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: src\\api.py /ask route has been fixed.")