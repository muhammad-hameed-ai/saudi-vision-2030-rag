import os

with open(r"src\api.py", "r", encoding="utf-8") as f:
    content = f.read()

# 1. Add the import at the top
if "generate_hypothesis" not in content:
    content = content.replace(
        "from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query",
        "from src.logging_middleware import StructuredLoggingMiddleware, log_rag_query\nfrom src.hyde_retriever import generate_hypothesis"
    )

# 2. Update the /api/chat endpoint
old_chat = """        # Stage 1: Async Hybrid retrieval
        candidates = await asyncio.to_thread(
            retriever.retrieve, request.question, k=RETRIEVAL_K
        )"""

new_chat = """        # Stage 1: Async Hybrid retrieval
        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(request.question) if use_hyde else request.question
        candidates = await asyncio.to_thread(
            retriever.retrieve, search_query, k=RETRIEVAL_K
        )"""

content = content.replace(old_chat, new_chat)

# 3. Update the /ask endpoint
old_ask = """        candidates = await asyncio.to_thread(retriever.retrieve, request.question, k=RETRIEVAL_K)"""

new_ask = """        use_hyde = os.environ.get("USE_HYDE", "false").lower() == "true"
        search_query = await generate_hypothesis(request.question) if use_hyde else request.question
        candidates = await asyncio.to_thread(retriever.retrieve, search_query, k=RETRIEVAL_K)"""

content = content.replace(old_ask, new_ask)

with open(r"src\api.py", "w", encoding="utf-8") as f:
    f.write(content)

print("SUCCESS: src\\api.py has been updated with the USE_HYDE integration.")