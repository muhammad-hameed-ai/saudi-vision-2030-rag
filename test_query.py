import asyncio
from src.api import ask, ChatRequest
import json

async def main():
    try:
        req = ChatRequest(question="What is the target for non-oil GDP growth?", k=3)
        resp = await ask(req)
        print("SUCCESS_OUTPUT:")
        print(resp.model_dump_json(indent=2))
    except Exception as e:
        print(f"FAILED: {e}")

if __name__ == "__main__":
    asyncio.run(main())
