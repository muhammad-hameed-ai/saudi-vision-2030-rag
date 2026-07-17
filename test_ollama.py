import httpx
import asyncio
import os

async def test_ollama():
    print("--- OLLAMA DIAGNOSTIC SCRIPT ---")
    
    urls = [
        "http://127.0.0.1:11434/api/tags",
        "http://localhost:11434/api/tags",
        "http://[::1]:11434/api/tags",
    ]
    
    # Also test the environment variable we've been using
    env_host = os.environ.get("OLLAMA_HOST", "NOT SET")
    print(f"Current OLLAMA_HOST env var: {env_host}\n")

    for url in urls:
        print(f"Testing {url} ...")
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url)
                print(f"  SUCCESS! Status: {resp.status_code}")
                # print(f"  Response snippet: {resp.text[:100]}")
        except httpx.ConnectError as e:
            print(f"  FAILED (ConnectError): {e}")
        except Exception as e:
            print(f"  FAILED (Other): type={type(e).__name__}, msg={e}")
        print()

if __name__ == "__main__":
    asyncio.run(test_ollama())