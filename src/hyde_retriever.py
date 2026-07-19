import asyncio
import ollama
from src.retriever import HybridRetriever

async def generate_hypothesis(query: str) -> str:
    prompt = (
        "Write a concise, 1-paragraph hypothetical answer to this policy question "
        "as if it were extracted directly from an official Saudi Vision 2030 document. Keep it extremely brief.\n"
        f"Question: {query}"
    )
    client = ollama.AsyncClient(host="http://localhost:11434", timeout=180.0)
    response = await client.chat(
        model="llama3.2:1b",
        messages=[{"role": "user", "content": prompt}],
        options={"num_predict": 75, "temperature": 0.3}
    )
    return response["message"]["content"].strip()

async def hyde_retrieve(query: str, k: int = 20):
    hypothesis = await generate_hypothesis(query)
    print(f"\n--- HYPOTHESIS GENERATED ---\n{hypothesis}\n----------------------------\n")
    
    retriever = HybridRetriever()
    results = await asyncio.to_thread(retriever.retrieve, hypothesis, k=k)
    return results

async def main():
    query = "What are the main economic goals for the private sector?"
    print(f"Original Query: {query}")
    
    retriever = HybridRetriever()
    
    print("\n1. Running standard retrieval...")
    standard_results = await asyncio.to_thread(retriever.retrieve, query, k=3)
    for i, res in enumerate(standard_results):
        print(f"Standard Result {i+1} Score: {res.score:.4f} | Preview: {res.content[:80]}...")

    print("\n2. Running HyDE retrieval...")
    hyde_results = await hyde_retrieve(query, k=3)
    for i, res in enumerate(hyde_results):
        print(f"HyDE Result {i+1} Score: {res.score:.4f} | Preview: {res.content[:80]}...")

if __name__ == "__main__":
    asyncio.run(main())