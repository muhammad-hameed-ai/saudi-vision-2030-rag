import qdrant_client
import requests
from langchain_huggingface import HuggingFaceEmbeddings

# 1. Setup Models & Database
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
client = qdrant_client.QdrantClient(url="http://localhost:6333")

query = "What does the report say about volunteering and the Orphan Day initiative?"
print(f"\n[1] User Query: {query}")

# 2. Retrieve Context from Qdrant
print("[2] Retrieving context from Qdrant...")
query_vector = embeddings.embed_query(query)
results = client.query_points(
    collection_name="saudi_vision_2030",
    query=query_vector,
    limit=2
).points

context = "\n\n".join([hit.payload.get('page_content', '') for hit in results])

# 3. Generate Final Answer with Llama
print("[3] Generating final answer with local Llama 3.2 model...\n")
prompt = f"""Use the following context from the Saudi Vision 2030 documents to answer the user's question accurately. Do not make up information.

Context:
{context}

Question: {query}
Answer:"""

url = "http://localhost:11434/api/generate"
payload = {
    "model": "llama3.2:1b",
    "prompt": prompt,
    "stream": False
}

try:
    response = requests.post(url, json=payload)
    response.raise_for_status()
    final_answer = response.json().get('response', '')
    print("=== FINAL AI RESPONSE ===")
    print(final_answer.strip())
except Exception as e:
    print(f"Error connecting to Ollama: {e}")
    print("Please make sure your Ollama application is open and running in the background!")