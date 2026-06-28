import qdrant_client
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize local embedding model
print("Loading embedding model...")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Connect to running Qdrant container
print("Connecting to Qdrant...")
client = qdrant_client.QdrantClient(url="http://localhost:6333")

# Define a test query related to your data
query = "What does the report say about volunteering or the Orphan Day initiative?"
print(f"\nUser Query: {query}")

# Embed query and search Qdrant using the updated API
query_vector = embeddings.embed_query(query)
search_results = client.query_points(
    collection_name="saudi_vision_2030",
    query=query_vector,
    limit=2
).points

# Display retrieved results
print("\n=== RETRIEVED CONTEXT FROM QDRANT ===")
for i, hit in enumerate(search_results, 1):
    print(f"\n[Match {i}] Score: {hit.score:.4f}")
    print(f"Source: {hit.payload.get('metadata', {}).get('source')}")
    print(f"Page: {hit.payload.get('metadata', {}).get('page')}")
    print(f"Content: {hit.payload.get('page_content')[:300]}...")