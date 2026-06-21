import os
import pickle
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore

PROCESSED_DATA_DIR = "data/processed_data"
CHUNKS_PATH = os.path.join(PROCESSED_DATA_DIR, "document_chunks.pkl")

def main():
    print("Loading production text chunks from disk...")
    with open(CHUNKS_PATH, "rb") as f:
        chunks = pickle.load(f)
        
    print(f"Loaded {len(chunks)} chunks ready for vectorization.")

    print("Initializing the Embedding Model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("Connecting to local Qdrant Vector Database on port 6333...")
    print("Uploading data... This will take a few minutes depending on your CPU.")
    
    # Using the modern QdrantVectorStore class
    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name="saudi_vision_2030",
    )
    
    print("\n✅ Success! All chunks are embedded and securely stored in Qdrant.")

if __name__ == "__main__":
    main()