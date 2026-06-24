import os
import pickle
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import QdrantVectorStore


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    chunk_cfg = params["chunk"]
    embed_cfg  = params["embed"]

    print(f"Loading chunks from {chunk_cfg['output_path']}...")
    with open(chunk_cfg["output_path"], "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print(f"Initializing embedding model: {embed_cfg['model_name']}")
    embeddings = HuggingFaceEmbeddings(
        model_name=embed_cfg["model_name"],
        model_kwargs={"device": "cpu"}
    )

    print(f"Connecting to Qdrant at {embed_cfg['qdrant_url']}...")
    print(f"Uploading {len(chunks)} chunks to collection '{embed_cfg['collection_name']}'...")

    qdrant = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url=embed_cfg["qdrant_url"],
        collection_name=embed_cfg["collection_name"],
        force_recreate=True,
    )

    print(f"Successfully embedded {len(chunks)} chunks into Qdrant")
    return len(chunks)


if __name__ == "__main__":
    main()