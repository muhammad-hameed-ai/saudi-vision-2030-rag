"""
Embedding pipeline: generates both dense (MiniLM) and sparse (BM25) vectors
and upserts them into a Qdrant collection with named vector support.
"""

import os
import pickle
import yaml
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


def load_params():
    with open("params.yaml") as f:
        return yaml.safe_load(f)


def main():
    params = load_params()
    chunk_cfg = params["chunk"]
    embed_cfg = params["embed"]

    # --- Load chunks ---
    print(f"Loading chunks from {chunk_cfg['output_path']}...")
    with open(chunk_cfg["output_path"], "rb") as f:
        chunks = pickle.load(f)
    print(f"Loaded {len(chunks)} chunks")

    # --- Initialize models ---
    dense_model_name = embed_cfg["model_name"]
    sparse_model_name = embed_cfg.get("sparse_model", "Qdrant/bm25")

    print(f"Initializing dense model: {dense_model_name}")
    dense_model = HuggingFaceEmbeddings(
        model_name=dense_model_name,
        model_kwargs={"device": "cpu"},
    )

    print(f"Initializing sparse model: {sparse_model_name}")
    sparse_model = SparseTextEmbedding(sparse_model_name)

    # --- Connect to Qdrant ---
    qdrant_url = embed_cfg["qdrant_url"]
    collection_name = embed_cfg["collection_name"]

    print(f"Connecting to Qdrant at {qdrant_url}...")
    client = QdrantClient(url=qdrant_url, timeout=60)

    # --- Recreate collection with dual-vector config ---
    print(f"Creating collection '{collection_name}' with dense + sparse vectors...")
    if client.collection_exists(collection_name):
        client.delete_collection(collection_name)

    client.create_collection(
        collection_name=collection_name,
        vectors_config={
            "dense": models.VectorParams(
                size=384,  # MiniLM-L6-v2 output dimension
                distance=models.Distance.COSINE,
            ),
        },
        sparse_vectors_config={
            "sparse": models.SparseVectorParams(),
        },
    )
    print("Collection created successfully.")

    # --- Batch embed and upsert ---
    batch_size = embed_cfg.get("batch_size", 100)
    texts = [chunk.page_content for chunk in chunks]
    total = len(texts)

    print(f"Embedding and uploading {total} chunks in batches of {batch_size}...")

    for batch_start in range(0, total, batch_size):
        batch_end = min(batch_start + batch_size, total)
        batch_texts = texts[batch_start:batch_end]
        batch_chunks = chunks[batch_start:batch_end]

        # Generate dense embeddings
        dense_vectors = dense_model.embed_documents(batch_texts)

        # Generate sparse embeddings
        sparse_results = list(sparse_model.embed(batch_texts))

        # Build points
        points = []
        for i, (chunk, dense_vec, sparse_vec) in enumerate(
            zip(batch_chunks, dense_vectors, sparse_results)
        ):
            point_id = batch_start + i
            points.append(
                models.PointStruct(
                    id=point_id,
                    vector={
                        "dense": dense_vec,
                        "sparse": models.SparseVector(
                            indices=sparse_vec.indices.tolist(),
                            values=sparse_vec.values.tolist(),
                        ),
                    },
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        client.upsert(collection_name=collection_name, points=points)
        progress = round(batch_end / total * 100, 1)
        print(f"  Uploaded {batch_end}/{total} ({progress}%)")

    # --- Verify ---
    info = client.get_collection(collection_name)
    print(f"\nCollection '{collection_name}' ready:")
    print(f"  Points: {info.points_count}")
    print(f"  Vectors: dense (384-dim cosine) + sparse (BM25)")
    print(f"  Status: {info.status}")
    return info.points_count


if __name__ == "__main__":
    main()