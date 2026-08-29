"""
Embedding pipeline: generates dense (MiniLM) and sparse (BM25) vectors and upserts
them into the Qdrant collection the API serves from.

This is the reproducible path from data/raw_pdfs to the live vector store. It targets
Qdrant Cloud by default using the same environment variables the API reads, so the
corpus can be rebuilt if the cluster is ever lost.

Safety: the collection is NEVER dropped unless --recreate is passed explicitly. The
previous version called delete_collection unconditionally, so pointing it at the cloud
would have destroyed the live corpus before re-uploading anything.

Embeddings come from FastEmbed, the same implementation the query path uses. The old
version used langchain HuggingFaceEmbeddings, which is not in requirements.txt and
truncates at a different point, so the two produce noticeably different vectors for
chunks near the model's 256-token limit. Rebuild the WHOLE collection in one run
rather than mixing vectors from both implementations.

Usage:
    python -m src.create_embeddings --dry-run     # report the plan, change nothing
    python -m src.create_embeddings               # upsert in place (idempotent)
    python -m src.create_embeddings --recreate    # drop and rebuild from scratch
"""

import argparse
import os
import pickle
import sys

import yaml
from dotenv import load_dotenv
from fastembed import TextEmbedding, SparseTextEmbedding
from qdrant_client import QdrantClient, models

from src.retriever import HybridRetriever

DENSE_VECTOR_NAME = "dense"
SPARSE_VECTOR_NAME = "sparse"
DENSE_DIMENSIONS = 384

# Indexed as KEYWORD because the document registry facets on source and the retriever
# filters both fields by exact value. See the note in retriever._get_vision_sources
# for why a TEXT index cannot be used on metadata.source.
KEYWORD_INDEXED_FIELDS = ("metadata.source", "metadata.section")


def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_target(embed_cfg):
    """
    Environment first, params.yaml second.

    params.yaml still points at http://localhost:6333, which is why this pipeline could
    never rebuild the cloud collection. Reading the same variables the API reads means
    one configuration drives both sides.
    """
    url = (
        os.getenv("QDRANT_CLOUD_URL")
        or os.getenv("QDRANT_URL")
        or embed_cfg.get("qdrant_url", "http://localhost:6333")
    )
    api_key = os.getenv("QDRANT_CLOUD_API_KEY") or os.getenv("QDRANT_API_KEY")
    return url, api_key


def ensure_collection(client, name, recreate):
    exists = client.collection_exists(name)
    if exists and recreate:
        print(f"  --recreate: dropping existing collection {name}")
        client.delete_collection(name)
        exists = False
    if not exists:
        print(f"  creating collection {name} (dense {DENSE_DIMENSIONS}d cosine + sparse BM25)")
        client.create_collection(
            collection_name=name,
            vectors_config={
                DENSE_VECTOR_NAME: models.VectorParams(
                    size=DENSE_DIMENSIONS, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={SPARSE_VECTOR_NAME: models.SparseVectorParams()},
        )
    else:
        print(f"  collection {name} exists - upserting in place (pass --recreate to rebuild)")


def ensure_payload_indexes(client, name):
    """Without these the registry facet, document deletion and the booster all fail."""
    for field in KEYWORD_INDEXED_FIELDS:
        try:
            client.create_payload_index(
                collection_name=name,
                field_name=field,
                field_schema=models.PayloadSchemaType.KEYWORD,
                wait=True,
            )
            print(f"  payload index ready: {field} (keyword)")
        except Exception as e:
            print(f"  payload index note for {field} (safe if it already exists): {e}")


def main():
    parser = argparse.ArgumentParser(description="Index the chunked corpus into Qdrant.")
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="DESTRUCTIVE: drop the collection and rebuild it from scratch.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would happen without contacting Qdrant.",
    )
    parser.add_argument("--batch-size", type=int, default=None)
    args = parser.parse_args()

    load_dotenv()
    params = load_params()
    chunk_cfg = params["chunk"]
    embed_cfg = params["embed"]
    collection = embed_cfg["collection_name"]
    url, api_key = resolve_target(embed_cfg)
    batch_size = args.batch_size or embed_cfg.get("batch_size", 100)

    chunk_path = chunk_cfg["output_path"]
    if not os.path.exists(chunk_path):
        print(
            f"ERROR: {chunk_path} not found. Run the ingest and chunk stages first "
            f"(dvc repro), which build it from data/raw_pdfs.",
            file=sys.stderr,
        )
        return 1

    with open(chunk_path, "rb") as f:
        chunks = pickle.load(f)

    print(f"Target      : {url}")
    print(f"Collection  : {collection}")
    print(f"Chunks      : {len(chunks)} from {chunk_path}")
    print(f"Mode        : {'RECREATE (destructive)' if args.recreate else 'upsert (idempotent)'}")

    is_remote = "localhost" not in url and "127.0.0.1" not in url
    if is_remote and not api_key:
        print(
            "WARNING: no API key resolved for a remote cluster; the connection will "
            "likely be rejected. Set QDRANT_CLOUD_API_KEY.",
            file=sys.stderr,
        )

    if args.dry_run:
        print("\n--dry-run: nothing was sent to Qdrant.")
        return 0

    if args.recreate:
        confirm = input(
            f"\nThis DELETES every point in {collection}. "
            f"Type the collection name to confirm: "
        )
        if confirm.strip() != collection:
            print("Aborted.")
            return 1

    dense_name = embed_cfg["model_name"]
    sparse_name = embed_cfg.get("sparse_model", "Qdrant/bm25")
    print(f"\nLoading embedding models ({dense_name} + {sparse_name})...")
    dense_model = TextEmbedding(dense_name)
    # Match the width the serving path and the existing corpus use, or newly indexed
    # chunks get vectors inconsistent with everything already stored.
    HybridRetriever._widen_window(dense_model)
    sparse_model = SparseTextEmbedding(sparse_name)

    client = QdrantClient(url=url, api_key=api_key, timeout=120)
    ensure_collection(client, collection, args.recreate)

    total = len(chunks)
    print(f"\nEmbedding and upserting {total} chunks in batches of {batch_size}...")
    for start in range(0, total, batch_size):
        batch = chunks[start : start + batch_size]
        texts = [c.page_content for c in batch]
        dense_vectors = list(dense_model.embed(texts))
        sparse_vectors = list(sparse_model.embed(texts))

        points = []
        for i, chunk in enumerate(batch):
            points.append(
                models.PointStruct(
                    # Positional ids keep the run idempotent: re-running overwrites the
                    # same points instead of duplicating the corpus.
                    id=start + i,
                    vector={
                        DENSE_VECTOR_NAME: dense_vectors[i].tolist(),
                        SPARSE_VECTOR_NAME: models.SparseVector(
                            indices=sparse_vectors[i].indices.tolist(),
                            values=sparse_vectors[i].values.tolist(),
                        ),
                    },
                    payload={
                        "page_content": chunk.page_content,
                        "metadata": chunk.metadata,
                    },
                )
            )

        client.upsert(collection_name=collection, points=points, wait=True)
        done = start + len(batch)
        print(f"  {done}/{total} ({done / total * 100:.1f}%)")

    ensure_payload_indexes(client, collection)

    info = client.get_collection(collection)
    print(f"\nCollection {collection} ready:")
    print(f"  points : {info.points_count}")
    print(f"  status : {info.status}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
