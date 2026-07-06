"""
Hybrid Retriever: Dense (MiniLM) + Sparse (BM25) search with Qdrant's
Universal Query API and Reciprocal Rank Fusion (RRF).
"""

import os
from dataclasses import dataclass, field
from langchain_huggingface import HuggingFaceEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models


class QdrantUnavailableError(Exception):
    """Raised when Qdrant is unreachable."""
    pass


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with metadata."""
    content: str
    source: str
    page: int
    section: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Two-stage retriever:
      1. Dense cosine similarity via sentence-transformers/all-MiniLM-L6-v2
      2. Sparse BM25 keyword matching via Qdrant/bm25 (fastembed)
      3. Fused via Reciprocal Rank Fusion (RRF)
    """

    COLLECTION_NAME = "saudi_vision_2030"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, qdrant_url: str = None):
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_URL", "http://localhost:6333")
        self._client = None
        self._dense_model = None
        self._sparse_model = None

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            try:
                self._client = QdrantClient(url=self.qdrant_url, timeout=10)
            except Exception as e:
                raise QdrantUnavailableError(f"Cannot connect to Qdrant at {self.qdrant_url}: {e}")
        return self._client

    def _get_dense_model(self):
        if self._dense_model is None:
            print("Loading dense embedding model: sentence-transformers/all-MiniLM-L6-v2")
            self._dense_model = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={"device": "cpu"},
            )
        return self._dense_model

    def _get_sparse_model(self):
        if self._sparse_model is None:
            print("Loading sparse BM25 model: Qdrant/bm25")
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return self._sparse_model

    def health_check(self) -> bool:
        """Check if Qdrant is reachable and the collection exists."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            return info.points_count > 0
        except Exception:
            return False

    def get_collection_info(self) -> dict:
        """Return collection metadata for the /info endpoint."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            return {
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"points_count": 0, "status": "unavailable"}

    def retrieve(self, query: str, k: int = 20) -> list[RetrievedChunk]:
        """
        Run hybrid search: dense + sparse prefetch, fused with RRF.
        Returns top-k RetrievedChunk objects sorted by fused score.
        """
        try:
            client = self._get_client()
        except Exception as e:
            raise QdrantUnavailableError(str(e))

        try:
            # Generate dense query vector
            dense_vector = self._get_dense_model().embed_query(query)

            # Generate sparse query vector
            sparse_result = list(self._get_sparse_model().embed([query]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )

            # Hybrid search with prefetch + RRF fusion
            results = client.query_points(
                collection_name=self.COLLECTION_NAME,
                prefetch=[
                    models.Prefetch(
                        query=dense_vector,
                        using=self.DENSE_VECTOR_NAME,
                        limit=k,
                    ),
                    models.Prefetch(
                        query=sparse_vector,
                        using=self.SPARSE_VECTOR_NAME,
                        limit=k,
                    ),
                ],
                query=models.FusionQuery(fusion=models.Fusion.RRF),
                limit=k,
                with_payload=True,
            )

            chunks = []
            for point in results.points:
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                chunks.append(RetrievedChunk(
                    content=payload.get("page_content", ""),
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page", 0),
                    section=metadata.get("section", "General"),
                    score=round(point.score, 4) if point.score else 0.0,
                    metadata=metadata,
                ))
            return chunks

        except (ConnectionError, TimeoutError, OSError) as e:
            raise QdrantUnavailableError(f"Qdrant query failed: {e}")
        except Exception as e:
            # Re-raise unexpected errors as-is for debugging
            raise
