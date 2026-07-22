"""
Hybrid Retriever: Dense (MiniLM API) + Sparse (BM25) search with Qdrant's
Universal Query API and Reciprocal Rank Fusion (RRF).
"""

import os
import logging
import traceback
from dataclasses import dataclass, field
from typing import List, Optional
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient, models

logger = logging.getLogger("vision2030.retriever")


class QdrantUnavailableError(Exception):
    """Raised when Qdrant cluster infrastructure is unreachable."""
    pass


@dataclass
class RetrievedChunk:
    """A single retrieved document chunk with production metadata mapping."""
    content: str
    source: str
    page: int
    section: str
    score: float
    metadata: dict = field(default_factory=dict)


class HybridRetriever:
    """
    Two-stage production-grade retriever:
      1. Serverless Dense inference via sentence-transformers/all-MiniLM-L6-v2 (Low RAM)
      2. Sparse BM25 keyword matching via Qdrant/bm25 (fastembed)
      3. Dynamic multi-vector fusion via Reciprocal Rank Fusion (RRF)
    """

    COLLECTION_NAME = "saudi_vision_2030"
    DENSE_VECTOR_NAME = "dense"
    SPARSE_VECTOR_NAME = "sparse"

    def __init__(self, qdrant_url: Optional[str] = None):
        # Dynamically read cloud variables first, fallback gracefully to localhost
        self.qdrant_url = qdrant_url or os.getenv("QDRANT_CLOUD_URL") or os.getenv("QDRANT_URL", "http://localhost:6333")
        self.qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY")
        
        self._client = None
        self._dense_model = None
        self._sparse_model = None

    def _get_client(self) -> QdrantClient:
        """Initializes a secured QdrantClient instance with token authentication mapping."""
        if self._client is None:
            try:
                # Inject token key for secure Qdrant Cloud handshakes
                self._client = QdrantClient(
                    url=self.qdrant_url, 
                    api_key=self.qdrant_api_key,
                    timeout=60.0
                )
            except Exception as e:
                logger.error(f"[Retriever] Failed to bind to Qdrant cluster:\n{traceback.format_exc()}")
                raise QdrantUnavailableError(f"Cannot bind socket to Qdrant cluster host: {e}")
        return self._client

    def _get_dense_model(self) -> HuggingFaceEndpointEmbeddings:
        """Initializes serverless cloud embedding client to eliminate local RAM usage spikes."""
        if self._dense_model is None:
            print("[INFO] Mounting serverless inference handler: sentence-transformers/all-MiniLM-L6-v2")
            self._dense_model = HuggingFaceEndpointEmbeddings(
                model="sentence-transformers/all-MiniLM-L6-v2",
                huggingfacehub_api_token=os.getenv("HF_TOKEN"),
            )
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Loads lightweight BM25 sparse tokenizer layer."""
        if self._sparse_model is None:
            print("[INFO] Initializing sparse BM25 vocabulary index Matrix...")
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return self._sparse_model

    def health_check(self) -> bool:
        """Verifies operational status of the upstream vector network pipeline."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            return info.points_count > 0
        except Exception:
            return False

    def get_collection_info(self) -> dict:
        """Fetches runtime structural stats directly from remote cluster maps."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            return {
                "points_count": info.points_count,
                "status": str(info.status),
            }
        except Exception:
            return {"points_count": 0, "status": "unavailable"}

    def retrieve(self, query: str, k: int = 20) -> List[RetrievedChunk]:
        """
        Executes non-blocking hybrid vector extraction fused via Reciprocal Rank Fusion.
        """
        try:
            client = self._get_client()
        except Exception as e:
            logger.error(f"[Retriever] Qdrant Client initialization failed:\n{traceback.format_exc()}")
            raise QdrantUnavailableError(str(e))

        try:
            # 1. Dispatch cloud request for dense matrix vectors
            dense_vector = self._get_dense_model().embed_query(query)

            # 2. Local vocabulary tokenizer tokenization for keywords
            sparse_result = list(self._get_sparse_model().embed([query]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )

            # 3. Parallel dual-query prefetch step mapped to unified RRF compiler
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
            logger.error(f"[Retriever] Network error communicating with Qdrant:\n{traceback.format_exc()}")
            raise QdrantUnavailableError(f"Network error communicating with Qdrant: {e}")
        except Exception as e:
            logger.error(f"[Retriever] Unexpected pipeline break inside retriever block:\n{traceback.format_exc()}")
            raise RuntimeError(f"Unexpected pipeline trace break inside retriever block: {e}")