"""
Hybrid Retriever: Dense (MiniLM API) + Sparse (BM25) search with Qdrant's
Universal Query API and Reciprocal Rank Fusion (RRF).
"""

import os
import logging
import traceback
import httpx
from dataclasses import dataclass, field
from typing import List, Optional
from fastembed import TextEmbedding, SparseTextEmbedding
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
        self.qdrant_api_key = os.getenv("QDRANT_CLOUD_API_KEY") or os.getenv("QDRANT_API_KEY")
        
        self._client = None
        self._dense_model = None
        self._sparse_model = None
        self._vision_sources = None

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
                
                # Zero-Touch Programmatic Payload Indexing
                try:
                    self._client.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name="metadata.section",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    self._client.create_payload_index(
                        collection_name=self.COLLECTION_NAME,
                        field_name="metadata.source",
                        field_schema=models.PayloadSchemaType.KEYWORD,
                    )
                    logger.info(f"[Retriever] Payload indexes verified/created for {self.COLLECTION_NAME}")
                except Exception as e:
                    # Safely handle case if index already exists or user lacks permissions
                    logger.info(f"[Retriever] Payload index setup note (safe to ignore if already exists): {e}")

            except Exception as e:
                logger.error(f"[Retriever] Failed to bind to Qdrant cluster:\n{traceback.format_exc()}")
                raise QdrantUnavailableError(f"Cannot bind socket to Qdrant cluster host: {e}")
        return self._client

    def _get_dense_model(self) -> TextEmbedding:
        """Initializes local FastEmbed dense embedding model (zero external API dependency)."""
        if self._dense_model is None:
            print("[INFO] Loading local dense embedding model: sentence-transformers/all-MiniLM-L6-v2")
            self._dense_model = TextEmbedding("sentence-transformers/all-MiniLM-L6-v2")
        return self._dense_model

    def _get_sparse_model(self) -> SparseTextEmbedding:
        """Loads lightweight BM25 sparse tokenizer layer."""
        if self._sparse_model is None:
            print("[INFO] Initializing sparse BM25 vocabulary index Matrix...")
            self._sparse_model = SparseTextEmbedding("Qdrant/bm25")
        return self._sparse_model

    def _get_vision_sources(self) -> List[str]:
        """
        Resolves the exact `metadata.source` values of the core Vision 2030 policy docs.

        `metadata.source` must carry a KEYWORD index because the document registry
        facets on it and delete_document filters on it with MatchValue. Qdrant allows
        only one index per field, and KEYWORD matching is whole-value only -- so no
        substring predicate (MatchText, or MatchAny on a bare token) can match here.
        We resolve the full values once and match them exactly instead.
        """
        if self._vision_sources is not None:
            return self._vision_sources
        try:
            facet_result = self._get_client().facet(
                collection_name=self.COLLECTION_NAME,
                key="metadata.source",
                limit=1000,
            )
            self._vision_sources = [
                hit.value for hit in (facet_result.hits or [])
                if isinstance(hit.value, str) and "vision2030" in hit.value.lower()
            ]
            logger.info(
                f"[Retriever] Policy booster resolved {len(self._vision_sources)} Vision 2030 source(s)."
            )
        except Exception as e:
            logger.warning(f"[Retriever] Could not resolve Vision 2030 sources for booster: {e}")
            self._vision_sources = []
        return self._vision_sources

    @staticmethod
    def _cosine_scores(query_vector, points) -> List[float]:
        """
        True cosine similarity between the query and each returned chunk.

        The score Qdrant returns from a FusionQuery is an RRF score -- sum(1/(60+rank))
        -- which depends only on rank position, so it is identical for every query and
        carries no relevance signal. The dense vectors come back on the same round trip,
        so we compute the real similarity locally instead.
        """
        import numpy as np
        qv = np.asarray(query_vector, dtype=np.float32)
        qn = float(np.linalg.norm(qv)) or 1.0
        scores = []
        for point in points:
            vec = None
            if isinstance(point.vector, dict):
                vec = point.vector.get(HybridRetriever.DENSE_VECTOR_NAME)
            elif point.vector is not None:
                vec = point.vector
            if vec is None:
                # Vector withheld by the server: fall back to the fused score.
                scores.append(float(point.score or 0.0))
                continue
            cv = np.asarray(vec, dtype=np.float32)
            cn = float(np.linalg.norm(cv)) or 1.0
            scores.append(max(0.0, min(1.0, float(np.dot(cv, qv)) / (cn * qn))))
        return scores

    def health_check(self) -> bool:
        """Verifies operational status of the upstream vector network pipeline without blocking/crashing."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            return info.points_count > 0
        except (httpx.ConnectError, ConnectionError, TimeoutError, OSError) as e:
            logger.warning(f"[Retriever] Health check network timeout/failure: {e}")
            return False
        except Exception as e:
            logger.warning(f"[Retriever] Health check unexpected failure: {e}")
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

    def get_telemetry_stats(self) -> dict:
        """O(1) high-performance telemetry lookup directly from Qdrant Cloud."""
        try:
            client = self._get_client()
            info = client.get_collection(self.COLLECTION_NAME)
            
            # Qdrant Facet API: Fetches unique payload counts instantly
            facet_result = client.facet(
                collection_name=self.COLLECTION_NAME,
                key="metadata.source",
                limit=1000
            )
            unique_pdfs = len(facet_result.hits or [])
            
            return {
                "points_count": info.points_count,
                "unique_sources": unique_pdfs
            }
        except Exception as e:
            logger.warning(f"Telemetry fetch failed: {e}")
            # Report the outage rather than inventing plausible-looking counts.
            return {
                "points_count": None,
                "unique_sources": None,
                "available": False,
            }

    def get_document_registry(self) -> list:
        """Retrieves a registry of all uniquely uploaded documents and their chunk counts."""
        try:
            client = self._get_client()
            facet_result = client.facet(
                collection_name=self.COLLECTION_NAME,
                key="metadata.source",
                limit=1000
            )
            registry = []
            if facet_result.hits:
                for hit in facet_result.hits:
                    registry.append({
                        "filename": hit.value,
                        "chunks": hit.count
                    })
            return registry
        except Exception as e:
            logger.error(f"Failed to fetch document registry: {e}")
            return []

    def delete_document(self, filename: str) -> bool:
        """Atomic vector deletion based on metadata.source matching."""
        try:
            client = self._get_client()
            client.delete(
                collection_name=self.COLLECTION_NAME,
                points_selector=models.Filter(
                    must=[
                        models.FieldCondition(
                            key="metadata.source",
                            match=models.MatchValue(value=filename)
                        )
                    ]
                )
            )
            logger.info(f"Successfully purged vectors for document: {filename}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete document {filename}: {e}")
            return False

    async def upsert_in_memory_chunks(self, chunks: List[dict]):
        """
        Asynchronously vectorizes and upserts a batch of chunk dictionaries.
        """
        import asyncio
        await asyncio.to_thread(self._upsert_in_memory_chunks_sync, chunks)

    def _upsert_in_memory_chunks_sync(self, chunks: List[dict]):
        if not chunks:
            return

        import uuid
        client = self._get_client()
        texts = [chunk["text"] for chunk in chunks]
        
        # 1. Generate Dense Vectors
        dense_vectors = list(self._get_dense_model().embed(texts))
        
        # 2. Generate Sparse Vectors
        sparse_embeddings = list(self._get_sparse_model().embed(texts))
        
        points = []
        for idx, chunk in enumerate(chunks):
            sparse = sparse_embeddings[idx]
            # Create a deterministic UUID based on chunk_id or generate random if missing
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk["metadata"].get("chunk_id", str(uuid.uuid4()))))
            
            payload = {
                "page_content": chunk["text"],
                "metadata": chunk["metadata"]
            }
            
            point = models.PointStruct(
                id=point_id,
                payload=payload,
                vector={
                    self.DENSE_VECTOR_NAME: dense_vectors[idx],
                    self.SPARSE_VECTOR_NAME: models.SparseVector(
                        indices=sparse.indices.tolist(),
                        values=sparse.values.tolist(),
                    ),
                }
            )
            points.append(point)
            
        client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=points
        )

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
            # 1. Generate dense embedding vector locally
            dense_vector = list(self._get_dense_model().embed([query]))[0]

            # 2. Local vocabulary tokenizer tokenization for keywords
            sparse_result = list(self._get_sparse_model().embed([query]))[0]
            sparse_vector = models.SparseVector(
                indices=sparse_result.indices.tolist(),
                values=sparse_result.values.tolist(),
            )

            # 3. Hybrid prefetch (dense + sparse) fused via Reciprocal Rank Fusion.
            #    The third leg over-indexes on the core policy documents to counteract
            #    term-frequency dominance from the long financial circulars.
            booster_should = [
                models.FieldCondition(
                    key="metadata.section",
                    match=models.MatchValue(value="General"),
                ),
            ]
            vision_sources = self._get_vision_sources()
            if vision_sources:
                booster_should.append(
                    models.FieldCondition(
                        key="metadata.source",
                        match=models.MatchAny(any=vision_sources),
                    )
                )

            base_prefetch = [
                # Primary dense query (semantic matches)
                models.Prefetch(query=dense_vector, using=self.DENSE_VECTOR_NAME, limit=k),
                # Primary sparse query (exact keyword matches)
                models.Prefetch(query=sparse_vector, using=self.SPARSE_VECTOR_NAME, limit=k),
            ]
            boosted_prefetch = base_prefetch + [
                models.Prefetch(
                    query=dense_vector,
                    using=self.DENSE_VECTOR_NAME,
                    filter=models.Filter(should=booster_should),
                    limit=max(1, k // 2),
                ),
            ]

            def _run(prefetch):
                # with_vectors returns the dense vectors on this same round trip so we
                # can compute real cosine similarity without a second query.
                return client.query_points(
                    collection_name=self.COLLECTION_NAME,
                    prefetch=prefetch,
                    query=models.FusionQuery(fusion=models.Fusion.RRF),
                    limit=k,
                    with_payload=True,
                    with_vectors=[self.DENSE_VECTOR_NAME],
                )

            try:
                results = _run(boosted_prefetch)
            except Exception as e:
                # A missing payload index on a booster field is recoverable: the two
                # primary legs still answer the query correctly on their own.
                status = getattr(getattr(e, "response", None), "status_code", None)
                if status == 400 or "400" in str(e) or "Index required" in str(e):
                    logger.warning(
                        f"[Retriever] Booster prefetch rejected ({e}). "
                        "Falling back to pure Dense/Sparse fusion."
                    )
                    results = _run(base_prefetch)
                else:
                    raise

            scores = self._cosine_scores(dense_vector, results.points)

            chunks = []
            for point, score in zip(results.points, scores):
                payload = point.payload or {}
                metadata = payload.get("metadata", {})
                chunks.append(RetrievedChunk(
                    content=payload.get("page_content", ""),
                    source=metadata.get("source", "unknown"),
                    page=metadata.get("page", 0),
                    section=metadata.get("section", "General"),
                    score=round(score, 4),
                    metadata=metadata,
                ))
            # Fusion orders by RRF rank; re-order by true semantic similarity so the
            # strongest chunks lead the context window handed to the LLM.
            chunks.sort(key=lambda c: c.score, reverse=True)
            return chunks

        except (httpx.ConnectError, ConnectionError, TimeoutError, OSError) as e:
            logger.error(f"[Retriever] Network error communicating with Qdrant Cloud:\n{traceback.format_exc()}")
            raise QdrantUnavailableError(f"Network error communicating with Qdrant Cloud: {e}")
        except Exception as e:
            logger.error(f"[Retriever] Unexpected pipeline break inside retriever block:\n{traceback.format_exc()}")
            raise RuntimeError(f"Unexpected pipeline trace break inside retriever block: {e}")