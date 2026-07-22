"""
Saudi Vision 2030 Policy Intelligence Hub — ONNX Reranker Module
Handles cross-encoder relevance scoring with dynamic return-type extraction.
"""

import os
import gc
import logging
import traceback
from typing import List, Any

logger = logging.getLogger("vision2030.reranker")

class Reranker:
    def __init__(self, model_name: str = "Xenova/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model = None
        self._tokenizer = None

    def _load_model(self):
        """Lazy loader for ONNX cross-encoder model."""
        if self._model is None:
            logger.info(f"[Reranker] Loading lightweight ONNX model: {self.model_name}")
            try:
                from fastembed.rerank import TextReRank
                self._model = TextReRank(model_name=self.model_name)
                logger.info("[Reranker] Model loaded successfully.")
            except Exception as e:
                logger.error(f"[Reranker] Failed to load FastEmbed reranker: {e}")
                raise e

    def _extract_score(self, res: Any) -> float:
        """Polymorphic score extractor handling floats, dicts, objects, and tuples."""
        if isinstance(res, (int, float)):
            return float(res)
        if hasattr(res, "score"):
            return float(res.score)
        if isinstance(res, dict):
            return float(res.get("score", res.get("relevance_score", 0.0)))
        if isinstance(res, (tuple, list)) and len(res) > 1:
            return float(res[1])
        try:
            return float(res)
        except (ValueError, TypeError):
            return 0.0

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """Reranks retrieved document chunks using query cross-attention."""
        if not chunks:
            return []

        self._load_model()

        try:
            documents = [getattr(c, 'content', str(c)) for c in chunks]
            
            # Execute cross-encoder scoring
            raw_results = self._model.rerank(query=query, documents=documents)
            
            # Extract numerical scores safely across all potential return types
            extracted_scores = [self._clean_score_value(res) for res in raw_results]

            # Assign score back to chunk objects
            for chunk, score in zip(chunks, extracted_scores):
                chunk.score = score

            # Sort descending by relevance score
            sorted_chunks = sorted(
                chunks, 
                key=lambda x: getattr(x, 'score', 0.0), 
                reverse=True
            )
            
            return sorted_chunks[:top_k]

        except Exception as e:
            logger.error(f"[Reranker Exception] Error during scoring:\n{traceback.format_exc()}")
            # Safe Fallback: Return original chunks without crashing
            return chunks[:top_k]

    def _clean_score_value(self, res: Any) -> float:
        """Safely unpacks FastEmbed result objects or raw floats."""
        if isinstance(res, (int, float)):
            return float(res)
        if hasattr(res, "score"):
            return float(res.score)
        if isinstance(res, dict):
            return float(res.get("score", 0.0))
        return 0.0