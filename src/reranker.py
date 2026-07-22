"""
Saudi Vision 2030 Policy Intelligence Hub — ONNX Reranker Module
Handles cross-encoder relevance scoring with dynamic return-type extraction and
graceful fallbacks for OOM/Load failures.
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
        self._is_degraded = False

    def _load_model(self):
        """Multi-path dynamic loader for ONNX cross-encoder model to handle upstream API drift."""
        if self._model is not None or self._is_degraded:
            return

        logger.info(f"[Reranker] Loading lightweight ONNX model: {self.model_name}")
        try:
            # Multi-path import resolution to combat FastEmbed API drift
            try:
                from fastembed.rerank.cross_encoder import TextCrossEncoder as TextReRank
            except ImportError:
                from fastembed.rerank import TextReRank

            self._model = TextReRank(model_name=self.model_name)
            logger.info("[Reranker] Model loaded successfully.")
            
            # Force GC sweep to drop initialization memory spikes
            gc.collect()
            
        except Exception as e:
            logger.error(f"[Reranker] FATAL: Failed to load FastEmbed reranker. Entering degraded fallback mode.\n{traceback.format_exc()}")
            self._model = None
            self._is_degraded = True

    def rerank(self, query: str, chunks: List[Any], top_k: int = 5) -> List[Any]:
        """Reranks retrieved document chunks using query cross-attention with safe fallbacks."""
        if not chunks:
            return []

        self._load_model()

        # Graceful Pass-Through: If model failed to load (e.g., OOM), return chunks as-is
        if self._is_degraded or self._model is None:
            logger.warning("[Reranker] Degraded mode active. Bypassing cross-encoder and returning raw chunks.")
            for chunk in chunks:
                chunk.score = 0.5  # Assign default neutral score
            return chunks[:top_k]

        try:
            documents = [getattr(c, 'content', str(c)) for c in chunks]
            
            # Execute cross-encoder scoring
            raw_results = list(self._model.rerank(query=query, documents=documents))
            
            # Extract numerical scores safely across all potential return types
            extracted_scores = [self._clean_score_value(res) for res in raw_results]

            # Assign score back to chunk objects safely
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
            logger.error(f"[Reranker Exception] Error during scoring pipeline:\n{traceback.format_exc()}")
            # Safe Fallback: Return original chunks without crashing
            for chunk in chunks:
                if not hasattr(chunk, 'score'):
                    chunk.score = 0.5
            return chunks[:top_k]

    def _clean_score_value(self, res: Any) -> float:
        """Bulletproof polymorphic score extractor handling floats, dicts, objects, and tuples."""
        try:
            if isinstance(res, (int, float)):
                return float(res)
            if hasattr(res, "score"):
                return float(res.score)
            if isinstance(res, dict):
                return float(res.get("score", res.get("relevance_score", 0.0)))
            if isinstance(res, (tuple, list)) and len(res) > 1:
                return float(res[1])
            return float(res)
        except (ValueError, TypeError, AttributeError):
            logger.warning(f"[Reranker] Score extraction failed for type {type(res)}. Defaulting to 0.0")
            return 0.0