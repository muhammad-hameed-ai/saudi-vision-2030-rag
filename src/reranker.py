"""
Cross-Encoder Reranker: takes the top-k retrieved chunks from hybrid search
and re-scores them using a cross-encoder model to produce a tighter, more
relevant top-n for the LLM context window.

Optimized for ultra-low-RAM environments (Render Free Tier / 512MB):
  - Replaced heavy PyTorch/sentence-transformers with fastembed's ONNX runtime.
  - Implements forced garbage collection (gc) after model initialization to clear spikes.
"""

import os
import gc
import warnings
from typing import List

# Suppress noisy ONNX/tokenizer warnings before any model imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

from fastembed.rerank.cross_encoder import TextCrossEncoder
from src.retriever import RetrievedChunk


class Reranker:
    """
    Lazy-loaded ONNX cross-encoder reranker.
    Model is loaded on first call to avoid blocking server startup.
    """

    DEFAULT_MODEL = "Xenova/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _get_model(self) -> TextCrossEncoder:
        if self._model is None:
            print(f"[Reranker] Loading lightweight ONNX model: {self.model_name}")
            self._model = TextCrossEncoder(model_name=self.model_name)
            print("[Reranker] Model loaded successfully.")
            
            # Force garbage collection immediately after the model loads to clear
            # initialization spikes and keep the footprint under 400MB.
            gc.collect()
            print("[Reranker] GC sweep completed.")
            
        return self._model

    def rerank(
        self,
        query: str,
        chunks: List[RetrievedChunk],
        top_k: int = 5,
    ) -> List[RetrievedChunk]:
        """
        Re-score chunks using the ONNX cross-encoder and return the top-k
        most relevant ones, sorted by reranker score (descending).
        """
        if not chunks:
            return []

        model = self._get_model()

        # Build list of strings representing the document content
        documents = [chunk.content for chunk in chunks]

        # Get relevance scores directly via fastembed
        # fastembed takes the query and a list of documents and returns an iterable of results
        results = list(model.rerank(query, documents))

        # Attach reranker scores and sort descending
        scored_chunks = list(zip(chunks, [res.score for res in results]))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Update the score field with the reranker score and return top-k
        result = []
        for chunk, reranker_score in scored_chunks[:top_k]:
            chunk.score = round(float(reranker_score), 4)
            result.append(chunk)

        return result
