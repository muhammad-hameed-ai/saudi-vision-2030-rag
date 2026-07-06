"""
Cross-Encoder Reranker: takes the top-k retrieved chunks from hybrid search
and re-scores them using a cross-encoder model to produce a tighter, more
relevant top-n for the LLM context window.
"""

from sentence_transformers import CrossEncoder
from src.retriever import RetrievedChunk


class Reranker:
    """
    Lazy-loaded cross-encoder reranker.
    Model is loaded on first call to avoid blocking server startup.
    """

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self, model_name: str = None):
        self.model_name = model_name or self.DEFAULT_MODEL
        self._model = None

    def _get_model(self) -> CrossEncoder:
        if self._model is None:
            print(f"Loading reranker model: {self.model_name}")
            self._model = CrossEncoder(self.model_name)
            print("Reranker model loaded.")
        return self._model

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int = 5,
    ) -> list[RetrievedChunk]:
        """
        Re-score chunks using the cross-encoder and return the top-k
        most relevant ones, sorted by reranker score (descending).

        Args:
            query: The user's original question.
            chunks: List of RetrievedChunk from hybrid search.
            top_k: Number of chunks to return after reranking.

        Returns:
            List of top_k RetrievedChunk objects with updated scores.
        """
        if not chunks:
            return []

        model = self._get_model()

        # Build (query, document) pairs for cross-encoder scoring
        pairs = [(query, chunk.content) for chunk in chunks]

        # Get relevance scores from the cross-encoder
        scores = model.predict(pairs)

        # Attach reranker scores and sort descending
        scored_chunks = list(zip(chunks, scores))
        scored_chunks.sort(key=lambda x: x[1], reverse=True)

        # Update the score field with the reranker score and return top-k
        result = []
        for chunk, reranker_score in scored_chunks[:top_k]:
            chunk.score = round(float(reranker_score), 4)
            result.append(chunk)

        return result
