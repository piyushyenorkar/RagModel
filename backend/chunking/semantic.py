"""
Strategy 2: Semantic chunking.
Embed sentences, split where cosine similarity between consecutive
sentence embeddings drops below a threshold (topic boundary detection).
"""

import re
import numpy as np
from backend.models import Chunk
from backend.chunking import ChunkStrategy, register_strategy


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using regex (handles Hindi punctuation too)."""
    # Split on period, question mark, exclamation, or Hindi danda (।)
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


@register_strategy
class SemanticChunking(ChunkStrategy):
    """Split by meaning — detects topic boundaries via embedding similarity drops."""
    
    strategy_name = "semantic"
    
    def __init__(self, similarity_threshold: float = 0.5, min_chunk_sentences: int = 2):
        self.similarity_threshold = similarity_threshold
        self.min_chunk_sentences = min_chunk_sentences
        self._model = None
    
    def _get_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        return self._model
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        dot = np.dot(a, b)
        norm = np.linalg.norm(a) * np.linalg.norm(b)
        if norm == 0:
            return 0.0
        return float(dot / norm)
    
    def chunk(self, text: str, passage_id: str, **kwargs) -> list[Chunk]:
        sentences = _split_sentences(text)
        
        if len(sentences) <= self.min_chunk_sentences:
            return [Chunk(
                text=text,
                strategy="semantic",
                passage_id=passage_id,
                chunk_index=0,
                metadata={"sentence_count": len(sentences), **kwargs}
            )]
        
        # Embed all sentences
        model = self._get_model()
        embeddings = model.encode(sentences, show_progress_bar=False)
        
        # Find split points where similarity drops
        split_points = [0]
        for i in range(1, len(embeddings)):
            sim = self._cosine_similarity(embeddings[i - 1], embeddings[i])
            if sim < self.similarity_threshold:
                split_points.append(i)
        split_points.append(len(sentences))
        
        # Build chunks from split points
        chunks = []
        for idx in range(len(split_points) - 1):
            start = split_points[idx]
            end = split_points[idx + 1]
            chunk_sentences = sentences[start:end]
            
            if not chunk_sentences:
                continue
            
            chunk_text = " ".join(chunk_sentences)
            chunks.append(Chunk(
                text=chunk_text,
                strategy="semantic",
                passage_id=passage_id,
                chunk_index=idx,
                metadata={
                    "sentence_count": len(chunk_sentences),
                    "start_sentence": start,
                    "end_sentence": end,
                    **kwargs,
                }
            ))
        
        return chunks if chunks else [Chunk(
            text=text,
            strategy="semantic",
            passage_id=passage_id,
            chunk_index=0,
            metadata={"sentence_count": len(sentences), **kwargs}
        )]
