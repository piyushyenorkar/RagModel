"""
Strategy 1: Fixed-size chunking with overlap.
Baseline — split passages into ~200-token windows with 20% overlap.
"""

from backend.models import Chunk
from backend.chunking import ChunkStrategy, register_strategy


@register_strategy
class FixedSizeChunking(ChunkStrategy):
    """Fixed-size text windows with configurable overlap."""
    
    strategy_name = "fixed"
    
    def __init__(self, chunk_size: int = 200, overlap_ratio: float = 0.2):
        self.chunk_size = chunk_size  # in words (approximation of tokens)
        self.overlap = int(chunk_size * overlap_ratio)
    
    def chunk(self, text: str, passage_id: str, **kwargs) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunks.append(Chunk(
                text=chunk_text,
                strategy="fixed",
                passage_id=passage_id,
                chunk_index=chunk_index,
                metadata={
                    "word_count": end - start,
                    "start_word": start,
                    "end_word": end,
                    **kwargs,
                }
            ))
            
            chunk_index += 1
            start += self.chunk_size - self.overlap
            
            # Avoid creating a tiny trailing chunk
            if start >= len(words):
                break
            if len(words) - start < self.overlap:
                break
        
        return chunks
