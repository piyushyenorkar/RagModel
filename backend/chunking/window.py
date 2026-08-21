"""
Strategy 3: Sentence-window (small-to-big) chunking.
Index individual sentences for precise matching, but when retrieved,
expand to return the sentence plus N neighbors on each side as context.
Best of both: precise search, rich generation context.
"""

import re
from backend.models import Chunk
from backend.chunking import ChunkStrategy, register_strategy


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences."""
    sentences = re.split(r'(?<=[.!?।])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]


@register_strategy
class SentenceWindowChunking(ChunkStrategy):
    """
    Index single sentences, but store the surrounding window in metadata
    so retrieval can expand context at query time.
    """
    
    strategy_name = "window"
    
    def __init__(self, window_size: int = 2):
        self.window_size = window_size  # sentences on each side
    
    def chunk(self, text: str, passage_id: str, **kwargs) -> list[Chunk]:
        sentences = _split_sentences(text)
        
        if not sentences:
            return []
        
        chunks = []
        for i, sentence in enumerate(sentences):
            # Build the expanded window context
            window_start = max(0, i - self.window_size)
            window_end = min(len(sentences), i + self.window_size + 1)
            window_text = " ".join(sentences[window_start:window_end])
            
            chunks.append(Chunk(
                text=sentence,  # Index the single sentence for precise matching
                strategy="window",
                passage_id=passage_id,
                chunk_index=i,
                metadata={
                    "window_text": window_text,  # Expanded context for generation
                    "window_start": window_start,
                    "window_end": window_end,
                    "total_sentences": len(sentences),
                    **kwargs,
                }
            ))
        
        return chunks
