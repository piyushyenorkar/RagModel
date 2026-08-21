"""
Strategy 4: Metadata-aware chunking.
Tags every chunk with passage_id, query relevance label (from MS MARCO),
and source language. Uses metadata for filtering/boosting at retrieval time.
"""

from backend.models import Chunk
from backend.chunking import ChunkStrategy, register_strategy


@register_strategy
class MetadataAwareChunking(ChunkStrategy):
    """
    Chunks passages while preserving rich metadata from the dataset.
    MS MARCO already has relevance judgments — we tag chunks with them
    so retrieval can filter/boost using metadata fields.
    """
    
    strategy_name = "metadata"
    
    def __init__(self, chunk_size: int = 150, overlap_ratio: float = 0.15):
        self.chunk_size = chunk_size  # slightly smaller for metadata-rich chunks
        self.overlap = int(chunk_size * overlap_ratio)
    
    def chunk(self, text: str, passage_id: str, **kwargs) -> list[Chunk]:
        words = text.split()
        if not words:
            return []
        
        # Extract metadata from kwargs
        relevance_label = kwargs.get("relevance_label", 0)
        source_lang = kwargs.get("source_lang", "hi")
        query_text = kwargs.get("query_text", "")
        is_selected = kwargs.get("is_selected", False)
        
        chunks = []
        start = 0
        chunk_index = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk_text = " ".join(words[start:end])
            
            chunks.append(Chunk(
                text=chunk_text,
                strategy="metadata",
                passage_id=passage_id,
                chunk_index=chunk_index,
                metadata={
                    "relevance_label": relevance_label,
                    "source_lang": source_lang,
                    "is_selected": is_selected,
                    "query_text": query_text,
                    "word_count": end - start,
                    "passage_total_words": len(words),
                }
            ))
            
            chunk_index += 1
            start += self.chunk_size - self.overlap
            
            if start >= len(words) or len(words) - start < self.overlap:
                break
        
        return chunks
