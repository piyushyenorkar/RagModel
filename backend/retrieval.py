"""
Retrieval module — embeds query text and searches Qdrant.
Times the retrieval precisely with time.perf_counter() — this latency
is what gets reported for the 200ms requirement.
"""

import os
import time
import logging
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from dotenv import load_dotenv
from backend.models import Chunk, RetrievalResult

load_dotenv()
logger = logging.getLogger(__name__)

# ── Globals (loaded once, reused) ────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
QDRANT_PATH = os.getenv("QDRANT_PATH", os.path.join(BASE_DIR, "qdrant_local"))
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "msmarco_hi")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "paraphrase-multilingual-MiniLM-L12-v2")

_embedding_model: SentenceTransformer | None = None
_qdrant_client: QdrantClient | None = None


def get_embedding_model():
    """Lazy-load the embedding model (runs on CPU, no network call)."""
    global _embedding_model
    if _embedding_model is None:
        logger.info(f"Loading embedding model: {EMBEDDING_MODEL}")
        from sentence_transformers import SentenceTransformer
        _embedding_model = SentenceTransformer(EMBEDDING_MODEL)
        logger.info("Embedding model loaded successfully")
    return _embedding_model


def get_qdrant_client() -> QdrantClient:
    """Lazy-load the Qdrant client (local/embedded mode)."""
    global _qdrant_client
    if _qdrant_client is None:
        logger.info(f"Connecting to Qdrant at: {QDRANT_PATH}")
        _qdrant_client = QdrantClient(path=QDRANT_PATH)
        logger.info("Qdrant client ready")
    return _qdrant_client


def embed_text(text: str) -> list[float]:
    """Embed a single text string. Returns 384-dim vector."""
    model = get_embedding_model()
    embedding = model.encode(text, show_progress_bar=False)
    return embedding.tolist()


async def retrieve(
    query_text: str,
    strategy: str = "fixed",
    top_k: int = 5,
    filter_strategy: bool = True,
) -> RetrievalResult:
    """
    Embed the query and search Qdrant for relevant chunks.
    
    This is the function we time precisely — it's the 200ms number.
    
    Args:
        query_text: The transcribed question text
        strategy: Which chunking strategy to filter by
        top_k: Number of results to return
        filter_strategy: Whether to filter by strategy name in Qdrant
    
    Returns:
        RetrievalResult with chunks, top similarity score, and latency
    """
    t0 = time.perf_counter()
    
    # Step 1: Embed the query (local model, no network call)
    query_vector = embed_text(query_text)
    
    # Step 2: Search Qdrant
    client = get_qdrant_client()
    
    search_filter = None
    if filter_strategy:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="strategy",
                    match=MatchValue(value=strategy),
                )
            ]
        )
    
    response = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=top_k,
        query_filter=search_filter,
    )
    results = response.points
    
    latency_ms = (time.perf_counter() - t0) * 1000
    
    # Step 3: Convert Qdrant results to Chunk models
    chunks = []
    top_similarity = 0.0
    
    for hit in results:
        payload = hit.payload or {}
        
        # For sentence-window strategy, use the expanded window text for generation
        chunk_text = payload.get("text", "")
        if strategy == "window" and "window_text" in payload.get("metadata", {}):
            chunk_text = payload["metadata"]["window_text"]
        
        chunks.append(Chunk(
            text=chunk_text,
            strategy=payload.get("strategy", strategy),
            passage_id=payload.get("passage_id", ""),
            chunk_index=payload.get("chunk_index", 0),
            metadata=payload.get("metadata", {}),
        ))
        
        if hit.score > top_similarity:
            top_similarity = hit.score
    
    logger.info(
        f"Retrieval ({strategy}): {len(chunks)} chunks, "
        f"top_sim={top_similarity:.3f}, latency={latency_ms:.1f}ms"
    )
    
    return RetrievalResult(
        chunks=chunks,
        top_similarity=top_similarity,
        latency_ms=latency_ms,
        strategy_used=strategy,
    )


def retrieve_sync(
    query_text: str,
    strategy: str = "fixed",
    top_k: int = 5,
    filter_strategy: bool = True,
) -> RetrievalResult:
    """Synchronous version for benchmarking."""
    t0 = time.perf_counter()
    
    query_vector = embed_text(query_text)
    client = get_qdrant_client()
    
    search_filter = None
    if filter_strategy:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="strategy",
                    match=MatchValue(value=strategy),
                )
            ]
        )
    
    results = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=top_k,
        query_filter=search_filter,
    )
    
    latency_ms = (time.perf_counter() - t0) * 1000
    
    chunks = []
    top_similarity = 0.0
    
    for hit in results:
        payload = hit.payload or {}
        chunk_text = payload.get("text", "")
        if strategy == "window" and "window_text" in payload.get("metadata", {}):
            chunk_text = payload["metadata"]["window_text"]
        
        chunks.append(Chunk(
            text=chunk_text,
            strategy=payload.get("strategy", strategy),
            passage_id=payload.get("passage_id", ""),
            chunk_index=payload.get("chunk_index", 0),
            metadata=payload.get("metadata", {}),
        ))
        
        if hit.score > top_similarity:
            top_similarity = hit.score
    
    return RetrievalResult(
        chunks=chunks,
        top_similarity=top_similarity,
        latency_ms=latency_ms,
        strategy_used=strategy,
    )
