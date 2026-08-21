"""
FastAPI application — the single /ask endpoint.
Accepts multipart audio file upload, runs the full pipeline,
returns structured JSON with all latency numbers.
"""

import time
import logging
import os
import sys

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

load_dotenv()

from backend.models import PipelineState, AskResponse
from backend.pipeline import run_pipeline, run_pipeline_text

# ── Logging setup ────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── FastAPI app ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="Voice-Enabled RAG — HH Goa 2026",
    description="Speak a question, get a grounded answer from MS MARCO Hindi dataset.",
    version="1.0.0",
)

# CORS — allow frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup event — preload models ──────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Preload the embedding model and Qdrant client on startup."""
    logger.info("Starting Voice-Enabled RAG server...")
    try:
        from backend.retrieval import get_embedding_model, get_qdrant_client
        # get_embedding_model()  # Preload — takes ~2s first time
        # get_qdrant_client()
        logger.info("Models and Qdrant preloaded successfully")
    except Exception as e:
        logger.warning(f"Preload warning (may not be indexed yet): {e}")


# ── Health check ─────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "healthy", "service": "voice-rag-hhgoa"}


# ── Main endpoint ────────────────────────────────────────────────────────────

@app.post("/ask", response_model=AskResponse)
async def ask(
    audio: UploadFile = File(...),
    strategy: str = Form(default="fixed"),
):
    """
    Voice-to-answer endpoint.
    
    Accepts an audio file (wav/mp3/webm), transcribes it,
    retrieves relevant chunks, and generates a grounded answer.
    
    Args:
        audio: Audio file upload (wav, mp3, webm, ogg)
        strategy: Chunking strategy to use (fixed/semantic/window/metadata)
    
    Returns:
        AskResponse with transcript, answer, latencies, and abstain info
    """
    # Validate strategy
    valid_strategies = ["fixed", "semantic", "window", "metadata"]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Must be one of: {valid_strategies}"
        )
    
    # Read the audio bytes
    try:
        audio_bytes = await audio.read()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read audio: {e}")
    
    if len(audio_bytes) < 100:
        raise HTTPException(status_code=400, detail="Audio file is too small or empty")
    
    logger.info(
        f"Received audio: {audio.filename}, size={len(audio_bytes)} bytes, strategy={strategy}"
    )
    
    # Build initial pipeline state
    state = PipelineState(
        audio_path=audio.filename or "audio.webm",
        audio_bytes=audio_bytes,
        strategy=strategy,
    )
    
    # Run the full pipeline
    state = await run_pipeline(state)
    
    # Build response
    return AskResponse(
        transcript=state.transcript or "",
        answer=state.answer or state.abstain_reason or "",
        abstained=state.abstained,
        abstain_reason=state.abstain_reason,
        strategy_used=strategy,
        retrieval_latency_ms=state.stage_latencies_ms.get("retrieval", 0.0),
        total_latency_ms=state.stage_latencies_ms.get("total", 0.0),
        stage_latencies_ms=state.stage_latencies_ms,
        groundedness_score=state.groundedness_score,
    )


# ── Text-only endpoint (for testing without mic) ────────────────────────────

@app.post("/ask-text", response_model=AskResponse)
async def ask_text(
    query: str = Form(...),
    strategy: str = Form(default="fixed"),
):
    """
    Text-based query endpoint (skips STT, for testing).
    """
    valid_strategies = ["fixed", "semantic", "window", "metadata"]
    if strategy not in valid_strategies:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid strategy: {strategy}. Must be one of: {valid_strategies}"
        )
    
    state = await run_pipeline_text(query=query, strategy=strategy)
    
    return AskResponse(
        transcript=state.transcript or query,
        answer=state.answer or state.abstain_reason or "",
        abstained=state.abstained,
        abstain_reason=state.abstain_reason,
        strategy_used=strategy,
        retrieval_latency_ms=state.stage_latencies_ms.get("retrieval", 0.0),
        total_latency_ms=state.stage_latencies_ms.get("total", 0.0),
        stage_latencies_ms=state.stage_latencies_ms,
        groundedness_score=state.groundedness_score,
    )


# ── Run with uvicorn ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    # Need to pass import string when using reload=True
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
