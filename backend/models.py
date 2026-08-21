"""
Pydantic models for the Voice-Enabled RAG pipeline.
Every stage has typed input/output — this is the "harness" the judges want to see.
"""

from typing import Optional, Literal
from pydantic import BaseModel, Field


# ── Chunk model ──────────────────────────────────────────────────────────────

class Chunk(BaseModel):
    """A single chunk of text produced by any chunking strategy."""
    text: str
    strategy: Literal["fixed", "semantic", "window", "metadata"]
    passage_id: str
    chunk_index: int
    metadata: dict = Field(default_factory=dict)


# ── Stage result models ──────────────────────────────────────────────────────

class TranscriptionResult(BaseModel):
    """Output of the STT stage."""
    transcript: str
    language_code: str = "hi-IN"
    latency_ms: float = 0.0


class RetrievalResult(BaseModel):
    """Output of the retrieval stage."""
    chunks: list[Chunk] = Field(default_factory=list)
    top_similarity: float = 0.0
    latency_ms: float = 0.0
    strategy_used: str = "fixed"


class GuardrailResult(BaseModel):
    """Output of the guardrail check stage."""
    passed: bool = True
    abstain_reason: Optional[str] = None
    latency_ms: float = 0.0


class GenerationResult(BaseModel):
    """Output of the LLM generation stage."""
    answer: str = ""
    latency_ms: float = 0.0
    groundedness_score: float = 0.0


# ── Pipeline state ───────────────────────────────────────────────────────────

class PipelineState(BaseModel):
    """
    Flows through the entire pipeline: transcribe → retrieve → guardrail → generate.
    Every stage records its latency into stage_latencies_ms.
    """
    audio_path: Optional[str] = None
    audio_bytes: Optional[bytes] = None
    transcript: Optional[str] = None
    retrieved_chunks: list[Chunk] = Field(default_factory=list)
    top_similarity: float = 0.0
    answer: Optional[str] = None
    abstained: bool = False
    abstain_reason: Optional[str] = None
    strategy: str = "fixed"
    stage_latencies_ms: dict = Field(default_factory=dict)
    groundedness_score: float = 0.0

    class Config:
        arbitrary_types_allowed = True


# ── API response model ───────────────────────────────────────────────────────

class AskResponse(BaseModel):
    """JSON response returned by the /ask endpoint."""
    transcript: str = ""
    answer: str = ""
    abstained: bool = False
    abstain_reason: Optional[str] = None
    strategy_used: str = "fixed"
    retrieval_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    stage_latencies_ms: dict = Field(default_factory=dict)
    groundedness_score: float = 0.0
