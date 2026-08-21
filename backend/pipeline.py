"""
Pipeline orchestrator — wires all stages together.
transcribe → retrieve → guardrail check → (generate OR abstain)
Every stage's latency gets recorded into stage_latencies_ms.
This is the "harness" — structured state flow with error handling.
"""

import time
import logging
from backend.models import PipelineState
from backend.stt import transcribe, SarvamSTTError
from backend.retrieval import retrieve
from backend.guardrails import check_transcript, check_retrieval_confidence, check_groundedness
from backend.generation import generate, GroqGenerationError

logger = logging.getLogger(__name__)


async def transcribe_stage(state: PipelineState) -> PipelineState:
    """Stage 1: Convert voice to text using Sarvam AI."""
    try:
        result = await transcribe(
            audio_bytes=state.audio_bytes,
            filename=state.audio_path or "audio.webm",
        )
        state.transcript = result.transcript
        state.stage_latencies_ms["stt"] = result.latency_ms
        logger.info(f"[STT] transcript='{state.transcript[:60]}...' latency={result.latency_ms:.1f}ms")
    except SarvamSTTError as e:
        logger.error(f"[STT] Failed: {e}")
        state.abstained = True
        state.abstain_reason = f"Speech-to-text failed: {str(e)}"
        state.stage_latencies_ms["stt"] = 0.0
    return state


async def retrieve_stage(state: PipelineState) -> PipelineState:
    """Stage 2: Embed query and search Qdrant. THIS IS THE 200ms PIPELINE."""
    try:
        result = await retrieve(
            query_text=state.transcript,
            strategy=state.strategy,
            top_k=5,
        )
        state.retrieved_chunks = result.chunks
        state.top_similarity = result.top_similarity
        state.stage_latencies_ms["retrieval"] = result.latency_ms
        logger.info(
            f"[Retrieval] chunks={len(result.chunks)} "
            f"top_sim={result.top_similarity:.3f} latency={result.latency_ms:.1f}ms"
        )
    except Exception as e:
        logger.error(f"[Retrieval] Failed: {e}")
        state.abstained = True
        state.abstain_reason = f"Retrieval failed: {str(e)}"
        state.stage_latencies_ms["retrieval"] = 0.0
    return state


async def guardrail_stage(state: PipelineState) -> PipelineState:
    """Stage 3: Check if we should answer or abstain."""
    t0 = time.perf_counter()
    
    # Check 1: Transcript quality
    transcript_check = check_transcript(state.transcript)
    if not transcript_check.passed:
        state.abstained = True
        state.abstain_reason = transcript_check.abstain_reason
        state.stage_latencies_ms["guardrail"] = (time.perf_counter() - t0) * 1000
        logger.info(f"[Guardrail] Abstained: {state.abstain_reason}")
        return state
    
    # Check 2: Retrieval confidence
    confidence_check = check_retrieval_confidence(
        state.top_similarity,
        state.retrieved_chunks,
    )
    if not confidence_check.passed:
        state.abstained = True
        state.abstain_reason = confidence_check.abstain_reason
        state.stage_latencies_ms["guardrail"] = (time.perf_counter() - t0) * 1000
        logger.info(f"[Guardrail] Abstained: {state.abstain_reason}")
        return state
    
    state.stage_latencies_ms["guardrail"] = (time.perf_counter() - t0) * 1000
    logger.info(f"[Guardrail] Passed — proceeding to generation")
    return state


async def generate_stage(state: PipelineState) -> PipelineState:
    """Stage 4: Generate answer using Groq LLM."""
    try:
        result = await generate(
            question=state.transcript,
            chunks=state.retrieved_chunks,
        )
        state.answer = result.answer
        state.stage_latencies_ms["generation"] = result.latency_ms
        
        # Post-generation groundedness check
        score, is_grounded = check_groundedness(result.answer, state.retrieved_chunks)
        state.groundedness_score = score
        
        if not is_grounded:
            logger.warning(f"[Guardrail] Low groundedness score: {score:.2f}")
            # Don't abstain, but flag it — let the user see the answer with the score
        
        logger.info(
            f"[Generation] answer='{state.answer[:60]}...' "
            f"latency={result.latency_ms:.1f}ms groundedness={score:.2f}"
        )
    except GroqGenerationError as e:
        logger.error(f"[Generation] Failed: {e}")
        state.abstained = True
        state.abstain_reason = f"Generation failed: {str(e)}"
        state.stage_latencies_ms["generation"] = 0.0
    return state


async def run_pipeline(state: PipelineState) -> PipelineState:
    """
    Run the full voice-to-answer pipeline.
    
    Flow: STT → Retrieve → Guardrail → Generate (or Abstain)
    """
    t0 = time.perf_counter()
    
    logger.info(f"[Pipeline] Starting — strategy={state.strategy}")
    
    # Stage 1: Speech-to-Text
    state = await transcribe_stage(state)
    if state.abstained:
        state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
        return state
    
    # Stage 2: Retrieval (THE 200ms PART)
    state = await retrieve_stage(state)
    if state.abstained:
        state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
        return state
    
    # Stage 3: Guardrail check
    state = await guardrail_stage(state)
    if state.abstained:
        state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
        return state
    
    # Stage 4: Generate answer
    state = await generate_stage(state)
    
    state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
    
    logger.info(
        f"[Pipeline] Complete — total={state.stage_latencies_ms['total']:.1f}ms "
        f"abstained={state.abstained}"
    )
    
    return state


async def run_pipeline_text(query: str, strategy: str = "fixed") -> PipelineState:
    """
    Run the pipeline with text input (skip STT).
    Useful for testing and benchmarking.
    """
    t0 = time.perf_counter()
    
    state = PipelineState(transcript=query, strategy=strategy)
    
    # Skip STT, go directly to retrieval
    state = await retrieve_stage(state)
    if state.abstained:
        state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
        return state
    
    state = await guardrail_stage(state)
    if state.abstained:
        state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
        return state
    
    state = await generate_stage(state)
    state.stage_latencies_ms["total"] = (time.perf_counter() - t0) * 1000
    
    return state
