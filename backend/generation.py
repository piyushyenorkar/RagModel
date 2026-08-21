"""
Groq LLM generation wrapper.
Strict "only answer from context" system prompt.
Retry logic with exponential backoff on rate limits.
"""

import os
import time
import logging
from groq import AsyncGroq
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from backend.models import Chunk, GenerationResult

load_dotenv()
logger = logging.getLogger(__name__)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

# ── System prompt — forces grounded answers only ─────────────────────────────

SYSTEM_PROMPT = """You are a helpful question-answering assistant. You MUST follow these rules strictly:

1. ONLY answer using the provided context passages. Do NOT use any outside knowledge.
2. If the context does not contain enough information to answer the question, say: "इस संदर्भ में इस प्रश्न का उत्तर उपलब्ध नहीं है। (The answer to this question is not available in the given context.)"
3. Answer in the same language as the question (Hindi if asked in Hindi, English if asked in English).
4. Keep answers concise, factual, and directly supported by the context.
5. Do NOT make up facts, speculate, or add information not present in the context.
6. If only a partial answer is possible from the context, provide what you can and clearly state what information is missing."""


def _build_prompt(question: str, chunks: list[Chunk]) -> str:
    """Build the generation prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Passage {i}]:\n{chunk.text}")
    
    context_block = "\n\n".join(context_parts)
    
    return f"""Context passages:
{context_block}

Question: {question}

Answer (based ONLY on the above context):"""


class GroqGenerationError(Exception):
    """Custom error for Groq generation failures."""
    pass


class GroqRetryableError(GroqGenerationError):
    """Retryable errors (rate limits, server errors)."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=15),
    retry=retry_if_exception_type(GroqRetryableError),
    reraise=True,
)
async def generate(question: str, chunks: list[Chunk]) -> GenerationResult:
    """
    Generate a grounded answer using Groq LLM.
    
    Args:
        question: The user's transcribed question
        chunks: Retrieved context chunks from Qdrant
    
    Returns:
        GenerationResult with answer and latency
    """
    if not GROQ_API_KEY:
        raise GroqGenerationError("GROQ_API_KEY not set in environment variables")
    
    if not chunks:
        return GenerationResult(
            answer="कोई संदर्भ उपलब्ध नहीं है। (No context available.)",
            latency_ms=0.0,
            groundedness_score=0.0,
        )
    
    t0 = time.perf_counter()
    
    try:
        client = AsyncGroq(api_key=GROQ_API_KEY)
        
        user_prompt = _build_prompt(question, chunks)
        
        response = await client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.3,  # Low temperature for factual, grounded answers
            max_tokens=512,
            top_p=0.9,
        )
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        answer = response.choices[0].message.content or ""
        
        logger.info(f"Generation completed in {latency_ms:.1f}ms: '{answer[:80]}...'")
        
        return GenerationResult(
            answer=answer.strip(),
            latency_ms=latency_ms,
            groundedness_score=0.0,  # Will be set by guardrails post-check
        )
    
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        error_str = str(e).lower()
        
        # Check if it's a rate limit or server error → retry
        if "rate_limit" in error_str or "429" in error_str or "503" in error_str:
            raise GroqRetryableError(f"Groq rate limited: {e}")
        
        raise GroqGenerationError(f"Generation failed after {latency_ms:.0f}ms: {e}")
