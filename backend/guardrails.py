"""
Guardrails — knows when *not* to answer.
Two checks, cheap and fast, no extra API calls needed:
1. Retrieval confidence threshold — abstain if top similarity is too low
2. Post-generation groundedness check — lexical overlap verification
"""

import re
import time
import logging
from backend.models import Chunk, GuardrailResult

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

SIMILARITY_THRESHOLD = 0.35  # Below this → abstain, don't generate
GROUNDEDNESS_THRESHOLD = 0.15  # Below this → flag answer as ungrounded
MIN_TRANSCRIPT_LENGTH = 3  # Characters — reject empty/noise transcripts


def _tokenize(text: str) -> set[str]:
    """Simple word tokenizer that works for Hindi and English."""
    # Remove punctuation, split on whitespace
    words = re.findall(r'[\w\u0900-\u097F]+', text.lower())
    return set(words)


# ── Pre-retrieval guardrails ─────────────────────────────────────────────────

def check_transcript(transcript: str | None) -> GuardrailResult:
    """
    Check if the transcript is valid enough to proceed.
    Catches: empty audio, noise-only recordings, extremely short inputs.
    """
    t0 = time.perf_counter()
    
    if not transcript or len(transcript.strip()) < MIN_TRANSCRIPT_LENGTH:
        return GuardrailResult(
            passed=False,
            abstain_reason="आपका प्रश्न समझ में नहीं आया। कृपया फिर से बोलें। (Your question was not clear. Please speak again.)",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    
    return GuardrailResult(
        passed=True,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ── Post-retrieval guardrails ────────────────────────────────────────────────

def check_retrieval_confidence(
    top_similarity: float,
    chunks: list[Chunk],
) -> GuardrailResult:
    """
    If the best-matching chunk's similarity score is below the threshold,
    the query isn't grounded in the dataset → abstain immediately.
    """
    t0 = time.perf_counter()
    
    if not chunks:
        return GuardrailResult(
            passed=False,
            abstain_reason="इस डेटासेट में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है। (I don't have enough information in this dataset to answer that.)",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    
    if top_similarity < SIMILARITY_THRESHOLD:
        return GuardrailResult(
            passed=False,
            abstain_reason=f"प्रश्न डेटासेट के दायरे से बाहर है (confidence: {top_similarity:.2f} < {SIMILARITY_THRESHOLD}). (Question is outside the dataset's scope.)",
            latency_ms=(time.perf_counter() - t0) * 1000,
        )
    
    return GuardrailResult(
        passed=True,
        latency_ms=(time.perf_counter() - t0) * 1000,
    )


# ── Post-generation guardrails ───────────────────────────────────────────────

def check_groundedness(
    answer: str,
    chunks: list[Chunk],
) -> tuple[float, bool]:
    """
    Check if the generated answer is grounded in the retrieved chunks.
    Uses lexical overlap — what fraction of answer words appear in the context.
    
    Returns:
        (groundedness_score, is_grounded) tuple
    """
    if not answer or not chunks:
        return 0.0, False
    
    answer_tokens = _tokenize(answer)
    if not answer_tokens:
        return 0.0, False
    
    # Combine all chunk texts
    context_text = " ".join(c.text for c in chunks)
    context_tokens = _tokenize(context_text)
    
    if not context_tokens:
        return 0.0, False
    
    # Calculate overlap
    overlap = answer_tokens & context_tokens
    
    # Remove common stop words from consideration (Hindi + English)
    stop_words = {
        "है", "हैं", "का", "के", "की", "में", "से", "को", "पर", "और", "एक",
        "यह", "वह", "इस", "उस", "ने", "हो", "कर", "या", "the", "is", "a",
        "an", "and", "or", "of", "to", "in", "for", "it", "this", "that",
        "not", "with", "on", "as", "by", "at", "be", "was", "are", "were",
        "नहीं", "जो", "तो", "भी", "ही", "कि", "जब", "अगर", "लेकिन",
    }
    
    meaningful_answer = answer_tokens - stop_words
    meaningful_overlap = overlap - stop_words
    
    if not meaningful_answer:
        return 1.0, True  # Only stop words in answer — consider it grounded
    
    score = len(meaningful_overlap) / len(meaningful_answer)
    is_grounded = score >= GROUNDEDNESS_THRESHOLD
    
    logger.info(f"Groundedness: {score:.2f} ({'grounded' if is_grounded else 'UNGROUNDED'})")
    
    return score, is_grounded
