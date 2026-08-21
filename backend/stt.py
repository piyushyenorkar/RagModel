"""
Sarvam AI Speech-to-Text wrapper.
Uses the sarvamai SDK with retry logic (3 attempts, exponential backoff)
on 429/503 errors. Returns typed TranscriptionResult.
"""

import os
import time
import logging
import httpx
import base64
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from dotenv import load_dotenv
from backend.models import TranscriptionResult

load_dotenv()
logger = logging.getLogger(__name__)

SARVAM_API_KEY = os.getenv("SARVAM_API_KEY", "")
SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"


class SarvamSTTError(Exception):
    """Custom error for Sarvam STT failures."""
    pass


class SarvamRetryableError(SarvamSTTError):
    """Retryable Sarvam errors (429, 503, etc.)."""
    pass


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=10),
    retry=retry_if_exception_type(SarvamRetryableError),
    reraise=True,
)
async def transcribe(audio_bytes: bytes, filename: str = "audio.wav") -> TranscriptionResult:
    """
    Transcribe audio using Sarvam AI's speech-to-text API.
    
    Args:
        audio_bytes: Raw audio file bytes (wav/mp3/webm)
        filename: Original filename for content-type detection
    
    Returns:
        TranscriptionResult with transcript, language, and latency
    """
    if not SARVAM_API_KEY:
        raise SarvamSTTError("SARVAM_API_KEY not set in environment variables")
    
    t0 = time.perf_counter()
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Determine content type from filename
            content_type = "audio/wav"
            if filename.endswith(".mp3"):
                content_type = "audio/mpeg"
            elif filename.endswith(".webm"):
                content_type = "audio/webm"
            elif filename.endswith(".ogg"):
                content_type = "audio/ogg"
            
            response = await client.post(
                SARVAM_STT_URL,
                headers={
                    "api-subscription-key": SARVAM_API_KEY,
                    "Content-Type": "application/json",
                },
                json={
                    "url": f"data:{content_type};base64,{base64.b64encode(audio_bytes).decode('utf-8')}",
                    "model": "saaras:v2",
                    "language_code": "hi-IN",
                    "with_timestamps": False,
                }
            )
        
        latency_ms = (time.perf_counter() - t0) * 1000
        
        if response.status_code in (429, 503):
            raise SarvamRetryableError(
                f"Sarvam API returned {response.status_code}: {response.text}"
            )
        
        if response.status_code != 200:
            raise SarvamSTTError(
                f"Sarvam API error {response.status_code}: {response.text}"
            )
        
        data = response.json()
        transcript = data.get("transcript", "")
        language = data.get("language_code", "hi-IN")
        
        logger.info(f"STT completed in {latency_ms:.1f}ms: '{transcript[:50]}...'")
        
        return TranscriptionResult(
            transcript=transcript,
            language_code=language,
            latency_ms=latency_ms,
        )
    
    except httpx.TimeoutException:
        latency_ms = (time.perf_counter() - t0) * 1000
        raise SarvamRetryableError(f"Sarvam API timeout after {latency_ms:.0f}ms")
    except (httpx.HTTPError, Exception) as e:
        if isinstance(e, (SarvamSTTError, SarvamRetryableError)):
            raise
        latency_ms = (time.perf_counter() - t0) * 1000
        raise SarvamSTTError(f"STT failed after {latency_ms:.0f}ms: {str(e)}")
