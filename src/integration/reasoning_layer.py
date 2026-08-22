"""
reasoning_layer.py

LLM reasoning layer over already-finalized events. Reasons over structured
signals P2/P2P3Bridge/event_adapter.py have ALREADY computed (detection,
motion, repetition, invigilator-exclusion, timing) to produce a smarter
risk_level and natural-language explanation than the current rule-based
severity_score/risk_label formula in event_adapter.py.

This does NOT replace detection/tracking/motion, and does NOT gate/suppress
events -- it's a pure enrichment on top of an already-complete event. On any
failure (disabled, missing/bad API key, network/timeout, malformed JSON), it
returns None and the caller keeps its existing rule-based severity_score/
risk_label/explanation untouched (see event_adapter.py's wiring).

Model: openai/gpt-oss-20b via Groq (OpenAI-compatible client, `groq` SDK).
Do NOT use llama-3.3-70b-versatile or llama-3.1-8b-instant -- both
deprecated by Groq as of this writing.

Rate limits: Groq free tier is ~30 requests/minute as of last check. This
is a prototype layered on a per-event call -- single attempt, no retry/
backoff. If this needs to scale up, add real rate limiting/backoff then,
not here.

GROQ_API_KEY is read from the environment only -- never hardcoded, never
logged, never printed. Failures are logged by exception type/message only.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("reasoning_layer")

MODEL = "openai/gpt-oss-20b"
REQUEST_TIMEOUT_SECONDS = 10.0

SYSTEM_PROMPT = """You are assisting exam-hall invigilation review. You will be given a single JSON object describing one already-detected candidate event (motion detected by a computer-vision pipeline, optionally with an object like a phone or paper chit, plus tracking/timing signals). This event has ALREADY been flagged by the detection pipeline -- your job is only to judge how suspicious it looks and explain why, not to decide whether it happened.

Weigh the signals the way an experienced human invigilator would:
- A single brief glance or one isolated instance is usually NOT suspicious on its own.
- A REPEATED pattern (high repetition_count), especially one that correlates with the invigilator being absent/excluded from the scene (invigilator_excluded=true), IS more suspicious.
- Object detection (object_detected=true) with reasonable confidence raises risk; a detected object with very low confidence is weaker evidence than one with high confidence.
- Longer duration and higher motion intensity generally indicate more deliberate activity than a brief flicker.
- exam_phase can matter: activity during "distribution" or "submission" phases (handing out/collecting papers) is often normal handling, not cheating; the same activity during "core" exam time is more notable.

Do NOT invent details that are not present in the input. If the input is sparse (most fields null/missing), say so plainly in your explanation rather than fabricating specifics -- e.g. "Limited signal available; only a single brief motion event with no object detection or repetition data."

Respond with STRICT JSON only -- no preamble, no markdown code fences, no extra keys, exactly this shape:
{"risk_level": "low"|"medium"|"high", "explanation": "1-2 sentence justification"}
"""

# Fields event_adapter.py has actually computed for a finalized event by the
# time this is called -- nothing here is invented or upstream-unavailable.
_INPUT_FIELDS = (
    "seat_id",
    "duration",
    "activities",
    "object_detected",
    "object_confidence",
    "avg_motion_intensity",
    "motion_intensity",
    "repetition_count",
    "invigilator_excluded",
    "exam_phase",
    "related_event_id",
)


def _reasoning_enabled() -> bool:
    """
    On/off switch, default True. Flip ENABLE_LLM_REASONING to false/0/no to
    kill this instantly (e.g. mid-demo) without a redeploy. Checked at call
    time (not import time) so it can be toggled without restarting.
    """
    return os.environ.get("ENABLE_LLM_REASONING", "true").strip().lower() not in ("false", "0", "no")


def _build_llm_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """
    Pulls the subset of _INPUT_FIELDS actually present (and not None) on
    `event`. Missing/None fields are omitted entirely, never fabricated as
    a default like 0 or "unknown" -- the prompt tells the model to call out
    sparse input rather than treating an omission as a real zero value.
    """
    return {f: event[f] for f in _INPUT_FIELDS if event.get(f) is not None}


def get_llm_reasoning(event: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Calls the Groq-hosted LLM to reason over one finalized event's already-
    computed signals.

    Returns {"risk_level": "low"|"medium"|"high", "explanation": str} on
    success, or None on ANY failure -- disabled via flag, missing API key,
    package not installed, network/timeout error, or a response that isn't
    valid/well-shaped JSON. Never raises: this is an optional enrichment on
    top of an already-working rule-based pipeline, not a dependency it can
    fail on.
    """
    if not _reasoning_enabled():
        return None

    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        logger.warning("reasoning_layer: GROQ_API_KEY not set, skipping LLM reasoning")
        return None

    try:
        from groq import Groq
    except ImportError:
        logger.warning("reasoning_layer: groq package not installed, skipping LLM reasoning")
        return None

    llm_input = _build_llm_input(event)

    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(llm_input)},
            ],
            temperature=0.2,
            response_format={"type": "json_object"},
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        raw = response.choices[0].message.content
    except Exception as e:
        # str(e) from the groq/httpx client doesn't include the key, but
        # stay defensive anyway: log exception type/message only, never
        # api_key, headers, or the client object itself.
        logger.warning(f"reasoning_layer: Groq call failed ({type(e).__name__}: {e})")
        return None

    try:
        parsed = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        logger.warning(f"reasoning_layer: malformed JSON from LLM ({type(e).__name__}): {raw!r}")
        return None

    risk_level = parsed.get("risk_level") if isinstance(parsed, dict) else None
    explanation = parsed.get("explanation") if isinstance(parsed, dict) else None
    if risk_level not in ("low", "medium", "high") or not isinstance(explanation, str) or not explanation.strip():
        logger.warning(f"reasoning_layer: LLM response failed shape check: {parsed!r}")
        return None

    return {"risk_level": risk_level, "explanation": explanation}
