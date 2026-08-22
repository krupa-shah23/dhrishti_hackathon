"""
test_reasoning_layer.py

Verification for reasoning_layer.get_llm_reasoning(). Two parts:

1. test_real_examples: runs 5 realistic events (mirroring documented entries
   in data/ground_truth/ground_truth_events.csv -- clip1's isolated copying
   event, clip6's clustered phone-out burst, clip7's seat-exchange sequence,
   clip8's seat_12 paper copying, and a deliberately sparse event) through
   get_llm_reasoning() and prints input/output side by side for a human to
   eyeball. Requires a real GROQ_API_KEY -- skipped otherwise (this hits a
   real, rate-limited external API; it is intentionally NOT part of a
   no-network CI run).

2. test_failure_paths: exercises the kill switch, a missing key, and a
   deliberately invalid key (a real call that Groq rejects with 401) and
   asserts get_llm_reasoning() returns None cleanly in all three cases,
   never raising. This part runs regardless of whether a real key is
   configured.

Run directly for the printed input/output comparison:
    uv run python -m src.integration.test_reasoning_layer
"""
import json
import logging
import os

import pytest

from .reasoning_layer import get_llm_reasoning, _build_llm_input

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

EVENTS = [
    {
        "name": "clip1_e2 - single copying instant (00:00:12-00:00:16)",
        "event": {
            "seat_id": None,
            "duration": 4.0,
            "activities": ["copying"],
            "object_detected": True,
            "object_confidence": 0.87,
            "avg_motion_intensity": 0.32,
            "repetition_count": 1,
            "invigilator_excluded": False,
            "exam_phase": "core",
            "related_event_id": None,
        },
    },
    {
        "name": "clip6_e9 - clustered phone-out burst (01:23:42-01:26:44, 8 instances)",
        "event": {
            "seat_id": None,
            "duration": 182.0,
            "activities": ["phone_out"],
            "object_detected": True,
            "object_confidence": 0.74,
            "avg_motion_intensity": 0.55,
            "repetition_count": 8,
            "invigilator_excluded": True,
            "exam_phase": "core",
            "related_event_id": None,
        },
    },
    {
        "name": "clip7_e4 - seat exchange, no object (01:13:11)",
        "event": {
            "seat_id": None,
            "duration": 91.0,
            "activities": ["talking", "seat_exchange"],
            "object_detected": False,
            "object_confidence": None,
            "avg_motion_intensity": 0.41,
            "repetition_count": 1,
            "invigilator_excluded": False,
            "exam_phase": "core",
            "related_event_id": "c7_e2",
        },
    },
    {
        "name": "clip8_e1 - seat_12 paper copying (00:00:26-00:01:22)",
        "event": {
            "seat_id": "seat_12",
            "duration": 56.0,
            "activities": ["copying"],
            "object_detected": True,
            "object_confidence": 0.91,
            "avg_motion_intensity": 0.28,
            "repetition_count": 1,
            "invigilator_excluded": False,
            "exam_phase": "core",
            "related_event_id": None,
        },
    },
    {
        "name": "sparse event - brief motion only, almost nothing else known",
        "event": {
            "seat_id": None,
            "duration": 1.5,
            "activities": [],
            "object_detected": False,
            "object_confidence": None,
            "avg_motion_intensity": None,
            "repetition_count": 0,
            "invigilator_excluded": False,
            "exam_phase": None,
            "related_event_id": None,
        },
    },
]


@pytest.mark.skipif(not os.environ.get("GROQ_API_KEY"), reason="needs a real GROQ_API_KEY")
def test_real_examples():
    for case in EVENTS:
        llm_input = _build_llm_input(case["event"])
        result = get_llm_reasoning(case["event"])
        print(f"\n--- {case['name']} ---")
        print("INPUT :", json.dumps(llm_input))
        print("OUTPUT:", json.dumps(result) if result else "None")
        assert result is not None, f"expected a real completion for: {case['name']}"
        assert result["risk_level"] in ("low", "medium", "high")
        assert isinstance(result["explanation"], str) and result["explanation"].strip()


def test_failure_paths():
    sample_event = EVENTS[0]["event"]

    # Kill switch: no network call attempted at all.
    os.environ["ENABLE_LLM_REASONING"] = "false"
    try:
        assert get_llm_reasoning(sample_event) is None
    finally:
        del os.environ["ENABLE_LLM_REASONING"]

    # Missing key.
    had_key = os.environ.pop("GROQ_API_KEY", None)
    try:
        assert get_llm_reasoning(sample_event) is None

        # Invalid key -- a real call Groq rejects with 401, caught cleanly.
        os.environ["GROQ_API_KEY"] = "gsk_this_is_a_deliberately_invalid_key_for_testing"
        assert get_llm_reasoning(sample_event) is None
    finally:
        os.environ.pop("GROQ_API_KEY", None)
        if had_key:
            os.environ["GROQ_API_KEY"] = had_key


if __name__ == "__main__":
    print("=" * 78)
    print("PART 1: 5 realistic input/output examples")
    print("=" * 78)
    if os.environ.get("GROQ_API_KEY"):
        test_real_examples()
    else:
        print("(skipped -- no GROQ_API_KEY in environment)")

    print("\n" + "=" * 78)
    print("PART 2: failure path tests")
    print("=" * 78)
    test_failure_paths()
    print("All failure-path assertions passed.")
