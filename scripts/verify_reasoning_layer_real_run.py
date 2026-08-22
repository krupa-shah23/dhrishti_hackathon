"""
verify_reasoning_layer_real_run.py

Runs the REAL pipeline (process_video_live -> P2P3Bridge -> event_adapter ->
reasoning_layer.get_llm_reasoning) against a real video clip and captures
EVERY finalized event's exact LLM input and exact output, via a spy wrapper
around get_llm_reasoning: it calls the real, unmodified function, records
(input, output), and returns the same result unmodified -- get_llm_reasoning()
itself is never touched, this only observes calls event_adapter.py makes.

Saves one JSON row per finalized event (event_id, timing, the exact fields
sent to the LLM, risk_level/explanation actually used on the Event, whether
that came from a real LLM call or the rule-based fallback, and the other
rule-based fields for cross-referencing) to
reasoning_layer_real_run_<video_id>.json in the repo root.

Usage:
    uv run python scripts/verify_reasoning_layer_real_run.py <video_path> <video_id>

Example:
    uv run python scripts/verify_reasoning_layer_real_run.py \
        "data/college_dataset/01.Candidate was found using a mobile phone in the examination hall..mkv" \
        clip1
"""
import argparse
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")

from src.integration import event_adapter
from src.integration.reasoning_layer import get_llm_reasoning as _real_get_llm_reasoning


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("video_path")
    parser.add_argument("video_id")
    args = parser.parse_args()

    captured = []

    def _spy_get_llm_reasoning(event):
        result = _real_get_llm_reasoning(event)
        captured.append({"input": dict(event), "output": result})
        return result

    # Monkeypatch the name event_adapter.py resolves at call time -- the real
    # get_llm_reasoning() function is untouched, this only observes calls.
    event_adapter.get_llm_reasoning = _spy_get_llm_reasoning

    from src.integration.p1_p2_tracker import process_video_live

    finalized_events = []

    def on_events_ready(events):
        finalized_events.extend(events)

    frame_count = 0
    for _ in process_video_live(args.video_path, clip_name=args.video_id, on_events_ready=on_events_ready):
        frame_count += 1

    print(f"\nProcessed {frame_count} frames. {len(finalized_events)} finalized events, "
          f"{len(captured)} get_llm_reasoning() calls captured.")

    if len(captured) != len(finalized_events):
        print(f"WARNING: mismatch between captured LLM calls ({len(captured)}) and "
              f"finalized events ({len(finalized_events)}) -- zip-by-index below may misalign, "
              f"treat per-row correctness with caution.")

    rows = []
    for i, ev in enumerate(finalized_events):
        call = captured[i] if i < len(captured) else None
        llm_input = call["input"] if call else None
        llm_output = call["output"] if call else None
        is_fallback_template = ev.explanation == f"Activity {ev.event_type} detected on track {getattr(ev, 'track_id', None) or 'unknown'}"
        rows.append({
            "event_id": ev.event_id,
            "start": ev.start,
            "end": ev.end,
            "duration": ev.duration,
            "input_fields_sent_to_llm": llm_input,
            "llm_call_succeeded": llm_output is not None,
            "risk_level": ev.risk_level,
            "explanation": ev.explanation,
            "explanation_is_rule_based_template": is_fallback_template,
            "severity_score": ev.severity_score,
            "risk_label": ev.risk_label,
            "color_tag": ev.color_tag,
            "object_detected": ev.object_detected,
            "object_confidence": ev.object_confidence,
            "repetition_count": ev.repetition_count,
            "invigilator_excluded": ev.invigilator_excluded,
            "avg_motion_intensity": ev.avg_motion_intensity,
            "peak_intensity": ev.peak_intensity,
            "mog2_foreground_ratio": ev.mog2_foreground_ratio,
            "exam_phase": ev.exam_phase,
            "thumbnail_path": ev.thumbnail_path,
            "heatmap_ref": ev.heatmap_ref,
        })

    out_path = REPO_ROOT / f"reasoning_layer_real_run_{args.video_id}.json"
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_success = sum(1 for r in rows if r["llm_call_succeeded"])
    n_events = len(rows)
    print(f"\n=== SUMMARY ===")
    print(f"Total finalized events: {n_events}")
    print(f"LLM calls that succeeded: {n_success}/{n_events}")
    print(f"Events with explanation missing/empty (real bug if >0): "
          f"{sum(1 for r in rows if not r['explanation'])}")
    print(f"Events with risk_level missing (real bug if llm_call_succeeded is also True): "
          f"{sum(1 for r in rows if r['risk_level'] is None and r['llm_call_succeeded'])}")
    print(f"Saved to: {out_path}")


if __name__ == "__main__":
    main()
