# src/integration/p3_p4_bridge.py
from src.outputs_eval.logger import format_timestamp, format_roi_box


def _compute_motion_score(motion_area: float, frame_width: int, frame_height: int) -> float:
    frame_area = frame_width * frame_height
    return min(100.0, (motion_area / frame_area) * 100.0) if frame_area > 0 else 0.0


def p3_event_to_csv_row(
    event: dict,
    features: dict,
    risk: float,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> dict:
    """
    Maps one completed P2P3Bridge event + its P3 features/risk score
    into a P4 logger.log_event()-compatible row.
    """
    frame_index = event["end_frame"]  # one row per event, at its end_frame
    box = event["boxes"][-1]

    motion_score = _compute_motion_score(features.get("motion_area", 0.0), frame_width, frame_height)

    confidence = 0.0
    for m in event.get("metadata", []):
        if m.get("confidence") is not None:
            confidence = m["confidence"]

    return {
        "timestamp": format_timestamp(frame_index, fps),
        "frame_index": frame_index,
        "roi_box": format_roi_box(tuple(int(v) for v in box)),
        "motion_score": motion_score,
        "audio_level": features.get("audio_energy", 0.0),
        "risk_score": risk,
        "confidence": confidence,
    }


def p3_events_to_timeline_arrays(
    events_with_scores: list[tuple[dict, dict, float]],  # (event, features, risk)
    fps: float,
    frame_width: int,
    frame_height: int,
) -> tuple[list[float], list[float], list[float]]:
    """
    Unzips a list of (event, features, risk) tuples into the three parallel
    arrays plot_timeline() expects: timestamps (sec), motion_scores [0-100],
    risk_scores [0.0-1.0]. Sorted by end_frame.
    """
    rows = []
    for event, features, risk in events_with_scores:
        frame_index = event["end_frame"]
        ts_seconds = frame_index / fps
        motion_score = _compute_motion_score(features.get("motion_area", 0.0), frame_width, frame_height)
        rows.append((ts_seconds, motion_score, risk))

    rows.sort(key=lambda r: r[0])
    timestamps = [r[0] for r in rows]
    motion_scores = [r[1] for r in rows]
    risk_scores = [r[2] for r in rows]
    return timestamps, motion_scores, risk_scores