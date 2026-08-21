import json
import pandas as pd


def load_p1_region_scores(csv_path):
    """P1's real fused motion signal — confirm exact column names once she sends it."""
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def load_p2_detections(json_path):
    """P2's real detect_objects() output (JSON format) — confirm exact shape once she sends it."""
    with open(json_path) as f:
        return json.load(f)


def load_p2_real_detections(csv_path):
    """
    P2's real detect_objects() output, CSV format (as actually sent for clip3).
    Columns: event_id, window_start_s, window_end_s, class, confidence, note
    """
    df = pd.read_csv(csv_path)
    return df.to_dict("records")


def build_real_event(seat_id, start_time, end_time, video_id, camera_id,
                      avg_motion_intensity=0.0, peak_intensity=0.0,
                      mog2_foreground_ratio=0.0, repetition_count=1,
                      object_detected=None, object_confidence=None,
                      invigilator_excluded=False, exam_phase="mid",
                      audio_path=None):
    """Assembles one real Event dict matching the frozen schema."""
    return {
        "event_id": f"{video_id}_{seat_id}_{int(start_time)}",
        "video_id": video_id,
        "camera_id": camera_id,
        "seat_id": seat_id,
        "start_time": start_time,
        "end_time": end_time,
        "grid_cells": [],
        "avg_motion_intensity": avg_motion_intensity,
        "peak_intensity": peak_intensity,
        "mog2_foreground_ratio": mog2_foreground_ratio,
        "duration": max(end_time - start_time, 0.0),
        "repetition_count": repetition_count,
        "object_detected": object_detected,
        "object_confidence": object_confidence,
        "invigilator_excluded": invigilator_excluded,
        "exam_phase": exam_phase,
        "exam_mode": "CBT",
        "severity_score": None,
        "risk_label": None,
        "explanation": None,
        "audio_corroborated": None,
        "intervention_detected": False,
        "related_event_id": None,
        "event_type": "anomaly",
        "audio_path": audio_path,
    }


def attach_detections_to_events(events: list, detections: list, time_tolerance=1.0):
    """
    Cross-reference P2's real JSON-style detections onto P3 events by seat_id + time proximity.
    Updates object_detected / object_confidence on each event in-place.
    """
    for event in events:
        best_match = None
        best_conf = -1
        for det in detections:
            det_seat = det.get("seat_id")
            det_time = det.get("timestamp")
            det_conf = det.get("confidence") or det.get("conf")
            if det_seat != event.get("seat_id") or det_time is None:
                continue
            if abs(det_time - event["start_time"]) <= time_tolerance:
                if det_conf and det_conf > best_conf:
                    best_match = det
                    best_conf = det_conf
        if best_match:
            event["object_detected"] = best_match.get("class") or best_match.get("object_class")
            event["object_confidence"] = best_conf
    return events


def build_events_from_p1_scores(region_scores, video_id, camera_id, exam_phase="mid"):
    """
    Converts P1's raw region_scores records into real Event dicts.
    Adjust field access below once P1's real column names are confirmed.
    """
    events = []
    for r in region_scores:
        seat_id = r.get("seat_id")
        start_time = r.get("timestamp") or r.get("start_time")
        end_time = r.get("end_time", start_time + 1.0 if start_time is not None else 0.0)
        score = r.get("score", 0.0)
        events.append(build_real_event(
            seat_id=seat_id,
            start_time=start_time,
            end_time=end_time,
            video_id=video_id,
            camera_id=camera_id,
            avg_motion_intensity=score,
            peak_intensity=score,
            exam_phase=exam_phase,
        ))
    return events


def merge_real_detections_into_gt(gt_csv_path, detections_csv_path, video_id, output_path=None):
    """
    Overlays P2's real window-based detection results (CSV format) onto
    ground_truth_labels.csv rows for matching video_id/time-window overlap.
    Only updates rows with a real (non-empty) detection — confirmed-empty
    windows are left as-is, not force-overwritten to None.
    """
    gt_df = pd.read_csv(gt_csv_path)
    det_df = pd.read_csv(detections_csv_path)

    if "object_detected" not in gt_df.columns:
        gt_df["object_detected"] = None
    if "object_confidence" not in gt_df.columns:
        gt_df["object_confidence"] = 0.0

    updated_count = 0
    for _, det in det_df.iterrows():
        cls = det.get("class")
        if pd.isna(cls) or cls == "":
            continue  # empty detection window — skip, don't overwrite GT with a false "confirmed empty"

        overlap_mask = (
            (gt_df["video_id"] == video_id) &
            (gt_df["start_time"] < det["window_end_s"]) &
            (gt_df["end_time"] > det["window_start_s"])
        )
        gt_df.loc[overlap_mask, "object_detected"] = cls
        gt_df.loc[overlap_mask, "object_confidence"] = det["confidence"]
        updated_count += overlap_mask.sum()

    out_path = output_path or gt_csv_path
    gt_df.to_csv(out_path, index=False)
    print(f"Updated {updated_count} ground-truth rows with real detections from {video_id}")
    return gt_df


if __name__ == "__main__":
    # --- P2's real clip3 detection output — actually available now ---
    try:
        merge_real_detections_into_gt(
            "../../data/ground_truth_labels.csv",
            "../../data/p3_detection_output_clip3.csv",
            video_id="clip3",
        )
    except FileNotFoundError:
        print("[load_real_events] p3_detection_output_clip3.csv not found in data/ — place it there first.")

    # --- P1's real fused motion signal — still pending ---
    try:
        scores = load_p1_region_scores("../../data/p1_real_scores_clip1.csv")
        print(f"Loaded {len(scores)} real region scores")

        events = build_events_from_p1_scores(scores, video_id="clip1", camera_id="Camera12")
        print(f"Built {len(events)} real Event dicts")

        try:
            detections = load_p2_detections("../../data/p2_real_detections_clip1.json")
            print(f"Loaded {len(detections)} real detections")
            events = attach_detections_to_events(events, detections)
            print("Attached P2 detections to events")
        except FileNotFoundError:
            print("[load_real_events] P2's real JSON detections not yet available for clip1.")

        print("Sample event:", events[0] if events else "none")

    except FileNotFoundError:
        print("[load_real_events] P1's real file not yet available — waiting on handoff.")