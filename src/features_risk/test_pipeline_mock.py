# ============================================================
# test_pipeline_mock.py
# ============================================================
from mock_event_data import generate_region_scores
from segment_events import segment_events
from cluster_incidents import cluster_incidents
from link_related_events import link_related_events
from extract_features_v2 import extract_features
from severity_score import severity_score
from explain import explain
from checkpoint_event import get_connection, checkpoint_event


def run():
    scores = generate_region_scores(seat_ids=["Desk4", "Desk5"], duration_sec=300)
    raw = segment_events(scores, threshold=25, n_on=2, m_off=2)
    print(f"[1] segment_events -> {len(raw)} raw events")

    clustered = cluster_incidents(raw, time_gap=15)
    print(f"[2] cluster_incidents -> {len(clustered)} incidents")

    linked = link_related_events(clustered)
    print(f"[3] link_related_events -> done")

    conn = get_connection("test_pipeline.db")

    for ev in linked:
        # pad fields required by extract_features/severity_score/explain
        ev.setdefault("video_id", "mock_clip_01")
        ev.setdefault("camera_id", "cam1")
        ev.setdefault("avg_motion_intensity", sum(ev.get("scores", [0])) / max(len(ev.get("scores", [1])), 1))
        ev.setdefault("peak_intensity", max(ev.get("scores", [0]), default=0))
        ev.setdefault("mog2_foreground_ratio", 0.1)
        ev.setdefault("duration", ev["end_time"] - ev["start_time"])
        ev.setdefault("repetition_count", ev.get("sub_event_count", 1))
        ev.setdefault("object_detected", None)
        ev.setdefault("object_confidence", None)
        ev.setdefault("invigilator_excluded", False)
        ev.setdefault("exam_phase", "phase2")
        ev.setdefault("exam_mode", "CBT")
        ev.setdefault("audio_corroborated", None)
        ev.setdefault("intervention_detected", False)
        ev.setdefault("event_type", "anomaly")

        feats = extract_features(ev)
        score = severity_score(feats)
        ev["severity_score"] = score
        ev["risk_label"] = "Suspicious" if score > 0.5 else "Normal"
        ev["explanation"] = explain(ev)

        checkpoint_event(ev, conn)
        print(f"[4] {ev['event_id']}: score={score:.2f} label={ev['risk_label']} -> checkpointed")

    conn.close()
    print("[5] Pipeline smoke test complete.")


if __name__ == "__main__":
    run()