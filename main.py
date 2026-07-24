"""
main.py — P4: Offline Exam-Hall Video Analytics Pipeline Orchestrator
Owned by P4. Integrates P1 (motion), P2 (tracking/detection), P3 (features/risk)
via frozen Day-1 function contracts and produces all P4 outputs.
"""

import os
import sys
import time
import argparse

import cv2  # type: ignore[import-not-found]
import numpy as np
import yaml

# ── Stub imports from P1, P2, P3 ─────────────────────────────────────────────
# These will be replaced by actual implementations as P1/P2/P3 deliver their code.
from src.motion import get_motion_mask, get_rois
from src.track_det import track, detect_objects
from src.features_risk import extract_features, risk_score

# ── P4 output generators ──────────────────────────────────────────────────────
from src.outputs_eval.heatmap import (
    accumulate_mask,
    generate_heatmap,
    overlay_heatmap,
    save_heatmap,
)
from src.outputs_eval.timeline import plot_timeline
from src.outputs_eval.logger import init_log, log_event, format_timestamp, format_roi_box
from src.outputs_eval.report import generate_report
from src.outputs_eval.benchmark import benchmark_video, save_benchmark_csv
from src.outputs_eval.ablation import run_ablation, save_ablation_csv


# ─────────────────────────────────────────────────────────────────────────────
# MOCK STUBS (remove / swap in as P1/P2/P3 deliver real implementations)
# ─────────────────────────────────────────────────────────────────────────────

def _mock_get_motion_mask(frame: np.ndarray) -> np.ndarray:
    """Placeholder: simple frame differencing (replace with P1 MOG2)."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thresh = cv2.threshold(blurred, 25, 255, cv2.THRESH_BINARY)
    return thresh


def _mock_get_rois(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Placeholder: contour-based ROI extraction (replace with P1 roi_boxes)."""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            boxes.append(cv2.boundingRect(cnt))
    return boxes


def _mock_track(boxes: list) -> list[dict]:
    """Placeholder: returns boxes as static tracks (replace with P2 ByteTrack)."""
    return [{"track_id": i, "box": b, "history": [b]} for i, b in enumerate(boxes)]


def _mock_detect_objects(frame: np.ndarray) -> list:
    """Placeholder: no-op detection (replace with P2 YOLO infer)."""
    return []


def _mock_extract_features(trk: dict) -> dict:
    """Placeholder: minimal features from box (replace with P3)."""
    x, y, w, h = trk["box"]
    return {"area": w * h, "cx": x + w // 2, "cy": y + h // 2}


def _mock_risk_score(features: dict) -> float:
    """Placeholder: heuristic risk based on area (replace with P3 XGBoost)."""
    area = features.get("area", 0)
    return min(1.0, area / 20000.0)


# ─────────────────────────────────────────────────────────────────────────────

def _resolve_fn(real_fn, mock_fn, use_mock: bool):
    """Returns the real function or falls back to the mock."""
    if use_mock:
        return mock_fn
    try:
        # Quick sanity check — real function must not raise on import
        return real_fn
    except Exception:
        return mock_fn


def load_config(config_path: str = "config.yaml") -> dict:
    """Loads pipeline configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def draw_overlays(
    frame: np.ndarray,
    tracks: list[dict],
    detections: list,
    risk_scores_map: dict[int, float],
    risk_threshold: float,
) -> np.ndarray:
    """
    Draws bounding boxes, track IDs, risk scores, and YOLO detections
    onto the video frame for the saved overlay video.
    """
    out = frame.copy()
    for trk in tracks:
        tid = trk["track_id"]
        x, y, w, h = trk["box"]
        risk = risk_scores_map.get(tid, 0.0)

        # Color: green (safe) → red (high risk)
        color = (
            int(risk * 255),           # B
            int((1 - risk) * 180),     # G
            int((1 - risk) * 255),     # R  (closer to red at high risk)
        )

        cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
        label = f"ID:{tid} R:{risk:.2f}"
        cv2.putText(out, label, (x, y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # YOLO detections — drawn in yellow
    for box, cls_name, conf in detections:
        x1, y1, x2, y2 = box
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(out, f"{cls_name} {conf:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

    return out


def run_pipeline(video_path: str, cfg: dict, use_mock: bool = True) -> None:
    """
    Main video processing loop. Reads frames, runs the full P1→P2→P3 pipeline,
    and produces all P4 output artefacts.

    Args:
        video_path (str):  Path to the source video file.
        cfg (dict):        Loaded config.yaml dictionary.
        use_mock (bool):   If True, uses mock stubs for P1/P2/P3 functions.
    """

    # ── Resolve functions (real or mock) ────────────────────────────────────
    fn_mask    = _mock_get_motion_mask if use_mock else get_motion_mask
    fn_rois    = _mock_get_rois        if use_mock else get_rois
    fn_track   = _mock_track           if use_mock else track
    fn_detect  = _mock_detect_objects  if use_mock else detect_objects
    fn_feats   = _mock_extract_features if use_mock else extract_features
    fn_risk    = _mock_risk_score      if use_mock else risk_score

    # ── Open video ──────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[main] ERROR: Cannot open video: {video_path}")
        sys.exit(1)

    fps      = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width    = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_fr = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"[main] Video: {video_path} | {width}x{height} @ {fps:.1f}fps | {total_fr} frames")

    # ── Output paths ────────────────────────────────────────────────────────
    out_dir    = cfg.get("paths", {}).get("output_dir", "outputs")
    vid_id     = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(out_dir, exist_ok=True)

    overlay_path  = os.path.join(out_dir, f"{vid_id}_overlay.mp4")
    heatmap_path  = os.path.join(out_dir, f"{vid_id}_heatmap.png")
    timeline_path = os.path.join(out_dir, f"{vid_id}_timeline.png")
    csv_path      = os.path.join(out_dir, f"{vid_id}_events.csv")

    # ── Video writer ────────────────────────────────────────────────────────
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(overlay_path, fourcc, fps, (width, height))

    # ── Initialise accumulators ─────────────────────────────────────────────
    heat_acc     = np.zeros((height, width), dtype=np.float32)
    timestamps   = []
    motion_scores= []
    risk_scores_ts = []
    risk_threshold = cfg.get("risk", {}).get("risk_threshold", 0.65)

    init_log(csv_path)

    # ── Frame processing loop ───────────────────────────────────────────────
    frame_idx = 0
    t_start   = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ts_str = format_timestamp(frame_idx, fps)

        # P1 — Motion mask & ROIs
        mask = fn_mask(frame)
        boxes = fn_rois(mask)

        # Accumulate heatmap
        heat_acc = accumulate_mask(heat_acc, mask)

        # Compute frame-level motion score (% of mask pixels active)
        motion_pct = (np.count_nonzero(mask) / mask.size) * 100.0

        # P2 — Tracking & detection
        tracks_list  = fn_track(boxes)
        detections   = fn_detect(frame)

        # P3 — Features & risk per track
        risk_scores_map: dict[int, float] = {}
        max_risk = 0.0
        for trk in tracks_list:
            feats = fn_feats(trk)
            r     = fn_risk(feats)
            risk_scores_map[trk["track_id"]] = r
            max_risk = max(max_risk, r)

            # Log each high-risk ROI event
            if r >= risk_threshold:
                log_event(csv_path, {
                    "timestamp":    ts_str,
                    "frame_index":  frame_idx,
                    "roi_box":      format_roi_box(trk["box"]),
                    "motion_score": round(motion_pct, 2),
                    "audio_level":  0.0,   # Placeholder until P3 audio fusion
                    "risk_score":   round(r, 4),
                    "confidence":   round(r * 0.9, 4),   # Placeholder formula
                })

        timestamps.append(frame_idx / fps)
        motion_scores.append(round(motion_pct, 2))
        risk_scores_ts.append(round(max_risk, 4))

        # P4 — Draw overlay frame and write to video
        out_frame = draw_overlays(frame, tracks_list, detections, risk_scores_map, risk_threshold)
        writer.write(out_frame)

        frame_idx += 1

        # Progress print every 100 frames
        if frame_idx % 100 == 0:
            elapsed = time.time() - t_start
            fps_proc = frame_idx / elapsed if elapsed > 0 else 0
            print(f"[main] Frame {frame_idx}/{total_fr} | {fps_proc:.1f} fps processing")

    cap.release()
    writer.release()

    elapsed_total = time.time() - t_start
    print(f"\n[main] Processing complete: {frame_idx} frames in {elapsed_total:.1f}s "
          f"({frame_idx/elapsed_total:.1f} fps)")

    # ── Save P4 artefacts ────────────────────────────────────────────────────
    if cfg.get("outputs", {}).get("save_heatmap", True):
        heatmap_img = generate_heatmap(heat_acc)
        save_heatmap(heatmap_img, heatmap_path)

    if cfg.get("outputs", {}).get("save_timeline", True):
        plot_timeline(
            timestamps=timestamps,
            motion_scores=motion_scores,
            risk_scores=risk_scores_ts,
            output_path=timeline_path,
            video_id=vid_id,
            risk_threshold=risk_threshold,
        )

    print(f"[main] Timeline PNG   -> {timeline_path}")

    # ── Optional: PDF report ──────────────────────────────────────────────────
    if cfg.get("outputs", {}).get("save_pdf_report", True):
        report_path = os.path.join(out_dir, f"{vid_id}_report.pdf")
        generate_report(
            video_path=video_path,
            csv_path=csv_path,
            heatmap_path=heatmap_path,
            timeline_path=timeline_path,
            output_path=report_path,
            processing_time_s=elapsed_total,
            risk_threshold=risk_threshold,
        )
        print(f"[main] PDF Report     -> {report_path}")


# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline Exam-Hall Video Analytics — P4 Pipeline Orchestrator"
    )
    parser.add_argument(
        "--video", "-v",
        type=str,
        default=None,
        help="Path to input video file. Overrides config.yaml path.",
    )
    parser.add_argument(
        "--config", "-c",
        type=str,
        default="config.yaml",
        help="Path to configuration YAML file. (default: config.yaml)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use mock stubs for P1/P2/P3 (default: True until real modules are ready).",
    )
    parser.add_argument(
        "--no-mock",
        dest="mock",
        action="store_false",
        help="Use real P1/P2/P3 implementations (requires all modules to be complete).",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        default=False,
        help="Run 4-method benchmark comparison and save benchmark_table.csv.",
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        default=False,
        help="Run ablation study and save ablation_table.csv.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args   = parse_args()
    config = load_config(args.config)

    video_path = args.video or config.get("paths", {}).get("input_video", "data/demo_clips/raw/sample_exam.mp4")

    print("=" * 60)
    print("  Offline Exam-Hall Video Analytics Pipeline")
    print(f"  Mode : {'MOCK (stub)' if args.mock else 'REAL'}")
    print(f"  Input: {video_path}")
    print("=" * 60)

    run_pipeline(video_path=video_path, cfg=config, use_mock=args.mock)

    if args.benchmark:
        print("\n" + "=" * 60)
        print("  Running Benchmark Comparison")
        print("=" * 60)
        bench_results = benchmark_video(video_path)
        save_benchmark_csv(bench_results, os.path.join(config.get("paths", {}).get("output_dir", "outputs"), "benchmark_table.csv"))

    if args.ablation:
        print("\n" + "=" * 60)
        print("  Running Ablation Study")
        print("=" * 60)
        ablation_rows = run_ablation(video_path, config)
        save_ablation_csv(ablation_rows, os.path.join(config.get("paths", {}).get("output_dir", "outputs"), "ablation_table.csv"))
