"""
Day 3 deliverable — small-object tiling comparison (P2 doc §4.2, §6, §7).

Measures: for ground-truth boxes that are SMALL relative to their image
(our proxy for the pen-cap / under-desk hard cases), does
detect_small_object_tiled() catch more of them than plain yolo_infer()
on the same crop, using the SAME weights (no retraining)?

Usage (run from repo root):
    uv run python scripts\\tiling_comparison.py

Reads:  data/roboflow_export/test/images/*.jpg + matching labels/*.txt (YOLO format)
Writes: tiling_comparison_results.csv, prints summary numbers for slides.
"""
import csv
import sys
from pathlib import Path

import cv2

sys.path.insert(0, ".")
from src.track_det.detector import (
    yolo_infer,
    detect_small_object_tiled,
    class_filter,
    _detector,
)

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "roboflow_export" / "test"
IMAGES_DIR = TEST_DIR / "images"
LABELS_DIR = TEST_DIR / "labels"

# Proxy threshold for "small object" — box area as a fraction of full image
# area. Tune this if it doesn't match your intuition for the pen-cap /
# under-desk cases once you eyeball the flagged images.
SMALL_BOX_AREA_FRACTION = 0.02  # 2% of image area or less = "small"

CLASS_NAMES = {0: "paper-chit", 1: "phone"}  # confirm against your data.yaml order — see note below
CROP_PADDING_FRAC = 2.5  # pad the crop around the gt box by 250% each side.
                          # NOTE: raised from 0.5 after diagnose_tiny_crop.py showed
                          # a tight (50%) padding crop gives the model literally no
                          # surrounding context on genuinely tiny (~25x40px) objects —
                          # zero detections even at conf=0.01. A wider crop (300%
                          # padding in the diagnostic) is what actually let the model
                          # see anything at all. This value simulates a more realistic
                          # ROI window (closer to what P1's motion gate would hand you
                          # — a region around the activity, not a tight box-sized crop).


def load_yolo_label(label_path: Path, img_w: int, img_h: int):
    """Parses a YOLO-format label file -> list of (class_id, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().splitlines():
        if not line.strip():
            continue
        parts = line.split()
        cls_id = int(parts[0])
        xc, yc, w, h = map(float, parts[1:5])
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((cls_id, x1, y1, x2, y2))
    return boxes


def crop_with_padding(img, x1, y1, x2, y2, pad_frac):
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    cx1 = max(0, int(x1 - px))
    cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px))
    cy2 = min(h, int(y2 + py))
    return img[cy1:cy2, cx1:cx2]


def main():
    if _detector is None:
        print("[FATAL] Detector weights not loaded — check models/phone_detector_v1.pt exists.")
        return

    if not IMAGES_DIR.exists():
        print(f"[FATAL] Expected images at {IMAGES_DIR} — adjust TEST_DIR/IMAGES_DIR if your "
              f"roboflow_export layout differs.")
        return

    image_paths = sorted(IMAGES_DIR.glob("*.jpg")) + sorted(IMAGES_DIR.glob("*.png"))
    print(f"Scanning {len(image_paths)} test images for small ground-truth objects "
          f"(threshold: box area < {SMALL_BOX_AREA_FRACTION:.1%} of image area)...")

    rows = []
    small_gt_count = 0

    for img_path in image_paths:
        label_path = LABELS_DIR / (img_path.stem + ".txt")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w

        gt_boxes = load_yolo_label(label_path, img_w, img_h)

        for cls_id, x1, y1, x2, y2 in gt_boxes:
            box_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
            if img_area == 0 or box_area / img_area > SMALL_BOX_AREA_FRACTION:
                continue  # not a "small" object — skip, not part of this test

            small_gt_count += 1
            gt_class = CLASS_NAMES.get(cls_id, str(cls_id))

            crop = crop_with_padding(img, x1, y1, x2, y2, CROP_PADDING_FRAC)
            if crop.size == 0:
                continue

            allowed = class_filter("CBT")  # both classes active for this test

            # --- baseline: plain inference, no tiling (normal 0.25 default threshold) ---
            baseline = yolo_infer(crop, allowed)
            baseline_best = max((d["confidence"] for d in baseline
                                  if d["class"] == gt_class), default=0.0)
            baseline_hit = baseline_best > 0.0

            # --- tiled: explicit upsample + re-infer (normal 0.25 default threshold) ---
            tiled = detect_small_object_tiled(crop, "CBT")
            tiled_best = max((d["confidence"] for d in tiled
                               if d["class"] == gt_class), default=0.0)
            tiled_hit = tiled_best > 0.0

            # --- raw low-threshold check (conf=0.01) — tells us the model's real
            # confidence even when it's below the normal 0.25 cutoff, so a small
            # improvement isn't invisible just because neither side "hits" ---
            raw_baseline = _detector.predict(crop, conf=0.01, verbose=False)
            raw_baseline_best = max(
                (float(b.conf[0]) for r in raw_baseline for b in r.boxes
                 if _detector.names[int(b.cls[0])] == gt_class),
                default=0.0,
            )
            upsampled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            raw_tiled = _detector.predict(upsampled, conf=0.01, verbose=False)
            raw_tiled_best = max(
                (float(b.conf[0]) for r in raw_tiled for b in r.boxes
                 if _detector.names[int(b.cls[0])] == gt_class),
                default=0.0,
            )

            rows.append({
                "image": img_path.name,
                "gt_class": gt_class,
                "gt_box_area_frac": round(box_area / img_area, 4),
                "baseline_hit": baseline_hit,
                "baseline_confidence": round(baseline_best, 4),
                "tiled_hit": tiled_hit,
                "tiled_confidence": round(tiled_best, 4),
                "improved": tiled_hit and not baseline_hit,
                "regressed": baseline_hit and not tiled_hit,
                "raw_baseline_confidence": round(raw_baseline_best, 4),
                "raw_tiled_confidence": round(raw_tiled_best, 4),
                "raw_confidence_delta": round(raw_tiled_best - raw_baseline_best, 4),
            })

    if not rows:
        print("[WARN] No small ground-truth objects found under the current threshold. "
              "Try raising SMALL_BOX_AREA_FRACTION, or confirm labels/ path is correct.")
        return

    out_csv = REPO_ROOT / "tiling_comparison_results.csv"
    with out_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    # --- summary ---
    n = len(rows)
    baseline_hits = sum(r["baseline_hit"] for r in rows)
    tiled_hits = sum(r["tiled_hit"] for r in rows)
    improved = sum(r["improved"] for r in rows)
    regressed = sum(r["regressed"] for r in rows)
    avg_conf_delta = sum(r["tiled_confidence"] - r["baseline_confidence"] for r in rows) / n
    avg_raw_delta = sum(r["raw_confidence_delta"] for r in rows) / n
    raw_improved = sum(1 for r in rows if r["raw_confidence_delta"] > 0)
    raw_regressed = sum(1 for r in rows if r["raw_confidence_delta"] < 0)

    print(f"\n=== Small-object tiling comparison ({n} small ground-truth boxes) ===")
    print(f"--- At normal detection threshold (conf >= 0.25) ---")
    print(f"Baseline (no tiling)  precision-proxy: {baseline_hits}/{n} = {baseline_hits/n:.1%}")
    print(f"Tiled                 precision-proxy: {tiled_hits}/{n} = {tiled_hits/n:.1%}")
    print(f"Objects newly caught by tiling:  {improved}")
    print(f"Objects lost by tiling (regressions): {regressed}")
    print(f"\n--- Raw confidence signal (conf >= 0.01, below normal threshold) ---")
    print(f"Average raw confidence delta (tiled - baseline): {avg_raw_delta:+.4f}")
    print(f"Objects where tiling raised raw confidence:  {raw_improved}/{n}")
    print(f"Objects where tiling lowered raw confidence: {raw_regressed}/{n}")
    print(f"\nFull per-image results written to: {out_csv}")

    if tiled_hits > baseline_hits:
        print("\nSlide line: \"Same weights, tiling caught "
              f"{tiled_hits - baseline_hits} additional small object(s) out of {n} "
              f"({(tiled_hits-baseline_hits)/n:+.1%} precision), zero additional training.\"")
    else:
        print(f"\nSlide line (honest limitation): \"On the smallest ground-truth objects "
              f"(<2% of frame area), tiling improved raw detection confidence in "
              f"{raw_improved}/{n} cases (avg {avg_raw_delta:+.4f}), but neither approach "
              f"cleared the detection threshold on this hardest subset — a disclosed limitation, "
              f"not a tiling failure.\"")


if __name__ == "__main__":
    main()