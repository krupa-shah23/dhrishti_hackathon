"""
FP/FN sweep — P2 status doc §3.5 & Confidence-vs-F1 Curve (§3.2).
Runs the trained detector across labeled frames in data/roboflow_export2.
Supports:
  - Default mode: single-threshold (0.25) sweep logging errors to fp_fn_sweep_results.csv
  - Confidence sweep mode (--confidence-sweep): sweeps thresholds 0.10 to 0.90 (step 0.05)
    computing per-class Precision/Recall/F1 on test split and all splits, saved to confidence_curve.csv.
"""
import csv
import argparse
from pathlib import Path
import cv2

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "roboflow_export2"
OUT_CSV = REPO_ROOT / "fp_fn_sweep_results.csv"
CURVE_CSV = REPO_ROOT / "confidence_curve.csv"

IOU_THRESHOLD = 0.5


def load_yolo_labels(label_path, img_w, img_h):
    """Returns list of (class_name, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes
    class_names = ["paper-chit", "phone"]
    for line in label_path.read_text().splitlines():
        if not line.strip():
            continue
        cls_id, xc, yc, w, h = map(float, line.split())
        cls_name = class_names[int(cls_id)]
        x1 = (xc - w / 2) * img_w
        y1 = (yc - h / 2) * img_h
        x2 = (xc + w / 2) * img_w
        y2 = (yc + h / 2) * img_h
        boxes.append((cls_name, x1, y1, x2, y2))
    return boxes


def sweep_split(split_name, rows_out):
    from src.track_det.detector import detect_objects_single_frame

    img_dir = DATASET_ROOT / split_name / "images"
    label_dir = DATASET_ROOT / split_name / "labels"
    if not img_dir.exists():
        print(f"[skip] {img_dir} not found")
        return

    for img_path in sorted(img_dir.glob("*.*")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, w, h)

        preds = detect_objects_single_frame(img, exam_mode="CBT")

        pred_classes = {p["class"] for p in preds}
        gt_classes = {b[0] for b in gt_boxes}

        false_negatives = gt_classes - pred_classes
        false_positives = pred_classes - gt_classes

        for cls in false_negatives:
            rows_out.append({
                "split": split_name, "image": img_path.name,
                "type": "FN", "class": cls, "note": "GT has object, detector missed"
            })
        for cls in false_positives:
            conf = next((p["confidence"] for p in preds if p["class"] == cls), None)
            rows_out.append({
                "split": split_name, "image": img_path.name,
                "type": "FP", "class": cls, "note": f"detector said {cls} @ {conf:.2f}, GT disagrees"
            })


def run_confidence_sweep(out_curve_path: Path = CURVE_CSV):
    from src.track_det.detector import _detector

    if _detector is None:
        print("[FATAL] Detector weights not loaded.")
        return

    print("Running confidence threshold sweep across roboflow_export2 (0.10 to 0.90, step 0.05)...")

    cached_data = {"train": [], "valid": [], "test": []}
    splits = ["train", "valid", "test"]

    for split in splits:
        img_dir = DATASET_ROOT / split / "images"
        lbl_dir = DATASET_ROOT / split / "labels"
        if not img_dir.exists():
            continue
        for img_path in sorted(img_dir.glob("*.*")):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            gt_boxes = load_yolo_labels(lbl_path, w, h)
            gt_classes = {b[0] for b in gt_boxes}

            results = _detector.predict(img, conf=0.01, verbose=False)
            preds = []
            for r in results:
                for b in r.boxes:
                    cls_name = _detector.names[int(b.cls[0])]
                    conf = float(b.conf[0])
                    preds.append((cls_name, conf))

            cached_data[split].append((img_path.name, gt_classes, preds))

    thresholds = [round(0.10 + i * 0.05, 2) for i in range(17)]  # 0.10, 0.15, ..., 0.90
    curve_rows = []

    def calc_metrics(items, thresh):
        phone_tp, phone_fp, phone_fn = 0, 0, 0
        chit_tp, chit_fp, chit_fn = 0, 0, 0
        for name, gt_classes, preds in items:
            p_classes = {c for (c, conf) in preds if conf >= thresh}
            # phone
            if "phone" in gt_classes and "phone" in p_classes:
                phone_tp += 1
            elif "phone" in gt_classes and "phone" not in p_classes:
                phone_fn += 1
            elif "phone" not in gt_classes and "phone" in p_classes:
                phone_fp += 1

            # paper-chit
            if "paper-chit" in gt_classes and "paper-chit" in p_classes:
                chit_tp += 1
            elif "paper-chit" in gt_classes and "paper-chit" not in p_classes:
                chit_fn += 1
            elif "paper-chit" not in gt_classes and "paper-chit" in p_classes:
                chit_fp += 1

        def prf(tp, fp, fn):
            p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
            return round(p, 4), round(r, 4), round(f1, 4), tp, fp, fn

        return prf(phone_tp, phone_fp, phone_fn), prf(chit_tp, chit_fp, chit_fn)

    for t in thresholds:
        test_items = cached_data["test"]
        all_items = cached_data["train"] + cached_data["valid"] + cached_data["test"]

        test_phone, test_chit = calc_metrics(test_items, t)
        all_phone, all_chit = calc_metrics(all_items, t)

        row = {
            "threshold": f"{t:.2f}",
            "test_phone_precision": test_phone[0],
            "test_phone_recall": test_phone[1],
            "test_phone_f1": test_phone[2],
            "test_phone_tp": test_phone[3],
            "test_phone_fp": test_phone[4],
            "test_phone_fn": test_phone[5],
            "test_chit_precision": test_chit[0],
            "test_chit_recall": test_chit[1],
            "test_chit_f1": test_chit[2],
            "test_chit_tp": test_chit[3],
            "test_chit_fp": test_chit[4],
            "test_chit_fn": test_chit[5],
            "all_phone_precision": all_phone[0],
            "all_phone_recall": all_phone[1],
            "all_phone_f1": all_phone[2],
            "all_phone_tp": all_phone[3],
            "all_phone_fp": all_phone[4],
            "all_phone_fn": all_phone[5],
            "all_chit_precision": all_chit[0],
            "all_chit_recall": all_chit[1],
            "all_chit_f1": all_chit[2],
            "all_chit_tp": all_chit[3],
            "all_chit_fp": all_chit[4],
            "all_chit_fn": all_chit[5],
        }
        curve_rows.append(row)

    with open(out_curve_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(curve_rows[0].keys()))
        writer.writeheader()
        writer.writerows(curve_rows)

    print(f"Confidence curve written to {out_curve_path}")
    return curve_rows


def main():
    parser = argparse.ArgumentParser(description="FP/FN Sweep and Confidence Curve Generator")
    parser.add_argument(
        "--confidence-sweep",
        action="store_true",
        help="Run full confidence threshold sweep (0.10 to 0.90) and write confidence_curve.csv",
    )
    args = parser.parse_args()

    if args.confidence_sweep:
        run_confidence_sweep()
    else:
        rows = []
        for split in ["train", "valid", "test"]:
            sweep_split(split, rows)

        with open(OUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["split", "image", "type", "class", "note"])
            writer.writeheader()
            writer.writerows(rows)

        fn_count = sum(1 for r in rows if r["type"] == "FN")
        fp_count = sum(1 for r in rows if r["type"] == "FP")
        print(f"Done. {fn_count} FN, {fp_count} FP. Written to {OUT_CSV}")
        print("FN rows = your next labeling priorities (model missed a real object).")


if __name__ == "__main__":
    main()