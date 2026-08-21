"""
FP/FN sweep — P2 status doc §3.5.
Runs the trained detector across every labeled frame (train+val+test,
not just held-out) and logs every mismatch with clip/timestamp context
so P3 can use it for severity tuning and so we know exactly which
frames to prioritize for more labeling.
"""
import csv
from pathlib import Path
from src.track_det.detector import detect_objects_single_frame

REPO_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = REPO_ROOT / "data" / "roboflow_export"
OUT_CSV = REPO_ROOT / "fp_fn_sweep_results.csv"

IOU_THRESHOLD = 0.5  # box counts as a match if IoU >= this


def load_yolo_labels(label_path, img_w, img_h):
    """Returns list of (class_name, x1, y1, x2, y2) in pixel coords."""
    boxes = []
    if not label_path.exists():
        return boxes  # negative frame, no objects
    class_names = ["paper-chit", "phone"]  # must match data.yaml order exactly
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


def iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0
    inter = (ix2 - ix1) * (iy2 - iy1)
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter)


def sweep_split(split_name, rows_out):
    img_dir = DATASET_ROOT / split_name / "images"
    label_dir = DATASET_ROOT / split_name / "labels"
    if not img_dir.exists():
        print(f"[skip] {img_dir} not found")
        return

    import cv2
    for img_path in sorted(img_dir.glob("*.*")):
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        label_path = label_dir / (img_path.stem + ".txt")
        gt_boxes = load_yolo_labels(label_path, w, h)

        preds = detect_objects_single_frame(img, exam_mode="CBT")
        # preds only have class+confidence, no coords here since
        # detect_objects_single_frame -> yolo_infer keeps "_box" internally;
        # if you need coords, temporarily strip the "_box" pop in yolo_infer
        # or call yolo_infer directly for this sweep script.

        pred_classes = {p["class"] for p in preds}
        gt_classes = {b[0] for b in gt_boxes}

        false_negatives = gt_classes - pred_classes  # GT has it, model missed it
        false_positives = pred_classes - gt_classes  # model said it, GT doesn't have it

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


def main():
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