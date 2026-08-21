"""
Diagnostic for the tiling comparison 0/14 result — checks ONE known tiny
ground-truth box: is the crop itself sane, and is the model seeing
anything at all below the default 0.25 confidence cutoff?

Usage:
    uv run python scripts\\diagnose_tiny_crop.py
"""
import sys
from pathlib import Path
import cv2

sys.path.insert(0, ".")
from src.track_det.detector import _detector, class_filter

REPO_ROOT = Path(__file__).resolve().parent.parent
TEST_DIR = REPO_ROOT / "data" / "roboflow_export" / "test"
IMAGES_DIR = TEST_DIR / "images"
LABELS_DIR = TEST_DIR / "labels"

# Pick the largest of the 14 tiny boxes so we give ourselves the best shot —
# if even this one fails, the smaller ones definitely will.
TARGET_IMAGE = "03-CCTV_Mobile_Usage_f0001322_t00109-4s_jpg.rf.4ef22af3e7aade63be47ed0952af85e0.jpg"
TARGET_CLASS = "phone"

CROP_PADDING_FRAC = 0.5
WIDE_PADDING_FRAC = 3.0  # much more context, for comparison


def load_yolo_label(label_path, img_w, img_h):
    boxes = []
    for line in label_path.read_text().strip().splitlines():
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
    cx1 = max(0, int(x1 - px)); cy1 = max(0, int(y1 - py))
    cx2 = min(w, int(x2 + px)); cy2 = min(h, int(y2 + py))
    return img[cy1:cy2, cx1:cx2]


def main():
    img_path = IMAGES_DIR / TARGET_IMAGE
    label_path = LABELS_DIR / (img_path.stem + ".txt")
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"[FATAL] Could not read {img_path}")
        return
    img_h, img_w = img.shape[:2]
    print(f"Full image size: {img_w}x{img_h}")

    gt_boxes = load_yolo_label(label_path, img_w, img_h)
    phone_boxes = [b for b in gt_boxes if b[0] == 1]  # 1 = phone per data.yaml
    if not phone_boxes:
        print("[WARN] No phone box found in this label — check TARGET_IMAGE.")
        return
    cls_id, x1, y1, x2, y2 = phone_boxes[0]
    bw, bh = x2 - x1, y2 - y1
    print(f"Ground-truth phone box: {bw:.1f}x{bh:.1f} px at ({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})")

    Path("debug_crops").mkdir(exist_ok=True)

    for label, pad in [("tight", CROP_PADDING_FRAC), ("wide", WIDE_PADDING_FRAC)]:
        crop = crop_with_padding(img, x1, y1, x2, y2, pad)
        ch, cw = crop.shape[:2]
        out_path = f"debug_crops/{label}_crop.png"
        cv2.imwrite(out_path, crop)
        print(f"\n--- {label} crop: {cw}x{ch} px, saved to {out_path} ---")

        allowed = class_filter("CBT")

        # Low-threshold raw inference — bypass the default 0.25 cutoff
        results = _detector.predict(crop, conf=0.01, verbose=False)
        found_any = False
        for r in results:
            for box in r.boxes:
                cname = _detector.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"    raw detection: class={cname} conf={conf:.4f}")
                found_any = True
        if not found_any:
            print("    (no detections at all, even at conf=0.01)")

        # Same, but tiled (3x upsample) at low threshold too
        upsampled = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        uh, uw = upsampled.shape[:2]
        results_tiled = _detector.predict(upsampled, conf=0.01, verbose=False)
        print(f"    (tiled to {uw}x{uh})")
        found_tiled = False
        for r in results_tiled:
            for box in r.boxes:
                cname = _detector.names[int(box.cls[0])]
                conf = float(box.conf[0])
                print(f"    tiled detection: class={cname} conf={conf:.4f}")
                found_tiled = True
        if not found_tiled:
            print("    (no tiled detections either, even at conf=0.01)")


if __name__ == "__main__":
    main()