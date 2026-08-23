"""
split_dataset.py

Builds the YOLO train/val/TEST folder structure per the final 5-day plan's
frozen protocol (P2 doc §2): 80/15/5 split, clip-level, with the 5% test
split held out and UNTOUCHED until Day 5's honest generalization number.

Splits are done at the CLIP level, not the frame level, for the same
reason as before: consecutive frames from one video are highly correlated,
so a frame-level random split leaks near-duplicates across train/val/test
and inflates every metric you'd report. This version extends that to a
three-way split, and specifically guarantees the test clip(s) never
overlap with train or val.

DO NOT touch data/phone_dataset/images/test (or labels/test) for anything
except the Day 5 final validation run. Not for training, not for
threshold-tuning, not for "let me just check something quickly" — per the
doc, that's the whole point of a held-out set.

Usage
-----
python scripts/split_dataset.py \
    --domain-dirs data/frames_raw/clip01_ev1_takeout data/frames_raw/clip01_ev2_copying \
                  data/frames_raw/clip01_ev3_copying2 data/frames_raw/clip02_ev1_takeout \
                  data/frames_raw/clip03_ev1_hiding data/frames_raw/clip03_ev2_retrieval \
                  data/frames_raw/clip03_ev3_photo_hide_desk data/frames_raw/clip03_ev4_photo_copy_burst \
                  data/frames_raw/clip03_ev5_photo data/frames_raw/clip03_ev6_photo_pointing_hide \
                  data/frames_raw/clip07_copying \
                  data/frames_raw/clip04_negative data/frames_raw/clip05_negative \
    --kaggle-dir data/kaggle_phone_dataset \
    --out data/phone_dataset \
    --val-fraction 0.15 --test-fraction 0.05
"""

import argparse
import random
import shutil
from pathlib import Path


def find_labeled_pairs(src_dir: Path):
    """Return list of (image_path, label_path) for a labeled folder.
    Supports both flat (img.jpg + img.txt) and images/+labels/ layouts."""
    pairs = []
    images_sub = src_dir / "images"
    labels_sub = src_dir / "labels"
    if images_sub.is_dir() and labels_sub.is_dir():
        for img in sorted(images_sub.glob("*.*")):
            lbl = labels_sub / (img.stem + ".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    else:
        for img in sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.png")):
            lbl = img.with_suffix(".txt")
            if lbl.exists():
                pairs.append((img, lbl))
    return pairs


def copy_pairs(pairs, images_out: Path, labels_out: Path, prefix: str = ""):
    images_out.mkdir(parents=True, exist_ok=True)
    labels_out.mkdir(parents=True, exist_ok=True)
    for img, lbl in pairs:
        name = f"{prefix}{img.name}" if prefix else img.name
        shutil.copy2(img, images_out / name)
        shutil.copy2(lbl, labels_out / (Path(name).stem + ".txt"))


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain-dirs", nargs="+", required=True,
                     help="one folder per clip-event, already labeled")
    ap.add_argument("--kaggle-dir", default=None,
                     help="optional Kaggle phone-detection set (images/labels)")
    ap.add_argument("--out", required=True, help="output dataset root")
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--test-fraction", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    out_root = Path(args.out)
    splits = {
        "train": (out_root / "images/train", out_root / "labels/train"),
        "val":   (out_root / "images/val",   out_root / "labels/val"),
        "test":  (out_root / "images/test",  out_root / "labels/test"),
    }

    domain_dirs = [Path(d) for d in args.domain_dirs]
    labeled_dirs = []
    for d in domain_dirs:
        pairs = find_labeled_pairs(d)
        if not pairs:
            print(f"  [WARN] no labeled image/label pairs found in {d} -- "
                  f"skipping (did you export labels here yet?)")
            continue
        labeled_dirs.append((d, pairs))

    if not labeled_dirs:
        print("No labeled domain clips found. Label your extracted frames "
              "first (Roboflow/LabelImg -> YOLO export), then re-run.")
        return

    random.shuffle(labeled_dirs)
    n = len(labeled_dirs)
    n_test = max(1, round(n * args.test_fraction))
    n_val = max(1, round(n * args.val_fraction))
    n_val = min(n_val, n - n_test - 1) if n - n_test > 1 else n_val  # keep >=1 for train if possible

    test_clips = labeled_dirs[:n_test]
    val_clips = labeled_dirs[n_test:n_test + n_val]
    train_clips = labeled_dirs[n_test + n_val:]

    if not train_clips:
        print("  [WARN] too few labeled clip-folders to reserve a separate "
              "test split without starving train — consider labeling more "
              "clip-event folders before finalizing the held-out set.")

    print(f"Clip-level split: {len(train_clips)} -> train, {len(val_clips)} -> val, "
          f"{len(test_clips)} -> TEST (held out, do not touch until Day 5)")

    counts = {"train": 0, "val": 0, "test": 0}
    for split_name, clips in [("train", train_clips), ("val", val_clips), ("test", test_clips)]:
        img_out, lbl_out = splits[split_name]
        for d, pairs in clips:
            copy_pairs(pairs, img_out, lbl_out, prefix=f"{d.name}__")
            counts[split_name] += len(pairs)
            marker = " [HELD OUT]" if split_name == "test" else ""
            print(f"  {split_name:5s} <- {d.name} ({len(pairs)} frames){marker}")

    # --- Kaggle pool: random image-level split is fine (not video frames),
    # but keep it OUT of test entirely -- test should be pure domain-matched
    # footage, since that's the number that actually answers "did this
    # generalize to our real cameras," not "did it generalize to Kaggle." ---
    if args.kaggle_dir:
        kaggle_pairs = find_labeled_pairs(Path(args.kaggle_dir))
        if kaggle_pairs:
            random.shuffle(kaggle_pairs)
            n_val_k = round(len(kaggle_pairs) * args.val_fraction)
            k_val, k_train = kaggle_pairs[:n_val_k], kaggle_pairs[n_val_k:]
            copy_pairs(k_train, *splits["train"], prefix="kaggle__")
            copy_pairs(k_val, *splits["val"], prefix="kaggle__")
            counts["train"] += len(k_train)
            counts["val"] += len(k_val)
            print(f"  Kaggle pool: {len(k_train)} -> train, {len(k_val)} -> val "
                  f"(kept OUT of test intentionally)")
        else:
            print(f"  [WARN] no labeled pairs found in --kaggle-dir {args.kaggle_dir}")

    print(f"\nFinal dataset: train={counts['train']}  val={counts['val']}  "
          f"test={counts['test']} (HELD OUT) -> {out_root}")
    print("phone_data.yaml already points train/val/test at this folder.")
    print("Reminder: do not touch images/test or labels/test until Day 5.")


if __name__ == "__main__":
    main()