from pathlib import Path
import cv2

DATA_DIR = Path("data")

print("=" * 70)
print(f"{'Category':<20} {'Sequence':<20} {'Resolution':<15} {'Frames':<10}")
print("=" * 70)

for category in sorted(DATA_DIR.iterdir()):
    if not category.is_dir():
        continue

    for sequence in sorted(category.iterdir()):
        input_dir = sequence / "input"
        if not input_dir.exists():
            continue

        images = sorted(input_dir.glob("*.jpg"))
        if not images:
            images = sorted(input_dir.glob("*.png"))

        if not images:
            continue

        img = cv2.imread(str(images[0]))
        h, w = img.shape[:2]

        print(f"{category.name:<20} {sequence.name:<20} {w}x{h:<11} {len(images):<10}")

print("=" * 70)
print("\nFPS: CDNet2014 image sequences do not contain embedded FPS metadata.")