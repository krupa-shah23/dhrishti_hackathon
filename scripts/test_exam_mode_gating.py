"""Quick synthetic check: exam_mode gating suppresses the right classes."""
import sys
sys.path.insert(0, ".")
from src.track_det.detector import class_filter

modes_to_check = {
    "CBT": {"phone", "paper-chit"},
    "paper_pen": {"phone"},
    "physical_fitness": set(),
}

all_pass = True
for mode, expected in modes_to_check.items():
    result = set(class_filter(mode))
    status = "PASS" if result == expected else "FAIL"
    if status == "FAIL":
        all_pass = False
    print(f"[{status}] exam_mode={mode!r} -> {sorted(result)} (expected {sorted(expected)})")

print("\nALL PASS" if all_pass else "\nSOME FAILED")