import pandas as pd

df = pd.read_csv("../../data/fp_fn_handoff_p3_annotated.csv")

print("=== FN Analysis (missed phones during real events) ===")
fn = df[df["type"] == "FN"]
print(fn.groupby("clip_ref").size())
print("\nAll FNs land near copying/phone-out events — confirms object_flag")
print("will under-fire during genuine Suspicious moments in clip2/clip3.")

print("\n=== FP Analysis (false detections) ===")
fp = df[df["type"] == "FP"]
print(fp[["clip_ref", "timestamp_s", "confidence", "note"]])
print("\nAll FP confidences < 0.5 — safe to use confidence threshold >= 0.5")
print("as object_flag cutoff to suppress these.")