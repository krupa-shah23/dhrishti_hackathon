import pandas as pd

# Each row: event_id, video_id, start_time, end_time, label, notes
rows = [
    # Clip 1
    {"video_id": "clip1", "start_time": 0, "end_time": 5, "label": "Normal", "notes": "calm before phone-out"},
    {"video_id": "clip1", "start_time": 6, "end_time": 8, "label": "Suspicious", "notes": "took out phone"},
    {"video_id": "clip1", "start_time": 12, "end_time": 16, "label": "Suspicious", "notes": "copying, photo of screen"},
    {"video_id": "clip1", "start_time": 16, "end_time": 50, "label": "Normal", "notes": "calm between copying events"},
    {"video_id": "clip1", "start_time": 51, "end_time": 55, "label": "Suspicious", "notes": "copying, photo of screen"},
    {"video_id": "clip1", "start_time": 55, "end_time": 56, "label": "Suspicious", "notes": "intervention after being seen copying"},
    {"video_id": "clip1", "start_time": 68, "end_time": 70, "label": "Suspicious", "notes": "accused left camera scene"},

    # Clip 2
    {"video_id": "clip2", "start_time": 0, "end_time": 60, "label": "Normal", "notes": "setup phase, calm"},
    {"video_id": "clip2", "start_time": 0, "end_time": 190, "label": "Normal", "notes": "setup phase, setup guy present"},
    {"video_id": "clip2", "start_time": 190, "end_time": 200, "label": "Normal", "notes": "setup guy leaves"},
    {"video_id": "clip2", "start_time": 150, "end_time": 190, "label": "Normal", "notes": "accused looking around before phone"},
    {"video_id": "clip2", "start_time": 206, "end_time": 210, "label": "Normal", "notes": "takes out phone from pocket, no copying"},
    {"video_id": "clip2", "start_time": 210, "end_time": 260, "label": "Normal", "notes": "after phone-out, no copying, calm"},

    # Clip 3
    {"video_id": "clip3", "start_time": 0, "end_time": 2, "label": "Suspicious", "notes": "hiding phone under tshirt"},
    {"video_id": "clip3", "start_time": 10, "end_time": 55, "label": "Normal", "notes": "calm, phone still hidden, no visible action"},
    {"video_id": "clip3", "start_time": 56, "end_time": 58, "label": "Suspicious", "notes": "takes phone out from under tshirt"},
    {"video_id": "clip3", "start_time": 65, "end_time": 67, "label": "Suspicious", "notes": "photo of screen, hides under desk"},
    {"video_id": "clip3", "start_time": 81, "end_time": 89, "label": "Suspicious", "notes": "takes out phone, photo of screen"},
    {"video_id": "clip3", "start_time": 93, "end_time": 100, "label": "Suspicious", "notes": "takes out phone, copying"},
    {"video_id": "clip3", "start_time": 107, "end_time": 109, "label": "Suspicious", "notes": "photo of screen"},
    {"video_id": "clip3", "start_time": 124, "end_time": 126, "label": "Suspicious", "notes": "photo of screen, phone pointing at screen"},
    {"video_id": "clip3", "start_time": 131, "end_time": 133, "label": "Suspicious", "notes": "hides phone in pant"},
    {"video_id": "clip3", "start_time": 133, "end_time": 251, "label": "Normal", "notes": "acts as studying, turning pages"},
    {"video_id": "clip3", "start_time": 150, "end_time": 200, "label": "Normal", "notes": "acts as studying, calm stretch"},

    # Clip 4
    {"video_id": "clip4", "start_time": 3, "end_time": 12, "label": "Suspicious", "notes": "first pair talking"},
    {"video_id": "clip4", "start_time": 15, "end_time": 70, "label": "Normal", "notes": "calm between talking bursts"},
    {"video_id": "clip4", "start_time": 72, "end_time": 87, "label": "Suspicious", "notes": "second pair talking, backs visible only"},
    {"video_id": "clip4", "start_time": 97, "end_time": 102, "label": "Suspicious", "notes": "same guy talking again, looking at other screen"},
    {"video_id": "clip4", "start_time": 105, "end_time": 135, "label": "Normal", "notes": "calm before final talking burst"},
    {"video_id": "clip4", "start_time": 140, "end_time": 143, "label": "Suspicious", "notes": "talking again till end"},

    # Clip 5 — EXCLUDED pending team clarification (Krupa unsure what's suspicious)
    # {"video_id": "clip5", "start_time": 0, "end_time": 60, "label": "Normal", "notes": "crowd scene, unclear"},

    # Clip 6 (long clip, times in seconds from start — 1hr11min = 4260s)
    {"video_id": "clip6", "start_time": 3600, "end_time": 3660, "label": "Normal", "notes": "calm stretch, ~1hr mark, no noted activity"},
    {"video_id": "clip6", "start_time": 4260, "end_time": 4338, "label": "Suspicious", "notes": "girl looking around, then talking about exchanging seats"},
    {"video_id": "clip6", "start_time": 4358, "end_time": 4391, "label": "Suspicious", "notes": "moves pc ahead so he can see"},
    {"video_id": "clip6", "start_time": 4391, "end_time": 4406, "label": "Suspicious", "notes": "get up to exchange seats"},
    {"video_id": "clip6", "start_time": 4406, "end_time": 4471, "label": "Normal", "notes": "guard checks in, intervention, told to leave"},

    # Clip 7
    {"video_id": "clip7", "start_time": 0, "end_time": 24, "label": "Normal", "notes": "calm before copying starts"},
    {"video_id": "clip7", "start_time": 26, "end_time": 82, "label": "Suspicious", "notes": "copying from seat 12, seat number identified"},
]

df = pd.DataFrame(rows)
df.insert(0, "event_id", [f"GT{i+1:04d}" for i in range(len(df))])

# Seat ID mapping — fill in as known, rest stays "Desk_unknown"
seat_map = {
    "clip7": "Seat12",  # explicitly mentioned by Krupa
}
df["seat_id"] = df["video_id"].map(seat_map).fillna("Desk_unknown")
df = df[["event_id", "video_id", "start_time", "end_time", "seat_id", "label", "notes"]]

df.to_csv("../../data/ground_truth_labels.csv", index=False)
print(f"Wrote {len(df)} real ground-truth events")
print(df["label"].value_counts())