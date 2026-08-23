"""
Tuning calculator — plug in P1's real FPS/resolution the moment she
sends them, and get an immediate max_distance/max_age recommendation.
Built ahead of time so there's no re-analysis needed under time
pressure once the numbers arrive.

Uses REAL data already measured from p1_rois.csv this session:
- Observed frame-to-frame jitter on a continuous track: ~5px (x1 values
  105,106,107,104,106,104,104,106,110,104,105,105,104,106,108 across
  frames 466-484)
- P3 confirmed (see master doc §7b): max_age only needs to cover
  realistic IN-BURST occlusion (e.g. a hand briefly blocked by another
  student), NOT the long between-burst stillness gaps (150-666 frames)
  found in the real data -- those SHOULD keep spawning new tracks.

Run from the REPO ROOT:
    python -m src.track_det.tuning_calculator --fps 30 --width 1920 --height 1080

Adjust --fps/--width/--height to whatever P1 actually sends.
"""
import argparse


def recommend(fps: float, width: int, height: int):
    print(f"Input: {width}x{height} @ {fps} fps\n")

    # --- max_distance recommendation ---
    # Observed real jitter on a continuous track was ~5px, but that was
    # measured on P1's CDNet-scale test clip (y2 topped out ~239px,
    # suggesting a low-resolution clip, possibly ~320x240 or similar).
    # Scale the jitter tolerance by resolution ratio so max_distance
    # stays meaningful if the real exam-hall camera is higher-res.
    reference_height = 240  # approx height of the clip jitter was measured on
    scale_factor = height / reference_height
    observed_jitter_px = 5.0
    scaled_jitter_px = observed_jitter_px * scale_factor

    # Recommend max_distance as a safety margin above scaled jitter --
    # NOT equal to it. A person walking normally covers real ground
    # frame to frame too, not just jitter; this margin is a starting
    # point for a first real test, not a final tuned value.
    safety_multiplier = 6.0  # generous margin: normal walking speed can
                              # easily be several x pure jitter
    recommended_max_distance = round(scaled_jitter_px * safety_multiplier, 1)

    print(f"--- max_distance recommendation ---")
    print(f"Reference: ~{observed_jitter_px}px jitter observed on a "
          f"continuous track in P1's real ROI data (measured on an "
          f"apparent ~{reference_height}px-tall clip).")
    print(f"Scaled to this resolution ({height}px tall): "
          f"~{scaled_jitter_px:.1f}px jitter expected.")
    print(f"RECOMMENDED max_distance (starting point): "
          f"{recommended_max_distance}px")
    print(f"  (This is a STARTING point for a first real test, not a "
          f"final value -- re-check against actual jitter once you can "
          f"run track() on this specific footage.)\n")

    # --- max_age recommendation ---
    # Per P3's confirmed decision: max_age should cover realistic
    # IN-BURST occlusion only (e.g. hand briefly blocked), not the long
    # between-burst stillness gaps. A reasonable real-world assumption:
    # brief occlusion lasts up to roughly 0.3-0.5 seconds.
    print(f"--- max_age recommendation ---")
    for occlusion_seconds in (0.3, 0.5, 1.0):
        frames = round(occlusion_seconds * fps)
        print(f"  If targeting ~{occlusion_seconds}s of in-burst occlusion "
              f"tolerance -> max_age = {frames} frames")

    print(
        f"\nPer P3's confirmed decision (master doc \u00a77b): max_age "
        f"should NOT be stretched to cover the 150-666 frame between-"
        f"burst stillness gaps found in the real data -- those should "
        f"keep spawning new track IDs. Pick a max_age from the options "
        f"above based on how long a believable in-burst occlusion (e.g. "
        f"one student's hand briefly blocked by another) would "
        f"realistically last on THIS footage -- 0.5s is a reasonable "
        f"default absent better information."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    args = parser.parse_args()
    recommend(args.fps, args.width, args.height)