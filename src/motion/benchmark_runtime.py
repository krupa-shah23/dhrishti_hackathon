import cv2
import time
import argparse

try:
    from .motion import MotionEstimator
    from .roi import get_rois
    from .exclusion_regions import get_exclusion_regions
except ImportError:
    from motion import MotionEstimator
    from roi import get_rois
    from exclusion_regions import get_exclusion_regions


def benchmark_video(video_path, min_area=500):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    estimator = MotionEstimator()

    frames = 0
    total_rois = 0

    start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = estimator.get_motion_mask(frame)
        rois = get_rois(mask, min_area=min_area)

        total_rois += len(rois)
        frames += 1

    elapsed = time.perf_counter() - start
    cap.release()

    fps = frames / elapsed
    ms_per_frame = elapsed / frames * 1000

    print("\n===== P1 RUNTIME BENCHMARK =====")
    print(f"Frames       : {frames}")
    print(f"Runtime      : {elapsed:.2f} s")
    print(f"Throughput   : {fps:.2f} FPS")
    print(f"Latency      : {ms_per_frame:.2f} ms/frame")
    print(f"Total ROIs   : {total_rois}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("video")
    parser.add_argument("--min-area", type=int, default=500)

    args = parser.parse_args()

    benchmark_video(args.video, args.min_area)