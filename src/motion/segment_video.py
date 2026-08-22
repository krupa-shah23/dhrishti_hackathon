import cv2
import json
import csv
import argparse
import sys
from pathlib import Path

def get_video_metadata(video_path: Path):
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    codec_float = cap.get(cv2.CAP_PROP_FOURCC)
    codec = "".join([chr((int(codec_float) >> 8 * i) & 0xFF) for i in range(4)]) if codec_float > 0 else "NONE"
    
    duration_sec = frame_count / fps if fps > 0 else 0
    
    cap.release()
    
    # Check for audio using imageio_ffmpeg's bundled ffmpeg
    has_audio = None
    import subprocess
    try:
        import imageio_ffmpeg
        ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        cmd = [ffmpeg_exe, "-i", str(video_path)]
        # ffmpeg outputs stream info to stderr when just given -i
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
        # Look for a line containing "Audio:" in stderr
        if "Audio:" in result.stderr:
            has_audio = True
        else:
            has_audio = False
    except Exception:
        pass # Fallback to None if imageio_ffmpeg fails or missing

    return {
        "source_video": str(video_path.name),
        "fps": fps,
        "frame_count": frame_count,
        "width": width,
        "height": height,
        "codec": codec,
        "duration_sec": duration_sec,
        "has_audio": has_audio
    }

def calculate_segments(duration_sec: float):
    calib_window = min(max(0.10 * duration_sec, 60.0), 300.0)
    
    phase_1_start = 0.0
    phase_1_end = 0.10 * duration_sec
    phase_2_start = phase_1_end
    phase_2_end = 0.90 * duration_sec
    phase_3_start = phase_2_end
    phase_3_end = duration_sec
    
    segment_duration = max(30.0, min(60.0, 0.05 * duration_sec))
    
    segments = []
    
    def add_segment(seg_id, label, start_target, end_target, phase, purpose):
        start_sec = max(0.0, min(start_target, duration_sec))
        end_sec = max(start_sec, min(end_target, duration_sec))
        
        segments.append({
            "segment_id": seg_id,
            "label": label,
            "start_sec": start_sec,
            "end_sec": end_sec,
            "duration_sec": end_sec - start_sec,
            "phase": phase,
            "purpose": purpose
        })
    
    # 1. Calibration
    # The first diagnostic segment MUST cover the complete calibration window, but we cap to practical length if very large.
    cap_calib = min(calib_window, 60.0) if calib_window > 60 else calib_window
    add_segment("01", "calibration", 0.0, cap_calib, "phase_1", "Calibration/background/overlay inspection")
    
    # 2. Early Phase
    # Must lie entirely within Phase 1
    p1_target = phase_1_start + 0.20 * (phase_1_end - phase_1_start)
    s2_start = p1_target
    s2_end = s2_start + segment_duration
    if s2_end > phase_1_end:
        s2_start = max(phase_1_start, phase_1_end - segment_duration)
        s2_end = phase_1_end
    s2_start = max(phase_1_start, s2_start)
    add_segment("02", "early_phase", s2_start, s2_end, "phase_1", "Early activity inspection")
    
    # 3. Early/Middle Phase
    # Target approx 25% of total duration
    s3_mid = 0.25 * duration_sec
    s3_start = s3_mid - segment_duration / 2
    s3_end = s3_mid + segment_duration / 2
    add_segment("03", "early_middle", s3_start, s3_end, "phase_2", "Normal candidate behavior")
    
    # 4. Midpoint
    # Target approx 50% of total duration
    s4_mid = 0.50 * duration_sec
    s4_start = s4_mid - segment_duration / 2
    s4_end = s4_mid + segment_duration / 2
    add_segment("04", "midpoint", s4_start, s4_end, "phase_2", "Representative middle-of-exam behavior")
    
    # 5. Late Phase
    # Target approx 75% of total duration
    s5_mid = 0.75 * duration_sec
    s5_start = s5_mid - segment_duration / 2
    s5_end = s5_mid + segment_duration / 2
    add_segment("05", "late_phase", s5_start, s5_end, "phase_2", "Changing candidate behavior")
    
    # 6. Final Phase
    # Must lie entirely within Phase 3
    s6_mid = phase_3_start + (phase_3_end - phase_3_start) / 2
    s6_start = max(phase_3_start, s6_mid - segment_duration / 2)
    s6_end = min(phase_3_end, s6_mid + segment_duration / 2)
    add_segment("06", "final_phase", s6_start, s6_end, "phase_3", "Late-exam conditions")
    
    # 7. End
    # Final short window ending at the actual video end
    s7_start = duration_sec - segment_duration
    s7_end = duration_sec
    add_segment("07", "end", s7_start, s7_end, "phase_3", "End of recording behavior")
            
    # Filter out empty or negligible segments
    segments = [s for s in segments if s['duration_sec'] >= 0.5]
    
    for s in segments:
        s['start_sec'] = round(s['start_sec'], 2)
        s['end_sec'] = round(s['end_sec'], 2)
        s['duration_sec'] = round(s['duration_sec'], 2)

    return {
        "calib_window_sec": round(calib_window, 2),
        "phase_1": {"start": round(phase_1_start, 2), "end": round(phase_1_end, 2)},
        "phase_2": {"start": round(phase_2_start, 2), "end": round(phase_2_end, 2)},
        "phase_3": {"start": round(phase_3_start, 2), "end": round(phase_3_end, 2)},
        "segment_duration_sec": round(segment_duration, 2),
        "segments": segments
    }

def export_segments(video_path: Path, out_dir: Path, segments_info: dict, metadata: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    preview_dir = out_dir / "previews"
    preview_dir.mkdir(exist_ok=True)
    
    # Write metadata.json
    with open(out_dir / "metadata.json", "w") as f:
        full_metadata = metadata.copy()
        full_metadata.update(segments_info)
        full_metadata["segmentation_algorithm"] = "v1.0"
        json.dump(full_metadata, f, indent=4)
        
    # Write segments.csv
    with open(out_dir / "segments.csv", "w", newline="") as f:
        fieldnames = ["segment_id", "label", "start_sec", "end_sec", "duration_sec", "phase", "purpose"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for seg in segments_info["segments"]:
            writer.writerow(seg)
            
    cap = cv2.VideoCapture(str(video_path))
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    
    # Attempt to write with a reliable cross-platform codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    
    write_success = True
    
    for seg in segments_info["segments"]:
        start_frame = int(seg["start_sec"] * fps)
        end_frame = int(seg["end_sec"] * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        
        out_name = f"{seg['segment_id']}_{seg['label']}.mkv"
        out_path = out_dir / out_name
        
        out = cv2.VideoWriter(str(out_path), fourcc, fps, (width, height))
        if not out.isOpened():
            print(f"Warning: Could not open VideoWriter for {out_name}")
            write_success = False
            continue
            
        preview_saved = False
        mid_frame = start_frame + (end_frame - start_frame) // 2
        
        for frame_idx in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            
            if not preview_saved and frame_idx == mid_frame:
                cv2.imwrite(str(preview_dir / f"{seg['segment_id']}_{seg['label']}.jpg"), frame)
                preview_saved = True
                
        out.release()
        
    cap.release()
    return write_success

def main():
    parser = argparse.ArgumentParser(description="Segment diagnostic clips for DRISHTI P1 inspection.")
    parser.add_argument("source_video", type=str, help="Path to original source video")
    parser.add_argument("--out-dir", type=str, required=True, help="Directory to output segments")
    
    args = parser.parse_args()
    
    video_path = Path(args.source_video)
    out_dir = Path(args.out_dir)
    
    print(f"Analyzing {video_path}...")
    try:
        metadata = get_video_metadata(video_path)
    except Exception as e:
        print(f"Error reading video metadata: {e}")
        sys.exit(1)
        
    print(f"Duration: {metadata['duration_sec']:.2f} sec | FPS: {metadata['fps']:.2f} | "
          f"Resolution: {metadata['width']}x{metadata['height']} | Frames: {metadata['frame_count']}")
          
    segments_info = calculate_segments(metadata["duration_sec"])
    
    print(f"Calculated calibration window: {segments_info['calib_window_sec']} sec")
    print(f"Phase boundaries:")
    print(f"  Phase 1: {segments_info['phase_1']['start']} - {segments_info['phase_1']['end']} sec")
    print(f"  Phase 2: {segments_info['phase_2']['start']} - {segments_info['phase_2']['end']} sec")
    print(f"  Phase 3: {segments_info['phase_3']['start']} - {segments_info['phase_3']['end']} sec")
    print(f"Diagnostic segment duration: {segments_info['segment_duration_sec']} sec")
    
    print("\nSelected Segments:")
    for seg in segments_info["segments"]:
        print(f"  {seg['segment_id']} - {seg['label']}: {seg['start_sec']} to {seg['end_sec']} ({seg['duration_sec']}s)")
        
    print(f"\nExporting to {out_dir}...")
    success = export_segments(video_path, out_dir, segments_info, metadata)
    
    if success:
        print("Export completed successfully.")
    else:
        print("Export finished with codec/writing issues.")

if __name__ == "__main__":
    main()
