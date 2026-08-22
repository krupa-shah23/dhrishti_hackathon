import argparse
import sys
import pandas as pd
from pathlib import Path
from src.motion.segment_video import get_video_metadata

# Explicit configuration for camera mappings based on burned-in labels
CAMERA_MAPPING = {
    "01_phone_use.mkv": "Camera04",
    "02_phone_use.mkv": "Camera04",
    "03_mobile_usage.mkv": "Camera12",
    "04_candidate_talking.mkv": "Camera12",
    "05_crowd_reception.mkv": "Mumbai04",
    "06_phone_use.mp4": "LUCKNOW1",
    "07_seat_exchange.mkv": "DAHISAR1",
    "08_seat12_copying.mkv": "AH003",
}

# Explicit configuration to map known videos to their properties
CONFIG = {
    "04_candidate_talking.mkv": {
        "near_empty_start": False
    }
}

# Clip number mapping and tracking notes
NOTES_MAPPING = {
    "01_phone_use.mkv": "Clip 1",
    "02_phone_use.mkv": "Clip 2",
    "03_mobile_usage.mkv": "Clip 3",
    "04_candidate_talking.mkv": "Clip 4",
    "05_crowd_reception.mkv": "Clip 5",
    "06_phone_use.mp4": "UNMAPPED/UNKNOWN (previously assumed clip 7, pending re-check)",
    "07_seat_exchange.mkv": "Clip 6",
    "08_seat12_copying.mkv": "Clip 7 (confirmed exact match: 19305056 bytes, 88.4s)",
}

# Default exam_mode mapping, can be overridden per clip
EXAM_MODE_MAPPING = {
    "default": "CBT"
}

def build_manifest(data_dir: str | Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    if not data_dir.exists() or not data_dir.is_dir():
        raise ValueError(f"Directory not found: {data_dir}")
        
    supported_exts = {".mp4", ".avi", ".mkv", ".mov"}
    exclude_dirs = {"analysis_segments"}
    
    rows = []
    
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() in supported_exts:
            if any(part in exclude_dirs for part in p.parts):
                continue
            try:
                meta = get_video_metadata(p)
                
                resolution = f"{meta['width']}x{meta['height']}"
                file_size_mb = p.stat().st_size / (1024 * 1024)
                
                file_config = CONFIG.get(p.name, {})
                camera_id = CAMERA_MAPPING.get(p.name, None)
                near_empty_start = file_config.get("near_empty_start", None)
                exam_mode = EXAM_MODE_MAPPING.get(p.name, EXAM_MODE_MAPPING["default"])
                notes = NOTES_MAPPING.get(p.name, "")
                
                fps = meta['fps']
                if fps <= 0:
                    fps = None
                    
                duration_sec = meta['duration_sec']
                
                calib_window_sec = None
                if duration_sec > 0:
                    calib_window_sec = min(max(0.10 * duration_sec, 60.0), 300.0)
                    
                row = {
                    "video_id": p.stem,
                    "filename": p.name,
                    "duration_sec": round(duration_sec, 2) if duration_sec else None,
                    "resolution": resolution,
                    "fps": round(fps, 2) if fps else None,
                    "has_audio": meta.get('has_audio', None),
                    "camera_id": camera_id,
                    "near_empty_start": near_empty_start,
                    "exam_mode": exam_mode,
                    "frame_count": meta['frame_count'],
                    "codec": meta['codec'],
                    "container": p.suffix.lower().strip('.'),
                    "file_size_mb": round(file_size_mb, 2),
                    "path": str(p),
                    "calib_window_sec": round(calib_window_sec, 2) if calib_window_sec else None,
                    "notes": notes
                }
                
                rows.append(row)
            except Exception as e:
                print(f"Failed to process {p}: {e}")
                
    df = pd.DataFrame(rows)
    mandatory_cols = ["video_id", "filename", "duration_sec", "resolution", "fps", "has_audio", "camera_id", "near_empty_start", "exam_mode", "notes"]
    if df.empty:
        df = pd.DataFrame(columns=mandatory_cols)
    else:
        cols = mandatory_cols + [c for c in df.columns if c not in mandatory_cols]
        df = df[cols]
        
    return df

def main():
    parser = argparse.ArgumentParser(description="Build video dataset manifest.")
    parser.add_argument("data_dir", type=str, help="Directory to scan for videos")
    parser.add_argument("--out", type=str, required=True, help="Output CSV path")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    out_path = Path(args.out)
    
    print(f"Scanning directory: {data_dir}...")
    try:
        df = build_manifest(data_dir)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
        
    supported_exts = {".mp4", ".avi", ".mkv", ".mov"}
    exclude_dirs = {"analysis_segments"}
    total_found = 0
    if data_dir.exists() and data_dir.is_dir():
        for p in data_dir.rglob("*"):
            if p.is_file() and p.suffix.lower() in supported_exts:
                if any(part in exclude_dirs for part in p.parts):
                    continue
                total_found += 1
                
    processed = len(df)
    failed = total_found - processed
    
    print(f"Videos found: {total_found}")
    print(f"Successfully processed: {processed}")
    print(f"Failed: {failed}")
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Output saved to: {out_path}")
    
    if processed > 0:
        print("\nConcise Metadata Summary:")
        # Convert None to readable NaN for text alignment
        summary_df = df[["video_id", "camera_id", "resolution", "duration_sec", "has_audio"]]
        print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()
