"""
DRISHTI ML Microservice — Pipeline Runner

Wraps the existing ML pipeline (from dhrishti_hackathon) and:
1. Updates video status at each pipeline stage
2. Writes detected events directly to MongoDB
3. Generates tracking data JSON for the frontend Canvas overlay
4. Generates heatmap images
5. Creates person entries and person-video mappings
"""
import sys
import os
import json
import time
import cv2
import numpy as np
from bson import ObjectId
from datetime import datetime

import config as cfg
from db import videos_col, events_col, persons_col, person_video_map_col
from status_updater import update_status

# Add the ML pipeline codebase to Python path
sys.path.insert(0, cfg.ML_PIPELINE_PATH)


def run_pipeline(video_id: str, video_path: str, filename: str):
    """
    Run the full ML pipeline on a video and write results to MongoDB.
    This function wraps the existing run_e2e_pipeline.py logic.
    """
    start_time = time.time()

    try:
        # ─── Stage 1: Ingesting ───
        update_status(video_id, "ingesting")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_seconds = total_frames / fps if fps > 0 else 0

        # Update video metadata
        update_status(video_id, "ingesting", duration=duration_seconds, fps=fps)

        # ─── Stage 2-4: Detection & Tracking ───
        update_status(video_id, "detecting")

        # Import ML pipeline modules
        from src.integration.p1_p2_tracker import P1P2TrackerPipeline
        from src.integration.p2_p3_bridge import P2P3Bridge
        from src.features_risk.extract_features import extract_features
        from src.integration.p3_p4_bridge import p3_event_to_csv_row
        from src.outputs_eval.heatmap import accumulate_mask, generate_heatmap, save_heatmap

        pipeline = P1P2TrackerPipeline()
        p2p3_bridge = P2P3Bridge(fps=fps)

        accumulator = np.zeros((height, width), dtype=np.float32)
        all_events = []
        tracking_frames = {}  # frame_index → list of bboxes

        frame_index = 0
        last_status_update = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Update status periodically (every 5% of frames)
            progress_pct = (frame_index / max(total_frames, 1)) * 100
            if progress_pct - last_status_update >= 5:
                if progress_pct < 30:
                    update_status(video_id, "detecting")
                elif progress_pct < 60:
                    update_status(video_id, "tracking")
                else:
                    update_status(video_id, "bridging")
                last_status_update = progress_pct

            # P1 & P2: Motion detection + YOLO + Tracking
            result = pipeline.process_frame(frame, frame_index)

            # Accumulate motion mask for heatmap
            try:
                mask = pipeline.motion_estimator.get_motion_mask(frame)
                accumulator = accumulate_mask(accumulator, mask)
            except Exception:
                pass  # Motion mask not critical

            # Collect tracking data for frontend Canvas overlay
            fused_tracks = result.get("fused_tracks", [])
            if fused_tracks:
                frame_bboxes = []
                for ft in fused_tracks:
                    box = ft["box"]
                    frame_bboxes.append({
                        "track_id": int(ft["track_id"]),
                        "x": float(box[0]),
                        "y": float(box[1]),
                        "w": float(box[2] - box[0]),
                        "h": float(box[3] - box[1]),
                        "class": ft.get("class"),
                        "confidence": float(ft.get("confidence", 0)),
                        "time": round(frame_index / fps, 3),
                    })
                tracking_frames[str(frame_index)] = frame_bboxes

            # P2→P3 Bridge
            p2p3_bridge.process_fused_tracks(fused_tracks, frame_index)

            # Process completed events
            completed_events = p2p3_bridge.get_completed_events()
            for event in completed_events:
                features = extract_features(event)
                risk_score = _compute_risk_score(event, features)
                all_events.append((event, features, risk_score))

            if frame_index % 500 == 0:
                print(f"  📊 Processed frame {frame_index}/{total_frames}")

            frame_index += 1

        cap.release()

        # Flush remaining tracks
        p2p3_bridge.flush()
        for event in p2p3_bridge.get_completed_events():
            features = extract_features(event)
            risk_score = _compute_risk_score(event, features)
            all_events.append((event, features, risk_score))

        # ─── Stage 6: Feature extraction complete ───
        update_status(video_id, "features")

        # ─── Stage 7: Risk scoring & write to MongoDB ───
        update_status(video_id, "scoring")

        person_map = {}  # track_id → person_id (MongoDB ObjectId)

        for event, features, risk_score in all_events:
            # Create or find Person
            person_oid = _get_or_create_person(
                event, video_id, person_map, fps, frame_index
            )

            # Determine activities
            activities = _classify_activities(event, features)

            # Write Event to MongoDB
            event_doc = {
                "videoId": ObjectId(video_id),
                "personIds": [person_oid],
                "activities": activities,
                "confidenceScore": round(risk_score * 100, 1),
                "timestamps": [{
                    "start": round(event["start_time"], 2),
                    "end": round(event["end_time"], 2),
                }],
                "duration": round(event["end_time"] - event["start_time"], 2),
                "objectDetected": _get_object_class(event),
                "seatId": None,  # TODO: seat grid mapping
                "colorTag": _get_color_for_track(event["track_id"]),
                "createdAt": datetime.utcnow(),
                "updatedAt": datetime.utcnow(),
            }

            events_col().insert_one(event_doc)

            # Update PersonVideoMap
            _update_person_video_map(person_oid, video_id, event_doc)

        # ─── Stage 8: Generate output artifacts ───
        update_status(video_id, "outputs")

        # Save tracking data JSON
        tracking_path = os.path.join(cfg.TRACKING_OUTPUT_DIR, f"{video_id}_tracking.json")
        with open(tracking_path, "w") as f:
            json.dump(tracking_frames, f)

        # Generate and save heatmap
        heatmap_path = os.path.join(cfg.HEATMAP_OUTPUT_DIR, f"{video_id}_heatmap.png")
        try:
            heatmap = generate_heatmap(accumulator)
            save_heatmap(heatmap, heatmap_path)
        except Exception as e:
            print(f"⚠️  Heatmap generation failed: {e}")
            heatmap_path = None

        # Update video document with output paths
        update_fields = {
            "trackingDataPath": tracking_path,
            "status": "done",
            "processingStage": "Processing complete",
            "processingProgress": 100,
        }
        if heatmap_path:
            update_fields["heatmapPath"] = heatmap_path

        videos_col().update_one(
            {"_id": ObjectId(video_id)},
            {"$set": update_fields}
        )

        elapsed = round(time.time() - start_time, 2)
        print(f"✅  Pipeline complete for {filename} in {elapsed}s — {len(all_events)} events detected")

        update_status(video_id, "done")

        return {
            "success": True,
            "events_count": len(all_events),
            "duration": elapsed,
            "tracking_path": tracking_path,
            "heatmap_path": heatmap_path,
        }

    except Exception as e:
        print(f"❌  Pipeline failed for {filename}: {e}")
        import traceback
        traceback.print_exc()
        update_status(video_id, "failed", error_message=str(e))
        return {"success": False, "error": str(e)}


def _compute_risk_score(event: dict, features: dict) -> float:
    """Compute risk score from event features (mirrors run_e2e_pipeline.py logic)."""
    risk = 0.5 + 0.1 * features.get("object_flag", 0) + 0.05 * (features.get("motion_area", 0) / 10000)
    return min(1.0, max(0.0, risk))


def _classify_activities(event: dict, features: dict) -> list:
    """Classify detected activities from event metadata."""
    activities = []

    # Check for object detection (phone, chit, etc.)
    for m in event.get("metadata", []):
        cls = m.get("class")
        if cls and cls not in activities:
            if "phone" in str(cls).lower() or "cell" in str(cls).lower():
                activities.append("phone")
            elif "book" in str(cls).lower() or "paper" in str(cls).lower():
                activities.append("chit")
            else:
                activities.append(str(cls))

    # Check motion-based activities
    motion_area = features.get("motion_area", 0)
    if motion_area > 5000:
        activities.append("head_movement")

    if not activities:
        activities.append("suspicious_movement")

    return activities


def _get_object_class(event: dict) -> str:
    """Extract the primary detected object class from event metadata."""
    for m in event.get("metadata", []):
        if m.get("class"):
            return str(m["class"])
    return None


def _get_color_for_track(track_id: int) -> str:
    """Generate a consistent color for a track ID (for canvas overlay)."""
    colors = [
        "#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4",
        "#FFEAA7", "#DDA0DD", "#98D8C8", "#F7DC6F",
        "#BB8FCE", "#85C1E9", "#F1948A", "#82E0AA",
    ]
    return colors[track_id % len(colors)]


def _get_or_create_person(event: dict, video_id: str, person_map: dict,
                          fps: float, total_frames: int) -> ObjectId:
    """
    Get or create a Person document for a tracked individual.
    Uses track_id → person_id mapping to avoid duplicates within a video.
    """
    track_id = event["track_id"]

    if track_id in person_map:
        return person_map[track_id]

    # Check if person already exists (by label)
    label = f"Person-{track_id:03d}"
    existing = persons_col().find_one({"personLabel": label})

    if existing:
        person_oid = existing["_id"]
    else:
        person_doc = {
            "personLabel": label,
            "embeddingVector": [],
            "firstSeenVideoId": ObjectId(video_id),
            "thumbnailPath": None,
            "seatId": None,
            "totalDetections": 1,
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow(),
        }
        result = persons_col().insert_one(person_doc)
        person_oid = result.inserted_id

    person_map[track_id] = person_oid
    return person_oid


def _update_person_video_map(person_oid: ObjectId, video_id: str, event_doc: dict):
    """Create or update the person↔video mapping."""
    try:
        person_video_map_col().update_one(
            {"personId": person_oid, "videoId": ObjectId(video_id)},
            {
                "$setOnInsert": {
                    "personId": person_oid,
                    "videoId": ObjectId(video_id),
                    "createdAt": datetime.utcnow(),
                },
                "$inc": {"totalDuration": event_doc.get("duration", 0)},
                "$push": {"eventIds": event_doc.get("_id")},
                "$set": {"updatedAt": datetime.utcnow()},
            },
            upsert=True,
        )
    except Exception as e:
        print(f"⚠️  PersonVideoMap update failed: {e}")
