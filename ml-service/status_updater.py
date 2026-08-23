"""
DRISHTI ML Microservice — Status Updater

Pushes processing status updates to Redis so the Node.js status worker
can update MongoDB in real-time (drives the frontend progress bars).

Also provides a direct MongoDB fallback in case the Node.js worker is down.
"""
import json
import redis
from bson import ObjectId
from db import videos_col
from config import REDIS_HOST, REDIS_PORT

_redis_client = None

# Maps pipeline stages to status enum values + progress percentages
STAGE_MAP = {
    "ingesting":   {"status": "ingesting",  "progress": 10,  "label": "Stage 1/8 – Ingesting video"},
    "motion":      {"status": "ingesting",  "progress": 20,  "label": "Stage 2/8 – Motion detection"},
    "detecting":   {"status": "detecting",  "progress": 35,  "label": "Stage 3/8 – YOLO detection"},
    "tracking":    {"status": "tracking",   "progress": 50,  "label": "Stage 4/8 – ByteTrack tracking"},
    "bridging":    {"status": "tracking",   "progress": 60,  "label": "Stage 5/8 – Event bridging"},
    "features":    {"status": "scoring",    "progress": 70,  "label": "Stage 6/8 – Feature extraction"},
    "scoring":     {"status": "scoring",    "progress": 80,  "label": "Stage 7/8 – Risk scoring"},
    "outputs":     {"status": "scoring",    "progress": 90,  "label": "Stage 8/8 – Generating outputs"},
    "done":        {"status": "done",       "progress": 100, "label": "Processing complete"},
    "failed":      {"status": "failed",     "progress": 0,   "label": "Processing failed"},
}


def _get_redis():
    global _redis_client
    if _redis_client is None:
        _redis_client = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis_client


def update_status(video_id: str, stage: str, error_message: str = None,
                  duration: float = None, fps: float = None):
    """
    Push a status update for a video. Tries Redis first (for BullMQ),
    falls back to direct MongoDB write.
    """
    stage_info = STAGE_MAP.get(stage, {"status": stage, "progress": 0, "label": stage})

    update_data = {
        "videoId": video_id,
        "status": stage_info["status"],
        "stage": stage_info["label"],
        "progress": stage_info["progress"],
    }
    if error_message:
        update_data["errorMessage"] = error_message
    if duration is not None:
        update_data["duration"] = duration
    if fps is not None:
        update_data["fps"] = fps

    # Try pushing to Redis (BullMQ ml-status-queue)
    try:
        r = _get_redis()
        # BullMQ uses a specific format — we push a simplified job
        job_data = json.dumps(update_data)
        r.rpush("bull:ml-status-queue:wait", job_data)
        # Also do a direct MongoDB update for reliability
        _direct_mongo_update(video_id, stage_info, error_message, duration, fps)
    except Exception as e:
        print(f"⚠️  Redis push failed ({e}), falling back to direct MongoDB update")
        _direct_mongo_update(video_id, stage_info, error_message, duration, fps)

    print(f"📡  [{video_id}] Status: {stage_info['label']} ({stage_info['progress']}%)")


def _direct_mongo_update(video_id: str, stage_info: dict, error_message: str = None,
                         duration: float = None, fps: float = None):
    """Directly update the video document in MongoDB."""
    update = {
        "$set": {
            "status": stage_info["status"],
            "processingStage": stage_info["label"],
            "processingProgress": stage_info["progress"],
        }
    }
    if error_message:
        update["$set"]["errorMessage"] = error_message
    if duration is not None:
        update["$set"]["duration"] = duration
    if fps is not None:
        update["$set"]["fps"] = fps

    try:
        videos_col().update_one({"_id": ObjectId(video_id)}, update)
    except Exception as e:
        print(f"❌  Direct MongoDB update failed for {video_id}: {e}")
