"""
DRISHTI ML Microservice — Configuration
"""
import os
from dotenv import load_dotenv

load_dotenv()

MONGODB_URI = os.getenv("MONGODB_URI", "mongodb://localhost:27017/drishti")
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
ML_SERVICE_PORT = int(os.getenv("ML_SERVICE_PORT", "8000"))

# Path to the existing DRISHTI ML pipeline codebase
ML_PIPELINE_PATH = os.getenv(
    "ML_PIPELINE_PATH",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "dri", "dhrishti_hackathon"))
)

# Output directories
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "./outputs")
TRACKING_OUTPUT_DIR = os.getenv("TRACKING_OUTPUT_DIR", os.path.join(OUTPUT_DIR, "tracking"))
HEATMAP_OUTPUT_DIR = os.getenv("HEATMAP_OUTPUT_DIR", os.path.join(OUTPUT_DIR, "heatmaps"))
THUMBNAIL_OUTPUT_DIR = os.getenv("THUMBNAIL_OUTPUT_DIR", os.path.join(OUTPUT_DIR, "thumbnails"))

# Ensure output dirs exist
for d in [OUTPUT_DIR, TRACKING_OUTPUT_DIR, HEATMAP_OUTPUT_DIR, THUMBNAIL_OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)
