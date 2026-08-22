"""
DRISHTI ML Microservice — MongoDB Connector
Provides direct access to the shared MongoDB collections.
"""
from pymongo import MongoClient
from config import MONGODB_URI

_client = None
_db = None


def get_db():
    """Get the MongoDB database instance (lazy singleton)."""
    global _client, _db
    if _db is None:
        _client = MongoClient(MONGODB_URI)
        _db = _client.get_default_database()
        print(f"✅  ML Service connected to MongoDB: {_db.name}")
    return _db


def get_collection(name: str):
    """Get a MongoDB collection by name."""
    return get_db()[name]


# Convenience accessors for the 4 core collections
def videos_col():
    return get_collection("videos")


def persons_col():
    return get_collection("persons")


def events_col():
    return get_collection("events")


def person_video_map_col():
    return get_collection("personvideomaps")
