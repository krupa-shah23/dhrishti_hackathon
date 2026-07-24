def centroid_track(boxes: list[tuple[int, int, int, int]]) -> list[dict]:
    """
    Fallback centroid tracker if ByteTrack fails to load.
    """
    raise NotImplementedError
