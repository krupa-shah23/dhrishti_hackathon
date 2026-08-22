import numpy as np

# Camera-specific grid configurations
# Each camera has a base resolution and a dict of seat boundary boxes [x1, y1, x2, y2).
# Overlap is expected in 2D perspective views (e.g., front row overlapping back row).
GRID_CONFIGS = {
    "Camera12": {
        "resolution": (640, 480),
        "seats": {
            "seat_66": (0, 120, 150, 480),    # Leftmost candidate, facing right
            "seat_65": (160, 120, 340, 280),  # Back row left
            "seat_64": (330, 130, 470, 380),  # Back row middle
            "seat_63": (470, 160, 640, 400),  # Back row right
            "seat_61": (80, 260, 310, 480),   # Front row left
            "seat_60": (280, 340, 480, 480),  # Front row right
        },
        "adjacency": {
            "seat_66": ["seat_61"],
            "seat_61": ["seat_66"],
            "seat_65": ["seat_64"],
            "seat_64": ["seat_65"]
        }
    },
    "DAHISAR1": {
        "resolution": (1280, 720),
        "seats": {
            # Measured from clip 07 (07_seat_exchange.mkv) frame 300.
            # Single row of 6 candidate seats along the left wall, perspective view.
            # seat_7 is the isolated invigilator/admin desk on the right wall — NOT in adjacency.
            "seat_1": (20, 430, 210, 720),    # Bottom-left, label '1'
            "seat_2": (70, 280, 290, 540),    # Label '2'
            "seat_3": (265, 165, 475, 410),   # Label '3'
            "seat_4": (420, 90, 630, 295),    # Label '4'
            "seat_5": (525, 35, 740, 210),    # Label '5'
            "seat_6": (610, 0, 840, 135),     # Label '6', upper-right of row
        },
        "adjacency": {
            # Direct-neighbor pairs only — no transitive chaining.
            # seat_7 deliberately excluded (not a candidate seat).
            "seat_1": ["seat_2"],
            "seat_2": ["seat_1", "seat_3"],
            "seat_3": ["seat_2", "seat_4"],
            "seat_4": ["seat_3", "seat_5"],
            "seat_5": ["seat_4", "seat_6"],
            "seat_6": ["seat_5"],
        }
    },
}

def get_grid_config(camera_id: str, target_width: int = -1, target_height: int = -1) -> dict:
    """
    Returns the grid configuration for a specific camera.
    If target_width and target_height are provided and differ from the configured resolution,
    the seat coordinates are proportionally scaled safely.
    Returns an empty dict if the camera is unknown.
    The returned seats use [x1, y1, x2, y2) coordinate conventions.
    """
    if camera_id not in GRID_CONFIGS:
        return {}
        
    config = GRID_CONFIGS[camera_id]
    base_w, base_h = config["resolution"]
    
    # If no target resolution provided or identical, return a copy of the seats
    if target_width <= 0 or target_height <= 0 or (target_width == base_w and target_height == base_h):
        return {
            "resolution": (base_w, base_h),
            "seats": config["seats"].copy()
        }
        
    # Proportional scaling
    scale_x = target_width / base_w if base_w > 0 else 1.0
    scale_y = target_height / base_h if base_h > 0 else 1.0
    
    scaled_seats = {}
    for seat_id, (x1, y1, x2, y2) in config["seats"].items():
        sx1 = max(0, int(x1 * scale_x))
        sy1 = max(0, int(y1 * scale_y))
        sx2 = min(target_width, int(x2 * scale_x))
        sy2 = min(target_height, int(y2 * scale_y))
        scaled_seats[seat_id] = (sx1, sy1, sx2, sy2)
        
    return {
        "resolution": (target_width, target_height),
        "seats": scaled_seats
    }

def validate_grid_config(config: dict) -> list[str]:
    """
    Validates a grid configuration dictionary.
    Returns a list of error strings if invalid, or an empty list if valid.
    """
    errors = []
    if not config:
        return errors
        
    w, h = config.get("resolution", (0, 0))
    seats = config.get("seats", {})
    
    for seat_id, box in seats.items():
        if len(box) != 4:
            errors.append(f"Seat {seat_id} has invalid coordinate format.")
            continue
            
        x1, y1, x2, y2 = box
        
        # Zero/negative area
        if x1 >= x2 or y1 >= y2:
            errors.append(f"Seat {seat_id} has zero or negative area: {box}")
            
        # Out of bounds
        if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
            errors.append(f"Seat {seat_id} is out of frame bounds: {box}")
            
    return errors

def get_seat_mask(frame_shape: tuple, seat_id: str, camera_id: str) -> np.ndarray:
    """
    Returns a binary mask (255 inside seat, 0 outside) for deterministic spatial membership.
    """
    h, w = frame_shape[:2]
    config = get_grid_config(camera_id, w, h)
    
    mask = np.zeros((h, w), dtype=np.uint8)
    if not config or seat_id not in config["seats"]:
        return mask
        
    x1, y1, x2, y2 = config["seats"][seat_id]
    mask[y1:y2, x1:x2] = 255
    return mask

def get_adjacent_seats(seat_id: str, camera_id: str) -> set[str]:
    """
    Returns a set of seat_ids that are physically adjacent to the given seat_id.
    Uses explicit adjacency maps if provided, otherwise falls back to empty set.
    """
    if camera_id not in GRID_CONFIGS:
        return set()
    
    config = GRID_CONFIGS[camera_id]
    if "adjacency" in config and seat_id in config["adjacency"]:
        return set(config["adjacency"][seat_id])
        
    return set()
