# ============================================================
# glance_persistence.py
# ============================================================
import statistics

def glance_persistence(flow_history, angle_tolerance=15.0, min_fraction=0.7):
    """
    flow_history: [{time, direction}] optical-flow direction samples (degrees).
    Returns True if direction is sustained (real glance toward neighbor),
    False if momentary/noisy (normal fidgeting).
    """
    if not flow_history or len(flow_history) < 3:
        return False

    directions = [f["direction"] for f in flow_history]
    ref = statistics.median(directions)

    def ang_diff(a, b):
        d = abs(a - b) % 360
        return min(d, 360 - d)

    within = sum(1 for d in directions if ang_diff(d, ref) <= angle_tolerance)
    fraction = within / len(directions)
    return fraction >= min_fraction


if __name__ == "__main__":
    from mock_event_data import generate_flow_history
    sustained = generate_flow_history(sustained=True)
    flicker = generate_flow_history(sustained=False)
    print("sustained ->", glance_persistence(sustained))  # expect True
    print("flicker   ->", glance_persistence(flicker))    # expect False