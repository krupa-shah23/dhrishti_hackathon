"""
src.integration package
P1 Motion/ROI -> P2 Tracker Live Integration
"""
from .p2_p3_bridge import P2P3Bridge
from .event_adapter import (
    adapt_bridge_event_to_schema,
    adapt_bridge_events_to_schema,
    DEFAULT_EVENT_TYPE_PLACEHOLDER,
)

__all__ = [
    "P2P3Bridge",
    "adapt_bridge_event_to_schema",
    "adapt_bridge_events_to_schema",
    "DEFAULT_EVENT_TYPE_PLACEHOLDER",
]
