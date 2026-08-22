# Project Scope Decisions

Record of deliberate scope cuts for Dhristi's detection module (owner: Krupa, P2). These are decisions, not limitations from running out of time — kept separate from technical to-do/status docs so they're visible immediately, not buried.

## Calculator detection — CANCELLED (2026-08-22)

Explicitly dropped from scope. Final detector is 2-class: phone, paper-chit only.

- Dataset/model reverted to `data/roboflow_export2` (Roboflow v4), `nc: 2`, `phone_detector_v3.pt` — confirmed via checksum + labels.cache timestamp match to the actual training run that produced the deployed weights.
- Prior calculator labeling work (~100 relabeled frames, `data/roboflow_export3`, `nc: 3`) is NOT deleted — left in place as-is in case of a future revisit, but is not part of the live config and should not be referenced by any active code path.
- `src/track_det/detector.py`'s `DEFAULT_MIN_CONFIDENCE` docstring still mentions calculator as a hypothetical future class — harmless (comment only), left as-is.
- Slide framing: present as a deliberate scope decision, not a time-constraint limitation.

## Smart-watch detection — CANCELLED (2026-08-22)

Was always a stretch/time-boxed item, zero work started. Formally dropped, not deferred. No cleanup required — nothing was ever built.

## repetition_count — SHIP AS DISCLOSED 0 (2026-08-22)

`repetition_count` always returns 0 in the current pipeline. Root cause: P2P3Bridge only retains a collapsed None→True latch on object_detected, not a full per-frame stream, so `compute_repetition_count()`'s burst-counting logic has no real data to run against. Confirmed via a real clip 1 run: all 202 finalized events showed `repetition_count: 0`.

Decision: do not add per-frame retention to fix this — judged too large an architectural change for remaining time. Ship as a disclosed, honest 0 rather than a fabricated or estimated value.

Separately unresolved, not part of this decision: whether "burst count within one event" (this field's original definition) is even the right concept vs. P1's cross-event gesture-repetition idea ("3+ repeated identical angular snaps within a 2-min window"). If repetition/pattern-detection matters for the demo narrative, that's likely better served by P1's metric later, not by resurrecting this one.

Slide framing: state plainly as a known, disclosed limitation — not silently zero, not implied as "working but returning 0 because nothing repeated."

## Why this file exists

Both of the above were previously tracked only in conversational planning docs that are not committed to this repo, which meant the actual repo config (`phone_data.yaml`) could silently drift out of sync with the real decision — which is exactly what happened here. Check this file first before assuming project scope from memory or from an external doc.
