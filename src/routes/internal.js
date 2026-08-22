/**
 * Internal ML Webhook Routes
 *
 * POST /internal/events              — upsert one ML event (idempotent on mlEventId)
 * POST /internal/complete/:video_id  — mark a video's ML processing done
 *
 * Not mounted under /api — these are pipeline-to-backend webhooks, not
 * frontend-facing endpoints (see server.js).
 */
const express = require('express');
const { Event, Video } = require('../models');

const router = express.Router();

// ─── POST /internal/events ───
// Body: { mlEventId, videoId, seatId, objectDetected, objectConfidence, timestamps, bboxOverlay }
// (shape produced by event_adapter.adapt_bridge_event_to_node_payload)
router.post('/events', async (req, res, next) => {
  try {
    const { mlEventId, videoId, seatId, objectDetected, objectConfidence, timestamps, bboxOverlay } = req.body;

    if (!mlEventId || !videoId) {
      const err = new Error('mlEventId and videoId are required');
      err.statusCode = 400;
      throw err;
    }

    const event = await Event.findOneAndUpdate(
      { mlEventId },
      {
        mlEventId,
        videoId,
        seatId: seatId ?? null,
        objectDetected: objectDetected ?? null,
        objectConfidence: objectConfidence ?? null,
        timestamps: timestamps ?? [],
        bboxOverlay: bboxOverlay ?? [],
      },
      { upsert: true, new: true, setDefaultsOnInsert: true }
    );

    res.json({ success: true, data: event });
  } catch (err) {
    next(err);
  }
});

// ─── POST /internal/complete/:video_id ───
// Body: { total_events } (per Backend Master Doc §6) for the success case.
// Body: { status: 'failed', error_message } for the pipeline-failed case --
// no separate error-status route existed, so this route now doubles as
// both the success and failure terminal signal (both are "the pipeline is
// done processing this video, here's the outcome").
router.post('/complete/:video_id', async (req, res, next) => {
  try {
    const { status, error_message } = req.body || {};
    const isFailure = status === 'failed';

    const video = await Video.findByIdAndUpdate(
      req.params.video_id,
      isFailure
        ? { status: 'failed', errorMessage: error_message ?? 'Unknown pipeline error', processingProgress: 100 }
        : { status: 'done', processingProgress: 100 },
      { new: true }
    );

    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    res.json({ success: true, data: video });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
