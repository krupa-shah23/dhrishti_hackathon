/**
 * Event (Incident) Routes
 * 
 * GET  /api/events            — List events (filterable by videoId)
 * GET  /api/events/:id        — Get single event details
 */
const express = require('express');
const { Event } = require('../models');

const router = express.Router();

// ─── GET /api/events ───
// Query params: ?videoId=xxx&page=1&limit=50&sort=confidenceScore&order=desc
// Frontend: populates the Incidents table.
router.get('/', async (req, res, next) => {
  try {
    const {
      videoId,
      page = 1,
      limit = 50,
      sort = 'confidenceScore',
      order = 'desc',
      activity,          // optional filter: ?activity=phone
      minConfidence,     // optional: ?minConfidence=60
    } = req.query;

    const filter = {};
    if (videoId) filter.videoId = videoId;
    if (activity) filter.activities = activity;
    if (minConfidence) filter.confidenceScore = { $gte: Number(minConfidence) };

    const skip = (Number(page) - 1) * Number(limit);
    const sortObj = { [sort]: order === 'asc' ? 1 : -1 };

    const [events, total] = await Promise.all([
      Event.find(filter)
        .populate('personIds', 'personLabel thumbnailPath seatId')
        .populate('videoId', 'filename originalName')
        .sort(sortObj)
        .skip(skip)
        .limit(Number(limit)),
      Event.countDocuments(filter),
    ]);

    res.json({
      success: true,
      count: events.length,
      total,
      page: Number(page),
      totalPages: Math.ceil(total / Number(limit)),
      data: events,
    });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/events/:id ───
router.get('/:id', async (req, res, next) => {
  try {
    const event = await Event.findById(req.params.id)
      .populate('personIds', 'personLabel thumbnailPath seatId')
      .populate('videoId', 'filename originalName');

    if (!event) {
      const err = new Error('Event not found');
      err.statusCode = 404;
      throw err;
    }

    res.json({ success: true, data: event });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
