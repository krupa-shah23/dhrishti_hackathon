/**
 * Person Routes
 * 
 * GET  /api/persons              — List all detected persons
 * GET  /api/persons/:id          — Person profile
 * GET  /api/persons/:id/timeline — Confidence over time for the person
 * GET  /api/persons/:id/videos   — Cross-session: all videos this person appears in
 */
const express = require('express');
const { Person, Event, PersonVideoMap } = require('../models');

const router = express.Router();

// ─── GET /api/persons ───
router.get('/', async (req, res, next) => {
  try {
    const persons = await Person.find()
      .select('-embeddingVector')  // don't send huge vectors to the client
      .sort({ createdAt: -1 });
    res.json({ success: true, count: persons.length, data: persons });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/persons/:id ───
router.get('/:id', async (req, res, next) => {
  try {
    const person = await Person.findById(req.params.id)
      .select('-embeddingVector');
    if (!person) {
      const err = new Error('Person not found');
      err.statusCode = 404;
      throw err;
    }
    res.json({ success: true, data: person });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/persons/:id/timeline ───
// Returns all events involving this person, with timestamps and confidence.
// Frontend: renders the "Person Timeline" confidence-vs-time graph.
router.get('/:id/timeline', async (req, res, next) => {
  try {
    const events = await Event.find({ personIds: req.params.id })
      .select('timestamps confidenceScore activities videoId')
      .populate('videoId', 'filename')
      .sort({ 'timestamps.0.start': 1 });

    const timeline = events.map((e) => ({
      eventId: e._id,
      videoId: e.videoId?._id,
      videoName: e.videoId?.filename,
      activities: e.activities,
      confidence: e.confidenceScore,
      timestamps: e.timestamps,
    }));

    res.json({ success: true, data: timeline });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/persons/:id/videos ───
// Cross-session tracking: all videos where this person was detected.
// Frontend: populates the Person Profile "Cross-Video Reference" list.
router.get('/:id/videos', async (req, res, next) => {
  try {
    const mappings = await PersonVideoMap.find({ personId: req.params.id })
      .populate('videoId', 'filename originalName duration status createdAt')
      .sort({ createdAt: -1 });

    res.json({ success: true, count: mappings.length, data: mappings });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
