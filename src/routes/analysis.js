/**
 * Analysis Routes
 * 
 * POST /api/analysis/summary/:videoId — Generate Gemini XAI natural-language summary
 */
const express = require('express');
const { Event, Video } = require('../models');
const config = require('../config');

const router = express.Router();

// ─── POST /api/analysis/summary/:videoId ───
// Pulls all events for a video, sends them to Gemini API,
// and returns a natural-language summary paragraph.
// Frontend: populates the XAI Summary Panel on the Incidents page.
router.post('/summary/:videoId', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.videoId);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    const events = await Event.find({ videoId: req.params.videoId })
      .populate('personIds', 'personLabel seatId')
      .lean();

    if (events.length === 0) {
      return res.json({
        success: true,
        data: {
          summary: 'No incidents were detected in this video.',
        },
      });
    }

    // Build a structured prompt for Gemini
    const eventsTable = events.map((e) => {
      const seat = e.seatId ? `Seat ${e.seatId}` : 'Unknown seat';
      const acts = e.activities.join(' and ');
      const duration = `${e.duration}s`;
      const objectStr = e.objectDetected ? `, ${e.objectDetected} detected ` : ' ';
      const conf = (e.confidenceScore / 100).toFixed(2);
      // Example: "Seat 14, hand movement toward neighbor, 6s, phone detected 0.62"
      return `${seat}, ${acts}, ${duration}${objectStr}${conf}`;
    });

    const prompt = `You are an AI analyst for an exam proctoring system called DRISHTI. 
Analyze the following detected incidents from video "${video.originalName}" and provide a concise, 
professional summary paragraph suitable for an exam supervisor. 
Highlight the most critical incidents, mention patterns, and note confidence levels.

Detected Incidents:
${JSON.stringify(eventsTable, null, 2)}

Provide a 3-5 sentence summary.`;

    // Check XAI Offline Toggle from Settings
    const { Setting } = require('../models');
    const xaiSetting = await Setting.findOne({ key: 'xaiToggle' });
    const isXaiEnabled = xaiSetting && xaiSetting.value === true;

    // Call Gemini API
    if (!config.geminiApiKey || !isXaiEnabled) {
      // Fallback: generate a basic summary without Gemini (Offline mode)
      const highConfidence = events.filter((e) => e.confidenceScore >= 70);
      return res.json({
        success: true,
        data: {
          summary: `[OFFLINE MODE] Analysis of "${video.originalName}": ${events.length} incidents detected involving ${new Set(events.flatMap((e) => e.personIds.map((p) => p.personLabel))).size} persons. ${highConfidence.length} incidents have high confidence (≥70%). Activities include: ${[...new Set(events.flatMap((e) => e.activities))].join(', ')}.`,
          source: 'fallback',
        },
      });
    }

    // Gemini API call
    const geminiResponse = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent?key=${config.geminiApiKey}`,
      {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          contents: [{ parts: [{ text: prompt }] }],
        }),
      }
    );

    const geminiData = await geminiResponse.json();
    const summary =
      geminiData?.candidates?.[0]?.content?.parts?.[0]?.text ||
      'Unable to generate summary.';

    res.json({ success: true, data: { summary, source: 'gemini' } });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
