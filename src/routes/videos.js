/**
 * Video Routes
 * 
 * GET    /api/videos          — List all videos (for Upload Manager page)
 * POST   /api/videos/upload   — Upload a new video
 * GET    /api/videos/:id      — Get single video details
 * DELETE /api/videos/:id      — Delete a video (free up a slot)
 * GET    /api/videos/:id/stream  — Stream video for the player
 * GET    /api/videos/:id/tracking — Get bounding box / tracking data JSON
 */
const express = require('express');
const fs = require('fs');
const path = require('path');
const { Video } = require('../models');
const upload = require('../middleware/upload');
const config = require('../config');

const router = express.Router();

// ─── GET /api/videos ───
// Returns all non-archived videos, sorted newest first.
// Frontend: populates the Upload Manager page slots.
router.get('/', async (req, res, next) => {
  try {
    // Exclude archived — they've been deleted from disk and freed their slot
    const videos = await Video.find({ status: { $ne: 'archived' } }).sort({ createdAt: -1 });
    res.json({ success: true, count: videos.length, data: videos });
  } catch (err) {
    next(err);
  }
});

// ─── POST /api/videos/register ───
// Lightweight registration: stores metadata only, NO file upload.
// The frontend can call this immediately after the user selects a file,
// then the ML service fetches/processes separately — making upload feel instant.
router.post('/register', async (req, res, next) => {
  try {
    const activeCount = await Video.countDocuments({
      status: { $nin: ['failed', 'archived'] },
    });
    if (activeCount >= config.maxVideoSlots) {
      const err = new Error(`Maximum ${config.maxVideoSlots} video slots are in use. Delete a video first.`);
      err.statusCode = 409;
      throw err;
    }

    const { originalName, size, mimetype } = req.body;
    if (!originalName) {
      const err = new Error('originalName is required.');
      err.statusCode = 400;
      throw err;
    }

    // Use a placeholder filename; filepath is empty until ML pulls the file
    const video = await Video.create({
      filename: originalName,
      originalName,
      filepath: '',
      mimetype: mimetype || 'video/mp4',
      size: size || 0,
      status: 'queued',
    });

    // Launch mock pipeline asynchronously (gated: see config.useMockPipeline)
    if (config.useMockPipeline) {
      try {
        const { executeMockPipeline } = require('../queues/mockPipeline');
        executeMockPipeline(video._id);
        console.log(`📤  Started mock pipeline for registered video: ${originalName}`);
      } catch (pipelineErr) {
        console.warn('⚠️  Failed to start mock pipeline:', pipelineErr.message);
      }
    }

    res.status(201).json({ success: true, data: video });
  } catch (err) {
    next(err);
  }
});

// ─── POST /api/videos/upload ───
// Accepts a single video file. Enforces max 3 active video slots.
router.post('/upload', upload.single('video'), async (req, res, next) => {
  try {
    // Enforce max video slot constraint
    const activeCount = await Video.countDocuments({
      status: { $nin: ['failed', 'archived'] },
    });
    if (activeCount >= config.maxVideoSlots) {
      // Clean up the file that was already saved by multer
      if (req.file) {
        fs.unlinkSync(req.file.path);
      }
      const err = new Error(`Maximum ${config.maxVideoSlots} video slots are in use. Delete a video first.`);
      err.statusCode = 409;
      throw err;
    }

    if (!req.file) {
      const err = new Error('No video file provided.');
      err.statusCode = 400;
      throw err;
    }

    const video = await Video.create({
      filename: req.file.filename,
      originalName: req.file.originalname,
      filepath: req.file.path,
      mimetype: req.file.mimetype,
      size: req.file.size,
      status: 'queued',
    });

    // Launch in-memory mock pipeline directly (bypassing Redis/BullMQ for local dev)
    // (gated: see config.useMockPipeline)
    if (config.useMockPipeline) {
      try {
        const { executeMockPipeline } = require('../queues/mockPipeline');
        executeMockPipeline(video._id); // Run asynchronously
        console.log(`📤  Started mock pipeline for video: ${video.originalName}`);
      } catch (pipelineErr) {
        console.warn('⚠️  Failed to start mock pipeline:', pipelineErr.message);
      }
    }

    res.status(201).json({ success: true, data: video });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/videos/:id ───
router.get('/:id', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
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

// ─── DELETE /api/videos/:id ───
// Archives the video record AND deletes the file from disk.
// This frees up the upload slot (since GET /api/videos excludes archived)
// but keeps the record in the DB so 'Total Videos Processed' count is preserved.
router.delete('/:id', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    // Delete video file from disk
    if (video.filepath && fs.existsSync(video.filepath)) {
      try { fs.unlinkSync(video.filepath); } catch (_) {}
    }

    // Delete tracking data file from disk
    if (video.trackingDataPath && fs.existsSync(video.trackingDataPath)) {
      try { fs.unlinkSync(video.trackingDataPath); } catch (_) {}
    }

    // Archive the video instead of deleting it
    video.status = 'archived';
    video.filepath = ''; // Clear file path since it's deleted from disk
    await video.save();

    res.json({ success: true, message: 'Video deleted and slot freed (archived for stats).' });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/videos/:id/stream ───
// Streams the video file with range-request support (seek support for <video>).
router.get('/:id/stream', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    const videoPath = path.resolve(video.filepath);
    if (!fs.existsSync(videoPath)) {
      const err = new Error('Video file not found on disk');
      err.statusCode = 404;
      throw err;
    }

    const stat = fs.statSync(videoPath);
    const fileSize = stat.size;
    const range = req.headers.range;

    if (range) {
      const parts = range.replace(/bytes=/, '').split('-');
      const start = parseInt(parts[0], 10);
      const end = parts[1] ? parseInt(parts[1], 10) : fileSize - 1;
      const chunkSize = end - start + 1;

      const stream = fs.createReadStream(videoPath, { start, end });
      res.writeHead(206, {
        'Content-Range': `bytes ${start}-${end}/${fileSize}`,
        'Accept-Ranges': 'bytes',
        'Content-Length': chunkSize,
        'Content-Type': video.mimetype || 'video/mp4',
      });
      stream.pipe(res);
    } else {
      res.writeHead(200, {
        'Content-Length': fileSize,
        'Content-Type': video.mimetype || 'video/mp4',
      });
      fs.createReadStream(videoPath).pipe(res);
    }
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/videos/:id/tracking ───
// Returns the bounding-box tracking JSON for the Canvas overlay.
router.get('/:id/tracking', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    if (!video.trackingDataPath || !fs.existsSync(video.trackingDataPath)) {
      return res.json({ success: true, data: [] });
    }

    const trackingData = JSON.parse(fs.readFileSync(video.trackingDataPath, 'utf8'));
    res.json({ success: true, data: trackingData });
  } catch (err) {
    next(err);
  }
});

// ─── POST /api/videos/:id/requeue ───
// Manually re-submit a video to the ML processing queue.
router.post('/:id/requeue', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    // Reset status
    video.status = 'queued';
    video.processingStage = null;
    video.processingProgress = 0;
    video.errorMessage = null;
    await video.save();

    // Launch in-memory mock pipeline directly (gated: see config.useMockPipeline)
    if (config.useMockPipeline) {
      try {
        const { executeMockPipeline } = require('../queues/mockPipeline');
        executeMockPipeline(video._id);
      } catch (pipelineErr) {
        console.warn('⚠️  Failed to start mock pipeline:', pipelineErr.message);
      }
    }

    res.json({ success: true, message: 'Video requeued for processing', data: video });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
