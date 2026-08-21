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
// Returns all videos, sorted newest first.
// Frontend: populates the Upload Manager page slots.
router.get('/', async (req, res, next) => {
  try {
    const videos = await Video.find().sort({ createdAt: -1 });
    res.json({ success: true, count: videos.length, data: videos });
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
      status: { $nin: ['failed'] },
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

    // Push ML processing job to BullMQ
    try {
      const { mlQueue } = require('../queues');
      const { Setting } = require('../models');
      const seatGridSetting = await Setting.findOne({ key: 'seatGrid' });

      await mlQueue.add('process-video', {
        videoId: video._id.toString(),
        filepath: video.filepath,
        filename: video.originalName,
        seatGrid: seatGridSetting ? seatGridSetting.value : null,
      }, {
        attempts: 2,
        backoff: { type: 'exponential', delay: 5000 },
        removeOnComplete: true,
        removeOnFail: false,
      });
      console.log(`📤  Queued ML job for video: ${video.originalName}`);
    } catch (queueErr) {
      console.warn('⚠️  Redis/BullMQ unavailable, ML job not queued:', queueErr.message);
      // Video is still saved — can be manually requeued later
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
// Deletes the video record AND the file from disk.
router.delete('/:id', async (req, res, next) => {
  try {
    const video = await Video.findById(req.params.id);
    if (!video) {
      const err = new Error('Video not found');
      err.statusCode = 404;
      throw err;
    }

    // Delete file from disk
    if (video.filepath && fs.existsSync(video.filepath)) {
      fs.unlinkSync(video.filepath);
    }

    // Delete tracking data file if it exists
    if (video.trackingDataPath && fs.existsSync(video.trackingDataPath)) {
      fs.unlinkSync(video.trackingDataPath);
    }

    await Video.findByIdAndDelete(req.params.id);
    res.json({ success: true, message: 'Video deleted' });
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

    const { mlQueue } = require('../queues');
    const { Setting } = require('../models');
    const seatGridSetting = await Setting.findOne({ key: 'seatGrid' });

    await mlQueue.add('process-video', {
      videoId: video._id.toString(),
      filepath: video.filepath,
      filename: video.originalName,
      seatGrid: seatGridSetting ? seatGridSetting.value : null,
    }, {
      attempts: 2,
      backoff: { type: 'exponential', delay: 5000 },
      removeOnComplete: true,
      removeOnFail: false,
    });

    res.json({ success: true, message: 'Video requeued for processing', data: video });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
