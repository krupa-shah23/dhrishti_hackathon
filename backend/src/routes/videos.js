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
const { getActiveVideoCount, getFreeDiskMB } = require('../services/uploadGuards');
const { processUploadedFile, describeError, withTimeout } = require('../services/uploadPipeline');

const router = express.Router();

// Runs BEFORE multer touches the request body, so an over-capacity upload is
// rejected without ever parsing the multipart body or writing a file to disk.
const checkSlotAvailable = async (req, res, next) => {
  try {
    const activeCount = await getActiveVideoCount();
    if (activeCount >= config.maxVideoSlots) {
      res.status(409).json({
        success: false,
        error: `Maximum ${config.maxVideoSlots} video slots are in use. Delete a video first.`,
      });
      // Stop the client from continuing to stream the file body into the void.
      res.on('finish', () => req.destroy());
      return;
    }
    next();
  } catch (err) {
    next(err);
  }
};

// Runs BEFORE multer, alongside checkSlotAvailable — rejects fast if the
// upload volume doesn't have enough headroom, before any bytes are written.
const checkDiskSpaceAvailable = async (req, res, next) => {
  try {
    const freeMB = await getFreeDiskMB();
    if (freeMB < config.minFreeDiskMB) {
      console.error(`❌  Rejecting upload — only ${freeMB}MB free on upload volume, need at least ${config.minFreeDiskMB}MB.`);
      res.status(507).json({
        success: false,
        error: `Insufficient storage: only ${freeMB}MB free, need at least ${config.minFreeDiskMB}MB.`,
      });
      res.on('finish', () => req.destroy());
      return;
    }
    next();
  } catch (err) {
    next(err);
  }
};

// Wraps multer so a write failure mid-transfer (e.g. disk fills up despite
// the precheck, from a concurrent upload) is caught here instead of falling
// through to the route handler with a half-written file and no req.file.
const uploadSingleVideo = (req, res, next) => {
  upload.single('video')(req, res, (err) => {
    if (!err) return next();

    // Clean up whatever multer managed to write before it failed.
    if (req._pendingUploadFilename) {
      const partialPath = path.join(path.resolve(config.uploadDir), req._pendingUploadFilename);
      if (fs.existsSync(partialPath)) {
        fs.unlinkSync(partialPath);
        console.warn(`🧹  Cleaned up partial upload after write error: ${partialPath}`);
      }
    }

    const reason = describeError(err);
    const isDiskFull = err.code === 'ENOSPC' || /ENOSPC|no space left/i.test(reason);
    if (isDiskFull) {
      console.error(`❌  Disk filled up mid-write, rejecting upload: ${reason}`);
      return res.status(507).json({ success: false, error: `Insufficient storage while writing the file: ${reason}` });
    }

    err.statusCode = err.statusCode || 400;
    next(err);
  });
};

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
// Accepts a single video file. Enforces max 3 active video slots and a
// minimum-free-disk-space floor — both checked before multer parses the
// body, so a rejected request never hits disk.
router.post('/upload', checkSlotAvailable, checkDiskSpaceAvailable, uploadSingleVideo, async (req, res, next) => {
  try {
    if (!req.file) {
      const err = new Error('No video file provided.');
      err.statusCode = 400;
      throw err;
    }

    const result = await processUploadedFile({
      filepath: req.file.path,
      filename: req.file.filename,
      originalName: req.file.originalname,
      mimetype: req.file.mimetype,
      size: req.file.size,
    });

    if (result.outcome === 'invalid') {
      return res.status(422).json({
        success: false,
        error: 'Uploaded file is not a valid or readable video.',
        details: result.details,
        data: result.video,
      });
    }
    if (result.outcome === 'duplicate') {
      return res.status(409).json({
        success: false,
        error: 'This video has already been uploaded.',
        details: `Matches existing video "${result.video.originalName}" (video_id: ${result.video._id}).`,
        data: result.video,
      });
    }

    res.status(201).json({ success: true, data: result.video });
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

    try {
      await withTimeout(mlQueue.add('process-video', {
        videoId: video._id.toString(),
        filepath: video.filepath,
        filename: video.originalName,
        seatGrid: seatGridSetting ? seatGridSetting.value : null,
      }, {
        attempts: 2,
        backoff: { type: 'exponential', delay: 5000 },
        removeOnComplete: true,
        removeOnFail: false,
      }), 5000, 'mlQueue.add');
    } catch (queueErr) {
      const reason = describeError(queueErr);
      console.error(`❌  Requeue failed for video ${video._id} — Redis/BullMQ unavailable or timed out: ${reason}`);
      const err = new Error(`ML queue unreachable, could not requeue: ${reason}`);
      err.statusCode = 503;
      throw err;
    }

    res.json({ success: true, message: 'Video requeued for processing', data: video });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
