/**
 * Shared post-write upload pipeline.
 *
 * Runs once a video file is fully, correctly present at rest on disk —
 * regardless of whether it got there via the single-POST multer route or a
 * completed tus resumable upload:
 *   1. ffprobe validation (corrupt/unreadable files are rejected)
 *   2. SHA-256 dedup (exact-content duplicates are rejected)
 *   3. Video.create(status: 'queued') + mlQueue.add, fail-fast/timeout wrapped
 *
 * Both upload routes call this instead of duplicating the logic.
 */
const fs = require('fs');
const crypto = require('crypto');
const { execFile } = require('child_process');
const ffprobePath = require('@ffprobe-installer/ffprobe').path;
const { Video, Setting } = require('../models');

// Bounds how long we wait on a BullMQ call before giving up, so a down Redis
// can never hang the caller indefinitely.
const withTimeout = (promise, ms, label) =>
  Promise.race([
    promise,
    new Promise((_, reject) => setTimeout(() => reject(new Error(`${label} timed out after ${ms}ms`)), ms)),
  ]);

// ioredis connection failures often surface as an AggregateError with an
// empty .message (the real reason is on .code, e.g. ECONNREFUSED).
const describeError = (err) => err.message || err.code || err.toString();

// Resolves to null if `filepath` is a valid, readable video; otherwise an
// error string describing why ffprobe rejected it (corrupt/unreadable file).
const probeVideo = (filepath) =>
  new Promise((resolve) => {
    execFile(
      ffprobePath,
      ['-v', 'error', '-show_entries', 'format=duration', '-of', 'json', filepath],
      { timeout: 15000 },
      (err, _stdout, stderr) => {
        if (err) {
          resolve((stderr && stderr.trim()) || err.message || 'ffprobe could not read the file');
        } else {
          resolve(null);
        }
      }
    );
  });

// Streams the file through SHA-256 instead of buffering it whole, so
// multi-GB uploads don't get loaded into memory just to hash them.
const hashFile = (filepath) =>
  new Promise((resolve, reject) => {
    const hash = crypto.createHash('sha256');
    const stream = fs.createReadStream(filepath);
    stream.on('error', reject);
    stream.on('data', (chunk) => hash.update(chunk));
    stream.on('end', () => resolve(hash.digest('hex')));
  });

/**
 * @param {{filepath: string, filename: string, originalName: string, mimetype: string, size: number}} file
 * @returns {Promise<{outcome: 'invalid'|'duplicate'|'created', video: object, details?: string}>}
 */
async function processUploadedFile({ filepath, filename, originalName, mimetype, size }) {
  const probeError = await probeVideo(filepath);
  if (probeError) {
    fs.unlinkSync(filepath);
    console.error(`❌  Rejected upload "${originalName}" — ffprobe validation failed: ${probeError}`);
    // Reuses the existing 'failed' status (not a new 'error' value) — the
    // frontend already special-cases 'failed' for red/error styling and
    // excludes it from the active-slot count, so this needs no frontend change.
    const failedVideo = await Video.create({
      filename,
      originalName,
      filepath, // file has been deleted above; kept for audit trail
      mimetype,
      size,
      status: 'failed',
      errorMessage: `Not a valid/readable video file: ${probeError}`,
    });
    return { outcome: 'invalid', video: failedVideo, details: probeError };
  }

  // Dedup by content, not filename — reject if these exact bytes were
  // already uploaded (regardless of what the file was named this time).
  const contentHash = await hashFile(filepath);
  const duplicate = await Video.findOne({ contentHash });
  if (duplicate) {
    fs.unlinkSync(filepath);
    console.warn(`⚠️  Duplicate upload rejected: "${originalName}" matches existing video ${duplicate._id} (${duplicate.originalName})`);
    return { outcome: 'duplicate', video: duplicate };
  }

  const video = await Video.create({
    filename,
    originalName,
    filepath,
    mimetype,
    size,
    contentHash,
    status: 'queued',
  });

  // Push ML processing job to BullMQ (bounded — must never hang the caller)
  try {
    const { mlQueue } = require('../queues');
    const seatGridSetting = await Setting.findOne({ key: 'seatGrid' });

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
    console.log(`📤  Queued ML job for video: ${video.originalName}`);
  } catch (queueErr) {
    const reason = describeError(queueErr);
    console.error(`❌  ML job NOT queued for video ${video._id} (${video.originalName}) — Redis/BullMQ unavailable or timed out: ${reason}`);
    video.errorMessage = `Queued for upload, but ML queue was unreachable at upload time (${reason}). Use POST /api/videos/${video._id}/requeue once it's back.`;
    await video.save();
    // Video stays status="queued" — it's a valid, complete upload; only
    // hand-off to the ML pipeline failed, and it can be retried via /requeue.
  }

  return { outcome: 'created', video };
}

module.exports = { processUploadedFile, hashFile, probeVideo, describeError, withTimeout };
