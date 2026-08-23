/**
 * Resumable (tus-protocol) video upload endpoint.
 *
 * Mounted at TUS_PATH (see server.js). Uses @tus/server + @tus/file-store —
 * tus-node-server is essentially the older, unmaintained iteration of the
 * same project; the tus team moved active development to the @tus/* scoped
 * packages, so those are what's installed here.
 *
 * The existing single-POST /api/videos/upload (multer) route is left in
 * place and fully working — this is an additive resumable path, not a hard
 * cutover, so any client that hasn't switched to tus-js-client yet keeps
 * working exactly as before.
 *
 * Lifecycle:
 *   - onUploadCreate (tus "creation" step, before any chunk is accepted):
 *     same slot-limit and disk-space checks as the multer route, reusing
 *     services/uploadGuards so both paths enforce identical rules.
 *   - chunks (PATCH) are handled entirely by @tus/server + FileStore.
 *   - onUploadFinish (all chunks received, file fully assembled): move the
 *     assembled file from the tus scratch directory into the real uploads
 *     directory, then run the exact same post-upload pipeline the multer
 *     route uses (services/uploadPipeline) — ffprobe, SHA-256 dedup,
 *     Video.create, mlQueue.add.
 *
 * Session persistence: @tus/file-store's default configstore (FileConfigstore)
 * writes each upload's offset/metadata to a JSON file next to the data file,
 * so an in-progress upload survives a server restart and a client can resume
 * it — no extra work needed for that. Sessions abandoned by the client
 * (never resumed) are swept via FileStore's built-in expiration
 * (see EXPIRATION_MS below) at startup, alongside the existing orphan
 * upload-directory reconciliation.
 */
const path = require('path');
const fs = require('fs');
const { v4: uuidv4 } = require('uuid');
const { Server } = require('@tus/server');
const { FileStore } = require('@tus/file-store');
const config = require('../config');
const { getActiveVideoCount, getFreeDiskMB } = require('../services/uploadGuards');
const { processUploadedFile } = require('../services/uploadPipeline');

const TUS_PATH = '/api/videos/upload/tus';
const EXPIRATION_MS = 24 * 60 * 60 * 1000; // abandoned tus sessions older than this get purged at startup

const tusUploadDir = path.resolve(config.tusUploadDir);
const datastore = new FileStore({ directory: tusUploadDir, expirationPeriodInMilliseconds: EXPIRATION_MS });

const jsonError = (status_code, error) => ({
  status_code,
  body: JSON.stringify({ success: false, error }),
});

const tusServer = new Server({
  path: TUS_PATH,
  datastore,
  // Matches the rest of this app's permissive `cors()` middleware — tus's
  // own CORS layer is separate because it needs to allow tus-specific
  // request headers (Upload-Offset, Tus-Resumable, etc.) on preflight.
  allowedOrigins: () => true,

  // Runs before any chunk is accepted — same admission rules as the multer route.
  onUploadCreate: async (req, upload) => {
    const activeCount = await getActiveVideoCount();
    if (activeCount >= config.maxVideoSlots) {
      throw jsonError(409, `Maximum ${config.maxVideoSlots} video slots are in use. Delete a video first.`);
    }
    const freeMB = await getFreeDiskMB();
    if (freeMB < config.minFreeDiskMB) {
      console.error(`❌  Rejecting tus upload — only ${freeMB}MB free on upload volume, need at least ${config.minFreeDiskMB}MB.`);
      throw jsonError(507, `Insufficient storage: only ${freeMB}MB free, need at least ${config.minFreeDiskMB}MB.`);
    }
    return {};
  },

  // Runs once all chunks have been received and the file is fully assembled.
  onUploadFinish: async (req, upload) => {
    // @tus/file-store writes the assembled file as `<tusUploadDir>/<upload.id>`
    // (no extension). upload.storage.path is only populated on the FileStore's
    // internal copy of the Upload object, NOT reliably on the `upload` arg
    // passed here — derive the path directly from upload.id instead.
    const assembledPath = path.join(tusUploadDir, upload.id);

    if (!fs.existsSync(assembledPath)) {
      console.error(`❌  tus onUploadFinish: assembled file not found at ${assembledPath}`);
      return jsonError(500, 'Assembled upload file not found on disk. Please retry the upload.');
    }

    const originalName = (upload.metadata && upload.metadata.filename) || upload.id;
    const mimetype = (upload.metadata && upload.metadata.filetype) || 'video/mp4';
    const ext = path.extname(originalName) || '.mp4';
    const finalFilename = `${uuidv4()}${ext}`;
    // Stored as a relative path (matches what multer's req.file.path already
    // produces for the non-resumable route) so both upload paths leave
    // Video.filepath in the same format.
    const finalRelPath = path.join(config.uploadDir, finalFilename);
    const finalAbsPath = path.resolve(finalRelPath);

    console.log(`📦  tus upload ${upload.id} complete — moving to ${finalAbsPath}`);

    // Same-volume rename: atomic, no copy overhead even for multi-GB files.
    try {
      fs.renameSync(assembledPath, finalAbsPath);
    } catch (renameErr) {
      console.error(`❌  tus rename failed: ${renameErr.message}`);
      return jsonError(500, `Could not finalise the upload: ${renameErr.message}`);
    }

    // Best-effort: drop the now-stale tus session bookkeeping for this
    // upload (the data file it pointed at no longer exists there).
    try {
      await datastore.configstore.delete(upload.id);
    } catch (err) {
      console.warn(`⚠️  Could not clean up tus session metadata for ${upload.id}:`, err.message);
    }

    const result = await processUploadedFile({
      filepath: finalRelPath,
      filename: finalFilename,
      originalName,
      mimetype,
      size: upload.size,
    });

    if (result.outcome === 'invalid') {
      return jsonError(422, `Uploaded file is not a valid or readable video: ${result.details}`);
    }
    if (result.outcome === 'duplicate') {
      return jsonError(409, `This video has already been uploaded (matches existing video "${result.video.originalName}", video_id: ${result.video._id}).`);
    }

    return {
      status_code: 200,
      body: JSON.stringify({ success: true, data: result.video }),
    };
  },
});

/** Purges tus sessions abandoned longer than EXPIRATION_MS. Call once at startup. */
async function cleanupExpiredTusSessions() {
  const removed = await datastore.deleteExpired();
  console.log(`🧹  tus startup cleanup: removed ${removed} abandoned/expired upload session(s).`);
  return removed;
}

/** Mounts the tus handler on the Express app at TUS_PATH (+ subpaths for chunk PATCHes). */
function mountTus(app) {
  // Express 4 (path-to-regexp v0.1.x) has no optional-group syntax for this,
  // so the bare creation path and the /:id chunk-PATCH path are registered
  // separately rather than as one pattern.
  app.all(TUS_PATH, (req, res) => tusServer.handle(req, res));
  app.all(`${TUS_PATH}/*`, (req, res) => tusServer.handle(req, res));
}

module.exports = { mountTus, cleanupExpiredTusSessions, TUS_PATH };
