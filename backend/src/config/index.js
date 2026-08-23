/**
 * DRISHTI Backend — Central Configuration
 * Reads from .env and exports a typed config object.
 */
require('dotenv').config();

module.exports = {
  port: parseInt(process.env.PORT, 10) || 5000,
  nodeEnv: process.env.NODE_ENV || 'development',

  // MongoDB
  mongoUri: process.env.MONGODB_URI || 'mongodb://localhost:27017/drishti',

  // Redis (BullMQ)
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT, 10) || 6379,
  },

  // Video uploads
  uploadDir: process.env.UPLOAD_DIR || './uploads',
  maxVideoSlots: parseInt(process.env.MAX_VIDEO_SLOTS, 10) || 3,
  // Minimum free space (MB) required on the upload volume before accepting a
  // new upload. Default 2048MB (2GB): comfortably above a typical single
  // exam-session recording for this app, well under multer's hard 10GB/file
  // cap in middleware/upload.js — low enough not to false-positive-block
  // normal uploads, high enough to catch a genuinely low-space disk early.
  minFreeDiskMB: parseInt(process.env.MIN_FREE_DISK_MB, 10) || 2048,
  // Scratch directory for in-progress tus (resumable) uploads. Deliberately
  // a sibling of uploadDir, not nested inside it — the startup orphan-cleanup
  // scan (utils/reconcileUploads.js) only knows about finished videos in
  // uploadDir, so keeping tus's own bookkeeping files out of that directory
  // avoids them ever being misidentified as orphans mid-transfer.
  tusUploadDir: process.env.TUS_UPLOAD_DIR || './tus-tmp',

  // Gemini XAI
  geminiApiKey: process.env.GEMINI_API_KEY || '',

  // ML microservice
  mlServiceUrl: process.env.ML_SERVICE_URL || 'http://localhost:8000',
};
