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

  // Gemini XAI
  geminiApiKey: process.env.GEMINI_API_KEY || '',

  // ML microservice
  mlServiceUrl: process.env.ML_SERVICE_URL || 'http://localhost:8000',
  // Gate for routes/videos.js's executeMockPipeline auto-trigger (register/
  // upload/requeue). Defaults true (current/unchanged behavior) so nothing
  // breaks for existing dev/demo flows that rely on the mock pipeline's
  // fake Person/Event/status progression when no real ML backend is wired
  // up. Set USE_MOCK_PIPELINE=false once the real ML service at
  // mlServiceUrl is actually driving these videos via /process, to stop it
  // racing on Video.status against real pipeline runs.
  useMockPipeline: process.env.USE_MOCK_PIPELINE !== 'false',
};
