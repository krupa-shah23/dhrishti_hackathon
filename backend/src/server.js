/**
 * DRISHTI Backend — Express Server Entry Point
 * Main server configuration and route initialization
 */
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');

const config = require('./config');
const connectDB = require('./config/db');
const reconcileUploads = require('./utils/reconcileUploads');
const { mountTus, cleanupExpiredTusSessions } = require('./routes/tusUpload');
const errorHandler = require('./middleware/errorHandler');
const {
  videoRoutes,
  eventRoutes,
  personRoutes,
  dashboardRoutes,
  analysisRoutes,
  settingsRoutes,
} = require('./routes');

const app = express();

// ─── Ensure uploads directory exists ───
const uploadsDir = path.resolve(config.uploadDir);
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// ─── Core Middleware ───
app.use(helmet());
app.use(cors());
app.use(morgan('dev'));
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true }));

// ─── Static files (thumbnails, heatmaps) ───
app.use('/static', express.static(uploadsDir));

// ─── Health Check ───
app.get('/api/health', (_req, res) => {
  res.json({
    success: true,
    service: 'drishti-backend',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
  });
});

// ─── Resumable (tus) upload endpoint — mounted before /api/videos so its
// multi-segment paths (/api/videos/upload/tus[/:id]) are never shadowed. ───
mountTus(app);

// ─── API Routes ───
app.use('/api/videos', videoRoutes);
app.use('/api/events', eventRoutes);
app.use('/api/persons', personRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/settings', settingsRoutes);

// ─── 404 handler ───
app.use((_req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

// ─── Error handler ───
app.use(errorHandler);

// ─── Start Server ───
const start = async () => {
  await connectDB();

  // Reconcile uploads/ vs. Video docs once at boot, before serving traffic —
  // catches orphaned files/docs left behind by a crash or restart mid-upload.
  try {
    await reconcileUploads();
  } catch (err) {
    console.warn('⚠️  Startup reconciliation failed (continuing anyway):', err.message);
  }

  // Purge tus (resumable upload) sessions nobody ever came back to finish.
  try {
    await cleanupExpiredTusSessions();
  } catch (err) {
    console.warn('⚠️  tus session cleanup failed (continuing anyway):', err.message);
  }

  // Start BullMQ status listener (if Redis is available)
  try {
    const startStatusWorker = require('./queues/statusWorker');
    startStatusWorker();
  } catch (err) {
    console.warn('⚠️  Redis/BullMQ not available. Status worker not started:', err.message);
    // Fall back to dev mock processor so videos still get processed end-to-end
    const { startMockProcessor } = require('./queues/mockProcessor');
    startMockProcessor();
  }

  app.listen(config.port, () => {
    console.log(`
╔══════════════════════════════════════════╗
║   🔱  DRISHTI Backend Server             ║
║   🌐  http://localhost:${config.port}            ║
║   📦  Environment: ${config.nodeEnv.padEnd(18)}║
╚══════════════════════════════════════════╝
    `);
  });
};

start();

module.exports = app;
