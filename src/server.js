/**
 * DRISHTI Backend — Express Server Entry Point
 */
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const morgan = require('morgan');
const path = require('path');
const fs = require('fs');

const config = require('./config');
const connectDB = require('./config/db');
const errorHandler = require('./middleware/errorHandler');
const {
  videoRoutes,
  eventRoutes,
  personRoutes,
  dashboardRoutes,
  analysisRoutes,
  settingsRoutes,
  internalRoutes,
} = require('./routes');

const app = express();

// ─── Ensure uploads directory exists ───
const uploadsDir = path.resolve(config.uploadDir);
if (!fs.existsSync(uploadsDir)) {
  fs.mkdirSync(uploadsDir, { recursive: true });
}

// ─── Core Middleware ───
app.use(helmet({ contentSecurityPolicy: false }));
app.use(cors({ origin: '*' }));
app.use(morgan('tiny')); // Use tiny format — avoids logging large multipart bodies
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

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

// ─── API Routes ───
app.use('/api/videos', videoRoutes);
app.use('/api/events', eventRoutes);
app.use('/api/persons', personRoutes);
app.use('/api/dashboard', dashboardRoutes);
app.use('/api/analysis', analysisRoutes);
app.use('/api/settings', settingsRoutes);
app.use('/internal', internalRoutes);

// ─── 404 handler ───
app.use((_req, res) => {
  res.status(404).json({ success: false, error: 'Route not found' });
});

// ─── Error handler ───
app.use(errorHandler);

// ─── Start Server ───
const start = async () => {
  await connectDB();

  // Start BullMQ status listener (if Redis is available)
  try {
    const startStatusWorker = require('./queues/statusWorker');
    startStatusWorker();
  } catch (err) {
    console.warn('⚠️  Redis/BullMQ not available. Status worker not started:', err.message);
  }

  const server = app.listen(config.port, () => {
    // Increase timeouts so large video uploads don't get killed mid-transfer
    server.keepAliveTimeout = 300000;   // 5 minutes
    server.headersTimeout  = 310000;   // slightly higher than keepAlive
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
