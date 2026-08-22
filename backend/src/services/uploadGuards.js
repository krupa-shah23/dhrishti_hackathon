/**
 * Shared upload admission checks — used by both the multer route
 * (routes/videos.js) and the tus resumable route (routes/tusUpload.js) so
 * the two upload paths enforce identical rules from one place.
 */
const path = require('path');
const checkDiskSpace = require('check-disk-space').default;
const { Video } = require('../models');
const config = require('../config');

async function getActiveVideoCount() {
  return Video.countDocuments({ status: { $nin: ['failed'] } });
}

async function getFreeDiskMB() {
  const { free } = await checkDiskSpace(path.resolve(config.uploadDir));
  return Math.floor(free / (1024 * 1024));
}

module.exports = { getActiveVideoCount, getFreeDiskMB };
