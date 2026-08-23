/**
 * Multer configuration for video uploads.
 * Stores files in the uploads/ directory with unique filenames.
 */
const multer = require('multer');
const path = require('path');
const { v4: uuidv4 } = require('uuid');
const config = require('../config');

const storage = multer.diskStorage({
  destination: (_req, _file, cb) => {
    cb(null, config.uploadDir);
  },
  filename: (req, file, cb) => {
    const ext = path.extname(file.originalname);
    const generated = `${uuidv4()}${ext}`;
    // Stash the name before the write starts: if the write fails partway
    // (e.g. disk fills up), multer never sets req.file, so this is the only
    // way the route can find and clean up the partial file afterward.
    req._pendingUploadFilename = generated;
    cb(null, generated);
  },
});

const fileFilter = (_req, file, cb) => {
  // Accept any video/* MIME type — covers MP4, MKV (video/matroska),
  // AVI, MOV, WebM, etc. A prefix check is more reliable than a
  // hardcoded list because browsers vary in which exact string they send.
  if (file.mimetype.startsWith('video/')) {
    cb(null, true);
  } else {
    cb(new Error(`Only video files are allowed. Received: ${file.mimetype}`), false);
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 10 * 1024 * 1024 * 1024, // 10 GB max
  },
});

module.exports = upload;
