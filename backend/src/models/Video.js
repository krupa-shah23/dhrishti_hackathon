/**
 * Video Model
 * Tracks uploaded videos, their processing status, and storage path.
 */
const mongoose = require('mongoose');

const videoSchema = new mongoose.Schema(
  {
    filename: { type: String, required: true },
    originalName: { type: String, required: true },
    filepath: { type: String, required: true },
    mimetype: { type: String, default: 'video/mp4' },
    size: { type: Number, default: 0 },            // bytes
    duration: { type: Number, default: null },       // seconds (populated by ML)
    fps: { type: Number, default: null },            // frames per second (populated by ML)
    status: {
      type: String,
      enum: ['uploading', 'queued', 'ingesting', 'detecting', 'tracking', 'scoring', 'done', 'failed'],
      default: 'uploading',
    },
    processingStage: { type: String, default: null },  // e.g. "Stage 3/8 – ByteTrack"
    processingProgress: { type: Number, default: 0 },  // 0-100 percentage
    errorMessage: { type: String, default: null },
    // Paths to ML output artifacts
    trackingDataPath: { type: String, default: null },  // JSON file with per-frame bboxes
    heatmapPath: { type: String, default: null },       // heatmap image path
  },
  {
    timestamps: true,  // createdAt, updatedAt
  }
);

// Virtual for video_id (maps _id → video_id for API consistency)
videoSchema.virtual('video_id').get(function () {
  return this._id.toHexString();
});

videoSchema.set('toJSON', { virtuals: true });

module.exports = mongoose.model('Video', videoSchema);
