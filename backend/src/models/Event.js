/**
 * Event Model
 * An incident/activity detected by the ML pipeline within a specific video.
 * This is the primary data source for the Incidents table in the frontend.
 */
const mongoose = require('mongoose');

const timestampRangeSchema = new mongoose.Schema(
  {
    start: { type: Number, required: true },  // seconds into video
    end: { type: Number, required: true },
  },
  { _id: false }
);

const bboxSchema = new mongoose.Schema(
  {
    frame: { type: Number },
    x: { type: Number },
    y: { type: Number },
    w: { type: Number },
    h: { type: Number },
    personId: { type: mongoose.Schema.Types.ObjectId, ref: 'Person' },
  },
  { _id: false }
);

const eventSchema = new mongoose.Schema(
  {
    videoId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Video',
      required: true,
      index: true,
    },
    personIds: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Person',
      },
    ],
    activities: [{ type: String }],  // e.g. ["phone", "talking", "head_movement"]
    confidenceScore: { type: Number, default: 0 },  // 0–100
    timestamps: [timestampRangeSchema],
    duration: { type: Number, default: 0 },  // total seconds of this activity
    thumbnailPath: { type: String, default: null },
    explanation: { type: String, default: null },  // Gemini-generated or ML explanation
    objectDetected: { type: String, default: null },  // e.g. "mobile_phone"
    seatId: { type: String, default: null },
    colorTag: { type: String, default: null },  // for canvas overlay color
    bboxOverlay: [bboxSchema],  // bounding boxes for frontend canvas
    heatmapRef: { type: String, default: null },
  },
  {
    timestamps: true,
  }
);

eventSchema.virtual('event_id').get(function () {
  return this._id.toHexString();
});

eventSchema.set('toJSON', { virtuals: true });

module.exports = mongoose.model('Event', eventSchema);
