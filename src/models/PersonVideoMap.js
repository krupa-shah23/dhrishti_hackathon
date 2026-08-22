/**
 * PersonVideoMap Model
 * Junction collection that links a Person to a Video.
 * Enables cross-session tracking: "which videos has this person appeared in?"
 */
const mongoose = require('mongoose');

const personVideoMapSchema = new mongoose.Schema(
  {
    personId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Person',
      required: true,
      index: true,
    },
    videoId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Video',
      required: true,
      index: true,
    },
    seatId: { type: String, default: null },
    totalDuration: { type: Number, default: 0 },  // seconds person was visible
    eventIds: [
      {
        type: mongoose.Schema.Types.ObjectId,
        ref: 'Event',
      },
    ],
  },
  {
    timestamps: true,
  }
);

// Compound index to prevent duplicate person-video links
personVideoMapSchema.index({ personId: 1, videoId: 1 }, { unique: true });

module.exports = mongoose.model('PersonVideoMap', personVideoMapSchema);
