/**
 * Person Model
 * Represents a unique individual detected & re-identified across videos.
 */
const mongoose = require('mongoose');

const personSchema = new mongoose.Schema(
  {
    personLabel: { type: String, required: true, unique: true }, // e.g. "Person-001"
    embeddingVector: { type: [Number], default: [] },            // Re-ID feature vector
    firstSeenVideoId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: 'Video',
      default: null,
    },
    thumbnailPath: { type: String, default: null },  // cropped face/body thumbnail
    seatId: { type: String, default: null },          // assigned seat grid position
    totalDetections: { type: Number, default: 0 },
  },
  {
    timestamps: true,
  }
);

personSchema.virtual('person_id').get(function () {
  return this._id.toHexString();
});

personSchema.set('toJSON', { virtuals: true });

module.exports = mongoose.model('Person', personSchema);
