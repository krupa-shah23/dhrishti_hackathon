/**
 * DRISHTI — Dev Mock Processor
 *
 * When the real ML service / Redis / BullMQ is unavailable, this lightweight
 * poller simulates the full pipeline so the UI can be demoed end-to-end.
 *
 * Flow per video:
 *  queued → ingesting (3 s) → detecting (5 s) → tracking (5 s) → scoring (3 s) → done
 *
 * It also seeds realistic mock Event documents so the Incidents page is populated.
 */

const mongoose = require('mongoose');
const { v4: uuidv4 } = require('uuid');

const STAGES = [
  { status: 'ingesting',  stage: 'Ingesting frames',       progress: 15, delay: 3000 },
  { status: 'detecting',  stage: 'Detecting objects',       progress: 40, delay: 5000 },
  { status: 'tracking',   stage: 'Tracking persons',        progress: 65, delay: 5000 },
  { status: 'scoring',    stage: 'Scoring incidents',       progress: 85, delay: 3000 },
];

const MOCK_ACTIVITIES = [
  ['phone_usage'],
  ['head_movement', 'talking'],
  ['chit_reference'],
  ['phone_usage', 'head_movement'],
  ['suspicious_gesture'],
];

const MOCK_OBJECTS = ['mobile_phone', 'paper_chit', null, 'mobile_phone', null];

const MOCK_EXPLANATIONS = [
  'Student appears to be operating a mobile device during the examination. High confidence based on hand position and screen glow.',
  'Repeated head movement towards neighbouring seat detected. Possible communication attempt.',
  'Student observed referencing paper material not part of exam kit. Object resembles chit or unauthorised note.',
  'Mobile phone detected in open position. Screen brightness and hand posture confirm active usage.',
  'Suspicious lateral head turn at 120° beyond normal writing posture. Flagged for manual review.',
];

/** Seed 2–4 mock events for the given videoId */
async function seedMockEvents(videoId) {
  const Event = mongoose.model('Event');
  const count = 2 + Math.floor(Math.random() * 3);
  const events = [];

  for (let i = 0; i < count; i++) {
    const startSec = 30 + i * 45 + Math.floor(Math.random() * 20);
    const actIdx = Math.floor(Math.random() * MOCK_ACTIVITIES.length);

    events.push({
      mlEventId: `mock-${uuidv4()}`,
      videoId,
      activities: MOCK_ACTIVITIES[actIdx],
      confidenceScore: 60 + Math.floor(Math.random() * 35),
      timestamps: [{ start: startSec, end: startSec + 8 + Math.floor(Math.random() * 12) }],
      duration: 8 + Math.floor(Math.random() * 12),
      objectDetected: MOCK_OBJECTS[actIdx],
      explanation: MOCK_EXPLANATIONS[actIdx],
      colorTag: ['#ef4444', '#f59e0b', '#8b5cf6'][Math.floor(Math.random() * 3)],
      thumbnailPath: null,
      heatmapRef: null,
    });
  }

  await Event.insertMany(events);
  console.log(`🧪  Mock processor: seeded ${events.length} events for video ${videoId}`);
}

/** Advance a single video through the pipeline stages */
async function processVideo(video) {
  const Video = mongoose.model('Video');
  try {
    for (const { status, stage, progress, delay } of STAGES) {
      await new Promise(r => setTimeout(r, delay));
      await Video.findByIdAndUpdate(video._id, {
        status,
        processingStage: stage,
        processingProgress: progress,
      });
      console.log(`🔄  Mock [${video.originalName}] → ${stage} (${progress}%)`);
    }

    await seedMockEvents(video._id);

    await Video.findByIdAndUpdate(video._id, {
      status: 'done',
      processingStage: 'Complete',
      processingProgress: 100,
    });

    console.log(`✅  Mock processor: finished ${video.originalName}`);
  } catch (err) {
    console.error(`❌  Mock processor error for ${video._id}:`, err.message);
    const Video = mongoose.model('Video');
    await Video.findByIdAndUpdate(video._id, {
      status: 'failed',
      errorMessage: err.message,
    });
  }
}

/** Set of videoIds currently being processed (avoid duplicate runs) */
const processing = new Set();

/** Poll every 4 seconds for queued videos */
function startMockProcessor() {
  console.log('🧪  Dev mock processor active — will auto-process queued videos');

  setInterval(async () => {
    try {
      const Video = mongoose.model('Video');
      const queued = await Video.find({ status: 'queued' });
      for (const v of queued) {
        const id = v._id.toString();
        if (!processing.has(id)) {
          processing.add(id);
          processVideo(v).finally(() => processing.delete(id));
        }
      }
    } catch (_err) {
      // Silently swallow poll errors (e.g. DB hiccup during restart)
    }
  }, 4000);
}

module.exports = { startMockProcessor };
