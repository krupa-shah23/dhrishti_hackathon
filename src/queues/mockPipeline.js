/**
 * In-Memory Mock ML Pipeline
 * Bypasses BullMQ/Redis so it works reliably on local dev environments without Redis.
 */
const { Video, Event, Person } = require('../models');

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const executeMockPipeline = async (videoId) => {
  console.log(`[Mock Pipeline] Started processing video ${videoId}`);

  const updateStatus = async (status, stage, progress) => {
    const update = {};
    if (status) update.status = status;
    if (stage) update.processingStage = stage;
    if (progress !== undefined) update.processingProgress = progress;
    await Video.findByIdAndUpdate(videoId, update);
  };

  try {
    // 1. Ingesting
    await updateStatus('ingesting', 'Ingesting video...', 10);
    await sleep(2000);

    // 2. Detecting
    await updateStatus('detecting', 'Detecting objects...', 35);
    await sleep(2500);

    // 3. Tracking
    await updateStatus('tracking', 'Tracking persons...', 65);
    await sleep(2500);

    // 4. Scoring & Fake Data Generation
    await updateStatus('scoring', 'Scoring incidents...', 85);
    
    // Create a fake person
    const person = await Person.create({
      personLabel: 'Student_' + Math.floor(Math.random() * 1000),
      seatId: 'A' + Math.floor(Math.random() * 20),
      totalDetections: 1
    });

    // Create a fake event
    await Event.create({
      videoId,
      personIds: [person._id],
      activities: ['mobile_phone', 'head_movement'],
      confidenceScore: 88,
      timestamps: [{ start: 5, end: 12 }],
      duration: 7,
      explanation: 'Suspicious sustained head movement followed by use of a mobile phone under the desk.',
      objectDetected: 'mobile_phone',
      seatId: person.seatId
    });

    // Generate XAI Summary automatically
    try {
      await updateStatus('scoring', 'Generating AI summary...', 95);
      const port = process.env.PORT || 5000;
      const res = await fetch(`http://localhost:${port}/api/analysis/summary/${videoId}`, {
        method: 'POST',
      });
      const data = await res.json();
      if (data.success && data.data?.summary) {
        await Video.findByIdAndUpdate(videoId, { xaiSummary: data.data.summary });
      }
    } catch (summaryErr) {
      console.warn(`[Mock Pipeline] Failed to auto-generate XAI summary for ${videoId}:`, summaryErr.message);
    }

    await sleep(1500);

    // 5. Done
    await updateStatus('done', 'Processing complete', 100);
    console.log(`[Mock Pipeline] Finished processing video ${videoId}`);

  } catch (err) {
    console.error('[Mock Pipeline] Error processing video', err);
    await updateStatus('failed', 'Error occurred', 0);
  }
};

module.exports = { executeMockPipeline };
