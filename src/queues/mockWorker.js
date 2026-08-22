/**
 * Mock ML Worker (Development Only)
 * Simulates the Python ML pipeline by consuming jobs from ml-processing-queue,
 * updating status, and generating fake events.
 */
const { Worker, Queue } = require('bullmq');
const { Video, Event, Person } = require('../models');
const { connection, statusQueue } = require('./index');

const sleep = (ms) => new Promise(resolve => setTimeout(resolve, ms));

const startMockWorker = () => {
  const worker = new Worker('ml-processing-queue', async (job) => {
    const { videoId } = job.data;
    console.log(`[Mock Worker] Picked up video ${videoId} for processing`);

    const updateStatus = async (status, stage, progress) => {
      await statusQueue.add('status', { videoId, status, stage, progress });
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

      await sleep(1500);

      // 5. Done
      const video = await Video.findById(videoId);
      await updateStatus('done', 'Processing complete', 100, { duration: 600, fps: 25 });
      
      console.log(`[Mock Worker] Finished processing video ${videoId}`);

    } catch (err) {
      console.error('[Mock Worker] Error processing video', err);
      await updateStatus('failed', 'Error occurred', 0, { errorMessage: err.message });
    }
  }, { connection });

  console.log('🤖 Mock ML Worker started. Listening on ml-processing-queue...');
  return worker;
};

module.exports = startMockWorker;
