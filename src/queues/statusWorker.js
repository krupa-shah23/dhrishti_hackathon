/**
 * BullMQ Worker — ML Status Listener
 * 
 * Listens on the ml-status-queue for status updates from the Python ML service.
 * Updates the Video document in MongoDB accordingly.
 */
const { Worker } = require('bullmq');
const { Video } = require('../models');
const { connection } = require('./index');

const startStatusWorker = () => {
  const worker = new Worker(
    'ml-status-queue',
    async (job) => {
      const { videoId, status, stage, progress, errorMessage, duration, fps } = job.data;

      console.log(`📡  ML status update for video ${videoId}: ${status} (${stage || 'N/A'})`);

      const update = {};
      if (status) update.status = status;
      if (stage) update.processingStage = stage;
      if (progress !== undefined) update.processingProgress = progress;
      if (errorMessage) update.errorMessage = errorMessage;
      if (duration) update.duration = duration;
      if (fps) update.fps = fps;

      await Video.findByIdAndUpdate(videoId, update);
    },
    { connection }
  );

  worker.on('completed', (job) => {
    console.log(`✅  Status job ${job.id} processed`);
  });

  worker.on('failed', (job, err) => {
    console.error(`❌  Status job ${job?.id} failed:`, err.message);
  });

  console.log('👂  ML status worker listening...');
  return worker;
};

module.exports = startStatusWorker;
