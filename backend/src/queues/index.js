/**
 * BullMQ Queue Setup
 * 
 * Exports the ML processing queue that Node.js uses to enqueue
 * video processing jobs for the Python ML microservice.
 */
const { Queue } = require('bullmq');
const config = require('../config');

// Producer connection — used by mlQueue/statusQueue.add() calls made inline
// during HTTP requests. Must fail fast: ioredis's default retryStrategy
// retries forever, which left `await mlQueue.add(...)` hanging indefinitely
// (and the HTTP response with it) whenever Redis was unreachable.
const producerConnection = {
  host: config.redis.host,
  port: config.redis.port,
  maxRetriesPerRequest: 1,
  connectTimeout: 1500,
  retryStrategy: (times) => (times > 1 ? null : 200),
};

// Worker connection — BullMQ requires maxRetriesPerRequest: null here
// (workers use blocking Redis commands). Only used by the background status
// worker, which is never awaited inside a request, so unbounded retries are fine.
const workerConnection = {
  host: config.redis.host,
  port: config.redis.port,
  maxRetriesPerRequest: null,
};

// Main queue: Node.js adds jobs here, Python ML worker consumes them
const mlQueue = new Queue('ml-processing-queue', { connection: producerConnection });

// Status update queue: Python ML worker pushes status updates here,
// Node.js listens and updates MongoDB
const statusQueue = new Queue('ml-status-queue', { connection: producerConnection });

module.exports = { mlQueue, statusQueue, connection: workerConnection };
