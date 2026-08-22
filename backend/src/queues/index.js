/**
 * BullMQ Queue Setup
 * 
 * Exports the ML processing queue that Node.js uses to enqueue
 * video processing jobs for the Python ML microservice.
 */
const { Queue } = require('bullmq');
const config = require('../config');

const connection = {
  host: config.redis.host,
  port: config.redis.port,
};

// Main queue: Node.js adds jobs here, Python ML worker consumes them
const mlQueue = new Queue('ml-processing-queue', { connection });

// Status update queue: Python ML worker pushes status updates here,
// Node.js listens and updates MongoDB
const statusQueue = new Queue('ml-status-queue', { connection });

module.exports = { mlQueue, statusQueue, connection };
