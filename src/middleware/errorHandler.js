/**
 * Global error-handling middleware
 */
const config = require('../config');

// eslint-disable-next-line no-unused-vars
const errorHandler = (err, req, res, _next) => {
  console.error('💥  Error:', err.message);
  if (config.nodeEnv === 'development') {
    console.error(err.stack);
  }

  const statusCode = err.statusCode || 500;
  res.status(statusCode).json({
    success: false,
    error: err.message || 'Internal Server Error',
    ...(config.nodeEnv === 'development' && { stack: err.stack }),
  });
};

module.exports = errorHandler;
