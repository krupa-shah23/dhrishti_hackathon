const mongoose = require('mongoose');
const config = require('../config');

console.log('Mongo URI loaded:', !!config.mongoUri);
console.log(
  'Mongo URI:',
  config.mongoUri.replace(/:\/\/([^:]+):([^@]+)@/, '://$1:*****@')
);

const connectDB = async () => {
  try {
    const conn = await mongoose.connect(config.mongoUri);

    console.log(
      `✅ MongoDB connected: conn.connection.host/{conn.connection.name}`
    );
  } catch (err) {
    console.error('❌ MongoDB connection error:', err.message);
    process.exit(1);
  }
};

module.exports = connectDB;

