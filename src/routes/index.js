/**
 * Routes barrel export
 */
const videoRoutes = require('./videos');
const eventRoutes = require('./events');
const personRoutes = require('./persons');
const dashboardRoutes = require('./dashboard');
const analysisRoutes = require('./analysis');
const settingsRoutes = require('./settings');
const internalRoutes = require('./internal');

module.exports = {
  videoRoutes,
  eventRoutes,
  personRoutes,
  dashboardRoutes,
  analysisRoutes,
  settingsRoutes,
  internalRoutes,
};
