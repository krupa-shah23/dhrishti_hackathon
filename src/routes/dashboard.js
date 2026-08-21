/**
 * Dashboard Routes
 * 
 * GET  /api/dashboard/stats     — Aggregate stats for the Overview Dashboard
 * GET  /api/dashboard/timeline  — Incident count grouped by date
 */
const express = require('express');
const { Video, Person, Event } = require('../models');

const router = express.Router();

// ─── GET /api/dashboard/stats ───
// Returns: totalVideos, totalPersonsCaught, incidentDistribution (for donut chart).
router.get('/stats', async (req, res, next) => {
  try {
    const [totalVideos, totalPersons, activityDistribution] = await Promise.all([
      Video.countDocuments({ status: 'done' }),
      Person.countDocuments(),
      Event.aggregate([
        { $unwind: '$activities' },
        {
          $group: {
            _id: '$activities',
            count: { $sum: 1 },
          },
        },
        { $sort: { count: -1 } },
      ]),
    ]);

    // Format for the donut chart
    const totalIncidents = activityDistribution.reduce((sum, a) => sum + a.count, 0);
    const incidentDistribution = activityDistribution.map((a) => ({
      activity: a._id,
      count: a.count,
      percentage: totalIncidents > 0 ? Math.round((a.count / totalIncidents) * 100) : 0,
    }));

    res.json({
      success: true,
      data: {
        totalVideosProcessed: totalVideos,
        totalPersonsCaught: totalPersons,
        totalIncidents,
        incidentDistribution,
      },
    });
  } catch (err) {
    next(err);
  }
});

// ─── GET /api/dashboard/timeline ───
// Returns incident counts grouped by date for the "Recent Incidents Timeline" chart.
// Query: ?range=week|month|year  (default: month)
router.get('/timeline', async (req, res, next) => {
  try {
    const range = req.query.range || 'month';

    let daysBack;
    switch (range) {
      case 'week':
        daysBack = 7;
        break;
      case 'year':
        daysBack = 365;
        break;
      case 'month':
      default:
        daysBack = 30;
        break;
    }

    const startDate = new Date();
    startDate.setDate(startDate.getDate() - daysBack);

    const timeline = await Event.aggregate([
      { $match: { createdAt: { $gte: startDate } } },
      {
        $group: {
          _id: {
            $dateToString: { format: '%Y-%m-%d', date: '$createdAt' },
          },
          count: { $sum: 1 },
          avgConfidence: { $avg: '$confidenceScore' },
        },
      },
      { $sort: { _id: 1 } },
    ]);

    res.json({
      success: true,
      range,
      data: timeline.map((t) => ({
        date: t._id,
        count: t.count,
        avgConfidence: Math.round(t.avgConfidence * 100) / 100,
      })),
    });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
