/**
 * Settings Routes
 * 
 * GET  /api/settings/:key — Get a specific setting
 * POST /api/settings/:key — Create or update a specific setting
 */
const express = require('express');
const { Setting } = require('../models');

const router = express.Router();

// ─── GET /api/settings/:key ───
router.get('/:key', async (req, res, next) => {
  try {
    const setting = await Setting.findOne({ key: req.params.key });
    if (!setting) {
      return res.json({ success: true, data: null });
    }
    res.json({ success: true, data: setting.value });
  } catch (err) {
    next(err);
  }
});

// ─── POST /api/settings/:key ───
router.post('/:key', async (req, res, next) => {
  try {
    const { value } = req.body;
    if (value === undefined) {
      const err = new Error('Value is required');
      err.statusCode = 400;
      throw err;
    }

    const setting = await Setting.findOneAndUpdate(
      { key: req.params.key },
      { value },
      { new: true, upsert: true }
    );

    res.json({ success: true, data: setting.value });
  } catch (err) {
    next(err);
  }
});

module.exports = router;
