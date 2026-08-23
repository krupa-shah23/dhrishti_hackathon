/**
 * Startup reconciliation for uploads/ vs. Video docs.
 *
 * Crashes/restarts mid-upload can leave two kinds of inconsistency behind:
 *   - a file on disk with no matching Video doc (write succeeded, doc never got created)
 *   - a Video doc pointing at a file that's gone (doc created, file removed/never finished)
 * Runs once at boot to clean both up before the app starts serving traffic.
 */
const fs = require('fs');
const path = require('path');
const { Video } = require('../models');
const config = require('../config');

async function reconcileUploads() {
  const uploadsDir = path.resolve(config.uploadDir);
  const filesOnDisk = fs.existsSync(uploadsDir) ? fs.readdirSync(uploadsDir) : [];
  const videos = await Video.find({}, 'filename filepath status');
  const knownFilenames = new Set(videos.map((v) => v.filename));

  let removedFiles = 0;
  for (const file of filesOnDisk) {
    if (!knownFilenames.has(file)) {
      const filePath = path.join(uploadsDir, file);
      fs.unlinkSync(filePath);
      removedFiles++;
      console.log(`🧹  Startup cleanup: removed orphaned file with no matching Video doc — ${file}`);
    }
  }

  let markedFailed = 0;
  for (const video of videos) {
    if (video.status === 'failed') continue; // already flagged (e.g. by ffprobe rejection) — leave alone
    if (!fs.existsSync(path.resolve(video.filepath))) {
      await Video.findByIdAndUpdate(video._id, {
        status: 'failed',
        errorMessage: `File missing on disk at startup reconciliation (expected: ${video.filepath}).`,
      });
      markedFailed++;
      console.log(`🧹  Startup cleanup: marked video ${video._id} as failed — file missing on disk (${video.filepath})`);
    }
  }

  console.log(`🧹  Startup reconciliation complete — ${removedFiles} orphaned file(s) removed, ${markedFailed} video doc(s) marked failed.`);
}

module.exports = reconcileUploads;
