/**
 * Session-local dashboard deltas. Uploading/removing a demo clip in the
 * Upload Manager nudges the Dashboard's totals and chart via localStorage,
 * so the numbers visibly move instead of sitting on a fixed backend value.
 */
const KEY = 'drishti_session_deltas';

function readAll() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

function writeAll(all) {
  localStorage.setItem(KEY, JSON.stringify(all));
}

export function addContribution(localId, demoClip) {
  const all = readAll();
  all[localId] = {
    videos: 1,
    persons: demoClip.personId ? 1 : 0,
    incidents: demoClip.tableRows.length,
    category: demoClip.dashboardCategory || null,
    count: demoClip.tableRows.length,
  };
  writeAll(all);
}

export function removeContribution(localId) {
  const all = readAll();
  delete all[localId];
  writeAll(all);
}

export function getAggregateDelta() {
  const all = readAll();
  const agg = { videos: 0, persons: 0, incidents: 0, categories: {} };
  Object.values(all).forEach((c) => {
    agg.videos += c.videos;
    agg.persons += c.persons;
    agg.incidents += c.incidents;
    if (c.category && c.count) {
      agg.categories[c.category] = (agg.categories[c.category] || 0) + c.count;
    }
  });
  return agg;
}
