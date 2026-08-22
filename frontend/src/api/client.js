/**
 * DRISHTI Frontend — API Client
 * Centralized Axios instance for all backend calls.
 */
import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 30000,
  headers: { 'Content-Type': 'application/json' },
});

// ─── Video APIs ───
export const videoApi = {
  getAll: () => api.get('/videos'),
  getById: (id) => api.get(`/videos/${id}`),

  // Instant registration — sends only metadata (name, size, mimetype), no file.
  // The slot is created immediately in DB and the mock pipeline starts right away.
  register: (file) => api.post('/videos/register', {
    originalName: file.name,
    size: file.size,
    mimetype: file.type || 'video/mp4',
  }),

  // Legacy full-file upload kept for reference / real ML integration
  upload: (file, onProgress) => {
    const formData = new FormData();
    formData.append('video', file);
    return api.post('/videos/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      timeout: 0, // no timeout for large uploads
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    });
  },
  delete: (id) => api.delete(`/videos/${id}`),
  requeue: (id) => api.post(`/videos/${id}/requeue`),
  getStreamUrl: (id) => `${API_BASE}/videos/${id}/stream`,
  getTracking: (id) => api.get(`/videos/${id}/tracking`),
};

// ─── Event APIs ───
export const eventApi = {
  getAll: (params) => api.get('/events', { params }),
  getById: (id) => api.get(`/events/${id}`),
};

// ─── Person APIs ───
export const personApi = {
  getAll: () => api.get('/persons'),
  getById: (id) => api.get(`/persons/${id}`),
  getTimeline: (id) => api.get(`/persons/${id}/timeline`),
  getVideos: (id) => api.get(`/persons/${id}/videos`),
};

// ─── Dashboard APIs ───
export const dashboardApi = {
  getStats: () => api.get('/dashboard/stats'),
  getTimeline: (range) => api.get('/dashboard/timeline', { params: { range } }),
};

// ─── Analysis APIs ───
export const analysisApi = {
  getSummary: (videoId) => api.post(`/analysis/summary/${videoId}`),
};

// ─── Settings APIs ───
// Used by SettingsPage to persist config (XAI toggle, seat grid, thresholds)
export const settingsApi = {
  get: (key) => api.get(`/settings/${key}`),
  save: (key, value) => api.post(`/settings/${key}`, { value }),
};

export default api;
