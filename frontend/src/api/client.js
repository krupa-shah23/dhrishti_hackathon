/**
 * DRISHTI Frontend — API Client
 * Centralized Axios instance for all backend calls.
 */
import axios from 'axios';
import * as tus from 'tus-js-client';

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:5000/api';
const RETRY_DELAYS = [0, 1000, 3000, 5000, 10000];

// tus-js-client wraps HTTP failures in a DetailedError with the raw response
// on .originalResponse — pull our backend's {success,error} JSON out of it so
// resumable-upload failures surface the same message text as the old
// single-POST path's err.response?.data?.error did.
function extractTusErrorMessage(error) {
  try {
    const body = error?.originalResponse?.getBody?.();
    if (body) {
      const parsed = JSON.parse(body);
      if (parsed?.error) return parsed.error;
    }
  } catch (_e) { /* not JSON — fall through to the raw error message */ }
  return error?.message || 'Upload failed';
}



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

  // File upload — uses raw axios (NOT the shared api instance) so the browser
  // can automatically set Content-Type: multipart/form-data with the correct boundary.
  // The shared api instance has Content-Type: application/json which breaks multer.
  // Kept working alongside uploadResumable below — not a hard cutover.
  upload: (file, onProgress) => {
    const formData = new FormData();
    formData.append('video', file);
    return axios.post(`${API_BASE}/videos/upload`, formData, {
      timeout: 0, // no timeout for large uploads
      onUploadProgress: (e) => {
        if (onProgress && e.total) {
          onProgress(Math.round((e.loaded / e.total) * 100));
        }
      },
    });
  },

  // Resumable, chunked upload via tus-js-client — automatically resumes after
  // a dropped connection or page reload (it fingerprints the file and looks
  // up any matching in-progress upload before starting a new one).
  // Returns the tus-js-client Upload instance so the caller can .abort() it.
  uploadResumable: (file, { onProgress, onResuming, onSuccess, onError } = {}) => {
    const upload = new tus.Upload(file, {
      endpoint: `${API_BASE}/videos/upload/tus`,
      chunkSize: 5 * 1024 * 1024, // 5MB — small enough to demonstrate real resume-from-last-chunk behavior
      retryDelays: RETRY_DELAYS,
      metadata: { filename: file.name, filetype: file.type || 'video/mp4' },
      onShouldRetry: (error) => {
        // Business-rule rejections (slot limit, duplicate, invalid file,
        // disk full) won't succeed on retry — only retry on transient/network
        // failures, same distinction the backend's own error paths already make.
        const status = error?.originalResponse?.getStatus?.();
        if (status && [409, 422, 507].includes(status)) return false;
        onResuming?.(true);
        return true;
      },
      onProgress: (bytesSent, bytesTotal) => {
        onResuming?.(false);
        onProgress?.(Math.round((bytesSent / bytesTotal) * 100));
      },
      onSuccess: () => onSuccess?.(upload),
      onError: (error) => onError?.(new Error(extractTusErrorMessage(error))),
    });

    upload.findPreviousUploads().then((previousUploads) => {
      if (previousUploads.length > 0) {
        upload.resumeFromPreviousUpload(previousUploads[0]);
        onResuming?.(true);
      }
      upload.start();
    });

    return upload;
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
