/**
 * DRISHTI — Upload Manager Page (/videos)
 * 
 * Drop-zone for video upload + 3 upload slot cards
 * Shows processing state with animated spinners and status text.
 */
import { useState, useEffect, useRef, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CloudUpload, X, RefreshCw, Trash2, Play, Loader,
  FileVideo, CheckCircle2, AlertCircle,
} from 'lucide-react';
import { videoApi } from '../api/client';

const STATUS_LABELS = {
  uploading: 'Uploading...',
  queued: 'Queued for processing',
  ingesting: 'Ingesting video...',
  detecting: 'Detecting objects...',
  tracking: 'Tracking persons...',
  scoring: 'Scoring incidents...',
  done: 'Processing complete ✓',
  failed: 'Processing failed ✗',
};

export default function UploadPage() {
  const [videos, setVideos] = useState([]);
  const [uploading, setUploading] = useState(false);
  const [uploadProgress, setUploadProgress] = useState(0);
  const [dragActive, setDragActive] = useState(false);
  const fileInputRef = useRef(null);
  const navigate = useNavigate();

  useEffect(() => {
    fetchVideos();
    // Poll for status updates every 5 seconds
    const interval = setInterval(fetchVideos, 5000);
    return () => clearInterval(interval);
  }, []);

  const fetchVideos = async () => {
    try {
      const res = await videoApi.getAll();
      setVideos(res.data.data || []);
    } catch (err) {
      console.error('Failed to fetch videos:', err);
    }
  };

  const handleUpload = async (file) => {
    if (!file) return;
    const activeCount = videos.filter(v => !['failed'].includes(v.status)).length;
    if (activeCount >= 3) {
      alert('Maximum 3 videos allowed. Delete a video first.');
      return;
    }

    try {
      setUploading(true);
      setUploadProgress(0);
      // Register the video by metadata only — no file transfer, near-instant
      await videoApi.register(file);
      setUploadProgress(100);
      await fetchVideos();
    } catch (err) {
      const msg = err.response?.data?.error || err.message;
      alert(`Registration failed: ${msg}`);
    } finally {
      setUploading(false);
      setUploadProgress(0);
    }
  };

  const handleDelete = async (id) => {
    if (!confirm('Delete this video and all its data?')) return;
    try {
      await videoApi.delete(id);
      await fetchVideos();
    } catch (err) {
      alert('Delete failed: ' + (err.response?.data?.error || err.message));
    }
  };

  const handleRequeue = async (id) => {
    try {
      await videoApi.requeue(id);
      await fetchVideos();
    } catch (err) {
      alert('Requeue failed: ' + (err.response?.data?.error || err.message));
    }
  };

  const onDrop = useCallback((e) => {
    e.preventDefault();
    setDragActive(false);
    const file = e.dataTransfer?.files?.[0];
    if (file) handleUpload(file);
  }, [videos]);

  const onDragOver = (e) => { e.preventDefault(); setDragActive(true); };
  const onDragLeave = () => setDragActive(false);

  const emptySlots = Math.max(0, 3 - videos.length);

  return (
    <div className="slide-up">
      {/* Drop Zone */}
      <div
        className={`drop-zone ${dragActive ? 'active' : ''}`}
        onDrop={onDrop}
        onDragOver={onDragOver}
        onDragLeave={onDragLeave}
        onClick={() => !uploading && fileInputRef.current?.click()}
        style={{ marginBottom: 32, cursor: uploading ? 'wait' : 'pointer' }}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept="video/*"
          style={{ display: 'none' }}
          onChange={(e) => handleUpload(e.target.files?.[0])}
        />

        {uploading ? (
          <>
            <Loader size={48} className="drop-zone-icon" style={{ animation: 'spin 1s linear infinite' }} />
            <div className="drop-zone-title">Registering video...</div>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 4 }}>
              This only takes a moment — no file transfer needed
            </div>
          </>
        ) : (
          <>
            <CloudUpload size={48} className="drop-zone-icon" />
            <div className="drop-zone-title">Upload Video for Analysis</div>
            <div className="drop-zone-subtitle">
              Max 3 videos allowed • Drag & drop or click to browse
            </div>
          </>
        )}
      </div>

      {/* Upload Slots */}
      <div className="grid-3">
        {videos.map((video) => (
          <VideoSlotCard
            key={video._id}
            video={video}
            onDelete={handleDelete}
            onRequeue={handleRequeue}
            onViewAnalysis={() => navigate(`/analysis/${video._id}`)}
            onViewVideo={() => navigate(`/videos/${video._id}`)}
          />
        ))}

        {/* Empty slot placeholders */}
        {Array.from({ length: emptySlots }).map((_, i) => (
          <div
            key={`empty-${i}`}
            className="card"
            style={{
              display: 'flex', alignItems: 'center', justifyContent: 'center',
              minHeight: 180, cursor: 'pointer', opacity: 0.5,
              border: '2px dashed var(--border-medium)',
              background: 'transparent', boxShadow: 'none',
            }}
            onClick={() => fileInputRef.current?.click()}
          >
            <div style={{ textAlign: 'center' }}>
              <CloudUpload size={32} style={{ color: 'var(--text-muted)', marginBottom: 8 }} />
              <div style={{ fontSize: 14, color: 'var(--text-muted)', fontWeight: 500 }}>
                + Add Video
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}

function VideoSlotCard({ video, onDelete, onRequeue, onViewAnalysis, onViewVideo }) {
  const isProcessing = ['queued', 'ingesting', 'detecting', 'tracking', 'scoring'].includes(video.status);
  const isDone = video.status === 'done';
  const isFailed = video.status === 'failed';

  return (
    <div className="card" style={{ position: 'relative' }}>
      <div className="card-body">
        {/* Header */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
          <div style={{
            width: 44, height: 44, borderRadius: 'var(--radius-md)',
            background: isDone ? 'var(--status-green-bg)' : isFailed ? 'var(--status-red-bg)' : 'var(--status-blue-bg)',
            display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
          }}>
            {isDone ? <CheckCircle2 size={22} color="var(--status-green)" /> :
             isFailed ? <AlertCircle size={22} color="var(--status-red)" /> :
             <FileVideo size={22} color="var(--status-blue)" />}
          </div>
          <div style={{ flex: 1, overflow: 'hidden' }}>
            <div style={{ fontWeight: 600, fontSize: 14, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
              {video.originalName}
            </div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
              {(video.size / (1024 * 1024)).toFixed(1)} MB
            </div>
          </div>
        </div>

        {/* Status */}
        <div style={{ marginBottom: 16 }}>
          <span className={`status-badge ${video.status}`}>
            {isProcessing && <span className="status-dot" />}
            {STATUS_LABELS[video.status] || video.status}
          </span>
        </div>

        {/* Progress Bar */}
        {isProcessing && (
          <div style={{ marginBottom: 16 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
              <span style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                {video.processingStage || 'Processing...'}
              </span>
              <span style={{ fontSize: 12, fontWeight: 600 }}>
                {video.processingProgress || 0}%
              </span>
            </div>
            <div className="progress-bar-container">
              <div className="progress-bar-fill" style={{ width: `${video.processingProgress || 0}%` }} />
            </div>
          </div>
        )}

        {/* Actions */}
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          {isDone && (
            <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={onViewAnalysis}>
              <Play size={14} /> Analysis Report
            </button>
          )}
          {isDone && (
            <button className="btn btn-ghost btn-sm" onClick={onViewVideo} title="Review raw video">
              <Play size={14} />
            </button>
          )}
          {isFailed && (
            <button className="btn btn-ghost btn-sm" style={{ flex: 1 }} onClick={() => onRequeue(video._id)}>
              <RefreshCw size={14} /> Retry
            </button>
          )}
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => onDelete(video._id)}
            style={{ color: 'var(--status-red)' }}
          >
            <Trash2 size={14} />
          </button>
        </div>
      </div>
    </div>
  );
}
