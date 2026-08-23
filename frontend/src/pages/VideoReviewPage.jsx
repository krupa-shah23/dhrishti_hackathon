/**
 * DRISHTI — Video Review Page (/videos/:videoId)
 * 
 * Seekable video player with Canvas bounding box overlay,
 * confidence graph timeline, and synchronized event sidebar.
 */
import { useState, useEffect, useRef } from 'react';
import { useParams, useSearchParams } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, ResponsiveContainer,
  CartesianGrid, Tooltip,
} from 'recharts';
import { EyeOff, Download, Play, Pause } from 'lucide-react';
import { videoApi, eventApi } from '../api/client';
import { getDemoClipById, parseClipTime } from '../data/demoClips';

export default function VideoReviewPage() {
  const { videoId } = useParams();
  const [searchParams] = useSearchParams();
  const seekTo = parseFloat(searchParams.get('t') || '0');

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [video, setVideo] = useState(null);
  const [events, setEvents] = useState([]);
  const [trackingData, setTrackingData] = useState({});
  const [currentTime, setCurrentTime] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const [privacyMode, setPrivacyMode] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, [videoId]);

  useEffect(() => {
    const videoEl = videoRef.current;
    if (!videoEl || seekTo <= 0) return;
    const seek = () => { videoEl.currentTime = seekTo; };
    if (videoEl.readyState >= 1) {
      // Metadata (and duration) already available — safe to seek now.
      seek();
    } else {
      videoEl.addEventListener('loadedmetadata', seek, { once: true });
      return () => videoEl.removeEventListener('loadedmetadata', seek);
    }
  }, [video, seekTo]);

  // Known demo clip for this video: reuses the hardcoded personId/personBbox
  // mapping so the flagged person's box can be highlighted without a real
  // ML tracking pipeline.
  const demoClip = getDemoClipById(videoId);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [videoRes, eventsRes, trackingRes] = await Promise.all([
        videoApi.getById(videoId),
        eventApi.getAll({ videoId, limit: 500, sort: 'timestamps.0.start', order: 'asc' }),
        videoApi.getTracking(videoId),
      ]);
      setVideo(videoRes.data.data);
      setEvents(eventsRes.data.data || []);
      setTrackingData(trackingRes.data.data || {});
    } catch (err) {
      console.error('Video review fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  // Draw bounding boxes on canvas
  useEffect(() => {
    if (!canvasRef.current || !videoRef.current) return;

    const drawFrame = () => {
      const canvas = canvasRef.current;
      const videoEl = videoRef.current;
      if (!canvas || !videoEl) return;

      const ctx = canvas.getContext('2d');
      canvas.width = videoEl.videoWidth || videoEl.clientWidth;
      canvas.height = videoEl.videoHeight || videoEl.clientHeight;
      ctx.clearRect(0, 0, canvas.width, canvas.height);

      // Find tracking data for current time
      const fps = video?.fps || 25;
      const frameIdx = Math.round(currentTime * fps);

      // Look for frame data within a small window
      for (let offset = 0; offset <= 2; offset++) {
        const boxes = trackingData[String(frameIdx + offset)] || trackingData[String(frameIdx - offset)];
        if (boxes && Array.isArray(boxes)) {
          boxes.forEach((box) => {
            const color = box.confidence >= 0.7 ? '#F04438' :
                          box.confidence >= 0.4 ? '#FFC107' : '#12B76A';

            ctx.strokeStyle = color;
            ctx.lineWidth = 2;
            ctx.strokeRect(box.x, box.y, box.w, box.h);

            // Label
            const label = `ID:${box.track_id} ${(box.confidence * 100).toFixed(0)}%`;
            ctx.fillStyle = color;
            ctx.fillRect(box.x, box.y - 18, ctx.measureText(label).width + 8, 18);
            ctx.fillStyle = '#fff';
            ctx.font = '12px Inter, sans-serif';
            ctx.fillText(label, box.x + 4, box.y - 5);
          });
          break;
        }
      }

      // Flagged-person highlight for known demo clips: red box over the
      // hardcoded personBbox while playback is inside that incident's window.
      if (demoClip?.personBbox) {
        const activeRow = demoClip.tableRows
          .filter((row) => {
            const start = parseClipTime(row.time);
            return currentTime >= start && currentTime <= start + 6;
          })
          .pop();

        if (activeRow) {
          const { x, y, w, h } = demoClip.personBbox;
          const boxX = x * canvas.width;
          const boxY = y * canvas.height;
          const boxW = w * canvas.width;
          const boxH = h * canvas.height;

          ctx.strokeStyle = '#F04438';
          ctx.lineWidth = 3;
          ctx.strokeRect(boxX, boxY, boxW, boxH);

          const label = demoClip.personId;
          ctx.font = '12px Inter, sans-serif';
          ctx.fillStyle = '#F04438';
          ctx.fillRect(boxX, boxY - 18, ctx.measureText(label).width + 8, 18);
          ctx.fillStyle = '#fff';
          ctx.fillText(label, boxX + 4, boxY - 5);
        }
      }

      requestAnimationFrame(drawFrame);
    };

    const animId = requestAnimationFrame(drawFrame);
    return () => cancelAnimationFrame(animId);
  }, [currentTime, trackingData, video, demoClip]);

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleSeekFromChart = (data) => {
    if (data?.activePayload?.[0]?.payload?.time != null && videoRef.current) {
      videoRef.current.currentTime = data.activePayload[0].payload.time;
    }
  };

  // Build confidence graph data from events
  const graphData = events.map((e) => ({
    time: e.timestamps?.[0]?.start || 0,
    confidence: e.confidenceScore,
    label: formatTime(e.timestamps?.[0]?.start),
  }));

  // Find events near current time for the sidebar highlight
  const activeEvents = events.filter((e) => {
    const start = e.timestamps?.[0]?.start || 0;
    const end = e.timestamps?.[0]?.end || start;
    return currentTime >= start - 1 && currentTime <= end + 1;
  });

  if (loading) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading video...</span>
      </div>
    );
  }

  if (!video) {
    return (
      <div className="empty-state">
        <div className="empty-state-title">Video not found</div>
      </div>
    );
  }

  return (
    <div className="slide-up">
      <div style={{ display: 'flex', gap: 24 }}>
        {/* Left: Player + Graph */}
        <div style={{ flex: '1 1 65%' }}>
          {/* Video Player with Canvas Overlay */}
          <div className="card" style={{ marginBottom: 24 }}>
            <div style={{ position: 'relative', background: '#000', borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0' }}>
              <video
                ref={videoRef}
                src={videoApi.getStreamUrl(videoId)}
                style={{ width: '100%', display: 'block', borderRadius: 'var(--radius-lg) var(--radius-lg) 0 0' }}
                onTimeUpdate={handleTimeUpdate}
                onPlay={() => setIsPlaying(true)}
                onPause={() => setIsPlaying(false)}
                controls
              />
              <canvas
                ref={canvasRef}
                style={{
                  position: 'absolute', top: 0, left: 0,
                  width: '100%', height: '100%',
                  pointerEvents: 'none',
                  filter: privacyMode ? 'blur(8px)' : 'none',
                }}
              />
            </div>
            {/* Player Controls */}
            <div style={{
              padding: '12px 16px', display: 'flex', alignItems: 'center',
              justifyContent: 'space-between', borderTop: '1px solid var(--border-light)',
            }}>
              <div style={{ display: 'flex', gap: 8 }}>
                <button
                  className={`btn btn-sm ${privacyMode ? 'btn-primary' : 'btn-ghost'}`}
                  onClick={() => setPrivacyMode(!privacyMode)}
                >
                  <EyeOff size={14} />
                  Privacy {privacyMode ? 'On' : 'Off'}
                </button>
                <button className="btn btn-ghost btn-sm">
                  <Download size={14} /> Export Clip
                </button>
              </div>
              <span style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                {formatTime(currentTime)} / {formatTime(video.duration || 0)}
              </span>
            </div>
          </div>

          {/* Confidence Graph Timeline */}
          <div className="card">
            <div className="card-header">
              <span className="card-title">Confidence Timeline</span>
            </div>
            <div className="card-body">
              {graphData.length > 0 ? (
                <ResponsiveContainer width="100%" height={160}>
                  <LineChart data={graphData} onClick={handleSeekFromChart}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="label" tick={{ fontSize: 11 }} stroke="#94A3B8" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#94A3B8" />
                    <Tooltip />
                    <Line
                      type="monotone" dataKey="confidence" stroke="#F04438"
                      strokeWidth={2} dot={{ r: 2 }}
                    />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state" style={{ padding: 30 }}>
                  <div className="empty-state-text">No events detected</div>
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Right: Synchronized Event Sidebar */}
        <div style={{ flex: '1 1 35%' }}>
          <div className="card" style={{ position: 'sticky', top: 'calc(var(--topbar-height) + 28px)' }}>
            <div className="card-header">
              <span className="card-title">Events ({events.length})</span>
            </div>
            <div style={{ maxHeight: 'calc(100vh - 200px)', overflowY: 'auto' }}>
              {events.length === 0 ? (
                <div className="empty-state" style={{ padding: 40 }}>
                  <div className="empty-state-text">No events detected</div>
                </div>
              ) : (
                events.map((event) => {
                  const isActive = activeEvents.includes(event);
                  return (
                    <div
                      key={event._id}
                      style={{
                        padding: '14px 20px',
                        borderBottom: '1px solid var(--border-light)',
                        background: isActive ? 'rgba(59, 130, 246, 0.06)' : 'transparent',
                        borderLeft: isActive ? '3px solid var(--accent-blue)' : '3px solid transparent',
                        cursor: 'pointer',
                        transition: 'all 150ms',
                      }}
                      onClick={() => {
                        if (videoRef.current) {
                          videoRef.current.currentTime = event.timestamps?.[0]?.start || 0;
                        }
                      }}
                    >
                      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                        <span style={{ fontSize: 13, fontWeight: 600 }}>
                          {formatTime(event.timestamps?.[0]?.start)}
                        </span>
                        <span className={`confidence-badge ${event.confidenceScore >= 70 ? 'high' : event.confidenceScore >= 40 ? 'medium' : 'low'}`}
                          style={{ fontSize: 11, padding: '2px 8px' }}
                        >
                          {event.confidenceScore?.toFixed(0)}%
                        </span>
                      </div>
                      <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
                        {(event.activities || []).map((act, i) => (
                          <span key={i} className={`tag ${act}`} style={{ fontSize: 11 }}>
                            {act.replace(/_/g, ' ')}
                          </span>
                        ))}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                        {event.personIds?.[0]?.personLabel || 'Unknown'} • {event.duration?.toFixed(1)}s
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

function formatTime(seconds) {
  if (seconds == null) return '00:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
