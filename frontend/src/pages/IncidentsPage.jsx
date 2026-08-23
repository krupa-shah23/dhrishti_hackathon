/**
 * DRISHTI — Incidents Page (/incidents)
 * 
 * Row 1: Event Timeline chart + XAI Summary Panel
 * Row 2: Detected Events table (the main output table)
 */
import { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { Brain, Filter, ChevronLeft, ChevronRight } from 'lucide-react';
import { eventApi, videoApi, analysisApi } from '../api/client';

export default function IncidentsPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const videoIdFilter = searchParams.get('videoId');

  const [events, setEvents] = useState([]);
  const [pagination, setPagination] = useState({ page: 1, totalPages: 1, total: 0 });
  const [summary, setSummary] = useState(null);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [loading, setLoading] = useState(true);
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(videoIdFilter || '');

  useEffect(() => {
    fetchVideos();
  }, []);

  useEffect(() => {
    fetchEvents(1);
  }, [selectedVideo]);

  useEffect(() => {
    if (selectedVideo) fetchSummary(selectedVideo);
  }, [selectedVideo]);

  const fetchVideos = async () => {
    try {
      const res = await videoApi.getAll();
      setVideos((res.data.data || []).filter((v) => v.status === 'done'));
    } catch (err) {
      console.error(err);
    }
  };

  const fetchEvents = async (page = 1) => {
    try {
      setLoading(true);
      const params = { page, limit: 25, sort: 'confidenceScore', order: 'desc' };
      if (selectedVideo) params.videoId = selectedVideo;
      const res = await eventApi.getAll(params);
      setEvents(res.data.data || []);
      setPagination({
        page: res.data.page,
        totalPages: res.data.totalPages,
        total: res.data.total,
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const fetchSummary = async (videoId) => {
    try {
      setSummaryLoading(true);
      const res = await analysisApi.getSummary(videoId);
      setSummary(res.data.data?.summary || null);
    } catch (err) {
      setSummary(null);
    } finally {
      setSummaryLoading(false);
    }
  };

  // Build timeline data from events
  const timelineData = events.map((e) => ({
    timestamp: e.timestamps?.[0]?.start?.toFixed(1) + 's',
    confidence: e.confidenceScore,
  }));

  const getConfidenceClass = (score) => {
    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  };

  const getRowClass = (score) => {
    if (score >= 70) return 'row-high';
    if (score >= 40) return 'row-medium';
    return '';
  };

  return (
    <div className="slide-up">
      {/* Row 1: Timeline + XAI Summary */}
      <div className="grid-65-35" style={{ marginBottom: 24 }}>
        {/* Event Timeline */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Event Timeline</span>
            <select
              style={{
                padding: '6px 12px', borderRadius: 'var(--radius-md)',
                border: '1px solid var(--border-light)', fontSize: 13,
                background: 'white', cursor: 'pointer',
              }}
              value={selectedVideo}
              onChange={(e) => setSelectedVideo(e.target.value)}
            >
              <option value="">All Videos</option>
              {videos.map((v) => (
                <option key={v._id} value={v._id}>{v.originalName}</option>
              ))}
            </select>
          </div>
          <div className="card-body">
            {timelineData.length > 0 ? (
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={timelineData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="timestamp" tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <Tooltip
                    contentStyle={{
                      borderRadius: 12, border: '1px solid #E2E8F0',
                      boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
                    }}
                  />
                  <Line
                    type="monotone" dataKey="confidence" stroke="#F04438"
                    strokeWidth={2} dot={{ r: 3 }} name="Confidence %"
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ padding: 40 }}>
                <div className="empty-state-text">No events to display</div>
              </div>
            )}
          </div>
        </div>

        {/* XAI Summary Panel */}
        <div className="card">
          <div className="card-header">
            <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
              <Brain size={18} /> XAI Summary
            </span>
          </div>
          <div className="card-body">
            {summaryLoading ? (
              <div className="loading-container" style={{ padding: 40 }}>
                <div className="spinner" />
                <span>Generating summary...</span>
              </div>
            ) : summary ? (
              <p style={{ fontSize: 14, lineHeight: 1.7, color: 'var(--text-secondary)' }}>
                {summary}
              </p>
            ) : (
              <div className="empty-state" style={{ padding: 40 }}>
                <Brain size={32} style={{ opacity: 0.3, marginBottom: 8 }} />
                <div className="empty-state-text">
                  Select a video to generate an AI summary
                </div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Row 2: Events Table */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">
            Detected Events {pagination.total > 0 && `(${pagination.total})`}
          </span>
          <Filter size={18} style={{ color: 'var(--text-muted)' }} />
        </div>

        {loading ? (
          <div className="loading-container">
            <div className="spinner" />
          </div>
        ) : events.length === 0 ? (
          <div className="empty-state">
            <div className="empty-state-title">No incidents detected</div>
            <div className="empty-state-text">Upload and process a video first</div>
          </div>
        ) : (
          <>
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Person ID</th>
                    <th>Photo</th>
                    <th>Activity</th>
                    <th>Duration</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event._id} className={getRowClass(event.confidenceScore)}>
                      <td>
                        <a
                          href="#"
                          onClick={(e) => {
                            e.preventDefault();
                            const vid = event.videoId?._id || event.videoId;
                            const t = event.timestamps?.[0]?.start || 0;
                            navigate(`/videos/${vid}?t=${t}`);
                          }}
                          style={{ fontWeight: 500 }}
                        >
                          {formatTimestamp(event.timestamps?.[0]?.start)}
                        </a>
                      </td>
                      <td style={{ fontWeight: 500 }}>
                        {event.personIds?.[0]?.personLabel || '—'}
                      </td>
                      <td>
                        {event.personIds?.[0]?.thumbnailPath ? (
                          <img
                            src={event.personIds[0].thumbnailPath}
                            alt="person"
                            className="person-avatar"
                          />
                        ) : (
                          <div className="person-avatar-placeholder">
                            {(event.personIds?.[0]?.personLabel || '?')[0]}
                          </div>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          {(event.activities || []).map((act, i) => (
                            <span key={i} className={`tag ${act}`}>
                              {act.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                      </td>
                      <td>{event.duration ? `${event.duration.toFixed(1)}s` : '—'}</td>
                      <td>
                        <span className={`confidence-badge ${getConfidenceClass(event.confidenceScore)}`}>
                          {event.confidenceScore?.toFixed(0)}%
                        </span>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {/* Pagination */}
            {pagination.totalPages > 1 && (
              <div style={{
                display: 'flex', justifyContent: 'center', alignItems: 'center',
                gap: 16, padding: 16, borderTop: '1px solid var(--border-light)',
              }}>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={pagination.page <= 1}
                  onClick={() => fetchEvents(pagination.page - 1)}
                >
                  <ChevronLeft size={16} /> Prev
                </button>
                <span style={{ fontSize: 14, color: 'var(--text-secondary)' }}>
                  Page {pagination.page} of {pagination.totalPages}
                </span>
                <button
                  className="btn btn-ghost btn-sm"
                  disabled={pagination.page >= pagination.totalPages}
                  onClick={() => fetchEvents(pagination.page + 1)}
                >
                  Next <ChevronRight size={16} />
                </button>
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

function formatTimestamp(seconds) {
  if (seconds == null) return '—';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}
