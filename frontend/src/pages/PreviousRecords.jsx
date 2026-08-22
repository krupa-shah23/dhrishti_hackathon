/**
 * DRISHTI — Previous Records Page (/previous-records)
 * Lists all completed videos as historical analysis records.
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { FileVideo, CheckCircle2, Calendar, Clock } from 'lucide-react';
import { videoApi } from '../api/client';

export default function PreviousRecords() {
  const navigate = useNavigate();
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await videoApi.getAll();
        // Show completed and archived videos
        const done = (res.data.data || []).filter((v) => v.status === 'done' || v.status === 'archived');
        setVideos(done);
      } catch (err) {
        console.error('Failed to load previous records:', err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) {
    return (
      <div className="loading-container" style={{ paddingTop: 80 }}>
        <div className="spinner" />
        <span>Loading previous records...</span>
      </div>
    );
  }

  if (videos.length === 0) {
    return (
      <div className="empty-state" style={{ paddingTop: 80 }}>
        <FileVideo size={48} style={{ opacity: 0.3, marginBottom: 12 }} />
        <div className="empty-state-title">No completed analyses yet</div>
        <div className="empty-state-text">Upload and process a video to see records here.</div>
        <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={() => navigate('/upload')}>
          Upload a Video
        </button>
      </div>
    );
  }

  return (
    <div className="slide-up">
      <div style={{ marginBottom: 24 }}>
        <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>Previous Records</h1>
        <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 14 }}>
          Review completed surveillance analysis reports ({videos.length} total)
        </p>
      </div>

      <div className="records-grid">
        {videos.map((video) => (
          <div key={video._id} className="card record">
            <div className="card-body">
              {/* Icon + Title */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
                <div style={{
                  width: 44, height: 44, borderRadius: 'var(--radius-md)',
                  background: 'var(--status-green-bg)',
                  display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0,
                }}>
                  <CheckCircle2 size={22} color="var(--status-green)" />
                </div>
                <div style={{ flex: 1, overflow: 'hidden' }}>
                  <div style={{ fontWeight: 600, fontSize: 15, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>
                    {video.originalName}
                  </div>
                  <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>
                    {(video.size / (1024 * 1024)).toFixed(1)} MB
                  </div>
                </div>
              </div>

              {/* Meta */}
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6, marginBottom: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)' }}>
                  <Calendar size={13} />
                  {new Date(video.createdAt).toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
                </div>
                {video.duration && (
                  <div style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 13, color: 'var(--text-secondary)' }}>
                    <Clock size={13} />
                    {Math.floor(video.duration / 60)}m {Math.floor(video.duration % 60)}s
                  </div>
                )}
              </div>

              {/* XAI Summary Preview */}
              {video.xaiSummary && (
                <div style={{ marginBottom: 16, padding: 12, background: 'var(--bg-primary)', borderRadius: 'var(--radius-md)', fontSize: 12, color: 'var(--text-secondary)', lineHeight: 1.5, display: '-webkit-box', WebkitLineClamp: 3, WebkitBoxOrient: 'vertical', overflow: 'hidden' }}>
                  <strong>AI Summary:</strong> {video.xaiSummary}
                </div>
              )}

              <span className="status-badge done">Processing complete ✓</span>

              <button
                className="btn btn-primary"
                style={{ width: '100%', marginTop: 16 }}
                onClick={() => navigate(`/analysis/${video._id}`)}
              >
                View Analysis Report
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
