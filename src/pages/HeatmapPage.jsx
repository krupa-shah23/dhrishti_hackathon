/**
 * DRISHTI — Heatmap Page (/heatmap)
 * Motion heatmap overlay view.
 */
import { useState, useEffect } from 'react';
import { Flame } from 'lucide-react';
import { videoApi } from '../api/client';

export default function HeatmapPage() {
  const [videos, setVideos] = useState([]);
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await videoApi.getAll();
        const done = (res.data.data || []).filter((v) => v.status === 'done');
        setVideos(done);
        if (done.length > 0) setSelectedVideo(done[0]);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) {
    return <div className="loading-container"><div className="spinner" /><span>Loading...</span></div>;
  }

  return (
    <div className="slide-up">
      {/* Video Selector */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-header">
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <Flame size={18} /> Motion Heatmap
          </span>
          <select
            style={{
              padding: '6px 12px', borderRadius: 'var(--radius-md)',
              border: '1px solid var(--border-light)', fontSize: 13,
              background: 'white', cursor: 'pointer',
            }}
            value={selectedVideo?._id || ''}
            onChange={(e) => setSelectedVideo(videos.find((v) => v._id === e.target.value))}
          >
            {videos.map((v) => (
              <option key={v._id} value={v._id}>{v.originalName}</option>
            ))}
          </select>
        </div>
        <div className="card-body" style={{ display: 'flex', justifyContent: 'center', minHeight: 400 }}>
          {selectedVideo?.heatmapPath ? (
            <img
              src={`http://localhost:5000/static/${selectedVideo.heatmapPath.split('/').pop()}`}
              alt="Motion Heatmap"
              style={{ maxWidth: '100%', maxHeight: 500, borderRadius: 'var(--radius-md)' }}
            />
          ) : (
            <div className="empty-state">
              <Flame size={48} className="empty-state-icon" />
              <div className="empty-state-title">No heatmap available</div>
              <div className="empty-state-text">
                {videos.length === 0
                  ? 'Process a video to generate heatmaps'
                  : 'This video has not generated a heatmap yet'}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
