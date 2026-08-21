/**
 * DRISHTI — Person Profile Page (/persons/:personId)
 * 
 * Person Timeline: confidence over time graph
 * Cross-Video Reference: list of all videos this person appeared in
 */
import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import { User, Video, ArrowLeft } from 'lucide-react';
import { personApi } from '../api/client';

export default function PersonProfilePage() {
  const { personId } = useParams();
  const navigate = useNavigate();
  const [person, setPerson] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [videos, setVideos] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchProfile();
  }, [personId]);

  const fetchProfile = async () => {
    try {
      setLoading(true);
      const [personRes, timelineRes, videosRes] = await Promise.all([
        personApi.getById(personId),
        personApi.getTimeline(personId),
        personApi.getVideos(personId),
      ]);
      setPerson(personRes.data.data);
      setTimeline(timelineRes.data.data || []);
      setVideos(videosRes.data.data || []);
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading-container"><div className="spinner" /><span>Loading profile...</span></div>;
  }

  if (!person) {
    return <div className="empty-state"><div className="empty-state-title">Person not found</div></div>;
  }

  const graphData = timeline.map((t) => ({
    time: t.timestamps?.[0]?.start?.toFixed(1) + 's',
    confidence: t.confidence,
    activities: t.activities?.join(', '),
  }));

  return (
    <div className="slide-up">
      {/* Back Button */}
      <button className="btn btn-ghost" style={{ marginBottom: 20 }} onClick={() => navigate('/persons')}>
        <ArrowLeft size={16} /> Back to Persons
      </button>

      {/* Person Header */}
      <div className="card" style={{ marginBottom: 24 }}>
        <div className="card-body" style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
          {person.thumbnailPath ? (
            <img src={person.thumbnailPath} alt={person.personLabel} className="person-avatar" style={{ width: 72, height: 72 }} />
          ) : (
            <div className="person-avatar-placeholder" style={{ width: 72, height: 72, fontSize: 28 }}>
              {person.personLabel?.charAt(person.personLabel.length - 1) || '?'}
            </div>
          )}
          <div>
            <h2 style={{ fontFamily: 'Poppins, sans-serif', fontSize: 24, fontWeight: 700 }}>
              {person.personLabel}
            </h2>
            <div style={{ fontSize: 14, color: 'var(--text-secondary)', marginTop: 4 }}>
              {person.seatId ? `Seat ${person.seatId}` : 'No seat assigned'} •
              {' '}{person.totalDetections || 0} detections •
              {' '}{videos.length} video{videos.length !== 1 ? 's' : ''}
            </div>
          </div>
        </div>
      </div>

      <div className="grid-2">
        {/* Person Timeline */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Confidence Over Time</span>
          </div>
          <div className="card-body">
            {graphData.length > 0 ? (
              <ResponsiveContainer width="100%" height={250}>
                <LineChart data={graphData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                  <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#94A3B8" />
                  <Tooltip />
                  <Line type="monotone" dataKey="confidence" stroke="#6366F1" strokeWidth={2} dot={{ r: 3 }} />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state" style={{ padding: 40 }}>
                <div className="empty-state-text">No timeline data</div>
              </div>
            )}
          </div>
        </div>

        {/* Cross-Video Reference */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Cross-Video References</span>
          </div>
          <div className="card-body">
            {videos.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {videos.map((mapping) => (
                  <div
                    key={mapping._id}
                    style={{
                      display: 'flex', alignItems: 'center', gap: 12,
                      padding: '12px 16px', borderRadius: 'var(--radius-md)',
                      border: '1px solid var(--border-light)', cursor: 'pointer',
                      transition: 'all 150ms',
                    }}
                    onClick={() => navigate(`/videos/${mapping.videoId?._id || mapping.videoId}`)}
                    onMouseOver={(e) => e.currentTarget.style.background = 'var(--bg-primary)'}
                    onMouseOut={(e) => e.currentTarget.style.background = 'transparent'}
                  >
                    <Video size={18} style={{ color: 'var(--accent-blue)', flexShrink: 0 }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ fontWeight: 500, fontSize: 14 }}>
                        {mapping.videoId?.originalName || mapping.videoId?.filename || 'Unknown video'}
                      </div>
                      <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                        Duration visible: {mapping.totalDuration?.toFixed(1)}s
                        {mapping.seatId && ` • Seat ${mapping.seatId}`}
                      </div>
                    </div>
                    <span className={`status-badge ${mapping.videoId?.status || 'done'}`}>
                      {mapping.videoId?.status || 'done'}
                    </span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state" style={{ padding: 40 }}>
                <div className="empty-state-text">No cross-video data</div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
