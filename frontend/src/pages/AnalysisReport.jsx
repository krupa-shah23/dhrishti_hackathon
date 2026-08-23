import { useState, useEffect, useRef } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import {
  BrainCircuit, Download, ArrowLeft, UserRound,
  AlertTriangle, Clock, TrendingUp, FileText,
} from 'lucide-react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts';
import { eventApi, analysisApi, videoApi } from '../api/client';
import html2pdf from 'html2pdf.js';

export default function AnalysisReport() {
  const { videoId } = useParams();
  const navigate = useNavigate();
  const reportRef = useRef(null);

  const [video, setVideo] = useState(null);
  const [events, setEvents] = useState([]);
  const [summary, setSummary] = useState('');
  const [loading, setLoading] = useState(true);
  const [summaryLoading, setSummaryLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!videoId) return;
    loadAll();
  }, [videoId]);

  const loadAll = async () => {
    try {
      setLoading(true);
      const [videoRes, eventsRes] = await Promise.all([
        videoApi.getById(videoId),
        eventApi.getAll({ videoId, limit: 200, sort: 'timestamps', order: 'asc' }),
      ]);
      setVideo(videoRes.data.data);
      setEvents(eventsRes.data.data || []);

      // Fetch XAI summary separately (can be slow)
      setSummaryLoading(true);
      try {
        const summaryRes = await analysisApi.getSummary(videoId);
        setSummary(summaryRes.data.data?.summary || '');
      } catch {
        setSummary('[AI Summary unavailable — check Gemini API key or XAI toggle in Settings]');
      } finally {
        setSummaryLoading(false);
      }
    } catch (err) {
      setError(err.message || 'Failed to load analysis data');
    } finally {
      setLoading(false);
    }
  };

  const handleDownloadPdf = () => {
    if (!reportRef.current) return;
    html2pdf()
      .set({
        margin: 0.6,
        filename: `DRISHTI_Report_${video?.originalName || videoId}.pdf`,
        image: { type: 'jpeg', quality: 0.98 },
        html2canvas: { scale: 2, useCORS: true },
        jsPDF: { unit: 'in', format: 'letter', orientation: 'portrait' },
      })
      .from(reportRef.current)
      .save();
  };

  const getConfidenceClass = (score) => {
    if (score >= 70) return 'high';
    if (score >= 40) return 'medium';
    return 'low';
  };

  const formatTime = (seconds) => {
    if (seconds == null) return '—';
    const m = Math.floor(seconds / 60);
    const s = Math.floor(seconds % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
  };

  // Timeline chart data
  const timelineData = events.map((e) => ({
    time: formatTime(e.timestamps?.[0]?.start),
    confidence: e.confidenceScore ?? 0,
  }));

  // Stats
  const highRisk   = events.filter((e) => e.confidenceScore >= 70).length;
  const mediumRisk = events.filter((e) => e.confidenceScore >= 40 && e.confidenceScore < 70).length;
  const lowRisk    = events.filter((e) => e.confidenceScore < 40).length;

  if (loading) {
    return (
      <div className="loading-container" style={{ paddingTop: 80 }}>
        <div className="spinner" />
        <span>Loading analysis report...</span>
      </div>
    );
  }

  if (error || !video) {
    return (
      <div className="empty-state" style={{ paddingTop: 80 }}>
        <AlertTriangle size={48} style={{ opacity: 0.4, marginBottom: 12 }} />
        <div className="empty-state-title">Analysis not found</div>
        <div className="empty-state-text">{error || 'This video ID does not exist.'}</div>
        <button className="btn btn-primary" style={{ marginTop: 20 }} onClick={() => navigate('/upload')}>
          <ArrowLeft size={16} /> Back to Uploads
        </button>
      </div>
    );
  }

  return (
    <div className="slide-up">
      {/* ── Page Header ── */}
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 24 }}>
        <div>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => navigate('/upload')}
            style={{ marginBottom: 8 }}
          >
            <ArrowLeft size={15} /> Back to Uploads
          </button>
          <h1 style={{ margin: 0, fontSize: 22, fontWeight: 700 }}>Analysis Report</h1>
          <p style={{ margin: '4px 0 0', color: 'var(--text-secondary)', fontSize: 14 }}>
            {video.originalName} &nbsp;·&nbsp; {events.length} incident{events.length !== 1 ? 's' : ''} detected
          </p>
        </div>
        <button className="btn btn-primary" onClick={handleDownloadPdf} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
          <Download size={16} /> Download PDF Report
        </button>
      </div>

      {/* ── PDF Content Wrapper ── */}
      <div ref={reportRef}>
        {/* PDF Header (only visible when printed) */}
        <div style={{ marginBottom: 20 }}>
          <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>
            DRISHTI AI Surveillance Suite &nbsp;·&nbsp; Report generated {new Date().toLocaleString()}
          </div>
          <div style={{ fontWeight: 700, fontSize: 18 }}>{video.originalName}</div>
        </div>

        {/* ── Stat Cards ── */}
        <div className="grid-3" style={{ marginBottom: 24 }}>
          <div className="stat-card">
            <div className="stat-icon red"><AlertTriangle size={24} /></div>
            <div>
              <div className="stat-value">{highRisk}</div>
              <div className="stat-label">High-Risk Incidents</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon yellow"><TrendingUp size={24} /></div>
            <div>
              <div className="stat-value">{mediumRisk}</div>
              <div className="stat-label">Medium-Risk Incidents</div>
            </div>
          </div>
          <div className="stat-card">
            <div className="stat-icon blue"><FileText size={24} /></div>
            <div>
              <div className="stat-value">{events.length}</div>
              <div className="stat-label">Total Events</div>
            </div>
          </div>
        </div>

        {/* ── Timeline + XAI Summary ── */}
        <div className="grid-65-35" style={{ marginBottom: 24 }}>
          {/* Timeline Chart */}
          <div className="card">
            <div className="card-header"><span className="card-title">Confidence Timeline</span></div>
            <div className="card-body">
              {timelineData.length > 0 ? (
                <ResponsiveContainer width="100%" height={220}>
                  <LineChart data={timelineData}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                    <XAxis dataKey="time" tick={{ fontSize: 11 }} stroke="#94A3B8" />
                    <YAxis domain={[0, 100]} tick={{ fontSize: 11 }} stroke="#94A3B8" tickFormatter={(v) => `${v}%`} />
                    <Tooltip
                      formatter={(v) => [`${v}%`, 'Confidence']}
                      contentStyle={{ borderRadius: 10, border: '1px solid #E2E8F0' }}
                    />
                    <Line type="monotone" dataKey="confidence" stroke="#F04438" strokeWidth={2.5} dot={{ r: 3 }} name="Confidence" />
                  </LineChart>
                </ResponsiveContainer>
              ) : (
                <div className="empty-state" style={{ padding: 40 }}>
                  <div className="empty-state-text">No timeline data</div>
                </div>
              )}
            </div>
          </div>

          {/* XAI Summary */}
          <div className="card">
            <div className="card-header">
              <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <BrainCircuit size={18} /> AI Explanatory Summary
              </span>
            </div>
            <div className="card-body">
              {summaryLoading ? (
                <div className="loading-container" style={{ padding: 40 }}>
                  <div className="spinner" />
                  <span style={{ fontSize: 13 }}>Generating AI summary...</span>
                </div>
              ) : (
                <p style={{ fontSize: 14, lineHeight: 1.75, color: 'var(--text-secondary)', margin: 0 }}>
                  {summary || 'No summary available.'}
                </p>
              )}
            </div>
          </div>
        </div>

        {/* ── Detected Events Table ── */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Detailed Incident Log</span>
            <span style={{ fontSize: 13, color: 'var(--text-muted)' }}>{events.length} events · sorted by timestamp</span>
          </div>

          {events.length === 0 ? (
            <div className="empty-state" style={{ padding: 60 }}>
              <div className="empty-state-title">No incidents detected</div>
              <div className="empty-state-text">The ML pipeline found no suspicious activity in this video.</div>
            </div>
          ) : (
            <div style={{ overflowX: 'auto' }}>
              <table className="data-table">
                <thead>
                  <tr>
                    <th>Timestamp</th>
                    <th>Person</th>
                    <th>Seat</th>
                    <th>Activity / What Happened</th>
                    <th>Object Detected</th>
                    <th>Duration</th>
                    <th>Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((event) => (
                    <tr key={event._id} className={event.confidenceScore >= 70 ? 'row-high' : event.confidenceScore >= 40 ? 'row-medium' : ''}>
                      <td style={{ fontWeight: 500, whiteSpace: 'nowrap' }}>
                        <Clock size={13} style={{ marginRight: 4, verticalAlign: 'middle', opacity: 0.5 }} />
                        {formatTime(event.timestamps?.[0]?.start)}
                        {event.timestamps?.[0]?.end != null && (
                          <span style={{ color: 'var(--text-muted)', fontSize: 12 }}>
                            &nbsp;→ {formatTime(event.timestamps[0].end)}
                          </span>
                        )}
                      </td>
                      <td>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                          <div className="person-avatar-placeholder" style={{ width: 28, height: 28, fontSize: 12 }}>
                            {(event.personIds?.[0]?.personLabel || '?')[0]}
                          </div>
                          <span style={{ fontWeight: 500 }}>
                            {event.personIds?.[0]?.personLabel || '—'}
                          </span>
                        </div>
                      </td>
                      <td style={{ color: 'var(--text-secondary)', fontSize: 13 }}>
                        {event.seatId ? `Seat ${event.seatId}` : '—'}
                      </td>
                      <td>
                        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
                          {(event.activities || []).map((act, i) => (
                            <span key={i} className={`tag ${act}`} style={{ textTransform: 'capitalize' }}>
                              {act.replace(/_/g, ' ')}
                            </span>
                          ))}
                        </div>
                        {event.explanation && (
                          <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                            {event.explanation}
                          </div>
                        )}
                      </td>
                      <td style={{ fontSize: 13 }}>
                        {event.objectDetected
                          ? <span className="tag phone">{event.objectDetected.replace(/_/g, ' ')}</span>
                          : <span style={{ color: 'var(--text-muted)' }}>None</span>}
                      </td>
                      <td style={{ fontSize: 13 }}>
                        {event.duration != null ? `${event.duration.toFixed(1)}s` : '—'}
                      </td>
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
          )}
        </div>

      </div>{/* end reportRef */}
    </div>
  );
}
