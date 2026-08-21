/**
 * DRISHTI — Overview Dashboard Page (/dashboard)
 * 
 * Row 1: Stat cards (Total Videos, Total Persons)
 * Row 2: Donut chart (Incident Type Distribution) + Sparklines
 * Row 3: Recent Incidents Timeline (area chart)
 */
import { useState, useEffect } from 'react';
import {
  PieChart, Pie, Cell, ResponsiveContainer, Tooltip, Legend,
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
} from 'recharts';
import { Video, Users, AlertTriangle, TrendingUp } from 'lucide-react';
import { dashboardApi } from '../api/client';

const DONUT_COLORS = ['#3B82F6', '#F04438', '#FFC107', '#6366F1', '#12B76A', '#EC4899', '#F97316'];

export default function DashboardPage() {
  const [stats, setStats] = useState(null);
  const [timeline, setTimeline] = useState([]);
  const [timeRange, setTimeRange] = useState('month');
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchData();
  }, [timeRange]);

  const fetchData = async () => {
    try {
      setLoading(true);
      const [statsRes, timelineRes] = await Promise.all([
        dashboardApi.getStats(),
        dashboardApi.getTimeline(timeRange),
      ]);
      setStats(statsRes.data.data);
      setTimeline(timelineRes.data.data);
    } catch (err) {
      console.error('Dashboard fetch error:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading && !stats) {
    return (
      <div className="loading-container">
        <div className="spinner" />
        <span>Loading dashboard...</span>
      </div>
    );
  }

  const donutData = stats?.incidentDistribution?.map((item) => ({
    name: item.activity?.replace(/_/g, ' ') || 'Unknown',
    value: item.count,
    percentage: item.percentage,
  })) || [];

  return (
    <div className="slide-up">
      {/* Row 1: Stat Cards */}
      <div className="grid-3" style={{ marginBottom: 24 }}>
        <div className="stat-card">
          <div className="stat-icon blue">
            <Video size={28} />
          </div>
          <div>
            <div className="stat-value">{stats?.totalVideosProcessed || 0}</div>
            <div className="stat-label">Total Videos Processed</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon red">
            <Users size={28} />
          </div>
          <div>
            <div className="stat-value">{stats?.totalPersonsCaught || 0}</div>
            <div className="stat-label">Total Persons Caught Copying</div>
          </div>
        </div>

        <div className="stat-card">
          <div className="stat-icon yellow">
            <AlertTriangle size={28} />
          </div>
          <div>
            <div className="stat-value">{stats?.totalIncidents || 0}</div>
            <div className="stat-label">Total Incidents Detected</div>
          </div>
        </div>
      </div>

      {/* Row 2: Donut Chart + Sparklines */}
      <div className="grid-2" style={{ marginBottom: 24 }}>
        <div className="card">
          <div className="card-header">
            <span className="card-title">Incident Type Distribution</span>
          </div>
          <div className="card-body">
            {donutData.length > 0 ? (
              <ResponsiveContainer width="100%" height={280}>
                <PieChart>
                  <Pie
                    data={donutData}
                    cx="50%"
                    cy="50%"
                    innerRadius={65}
                    outerRadius={110}
                    paddingAngle={3}
                    dataKey="value"
                  >
                    {donutData.map((_, idx) => (
                      <Cell key={idx} fill={DONUT_COLORS[idx % DONUT_COLORS.length]} />
                    ))}
                  </Pie>
                  <Tooltip
                    formatter={(value, name) => [`${value} incidents`, name]}
                    contentStyle={{
                      borderRadius: 12,
                      border: '1px solid #E2E8F0',
                      boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
                    }}
                  />
                  <Legend
                    verticalAlign="bottom"
                    formatter={(value) => (
                      <span style={{ fontSize: 13, color: '#64748B', textTransform: 'capitalize' }}>
                        {value}
                      </span>
                    )}
                  />
                </PieChart>
              </ResponsiveContainer>
            ) : (
              <div className="empty-state">
                <div className="empty-state-title">No incidents yet</div>
                <div className="empty-state-text">Upload and process a video to see data</div>
              </div>
            )}
          </div>
        </div>

        <div className="card">
          <div className="card-header">
            <span className="card-title">Top Categories</span>
            <TrendingUp size={18} style={{ color: 'var(--text-muted)' }} />
          </div>
          <div className="card-body">
            {donutData.length > 0 ? (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                {donutData.slice(0, 5).map((item, idx) => (
                  <div key={idx}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 6 }}>
                      <span style={{ fontSize: 14, fontWeight: 500, textTransform: 'capitalize' }}>
                        {item.name}
                      </span>
                      <span style={{ fontSize: 14, fontWeight: 600, color: DONUT_COLORS[idx] }}>
                        {item.percentage}%
                      </span>
                    </div>
                    <div className="progress-bar-container" style={{ height: 8 }}>
                      <div
                        className="progress-bar-fill"
                        style={{
                          width: `${item.percentage}%`,
                          background: DONUT_COLORS[idx],
                        }}
                      />
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-state">
                <div className="empty-state-text">No data available</div>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Row 3: Timeline */}
      <div className="card">
        <div className="card-header">
          <span className="card-title">Recent Incidents Timeline</span>
          <div style={{ display: 'flex', gap: 8 }}>
            {['week', 'month', 'year'].map((r) => (
              <button
                key={r}
                className={`btn btn-sm ${timeRange === r ? 'btn-primary' : 'btn-ghost'}`}
                onClick={() => setTimeRange(r)}
              >
                {r.charAt(0).toUpperCase() + r.slice(1)}
              </button>
            ))}
          </div>
        </div>
        <div className="card-body">
          {timeline.length > 0 ? (
            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={timeline}>
                <defs>
                  <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#3B82F6" stopOpacity={0.15} />
                    <stop offset="95%" stopColor="#3B82F6" stopOpacity={0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis dataKey="date" tick={{ fontSize: 12 }} stroke="#94A3B8" />
                <YAxis tick={{ fontSize: 12 }} stroke="#94A3B8" />
                <Tooltip
                  contentStyle={{
                    borderRadius: 12,
                    border: '1px solid #E2E8F0',
                    boxShadow: '0 4px 6px rgba(0,0,0,0.07)',
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="count"
                  stroke="#3B82F6"
                  strokeWidth={2.5}
                  fill="url(#areaGrad)"
                  name="Incidents"
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div className="empty-state" style={{ padding: '40px 20px' }}>
              <div className="empty-state-text">No timeline data for this range</div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
