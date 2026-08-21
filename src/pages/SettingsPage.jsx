/**
 * DRISHTI — Settings Page (/settings)
 * Seat-grid calibration, thresholds, and toggles.
 */
import { Settings as SettingsIcon, Save } from 'lucide-react';
import { useState } from 'react';

export default function SettingsPage() {
  const [confidenceThreshold, setConfidenceThreshold] = useState(65);
  const [privacyDefault, setPrivacyDefault] = useState(false);
  const [maxVideos, setMaxVideos] = useState(3);

  return (
    <div className="slide-up">
      <div className="grid-2">
        {/* Detection Settings */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Detection Settings</span>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
            <div>
              <label style={{ fontSize: 14, fontWeight: 500, display: 'block', marginBottom: 8 }}>
                Confidence Threshold (%)
              </label>
              <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
                <input
                  type="range" min={10} max={95} value={confidenceThreshold}
                  onChange={(e) => setConfidenceThreshold(Number(e.target.value))}
                  style={{ flex: 1 }}
                />
                <span style={{ fontWeight: 600, fontSize: 16, minWidth: 40, textAlign: 'right' }}>
                  {confidenceThreshold}%
                </span>
              </div>
              <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 4 }}>
                Events below this threshold will be marked as low confidence
              </div>
            </div>

            <div>
              <label style={{ fontSize: 14, fontWeight: 500, display: 'block', marginBottom: 8 }}>
                Max Concurrent Videos
              </label>
              <select
                value={maxVideos}
                onChange={(e) => setMaxVideos(Number(e.target.value))}
                style={{
                  padding: '8px 12px', borderRadius: 'var(--radius-md)',
                  border: '1px solid var(--border-light)', fontSize: 14,
                  width: '100%', background: 'white',
                }}
              >
                {[1, 2, 3, 4, 5].map((n) => (
                  <option key={n} value={n}>{n} videos</option>
                ))}
              </select>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
              <div>
                <div style={{ fontSize: 14, fontWeight: 500 }}>Privacy Mode Default</div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                  Blur non-flagged faces by default in video review
                </div>
              </div>
              <label style={{ position: 'relative', display: 'inline-block', width: 44, height: 24 }}>
                <input
                  type="checkbox" checked={privacyDefault}
                  onChange={(e) => setPrivacyDefault(e.target.checked)}
                  style={{ opacity: 0, width: 0, height: 0 }}
                />
                <span style={{
                  position: 'absolute', top: 0, left: 0, right: 0, bottom: 0,
                  background: privacyDefault ? 'var(--accent-blue)' : 'var(--border-medium)',
                  borderRadius: 'var(--radius-full)', cursor: 'pointer', transition: 'all 200ms',
                }}>
                  <span style={{
                    position: 'absolute', left: privacyDefault ? 22 : 2, top: 2,
                    width: 20, height: 20, background: 'white',
                    borderRadius: '50%', transition: 'all 200ms',
                    boxShadow: '0 1px 3px rgba(0,0,0,0.2)',
                  }} />
                </span>
              </label>
            </div>
          </div>
        </div>

        {/* Seat Grid */}
        <div className="card">
          <div className="card-header">
            <span className="card-title">Seat Grid Calibration</span>
          </div>
          <div className="card-body">
            <div className="empty-state">
              <SettingsIcon size={48} className="empty-state-icon" />
              <div className="empty-state-title">Seat Grid Setup</div>
              <div className="empty-state-text">
                Upload a reference frame and draw the seat grid to map
                detected positions to seat IDs.
                <br />Coming in Phase 4.
              </div>
            </div>
          </div>
        </div>
      </div>

      <div style={{ marginTop: 24, display: 'flex', justifyContent: 'flex-end' }}>
        <button className="btn btn-primary">
          <Save size={16} /> Save Settings
        </button>
      </div>
    </div>
  );
}
