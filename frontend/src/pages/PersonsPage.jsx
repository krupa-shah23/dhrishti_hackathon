/**
 * DRISHTI — Persons List Page (/persons)
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users } from 'lucide-react';
import { personApi } from '../api/client';

const REID_DEMO_PERSONS = [
  { _id: 'demo-RID-7F3A2C91', personLabel: 'RID-7F3A2C91', thumbnailPath: '/clip1.png', seatId: null, totalDetections: 9 },
  { _id: 'demo-RID-2B8E4D06', personLabel: 'RID-2B8E4D06', thumbnailPath: '/clip3.png', seatId: null, totalDetections: 10 },
  { _id: 'demo-RID-9C1D5A73', personLabel: 'RID-9C1D5A73', thumbnailPath: '/clip4.png', seatId: null, totalDetections: 6 },
  { _id: 'demo-RID-4E7B0F58', personLabel: 'RID-4E7B0F58', thumbnailPath: '/clip6.png', seatId: null, totalDetections: 26 },
  { _id: 'demo-RID-C63A19E4', personLabel: 'RID-C63A19E4', thumbnailPath: '/clip7.png', seatId: null, totalDetections: 6 },
  { _id: 'demo-RID-08D2F6BA', personLabel: 'RID-08D2F6BA', thumbnailPath: '/clip8.png', seatId: '12', totalDetections: 1 },
];

export default function PersonsPage() {
  const [persons, setPersons] = useState(REID_DEMO_PERSONS);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await personApi.getAll();
        // Only show persons that actually have a re-ID thumbnail — entries
        // without one are placeholder records, not real identified subjects.
        const withPhotos = (res.data.data || []).filter((p) => p.thumbnailPath);
        setPersons([...REID_DEMO_PERSONS, ...withPhotos]);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };
    fetch();
  }, []);

  if (loading) {
    return <div className="loading-container"><div className="spinner" /><span>Loading persons...</span></div>;
  }

  if (persons.length === 0) {
    return (
      <div className="empty-state">
        <Users size={48} className="empty-state-icon" />
        <div className="empty-state-title">No persons detected yet</div>
        <div className="empty-state-text">Process a video to see detected persons</div>
      </div>
    );
  }

  return (
    <div className="slide-up">
      <div className="grid-3">
        {persons.map((person) => (
          <div
            key={person._id}
            className="card"
            style={{ cursor: 'pointer' }}
            onClick={() => navigate(`/persons/${person._id}`)}
          >
            <div className="card-body" style={{ display: 'flex', alignItems: 'center', gap: 16 }}>
              {person.thumbnailPath ? (
                <img src={person.thumbnailPath} alt={person.personLabel} className="person-avatar" style={{ width: 56, height: 56 }} />
              ) : (
                <div className="person-avatar-placeholder" style={{ width: 56, height: 56, fontSize: 20 }}>
                  {person.personLabel?.charAt(person.personLabel.length - 1) || '?'}
                </div>
              )}
              <div>
                <div style={{ fontWeight: 600, fontSize: 16 }}>{person.personLabel}</div>
                <div style={{ fontSize: 13, color: 'var(--text-secondary)' }}>
                  {person.seatId ? `Seat ${person.seatId}` : 'No seat assigned'}
                </div>
                <div style={{ fontSize: 12, color: 'var(--text-muted)', marginTop: 2 }}>
                  {person.totalDetections || 0} detections
                </div>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
