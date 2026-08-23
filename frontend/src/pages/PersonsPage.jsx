/**
 * DRISHTI — Persons List Page (/persons)
 */
import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { Users } from 'lucide-react';
import { personApi } from '../api/client';

export default function PersonsPage() {
  const [persons, setPersons] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetch = async () => {
      try {
        const res = await personApi.getAll();
        setPersons(res.data.data || []);
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
