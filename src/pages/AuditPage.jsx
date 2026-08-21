/**
 * DRISHTI — Audit Log Page (/audit)
 * Offline/compliance audit log.
 */
import { ClipboardList, ShieldCheck } from 'lucide-react';

export default function AuditPage() {
  return (
    <div className="slide-up">
      <div className="card">
        <div className="card-header">
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <ClipboardList size={18} /> Audit Log
          </span>
        </div>
        <div className="card-body">
          <div className="empty-state">
            <ShieldCheck size={48} className="empty-state-icon" />
            <div className="empty-state-title">Compliance Audit Trail</div>
            <div className="empty-state-text">
              All system actions, user logins, video processing events,
              and data access records will be logged here for compliance.
              <br />This feature will be connected in Phase 4.
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
