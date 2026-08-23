/**
 * DRISHTI — Reports Page (/reports)
 * Export center for PDF dossiers and evidence packs.
 */
import { FileText, Download, FileArchive } from 'lucide-react';

export default function ReportsPage() {
  return (
    <div className="slide-up">
      <div className="card">
        <div className="card-header">
          <span className="card-title" style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <FileText size={18} /> Export Center
          </span>
        </div>
        <div className="card-body">
          <div className="empty-state">
            <FileArchive size={48} className="empty-state-icon" />
            <div className="empty-state-title">Reports & Evidence Packs</div>
            <div className="empty-state-text">
              Generate PDF dossiers and evidence packs from processed videos.
              <br />This feature will be connected in Phase 4.
            </div>
            <button className="btn btn-primary" style={{ marginTop: 16 }} disabled>
              <Download size={16} /> Generate Report
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
