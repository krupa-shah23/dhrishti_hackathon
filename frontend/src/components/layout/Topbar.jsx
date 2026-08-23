/**
 * DRISHTI — Top Bar Component
 */
import { Search } from 'lucide-react';

export default function Topbar({ title }) {
  return (
    <header className="topbar">
      <h1 className="topbar-title">{title}</h1>

      <div className="topbar-actions">
        <div className="search-bar">
          <Search size={16} style={{ color: 'var(--text-muted)', flexShrink: 0 }} />
          <input type="text" placeholder="Search incidents, persons..." />
        </div>

        <div className="user-avatar" title="Profile">
          P
        </div>
      </div>
    </header>
  );
}
