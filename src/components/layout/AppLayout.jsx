import { NavLink, Outlet, useLocation } from 'react-router-dom';
import {
  Bell, CalendarDays, LayoutDashboard, UploadCloud,
  AlertTriangle, Users, Settings, ShieldCheck, Search, Video,
} from 'lucide-react';

const nav = [
  { to: '/',          label: 'Dashboard',        icon: LayoutDashboard, exact: true },
  { to: '/upload',    label: 'Upload Manager',   icon: UploadCloud },
  { to: '/incidents', label: 'Incidents',        icon: AlertTriangle },
  { to: '/persons',   label: 'Persons',          icon: Users },
  { to: '/settings',  label: 'Settings',         icon: Settings },
];

export default function AppLayout() {
  const { pathname } = useLocation();

  const isActive = (item) => {
    if (item.exact) return pathname === '/';
    // Mark upload active also on /analysis/* (analysis report is an upload sub-flow)
    if (item.to === '/upload') {
      return pathname.startsWith('/upload') || pathname.startsWith('/analysis');
    }
    return pathname.startsWith(item.to);
  };

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark"><ShieldCheck size={25} /></div>
          <div>
            <strong>DRISHTI</strong>
            <span>SURVEILLANCE SUITE</span>
          </div>
        </div>

        <nav>
          {nav.map(({ to, label, icon: Icon }) => (
            <NavLink
              key={to}
              to={to}
              className={() => `nav-link ${isActive({ to, exact: to === '/' }) ? 'active' : ''}`}
            >
              <Icon size={19} />
              {label}
            </NavLink>
          ))}
        </nav>

        <div className="system-status">
          <span className="online-dot" />
          All systems operational
        </div>
      </aside>

      <main className="main">
        <header className="topbar">
          <label className="global-search">
            <Search size={18} />
            <input aria-label="Search records" placeholder="Search records, IDs..." />
          </label>
          <div className="top-actions">
            <NavLink to="/videos" className="icon-button" aria-label="Video Review">
              <Video size={18} />
            </NavLink>
            <button className="date-chip">
              <CalendarDays size={16} />
              {new Date().toLocaleDateString('en-GB', { day: '2-digit', month: 'short', year: 'numeric' })}
            </button>
            <button className="icon-button" aria-label="Notifications">
              <Bell size={18} />
            </button>
            <div className="avatar">A</div>
          </div>
        </header>

        <section className="page">
          <Outlet />
        </section>
      </main>
    </div>
  );
}
