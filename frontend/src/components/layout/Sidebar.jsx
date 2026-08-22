/**
 * DRISHTI — Sidebar Navigation Component
 */
import { useLocation, useNavigate } from 'react-router-dom';
import {
  LayoutDashboard,
  Upload,
  AlertTriangle,
  Video,
  Users,
  Flame,
  FileText,
  Settings,
  ClipboardList,
  Eye,
} from 'lucide-react';

const navItems = [
  { path: '/dashboard', label: 'Overview', icon: LayoutDashboard },
  { path: '/videos', label: 'Upload', icon: Upload },
  { path: '/incidents', label: 'Incidents', icon: AlertTriangle },
  { path: '/persons', label: 'Persons', icon: Users },
  { path: '/heatmap', label: 'Heatmap', icon: Flame },
  { path: '/reports', label: 'Reports', icon: FileText },
  { path: '/settings', label: 'Settings', icon: Settings },
  { path: '/audit', label: 'Audit Log', icon: ClipboardList },
];

export default function Sidebar() {
  const location = useLocation();
  const navigate = useNavigate();

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-logo">
          <Eye size={22} color="white" />
        </div>
        <span className="sidebar-brand-text">DRISHTI</span>
      </div>

      <nav className="sidebar-nav">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive =
            location.pathname === item.path ||
            location.pathname.startsWith(item.path + '/');

          return (
            <button
              key={item.path}
              className={`sidebar-link ${isActive ? 'active' : ''}`}
              onClick={() => navigate(item.path)}
            >
              <Icon size={20} />
              {item.label}
            </button>
          );
        })}
      </nav>
    </aside>
  );
}
