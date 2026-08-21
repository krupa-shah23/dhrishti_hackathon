/**
 * DRISHTI — App Shell Layout
 * Wraps all pages with the persistent Sidebar + Topbar.
 */
import { Outlet, useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import Topbar from './Topbar';

const pageTitles = {
  '/dashboard': 'Overview Dashboard',
  '/videos': 'Upload Manager',
  '/incidents': 'Detected Incidents',
  '/persons': 'Detected Persons',
  '/heatmap': 'Motion Heatmap',
  '/reports': 'Reports & Export',
  '/settings': 'Settings',
  '/audit': 'Audit Log',
};

export default function AppLayout() {
  const location = useLocation();

  // Match the longest prefix for dynamic routes like /videos/:id
  const title = Object.entries(pageTitles)
    .filter(([path]) => location.pathname.startsWith(path))
    .sort((a, b) => b[0].length - a[0].length)[0]?.[1] || 'DRISHTI';

  return (
    <div className="app-layout">
      <Sidebar />
      <main className="main-content">
        <Topbar title={title} />
        <div className="page-content fade-in">
          <Outlet />
        </div>
      </main>
    </div>
  );
}
