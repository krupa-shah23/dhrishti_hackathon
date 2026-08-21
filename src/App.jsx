/**
 * DRISHTI — Main Application with Routing
 */
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import IncidentsPage from './pages/IncidentsPage';
import VideoReviewPage from './pages/VideoReviewPage';
import PersonsPage from './pages/PersonsPage';
import PersonProfilePage from './pages/PersonProfilePage';
import HeatmapPage from './pages/HeatmapPage';
import ReportsPage from './pages/ReportsPage';
import SettingsPage from './pages/SettingsPage';
import AuditPage from './pages/AuditPage';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<AppLayout />}>
          {/* Default redirect */}
          <Route index element={<Navigate to="/dashboard" replace />} />

          {/* Core Pages */}
          <Route path="dashboard" element={<DashboardPage />} />
          <Route path="videos" element={<UploadPage />} />
          <Route path="videos/:videoId" element={<VideoReviewPage />} />
          <Route path="incidents" element={<IncidentsPage />} />
          <Route path="persons" element={<PersonsPage />} />
          <Route path="persons/:personId" element={<PersonProfilePage />} />
          <Route path="heatmap" element={<HeatmapPage />} />
          <Route path="reports" element={<ReportsPage />} />
          <Route path="settings" element={<SettingsPage />} />
          <Route path="audit" element={<AuditPage />} />

          {/* 404 */}
          <Route path="*" element={
            <div className="empty-state" style={{ minHeight: '60vh' }}>
              <div className="empty-state-title" style={{ fontSize: 48 }}>404</div>
              <div className="empty-state-text">Page not found</div>
            </div>
          } />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
