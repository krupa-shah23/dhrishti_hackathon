import { Navigate, Route, Routes } from 'react-router-dom';
import AppLayout from './components/layout/AppLayout';
import HomePage from './pages/HomePage';
import DashboardPage from './pages/DashboardPage';
import UploadPage from './pages/UploadPage';
import AnalysisReport from './pages/AnalysisReport';
import IncidentsPage from './pages/IncidentsPage';
import PersonsPage from './pages/PersonsPage';
import PersonProfilePage from './pages/PersonProfilePage';
import VideoReviewPage from './pages/VideoReviewPage';
import SettingsPage from './pages/SettingsPage';
import PreviousRecords from './pages/PreviousRecords';

export default function App() {
  return (
    <Routes>
      {/* ── Landing Page — NO sidebar (outside AppLayout) ── */}
      <Route path="/home" element={<HomePage />} />

      {/* ── App Shell — all routes with sidebar + topbar ── */}
      <Route element={<AppLayout />}>
        {/* Dashboard — root "/" shows the dashboard inside the app shell */}
        <Route path="/" element={<DashboardPage />} />

        {/* Upload Manager */}
        <Route path="/upload" element={<UploadPage />} />

        {/* Analysis Report — video ID from backend MongoDB _id */}
        <Route path="/analysis/:videoId" element={<AnalysisReport />} />

        {/* Incidents — live events table with XAI summary */}
        <Route path="/incidents" element={<IncidentsPage />} />

        {/* Persons */}
        <Route path="/persons" element={<PersonsPage />} />
        <Route path="/persons/:id" element={<PersonProfilePage />} />

        {/* Video Review / Canvas Overlay */}
        <Route path="/videos/:id" element={<VideoReviewPage />} />

        {/* Settings */}
        <Route path="/settings" element={<SettingsPage />} />

        {/* Previous Records */}
        <Route path="/previous-records" element={<PreviousRecords />} />

        {/* Redirects */}
        <Route path="/upload/*" element={<Navigate to="/upload" replace />} />
        <Route path="/analysis" element={<Navigate to="/upload" replace />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Route>
    </Routes>
  );
}
