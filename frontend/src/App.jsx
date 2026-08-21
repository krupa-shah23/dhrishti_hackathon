import { Navigate, Route, Routes } from 'react-router-dom';
import { VideoProvider } from './context/VideoContext'; // persisted mock video state
import AppLayout from './components/layout/AppLayout';
import Overview from './pages/Overview';
import UploadVideo from './pages/UploadVideo';
import ProcessingVideo from './pages/ProcessingVideo';
import AnalysisReport from './pages/AnalysisReport';
import PreviousRecords from './pages/PreviousRecords';

export default function App() {
  return <VideoProvider><Routes>
    <Route element={<AppLayout />}>
      <Route path="/" element={<Overview />} />
      <Route path="/upload" element={<UploadVideo />} />
      <Route path="/processing/:videoId" element={<ProcessingVideo />} />
      <Route path="/analysis/:videoId" element={<AnalysisReport />} />
      <Route path="/previous-records" element={<PreviousRecords />} />
      <Route path="/processing" element={<Navigate to="/upload" replace />} />
      <Route path="/analysis" element={<Navigate to="/previous-records" replace />} />
      <Route path="*" element={<Navigate to="/" replace />} />
    </Route>
  </Routes></VideoProvider>;
}
