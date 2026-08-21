import { createContext, useContext, useEffect, useState } from 'react';
import { analysisById } from '../data/mockData'; // mock report resolver

const VideoContext = createContext();
const key = 'wireframe-videos';
const read = () => { try { return JSON.parse(localStorage.getItem(key)) || {}; } catch { return {}; } };
export function VideoProvider({ children }) {
  const [videos, setVideos] = useState(read);
  useEffect(() => localStorage.setItem(key, JSON.stringify(videos)), [videos]);
  const addVideos = (files) => {
    const id = `upload-${Date.now()}`;
    setVideos(prev => ({...prev, [id]: { id, filename: files[0].name, size: `${(files[0].size / 1024 / 1024).toFixed(1)} MB`, duration:'--:--:--', status:'processing', progress:0, completed:false }}));
    return id;
  };
  const updateVideo = (id, patch) => setVideos(prev => prev[id] ? ({...prev, [id]: {...prev[id], ...patch}}) : prev);
  const getAnalysis = (id) => videos[id] ? ({...analysisById['econ-101-midterm'], id, title: videos[id].filename, events: analysisById['econ-101-midterm'].events}) : analysisById[id];
  return <VideoContext.Provider value={{videos, addVideos, updateVideo, getAnalysis}}>{children}</VideoContext.Provider>;
}
export const useVideos = () => useContext(VideoContext);
