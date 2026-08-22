import { useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { UploadCloud } from 'lucide-react';
import { UploadSlot } from '../components/common/Ui';
import { useVideos } from '../context/VideoContext';
const supported = ['mp4','avi','mov'];
export default function UploadVideo() { const [files,setFiles] = useState([]), [error,setError] = useState(''); const input = useRef(); const navigate=useNavigate(); const {addVideos}=useVideos();
  const add = list => { const incoming=Array.from(list || []); if (!incoming.length) return; const invalid=incoming.find(f => !supported.includes(f.name.split('.').pop().toLowerCase())); if(invalid) return setError('Unsupported file type. Please upload an MP4, AVI, or MOV video.'); if(files.length+incoming.length>3) return setError('You can upload a maximum of 3 videos.'); setError(''); setFiles(prev => [...prev,...incoming]); };
  const analyze = () => { if(files.length) navigate(`/processing/${addVideos(files)}`); };
  return <div className="upload-page"><div className="upload-card"><div className="page-title"><h1>Upload Video for Analysis</h1><p>Max 3 videos allowed. Supported formats: MP4, AVI, MOV.</p></div><div className="dropzone" onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();add(e.dataTransfer.files)}}><div className="upload-icon"><UploadCloud size={32}/></div><h2>Drag &amp; Drop Videos here</h2><p>or</p><button onClick={()=>input.current.click()}>BROWSE FILE</button><input ref={input} aria-label="Choose video files" type="file" multiple accept=".mp4,.avi,.mov,video/mp4,video/avi,video/quicktime" onChange={e=>add(e.target.files)}/></div>{error && <p className="form-error">{error}</p>}<div className="upload-slots">{[0,1,2].map(i=><UploadSlot key={i} file={files[i]} index={i} onPick={add} onRemove={()=>setFiles(f=>f.filter((_,n)=>n!==i))}/>)}</div><button className="primary wide" disabled={!files.length} onClick={analyze}>ANALYZE VIDEOS</button></div></div> }
