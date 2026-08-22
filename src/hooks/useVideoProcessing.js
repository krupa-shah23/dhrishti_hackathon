import { useEffect } from 'react';

export function useVideoProcessing({ video, onProgress, onComplete }) {
  useEffect(() => {
    if (!video || video.completed) return undefined;
    const timer = setInterval(() => onProgress(Math.min(100, (video.progress || 0) + 5)), 600);
    return () => clearInterval(timer);
  }, [video, onProgress]);

  useEffect(() => {
    if (video?.progress >= 100 && !video.completed) onComplete();
  }, [video, onComplete]);
}
