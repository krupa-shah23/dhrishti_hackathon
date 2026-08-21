import { analysisById } from '../data/mockData';

// Frontend-only service seam for a future analysis API.
export const getMockAnalysis = (videoId) => analysisById[videoId];
