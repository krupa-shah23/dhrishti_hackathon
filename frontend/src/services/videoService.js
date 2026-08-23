// Frontend-only service seam for a future video API.
export const createVideoProcessingRecord = (video) => ({ ...video, status: 'processing', progress: 0 });
