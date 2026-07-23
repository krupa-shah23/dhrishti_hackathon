# Feature Schema — extract_features(track) -> dict

| Feature      | Definition                              | Unit    |
|--------------|------------------------------------------|---------|
| motion_area  | avg foreground pixel count in ROI         | px^2    |
| duration     | event length (end - start)                | sec     |
| frequency    | number of motion bursts within event      | count   |
| speed        | avg centroid displacement per frame        | px/frame|
| roi_size     | bounding box area                          | px^2    |
| direction    | dominant motion angle                      | degrees |
| density      | motion_area / roi_size                     | ratio   |
| persistence  | fraction of frames with motion in event    | ratio   |
| audio_energy | RMS energy in event time window            | float   |
| object_flag  | phone/paper detected (from P2 detector)    | 0 or 1  |