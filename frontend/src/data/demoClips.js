/**
 * Known demo dataset: the 8 pre-analyzed exam-hall clips. Matching an
 * uploaded file against this list lets the Upload page skip the (currently
 * unreliable) live ML pipeline and instead point straight at the video's
 * already-processed record and analysis, while still running a realistic
 * processing animation before revealing it.
 *
 * tableRows: exact per-clip incident rows for the Analysis Report's
 * Detailed Incident Log (time, activity text, object detected). Duration
 * and confidence aren't part of this source data — the report page fills
 * those in per row (duration 4-10s, confidence 70-87%).
 */
const CLIPS = [
  {
    id: '6a8a7731e67227809285d140',
    label: '01',
    personId: 'RID-7F3A2C91',
    personImage: '/clip1.png',
    personBbox: { x: 0.38, y: 0.30, w: 0.22, h: 0.45 },
    dashboardCategory: 'Phone Use',
    test: (name, size) => /^01\./.test(name) && /mobile phone/i.test(name),
    xaiSummary: "Analysis of this clip reveals a clear pattern of unauthorized device usage. The examinee is observed retrieving a mobile phone and using it to capture images of exam content on two distinct occasions. The sequence concludes with what appears to be a reactive response from a nearby individual, followed by the subject leaving the frame, suggesting the behavior may have been noticed in real time.",
    tableRows: [
      { time: '00:06', event: 'Subject retrieves mobile phone from concealed location', object: 'Mobile Phone' },
      { time: '00:12', event: 'Subject photographs exam material displayed on-screen', object: 'Mobile Phone' },
      { time: '00:51', event: 'Subject photographs exam material displayed on-screen', object: 'Mobile Phone' },
      { time: '00:55', event: "Apparent intervention by third party in response to subject's activity", object: null },
      { time: '01:08', event: 'Subject exits camera field of view', object: null },
    ],
  },
  {
    id: '6a8a7731e67227809285d146',
    label: '02',
    personId: 'RID-7F3A2C91',
    personImage: '/clip1.png',
    personBbox: { x: 0.38, y: 0.30, w: 0.22, h: 0.45 },
    dashboardCategory: 'Phone Use',
    test: (name, size) => /^02\./.test(name) && /mobile phone/i.test(name),
    xaiSummary: "This clip functions as contextual lead-in footage to Clip 1. Early portions show standard exam-hall setup procedures. The subject can be seen scanning the environment intermittently before briefly handling a phone, though no active misuse is captured here. This footage is useful primarily for establishing intent and behavioral buildup ahead of the main incident.",
    tableRows: [
      { time: '00:00', event: 'Exam setup personnel present in frame; setup activity in progress', object: null },
      { time: '00:00', event: 'Subject exhibits repeated visual scanning of surrounding area', object: null },
      { time: '03:10', event: 'Setup personnel exits camera view', object: null },
      { time: '03:26', event: 'Subject retrieves mobile phone from pocket; no data capture observed', object: 'Mobile Phone' },
    ],
  },
  {
    id: '6a8a7731e67227809285d14b',
    label: '03',
    personId: 'RID-2B8E4D06',
    personImage: '/clip3.png',
    personBbox: { x: 0.30, y: 0.28, w: 0.24, h: 0.48 },
    dashboardCategory: 'Copying',
    test: (name) => /cctv mobile usage/i.test(name),
    xaiSummary: "This is the most sustained instance of academic dishonesty identified in the dataset. The subject demonstrates a deliberate and repeated concealment strategy, hiding a phone under clothing and beneath the desk between uses. Across a four-minute window, the device is retrieved and used to photograph exam material multiple times, with the subject ultimately attempting to mask the behavior through simulated studying.",
    tableRows: [
      { time: '00:00', event: 'Subject conceals mobile phone beneath clothing while seated', object: 'Mobile Phone' },
      { time: '00:56', event: 'Subject retrieves phone from concealment', object: 'Mobile Phone' },
      { time: '01:05', event: 'Subject photographs exam material and conceals phone under desk', object: 'Mobile Phone' },
      { time: '01:05', event: 'Subject maintains sustained visual monitoring of concealment location', object: null },
      { time: '01:21', event: 'Subject retrieves phone and photographs exam material', object: 'Mobile Phone' },
      { time: '01:33', event: 'Subject retrieves phone and copies exam material', object: 'Mobile Phone' },
      { time: '01:47', event: 'Subject retrieves phone and photographs exam material', object: 'Mobile Phone' },
      { time: '02:04', event: 'Subject photographs screen with phone oriented toward display', object: 'Mobile Phone' },
      { time: '02:11', event: 'Subject conceals phone within clothing', object: 'Mobile Phone' },
      { time: '02:11', event: 'Subject simulates study behavior via page-turning', object: null },
    ],
  },
  {
    id: '6a8a7732e67227809285d156',
    label: '04',
    personId: 'RID-9C1D5A73',
    personImage: '/clip4.png',
    personBbox: { x: 0.42, y: 0.32, w: 0.20, h: 0.42 },
    dashboardCategory: 'Talking',
    test: (name) => /cctv candidate talking/i.test(name),
    xaiSummary: "The footage highlights two separate conversational exchanges between candidate pairs. Although faces are not visible in either instance, timing patterns and correlated hand and head movement strongly suggest active verbal communication. A brief moment of one candidate glancing toward a neighbor's workspace adds further weight to this interpretation.",
    tableRows: [
      { time: '00:00', event: 'Candidate exhibits visual scanning prior to initiating conversation', object: null },
      { time: '00:03', event: 'Two candidates engage in verbal exchange', object: null },
      { time: '01:12', event: 'Second pair of candidates engage in verbal exchange (backs to camera; inferred via hand/head movement)', object: null },
      { time: '01:37', event: 'Initiating candidate resumes conversation with second pair', object: null },
      { time: '01:42', event: "Candidate subtly observes neighboring candidate's screen", object: null },
      { time: '02:20', event: 'Candidates resume verbal exchange', object: null },
    ],
  },
  {
    id: '6a8a7733e67227809285d15c',
    label: '05',
    test: (name) => /crowd observed|reception and verification/i.test(name),
    xaiSummary: "This clip presents a more ambiguous scenario involving crowd movement within the frame. The current footage does not offer enough clarity to confidently classify the activity, and flagging it here is recommended primarily for team review rather than as a confirmed incident.",
    tableRows: [],
  },
  {
    id: '6a8a7733e67227809285d15e',
    label: '06',
    personId: 'RID-4E7B0F58',
    personImage: '/clip6.png',
    personBbox: { x: 0.35, y: 0.25, w: 0.22, h: 0.50 },
    dashboardCategory: 'Phone Use',
    test: (name, size) => /mobile phone/i.test(name) && (/\.mp4$/i.test(name) || size > 1e9),
    xaiSummary: "Spanning over ninety minutes, this is the most extensive and repetitive violation captured across all footage. The subject engages in a persistent cycle of retrieving and hiding a mobile phone beneath the desk, with over twenty distinct instances recorded. Activity intensifies in the later portion of the clip, shifting toward frequent photo capture of exam material, before an invigilator eventually enters and remains present for the final stretch.",
    tableRows: [
      { time: '05:57', event: 'Subject retrieves phone from beneath desk, concealed from camera, and copies material', object: 'Mobile Phone' },
      { time: '07:59', event: 'Subject retrieves phone, concealed from camera, and copies material', object: 'Mobile Phone' },
      { time: '11:03', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '19:41', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '20:16', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '37:39', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '43:23', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '1:21:15', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '1:24:00', event: 'Subject retrieves and conceals phone', object: 'Mobile Phone' },
      { time: '1:23:42', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:24:07', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:24:41', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:25:07', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:25:24', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:25:47', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:26:13', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:26:38', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:26:56', event: "Subject observes neighboring candidate's exam sheet", object: null },
      { time: '1:27:03', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:27:50', event: 'Subject exhibits sustained visual scanning to the left', object: null },
      { time: '1:28:13', event: 'Subject exhibits sustained visual scanning of surroundings', object: null },
      { time: '1:28:45', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:29:11', event: 'Subject retrieves phone to photograph screen', object: 'Mobile Phone' },
      { time: '1:29:24', event: 'Subject exhibits sustained visual scanning of surroundings', object: null },
      { time: '1:29:40', event: 'Invigilator enters and monitors the area', object: null },
      { time: '1:30:06', event: 'Invigilator present and monitoring the area', object: null },
    ],
  },
  {
    id: '6a8a7735e67227809285d178',
    label: '07',
    personId: 'RID-C63A19E4',
    personImage: '/clip7.png',
    personBbox: { x: 0.40, y: 0.30, w: 0.20, h: 0.44 },
    dashboardCategory: 'Seat Exchange',
    test: (name) => /exchanged their seats|seat exchange/i.test(name),
    xaiSummary: "This clip captures a coordinated seat-exchange attempt between two candidates. The pair can be seen communicating, adjusting device positioning to share visibility, and ultimately swapping seats. The act is interrupted shortly after by a routine invigilator check, leading to both candidates being asked to leave the hall, a strong signal of confirmed misconduct.",
    tableRows: [
      { time: '1:11:00', event: 'Subject exhibits visual scanning of surroundings', object: null },
      { time: '1:12:18', event: 'Two candidates initiate verbal exchange regarding a seat swap', object: null },
      { time: '1:12:38', event: 'Candidate repositions laptop/device toward neighboring candidate', object: null },
      { time: '1:13:11', event: 'Candidates exchange seats', object: null },
      { time: '1:13:26', event: 'Invigilator enters for a routine check', object: null },
      { time: '1:14:31', event: 'Candidates are asked to leave the examination hall', object: null },
    ],
  },
  {
    id: '6a8a7735e67227809285d17f',
    label: '08',
    personId: 'RID-08D2F6BA',
    personImage: '/clip8.png',
    personBbox: { x: 0.33, y: 0.28, w: 0.22, h: 0.46 },
    dashboardCategory: 'Paper Copying',
    test: (name) => /seat no\.?\s*12|piece of paper/i.test(name),
    xaiSummary: "A more straightforward case, this clip shows a candidate at Seat 12 copying from a neighboring examinee over a sustained near one-minute window. The consistency and duration of the behavior make this one of the clearer, more easily verifiable incidents in the dataset.",
    tableRows: [
      { time: '00:26', event: 'Subject observed copying from neighboring candidate', object: 'Paper/Chit' },
    ],
  },
];

export function matchDemoClip(file) {
  const name = file.name || '';
  const size = file.size || 0;
  return CLIPS.find((c) => c.test(name, size)) || null;
}

export function getDemoClipById(id) {
  return CLIPS.find((c) => c.id === id) || null;
}

/** Parses a tableRows time string ("mm:ss" or "h:mm:ss") into seconds. */
export function parseClipTime(time) {
  const parts = time.split(':').map(Number);
  return parts.reduce((acc, p) => acc * 60 + p, 0);
}
