export const overviewStats = { videosProcessed: '1,284', personsCaughtCopying: 42 }; // dashboard summary
export const incidentTypes = [
  { name: 'Candidates Talking', value: 35, color: '#1e2a4a' }, { name: 'Detection (Phone/Paper)', value: 20, color: '#8a4cfc' },
  { name: 'Gestures (Hand/Touch)', value: 20, color: '#ba1a1a' }, { name: 'Constant Head Movement', value: 15, color: '#004395' }, { name: 'Exchanging Notes/Sheets', value: 10, color: '#712ae2' }
];
export const incidentsTrend = [{time:'Mon',incidents:5},{time:'Tue',incidents:9},{time:'Wed',incidents:7},{time:'Thu',incidents:15},{time:'Fri',incidents:12},{time:'Sat',incidents:18},{time:'Sun',incidents:10}];
export const records = [
  { id:'econ-101-midterm', title:'ECON 101 Midterm', date:'Oct 24, 2022', room:'Hall A', incidents:12, high:25, medium:35, low:40, color:'#7451ce' },
  { id:'math-202-final', title:'MATH 202 Final', date:'Oct 22, 2022', room:'Room 304', incidents:3, high:15, medium:30, low:55, color:'#286895' },
  { id:'cs-110-quiz', title:'CS 110 Quiz', date:'Oct 30, 2022', room:'Lab 6', incidents:8, high:20, medium:45, low:35, color:'#394b83' }
];
const baseEvents = [
  { timestamp:'11:13:50', person:'SUB-102', activity:'Copies Notes', duration:'12s', confidence:72 },
  { timestamp:'09:45:12', person:'SUB-105', activity:'Passes Note', duration:'3s', confidence:28 },
  { timestamp:'14:24:10', person:'SUB-093', activity:'Face Deviation', duration:'38s', confidence:87 },
  { timestamp:'15:07:00', person:'SUB-222', activity:'No Suspicion', duration:'4s', confidence:15 }
];
export const analysisById = Object.fromEntries(records.map((record, index) => [record.id, {
  ...record, events: baseEvents.map((e, i) => ({...e, confidence: Math.max(12, e.confidence - index * 5 + i)})),
  timeline: [{time:'08:00',confidence:20},{time:'10:00',confidence:36},{time:'12:00',confidence:57},{time:'14:00',confidence:91-index*7},{time:'16:00',confidence:52},{time:'18:00',confidence:29}],
  explanation:'The system detected several suspicious interactions between candidates during the selected time period. Movement and interaction patterns indicate potential coordinated behavior that merits review.'
}]));
