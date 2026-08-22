/**
 * DRISHTI — Home / Landing Page
 *
 * A full-page marketing/entry screen that sits at "/" with no sidebar.
 * Clicking any feature card or CTA navigates into the actual app.
 * All route links exactly match the router paths defined in App.jsx.
 */
import { useNavigate } from 'react-router-dom';
import {
  ShieldCheck,
  Upload,
  LayoutDashboard,
  AlertTriangle,
  Users,
  Video,
  Settings,
  FileBarChart2,
  ArrowRight,
  Eye,
  Cpu,
  Lock,
  Zap,
  CheckCircle2,
} from 'lucide-react';

/* ─── Feature cards wired to real backend-connected routes ─── */
const features = [
  {
    icon: Upload,
    title: 'Upload & Process',
    description:
      'Drop exam recordings and start AI-powered analysis instantly. Supports 3-hour+ videos processed frame-by-frame.',
    route: '/upload',
    cta: 'Go to Upload',
    color: 'from-teal-500 to-teal-700',
    bg: 'bg-teal-50',
    iconColor: 'text-teal-600',
  },
  {
    icon: AlertTriangle,
    title: 'Live Incidents',
    description:
      'Browse every flagged incident with confidence scores, timestamps, and per-event AI explanations.',
    route: '/incidents',
    cta: 'View Incidents',
    color: 'from-amber-400 to-orange-500',
    bg: 'bg-amber-50',
    iconColor: 'text-amber-600',
  },
  {
    icon: Users,
    title: 'Person Profiles',
    description:
      'Track individuals across sessions with re-identification embeddings and full activity timelines.',
    route: '/persons',
    cta: 'See Persons',
    color: 'from-violet-500 to-purple-700',
    bg: 'bg-violet-50',
    iconColor: 'text-violet-600',
  },
  {
    icon: LayoutDashboard,
    title: 'Dashboard',
    description:
      'Real-time overview of videos processed, incidents detected, and activity distribution charts.',
    route: '/',
    cta: 'Open Dashboard',
    color: 'from-teal-400 to-cyan-600',
    bg: 'bg-cyan-50',
    iconColor: 'text-cyan-600',
  },
  {
    icon: FileBarChart2,
    title: 'Analysis Reports',
    description:
      'Detailed per-video reports with Gemini-powered XAI summaries, bounding-box overlays, and export to PDF.',
    route: '/upload',
    cta: 'Generate Report',
    color: 'from-teal-600 to-emerald-700',
    bg: 'bg-emerald-50',
    iconColor: 'text-emerald-600',
  },
  {
    icon: Settings,
    title: 'Settings',
    description:
      'Configure seat grid, motion thresholds, Gemini XAI toggle, and system preferences.',
    route: '/settings',
    cta: 'Configure',
    color: 'from-slate-500 to-slate-700',
    bg: 'bg-slate-50',
    iconColor: 'text-slate-500',
  },
];

/* ─── Pipeline steps shown in the "How it works" section ─── */
const steps = [
  { icon: Upload,        label: 'Upload Video',         sub: 'Register any exam recording instantly' },
  { icon: Cpu,           label: 'ML Pipeline Runs',     sub: 'YOLOv8 + ByteTrack, frame-by-frame' },
  { icon: AlertTriangle, label: 'Incidents Scored',     sub: 'Confidence + severity, per event' },
  { icon: Eye,           label: 'Review & Export',      sub: 'Overlay playback + PDF report' },
];

/* ─── Trust badges ─── */
const badges = [
  { icon: Lock,         label: 'Offline-first' },
  { icon: Zap,          label: 'Real-time updates' },
  { icon: CheckCircle2, label: 'Explainable AI' },
  { icon: ShieldCheck,  label: 'Zero cloud dependency' },
];

export default function HomePage() {
  const navigate = useNavigate();

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-50 via-white to-teal-50/30 font-sans">

      {/* ── Ambient radial glow behind hero ── */}
      <div
        className="pointer-events-none fixed inset-0 z-0"
        style={{
          background:
            'radial-gradient(ellipse 70% 40% at 50% 0%, rgba(20,184,166,0.12), transparent)',
        }}
      />

      {/* ═══════════════════════════════════════
          TOPBAR
      ═══════════════════════════════════════ */}
      <header className="relative z-10 flex items-center justify-between px-8 py-5 bg-white/70 backdrop-blur-md border-b border-teal-100/60 shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-teal-500 to-teal-700 flex items-center justify-center shadow-teal-glow">
            <Eye size={20} color="white" />
          </div>
          <div>
          <span className="text-lg font-bold tracking-tight text-slate-900" style={{ fontFamily: 'Orbitron, sans-serif' }}>DRISHTI</span>
          <span className="block text-[10px] font-bold tracking-widest text-teal-600 uppercase" style={{ fontFamily: 'DM Sans, sans-serif', letterSpacing: '0.2em' }}>
            Surveillance Suite
          </span>
        </div>
        </div>
        <button
          onClick={() => navigate('/upload')}
          className="flex items-center gap-2 px-5 py-2.5 rounded-full text-sm font-semibold text-white
                     bg-gradient-to-r from-teal-500 to-teal-700
                     hover:shadow-[0_4px_20px_rgba(20,184,166,0.4)] transition-all duration-200 hover:-translate-y-0.5"
        >
          Get Started <ArrowRight size={16} />
        </button>
      </header>

      {/* ═══════════════════════════════════════
          HERO
      ═══════════════════════════════════════ */}
      <section className="relative z-10 text-center px-6 pt-20 pb-16">
        {/* pill badge */}
        <div className="inline-flex items-center gap-2 px-4 py-1.5 rounded-full bg-teal-50 border border-teal-200 text-teal-700 text-xs font-semibold mb-6 shadow-sm">
          <span className="w-2 h-2 rounded-full bg-teal-500 animate-pulse" />
          AI-Powered Exam Proctoring System
        </div>

        <h1 className="text-4xl md:text-5xl font-bold text-slate-900 tracking-tight leading-tight mb-5" style={{ fontFamily: 'Orbitron, sans-serif', letterSpacing: '-0.02em' }}>
          Detect Malpractice.{' '}
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-teal-500 to-teal-700">
            Automatically.
          </span>
        </h1>

        <p className="max-w-2xl mx-auto text-lg text-slate-500 leading-relaxed mb-10" style={{ fontFamily: 'DM Sans, sans-serif' }}>
          DRISHTI processes long exam recordings offline — detecting phones, chits, and suspicious
          gestures using YOLOv8 + ByteTrack, then delivers confidence-scored, explainable incident
          reports to invigilators.
        </p>

        <div className="flex items-center justify-center gap-4 flex-wrap">
          <button
            onClick={() => navigate('/upload')}
            className="flex items-center gap-2 px-8 py-3.5 rounded-full text-sm font-bold text-white
                       bg-gradient-to-r from-teal-500 to-teal-700
                       hover:shadow-[0_8px_30px_rgba(20,184,166,0.35)] transition-all duration-200 hover:-translate-y-0.5"
          >
            <Upload size={17} /> Upload a Video
          </button>
          <button
            onClick={() => navigate('/')}
            className="flex items-center gap-2 px-8 py-3.5 rounded-full text-sm font-bold text-slate-700
                       bg-white border border-slate-200
                       hover:border-teal-300 hover:text-teal-700 hover:shadow-md transition-all duration-200"
          >
            <LayoutDashboard size={17} /> View Dashboard
          </button>
        </div>

        {/* Trust badges */}
        <div className="flex items-center justify-center gap-6 mt-10 flex-wrap">
          {badges.map(({ icon: Icon, label }) => (
            <div key={label} className="flex items-center gap-1.5 text-xs text-slate-500 font-medium">
              <Icon size={14} className="text-teal-500" />
              {label}
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════
          HOW IT WORKS
      ═══════════════════════════════════════ */}
      <section className="relative z-10 px-6 py-14 bg-white/60 backdrop-blur-sm border-y border-teal-50">
        <h2 className="text-center text-2xl font-bold text-slate-800 mb-10" style={{ fontFamily: 'Orbitron, sans-serif' }}>How DRISHTI Works</h2>
        <div className="max-w-4xl mx-auto grid grid-cols-2 md:grid-cols-4 gap-6">
          {steps.map(({ icon: Icon, label, sub }, i) => (
            <div key={label} className="flex flex-col items-center text-center gap-3">
              <div className="relative">
                <div className="w-14 h-14 rounded-2xl bg-gradient-to-br from-teal-50 to-teal-100 border border-teal-200 flex items-center justify-center shadow-sm">
                  <Icon size={24} className="text-teal-600" />
                </div>
                <span className="absolute -top-2 -right-2 w-5 h-5 rounded-full bg-teal-500 text-white text-[10px] font-bold flex items-center justify-center">
                  {i + 1}
                </span>
              </div>
              <div>
                <p className="font-semibold text-slate-800 text-sm">{label}</p>
                <p className="text-xs text-slate-500 mt-0.5">{sub}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════
          FEATURE CARDS GRID
      ═══════════════════════════════════════ */}
      <section className="relative z-10 px-6 py-16 max-w-6xl mx-auto">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 mb-3" style={{ fontFamily: 'Orbitron, sans-serif' }}>Everything you need</h2>
            <p className="text-slate-500" style={{ fontFamily: 'DM Sans, sans-serif' }}>Click any module to jump directly into the app.</p>
          </div>

        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-6">
          {features.map(({ icon: Icon, title, description, route, cta, bg, iconColor }) => (
            <button
              key={title}
              onClick={() => navigate(route)}
              className="group text-left bg-white rounded-2xl border border-slate-100 p-6
                         shadow-sm hover:shadow-[0_8px_30px_rgba(20,184,166,0.15)]
                         hover:border-teal-200 transition-all duration-250 hover:-translate-y-1"
            >
              <div className={`w-12 h-12 rounded-xl ${bg} flex items-center justify-center mb-4`}>
                <Icon size={22} className={iconColor} />
              </div>
              <h3 className="font-bold text-slate-900 mb-2 text-[15px]">{title}</h3>
              <p className="text-sm text-slate-500 leading-relaxed mb-5">{description}</p>
              <div className={`inline-flex items-center gap-1.5 text-xs font-semibold ${iconColor} group-hover:gap-2.5 transition-all`}>
                {cta} <ArrowRight size={13} />
              </div>
            </button>
          ))}
        </div>
      </section>

      {/* ═══════════════════════════════════════
          BOTTOM CTA BANNER
      ═══════════════════════════════════════ */}
      <section className="relative z-10 mx-6 mb-12 rounded-3xl overflow-hidden">
        <div className="bg-gradient-to-r from-teal-600 to-teal-800 px-10 py-12 text-center">
          {/* subtle inner glow */}
          <div className="absolute inset-0 bg-[radial-gradient(ellipse_60%_80%_at_50%_120%,rgba(255,255,255,0.08),transparent)] pointer-events-none" />
          <h2 className="text-3xl font-bold text-white mb-3" style={{ fontFamily: 'Orbitron, sans-serif' }}>Ready to analyse an exam?</h2>
          <p className="text-teal-100 mb-8 text-base max-w-xl mx-auto" style={{ fontFamily: 'DM Sans, sans-serif' }}>
            Upload a video and the pipeline starts immediately — no cloud, no waiting, fully offline.
          </p>
          <button
            onClick={() => navigate('/upload')}
            className="inline-flex items-center gap-2 px-8 py-3.5 rounded-full bg-white text-teal-700 font-bold text-sm
                       hover:shadow-[0_8px_25px_rgba(255,255,255,0.3)] transition-all duration-200 hover:-translate-y-0.5"
          >
            <Upload size={17} /> Start Uploading
          </button>
        </div>
      </section>

      {/* ═══════════════════════════════════════
          FOOTER
      ═══════════════════════════════════════ */}
      <footer className="relative z-10 text-center py-6 text-xs text-slate-400 border-t border-slate-100">
        DRISHTI &mdash; AI-Powered Exam Surveillance &nbsp;|&nbsp; Built with YOLOv8 · ByteTrack · Gemini XAI
      </footer>
    </div>
  );
}
