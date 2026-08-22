"""
report.py -- P4/P2b: Outputs, Evaluation & Integration
Generates a professional PDF summary report using fpdf2.

Two report types:
  1. Per-video report (existing): metadata, event stats, heatmap/timeline,
     top high-risk events.
  2. P2b Day-5 deck report (new): detector metrics (from P2a), benchmark
     table, ablation table, event-level P/R/F1, Known Limitations.
"""

import os
import json
import csv
from datetime import datetime
from fpdf import FPDF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_events_csv(csv_path: str) -> list[dict]:
    """Reads all rows from the events CSV into a list of dicts."""
    rows = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return rows


def _load_table_csv(csv_path: str) -> tuple[list[str], list[dict]]:
    """Generic CSV loader for benchmark/ablation tables. Returns (columns, rows)."""
    rows = []
    cols: list[str] = []
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            cols = reader.fieldnames or []
            for row in reader:
                rows.append(row)
    except FileNotFoundError:
        pass
    return cols, rows


def _compute_summary_stats(events: list[dict], risk_threshold: float = 0.65) -> dict:
    """Derives summary statistics from the events list."""
    if not events:
        return {
            "total_events": 0,
            "high_risk_events": 0,
            "avg_motion_score": 0.0,
            "avg_risk_score": 0.0,
            "peak_risk_score": 0.0,
            "peak_risk_timestamp": "N/A",
        }

    motion_scores = [float(e.get("motion_score", 0)) for e in events]
    risk_scores   = [float(e.get("risk_score",   0)) for e in events]
    high_risk     = [e for e in events if float(e.get("risk_score", 0)) >= risk_threshold]

    peak_idx  = risk_scores.index(max(risk_scores))
    peak_ts   = events[peak_idx].get("timestamp", "N/A")

    return {
        "total_events":        len(events),
        "high_risk_events":    len(high_risk),
        "avg_motion_score":    round(sum(motion_scores) / len(motion_scores), 2),
        "avg_risk_score":      round(sum(risk_scores)   / len(risk_scores),   4),
        "peak_risk_score":     round(max(risk_scores), 4),
        "peak_risk_timestamp": peak_ts,
    }


# ---------------------------------------------------------------------------
# Default "Known, Stated Limitations" — edit inline as findings change
# ---------------------------------------------------------------------------

DEFAULT_KNOWN_LIMITATIONS = [
    "Phone recall (~0.5 on held-out test) is a disclosed, real limitation — "
    "confirmed three separate ways, not measurement noise, unless retrain moves it.",
    "Wrist/ear device detection: architecturally trivial, not built/validated — "
    "no real footage to test against.",
    "Small-object tiling: tested honestly, no improvement found — disclosed, not chased further.",
    "Clip 5 (crowd/reception-desk footage) excluded from ground truth/F1 — "
    "ambiguity never confirmed by anyone with knowledge of the footage.",
    "Smart-watch class (if added): validated against public/staged data only, "
    "not real invigilated footage.",
]


# ---------------------------------------------------------------------------
# PDF Report class
# ---------------------------------------------------------------------------

class ExamHallReport(FPDF):
    """Custom FPDF subclass with header and footer."""

    def __init__(self, title: str = "Offline Exam-Hall Video Analytics", *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.report_title = title

    def header(self):
        self.set_fill_color(20, 20, 50)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(230, 230, 255)
        self.set_xy(10, 4)
        self.cell(0, 10, f"{self.report_title} | P4/P2b Summary Report", align="L")
        self.ln(14)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(140, 140, 160)
        self.cell(0, 10, f"Page {self.page_no()} | Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}", align="C")

    def section_title(self, title: str):
        self.set_font("Helvetica", "B", 12)
        self.set_fill_color(30, 30, 70)
        self.set_text_color(200, 210, 255)
        self.cell(0, 8, f"  {title}", ln=True, fill=True)
        self.ln(3)

    def kv_row(self, key: str, value: str, shade: bool = False):
        self.set_font("Helvetica", "", 10)
        self.set_fill_color(245, 245, 252) if shade else self.set_fill_color(255, 255, 255)
        self.set_text_color(60, 60, 80)
        self.cell(75, 7, key, border=0, fill=True)
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 60)
        self.cell(0, 7, value, border=0, fill=True, ln=True)

    def table_header(self, cols: list[tuple[str, int]]):
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(40, 40, 80)
        self.set_text_color(220, 220, 255)
        for label, width in cols:
            self.cell(width, 7, label, border=0, fill=True, align="C")
        self.ln()

    def table_row(self, values: list[str], widths: list[int], shade: bool = False):
        self.set_font("Helvetica", "", 8)
        self.set_fill_color(240, 240, 250) if shade else self.set_fill_color(255, 255, 255)
        self.set_text_color(40, 40, 60)
        for val, width in zip(values, widths):
            self.cell(width, 6, val, border=0, fill=True, align="C")
        self.ln()

    def bullet_list(self, items: list[str]):
        self.set_font("Helvetica", "", 9)
        self.set_text_color(50, 50, 70)
        for item in items:
            self.set_x(14)
            self.multi_cell(180, 5.5, f"-  {item}")
            self.ln(1)


# ---------------------------------------------------------------------------
# 1. Per-video PDF report (existing)
# ---------------------------------------------------------------------------

def generate_report(
    video_path: str,
    csv_path: str,
    heatmap_path: str,
    timeline_path: str,
    output_path: str,
    extra_metrics: dict | None = None,
    processing_time_s: float = 0.0,
    risk_threshold: float = 0.65,
) -> None:
    """Compiles and saves a per-video PDF summary report."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    events   = _load_events_csv(csv_path)
    stats    = _compute_summary_stats(events, risk_threshold)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = ExamHallReport(title="Offline Exam-Hall Video Analytics", orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 60)
    pdf.ln(4)
    pdf.cell(0, 12, "Examination Hall Activity Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 7, f"Video ID: {video_id}", align="C", ln=True)
    pdf.cell(0, 7, f"Generated: {generated_at}", align="C", ln=True)
    pdf.ln(6)

    pdf.section_title("1. Video Metadata")
    meta_rows = [
        ("Source File",      os.path.basename(video_path)),
        ("Report Generated", generated_at),
        ("Processing Time",  f"{processing_time_s:.1f} seconds"),
        ("Risk Threshold",   str(risk_threshold)),
        ("Events CSV",       os.path.basename(csv_path)),
    ]
    for i, (k, v) in enumerate(meta_rows):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(6)

    pdf.section_title("2. Event Summary Statistics")
    stat_rows = [
        ("Total Logged Events", str(stats["total_events"])),
        ("High-Risk Events",    str(stats["high_risk_events"])),
        ("Avg. Motion Score",   f"{stats['avg_motion_score']:.2f} / 100"),
        ("Avg. Risk Score",     f"{stats['avg_risk_score']:.4f}"),
        ("Peak Risk Score",     f"{stats['peak_risk_score']:.4f}"),
        ("Peak Risk Timestamp", stats["peak_risk_timestamp"]),
    ]
    for i, (k, v) in enumerate(stat_rows):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(6)

    if extra_metrics:
        pdf.section_title("3. Quantitative Evaluation Metrics")
        for i, (k, v) in enumerate(extra_metrics.items()):
            pdf.kv_row(k.replace("_", " ").title(), str(v), shade=(i % 2 == 0))
        pdf.ln(6)

    if os.path.exists(timeline_path):
        pdf.section_title("4. Activity Timeline")
        pdf.ln(2)
        pdf.image(timeline_path, x=10, w=190)
        pdf.ln(4)

    if os.path.exists(heatmap_path):
        pdf.section_title("5. Motion Heatmap")
        pdf.ln(2)
        pdf.image(heatmap_path, x=10, w=190)
        pdf.ln(4)

    high_risk_events = sorted(
        [e for e in events if float(e.get("risk_score", 0)) >= risk_threshold],
        key=lambda e: float(e.get("risk_score", 0)),
        reverse=True,
    )[:20]

    if high_risk_events:
        pdf.add_page()
        pdf.section_title("6. Top High-Risk Events")
        pdf.ln(2)
        cols = [
            ("Timestamp", 32), ("Frame", 18), ("ROI Box", 48),
            ("Motion %", 24), ("Risk Score", 26), ("Confidence", 28),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols)
        for i, e in enumerate(high_risk_events):
            pdf.table_row(
                values=[
                    e.get("timestamp", ""), e.get("frame_index", ""),
                    e.get("roi_box", ""), e.get("motion_score", ""),
                    e.get("risk_score", ""), e.get("confidence", ""),
                ],
                widths=widths, shade=(i % 2 == 0),
            )
        pdf.ln(4)

    pdf.output(output_path)
    print(f"[report] PDF saved -> {output_path}")


def generate_json_report(
    video_path: str,
    csv_path: str,
    output_path: str,
    extra_metrics: dict | None = None,
    processing_time_s: float = 0.0,
    risk_threshold: float = 0.65,
) -> dict:
    """Generates a lightweight JSON summary report (alternative to PDF)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    events   = _load_events_csv(csv_path)
    stats    = _compute_summary_stats(events, risk_threshold)

    report = {
        "video_id":          video_id,
        "generated_at":      datetime.now().isoformat(),
        "processing_time_s": processing_time_s,
        "risk_threshold":    risk_threshold,
        "statistics":        stats,
        "metrics":           extra_metrics or {},
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[report] JSON saved -> {output_path}")
    return report


# ---------------------------------------------------------------------------
# 2. P2b Day-5 deck report (NEW) — detector + benchmark + ablation + limitations
# ---------------------------------------------------------------------------

def compile_slide_report(
    output_path: str,
    p2a_metrics: dict | None = None,
    benchmark_csv: str | None = None,
    ablation_csv: str | None = None,
    gt_stats: dict | None = None,
    known_limitations: list[str] | None = None,
) -> None:
    """
    Assembles the Day-5 slide-deck PDF: detector metrics (from P2a),
    benchmark table, ablation table, event-level F1, Known Limitations.

    Args:
        output_path:       Where to save the PDF.
        p2a_metrics:        dict e.g. {"map_val":0.648,"map_test":0.863,
                             "phone_recall":0.50,"paper_chit_recall":0.9, ...}
                             — paste in from P2a's handoff, or load from their JSON.
        benchmark_csv:      Path to outputs/benchmark_table.csv (from benchmark.py).
        ablation_csv:       Path to outputs/ablation_table.csv (from ablation.py).
        gt_stats:           dict e.g. {"total_events":45,"clips_covered":7,
                             "clip5_excluded":True}.
        known_limitations:  Override list; defaults to DEFAULT_KNOWN_LIMITATIONS.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    limitations = known_limitations or DEFAULT_KNOWN_LIMITATIONS
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = ExamHallReport(title="DRISHTI System Report", orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 60)
    pdf.ln(4)
    pdf.cell(0, 12, "DRISHTI — Day 5 System Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 7, f"Generated: {generated_at}", align="C", ln=True)
    pdf.ln(6)

    # 1. Detector metrics (P2a)
    if p2a_metrics:
        pdf.section_title("1. Detector Performance (P2a)")
        for i, (k, v) in enumerate(p2a_metrics.items()):
            pdf.kv_row(k.replace("_", " ").title(), str(v), shade=(i % 2 == 0))
        pdf.ln(6)

    # 2. Ground-truth log stats
    if gt_stats:
        pdf.section_title("2. Ground-Truth Event Log")
        for i, (k, v) in enumerate(gt_stats.items()):
            pdf.kv_row(k.replace("_", " ").title(), str(v), shade=(i % 2 == 0))
        pdf.ln(6)

    # 3. Benchmark table
    if benchmark_csv:
        cols, rows = _load_table_csv(benchmark_csv)
        if rows:
            pdf.section_title("3. Benchmark Table (frame-diff -> full pipeline)")
            n = len(cols)
            col_w = max(180 // n, 20)
            widths = [col_w] * n
            pdf.table_header([(c, col_w) for c in cols])
            for i, row in enumerate(rows):
                pdf.table_row([str(row.get(c, "")) for c in cols], widths, shade=(i % 2 == 0))
            pdf.ln(6)

    # 4. Ablation table
    if ablation_csv:
        cols, rows = _load_table_csv(ablation_csv)
        if rows:
            pdf.add_page()
            pdf.section_title("4. Ablation Table (F1 delta per component)")
            n = len(cols)
            col_w = max(180 // n, 18)
            widths = [col_w] * n
            pdf.table_header([(c, col_w) for c in cols])
            for i, row in enumerate(rows):
                pdf.table_row([str(row.get(c, "")) for c in cols], widths, shade=(i % 2 == 0))
            pdf.ln(6)

    # 5. Known Limitations
    pdf.add_page()
    pdf.section_title("5. Known, Stated Limitations")
    pdf.bullet_list(limitations)

    pdf.output(output_path)
    print(f"[report] Slide-deck PDF saved -> {output_path}")


def compile_slide_report_json(
    output_path: str,
    p2a_metrics: dict | None = None,
    benchmark_csv: str | None = None,
    ablation_csv: str | None = None,
    gt_stats: dict | None = None,
    known_limitations: list[str] | None = None,
) -> dict:
    """JSON equivalent of compile_slide_report — for programmatic reuse / web display."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    _, benchmark_rows = _load_table_csv(benchmark_csv) if benchmark_csv else ([], [])
    _, ablation_rows  = _load_table_csv(ablation_csv) if ablation_csv else ([], [])

    report = {
        "generated_at":       datetime.now().isoformat(),
        "detector_metrics":   p2a_metrics or {},
        "ground_truth_stats": gt_stats or {},
        "benchmark_table":    benchmark_rows,
        "ablation_table":     ablation_rows,
        "known_limitations":  known_limitations or DEFAULT_KNOWN_LIMITATIONS,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"[report] Slide-deck JSON saved -> {output_path}")
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="P4/P2b Report Generator")
    sub = parser.add_subparsers(dest="mode", required=True)

    p_video = sub.add_parser("video", help="Per-video PDF report")
    p_video.add_argument("--video", required=True)
    p_video.add_argument("--csv", required=True)
    p_video.add_argument("--heatmap", default="")
    p_video.add_argument("--timeline", default="")
    p_video.add_argument("--out", default="outputs/report.pdf")

    p_deck = sub.add_parser("deck", help="P2b Day-5 slide-deck report")
    p_deck.add_argument("--benchmark-csv", default=None)
    p_deck.add_argument("--ablation-csv", default=None)
    p_deck.add_argument("--p2a-json", default=None, help="JSON file with P2a's metrics")
    p_deck.add_argument("--out", default="outputs/deck_report.pdf")

    args = parser.parse_args()

    if args.mode == "video":
        generate_report(args.video, args.csv, args.heatmap, args.timeline, args.out)
    elif args.mode == "deck":
        p2a_metrics = None
        if args.p2a_json:
            with open(args.p2a_json) as f:
                p2a_metrics = json.load(f)
        compile_slide_report(
            args.out,
            p2a_metrics=p2a_metrics,
            benchmark_csv=args.benchmark_csv,
            ablation_csv=args.ablation_csv,
        )