"""
report.py -- P4: Outputs, Evaluation & Integration
Generates a professional PDF summary report using fpdf2.
Compiles video metadata, event statistics, metrics, and embeds
the heatmap and timeline PNG artefacts.
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
# PDF Report class
# ---------------------------------------------------------------------------

class ExamHallReport(FPDF):
    """Custom FPDF subclass with header and footer."""

    def __init__(self, video_id: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.video_id = video_id

    def header(self):
        self.set_fill_color(20, 20, 50)
        self.rect(0, 0, 210, 18, "F")
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(230, 230, 255)
        self.set_xy(10, 4)
        self.cell(0, 10, "Offline Exam-Hall Video Analytics | P4 Summary Report", align="L")
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
        if shade:
            self.set_fill_color(245, 245, 252)
        else:
            self.set_fill_color(255, 255, 255)
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
        if shade:
            self.set_fill_color(240, 240, 250)
        else:
            self.set_fill_color(255, 255, 255)
        self.set_text_color(40, 40, 60)
        for val, width in zip(values, widths):
            self.cell(width, 6, val, border=0, fill=True, align="C")
        self.ln()


# ---------------------------------------------------------------------------
# Main report generator
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
    """
    Compiles and saves a PDF summary report for a processed examination video.

    Args:
        video_path        (str):   Path to source video.
        csv_path          (str):   Path to events CSV log.
        heatmap_path      (str):   Path to heatmap PNG.
        timeline_path     (str):   Path to timeline PNG.
        output_path       (str):   Where to save the PDF.
        extra_metrics     (dict):  Optional dict with mask IoU / event F1 results.
        processing_time_s (float): Total pipeline processing time in seconds.
        risk_threshold    (float): Threshold used to define high-risk events.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    video_id = os.path.splitext(os.path.basename(video_path))[0]
    events   = _load_events_csv(csv_path)
    stats    = _compute_summary_stats(events, risk_threshold)
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    pdf = ExamHallReport(video_id=video_id, orientation="P", unit="mm", format="A4")
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ── Cover block ────────────────────────────────────────────────────────
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(20, 20, 60)
    pdf.ln(4)
    pdf.cell(0, 12, "Examination Hall Activity Report", align="C", ln=True)
    pdf.set_font("Helvetica", "", 11)
    pdf.set_text_color(100, 100, 130)
    pdf.cell(0, 7, f"Video ID: {video_id}", align="C", ln=True)
    pdf.cell(0, 7, f"Generated: {generated_at}", align="C", ln=True)
    pdf.ln(6)

    # ── Section 1: Video Metadata ──────────────────────────────────────────
    pdf.section_title("1. Video Metadata")
    meta_rows = [
        ("Source File",         os.path.basename(video_path)),
        ("Report Generated",    generated_at),
        ("Processing Time",     f"{processing_time_s:.1f} seconds"),
        ("Risk Threshold",      str(risk_threshold)),
        ("Events CSV",          os.path.basename(csv_path)),
    ]
    for i, (k, v) in enumerate(meta_rows):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(6)

    # ── Section 2: Event Summary Statistics ───────────────────────────────
    pdf.section_title("2. Event Summary Statistics")
    stat_rows = [
        ("Total Logged Events",    str(stats["total_events"])),
        ("High-Risk Events",       str(stats["high_risk_events"])),
        ("Avg. Motion Score",      f"{stats['avg_motion_score']:.2f} / 100"),
        ("Avg. Risk Score",        f"{stats['avg_risk_score']:.4f}"),
        ("Peak Risk Score",        f"{stats['peak_risk_score']:.4f}"),
        ("Peak Risk Timestamp",    stats["peak_risk_timestamp"]),
    ]
    for i, (k, v) in enumerate(stat_rows):
        pdf.kv_row(k, v, shade=(i % 2 == 0))
    pdf.ln(6)

    # ── Section 3: Evaluation Metrics ─────────────────────────────────────
    if extra_metrics:
        pdf.section_title("3. Quantitative Evaluation Metrics")
        for i, (k, v) in enumerate(extra_metrics.items()):
            pdf.kv_row(k.replace("_", " ").title(), str(v), shade=(i % 2 == 0))
        pdf.ln(6)

    # ── Section 4: Activity Timeline chart ────────────────────────────────
    if os.path.exists(timeline_path):
        pdf.section_title("4. Activity Timeline")
        pdf.ln(2)
        # Calculate image dimensions to fit page width
        pdf.image(timeline_path, x=10, w=190)
        pdf.ln(4)

    # ── Section 5: Motion Heatmap ─────────────────────────────────────────
    if os.path.exists(heatmap_path):
        pdf.section_title("5. Motion Heatmap")
        pdf.ln(2)
        pdf.image(heatmap_path, x=10, w=190)
        pdf.ln(4)

    # ── Section 6: Top 20 High-Risk Events table ──────────────────────────
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
            ("Timestamp", 32),
            ("Frame", 18),
            ("ROI Box",   48),
            ("Motion %",  24),
            ("Risk Score", 26),
            ("Confidence", 28),
        ]
        widths = [c[1] for c in cols]
        pdf.table_header(cols)
        for i, e in enumerate(high_risk_events):
            pdf.table_row(
                values=[
                    e.get("timestamp",    ""),
                    e.get("frame_index",  ""),
                    e.get("roi_box",      ""),
                    e.get("motion_score", ""),
                    e.get("risk_score",   ""),
                    e.get("confidence",   ""),
                ],
                widths=widths,
                shade=(i % 2 == 0),
            )
        pdf.ln(4)

    # ── Save ──────────────────────────────────────────────────────────────
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
    """
    Generates a lightweight JSON summary report (alternative to PDF).

    Returns:
        dict: Report dictionary (also saved to output_path).
    """
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
