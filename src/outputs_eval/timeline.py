"""
timeline.py — P4: Outputs, Evaluation & Integration
Generates a dual-axis matplotlib plot of motion score and risk score over
the video timeline. Saves PNG and optionally returns the figure for use
in the summary report.
"""

import os
import matplotlib
matplotlib.use("Agg")  # Headless backend — safe for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from typing import Optional


def plot_timeline(
    timestamps: list[float],
    motion_scores: list[float],
    risk_scores: list[float],
    output_path: str,
    video_id: str = "exam_video",
    risk_threshold: float = 0.65,
) -> None:
    """
    Generates and saves a dual-axis motion score / risk score timeline plot.

    Args:
        timestamps (list[float]):    Frame timestamps in seconds.
        motion_scores (list[float]): Normalized motion intensity per frame [0–100].
        risk_scores (list[float]):   Risk score per detected event [0.0–1.0].
        output_path (str):           Full output file path (e.g. 'outputs/video_timeline.png').
        video_id (str):              Identifier string shown in the plot title.
        risk_threshold (float):      Threshold line drawn on the risk axis. Default 0.65.
    """
    if not timestamps:
        print("[timeline] No data to plot — skipping.")
        return

    fig, ax1 = plt.subplots(figsize=(14, 5), facecolor="#1a1a2e")
    ax1.set_facecolor("#16213e")
    ax2 = ax1.twinx()

    # ── Motion score (left axis) ─────────────────────────────────────────────
    ax1.plot(
        timestamps,
        motion_scores,
        color="#00d4ff",
        linewidth=1.4,
        alpha=0.85,
        label="Motion Score",
    )
    ax1.fill_between(timestamps, motion_scores, alpha=0.15, color="#00d4ff")
    ax1.set_xlabel("Time (s)", color="#c8c8d4", fontsize=11)
    ax1.set_ylabel("Motion Score (0–100)", color="#00d4ff", fontsize=11)
    ax1.tick_params(axis="y", colors="#00d4ff")
    ax1.tick_params(axis="x", colors="#c8c8d4")
    ax1.set_ylim(0, 110)
    ax1.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.0f"))
    ax1.spines["bottom"].set_color("#333355")
    ax1.spines["left"].set_color("#00d4ff")
    ax1.spines["right"].set_color("#ff6b6b")
    ax1.spines["top"].set_color("#333355")

    # ── Risk score (right axis) ───────────────────────────────────────────────
    ax2.plot(
        timestamps,
        risk_scores,
        color="#ff6b6b",
        linewidth=1.6,
        linestyle="--",
        alpha=0.9,
        label="Risk Score",
    )
    ax2.axhline(
        y=risk_threshold,
        color="#ffcc44",
        linewidth=1.2,
        linestyle=":",
        label=f"Risk Threshold ({risk_threshold})",
    )
    ax2.set_ylabel("Risk Score (0–1)", color="#ff6b6b", fontsize=11)
    ax2.tick_params(axis="y", colors="#ff6b6b")
    ax2.set_ylim(0, 1.1)
    ax2.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.2f"))

    # ── Title and legend ─────────────────────────────────────────────────────
    plt.title(
        f"Activity Timeline — {video_id}",
        color="#e8e8f0",
        fontsize=14,
        fontweight="bold",
        pad=12,
    )
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines1 + lines2,
        labels1 + labels2,
        loc="upper right",
        facecolor="#1a1a2e",
        edgecolor="#333355",
        labelcolor="#e8e8f0",
        fontsize=9,
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[timeline] Saved -> {output_path}")
