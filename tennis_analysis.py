#!/usr/bin/env python3
"""
Tennis Activity Analyzer
Summarizes tennis sessions and tracks progression over time.

Usage:
    python3 tennis_analysis.py              # full analysis
    python3 tennis_analysis.py --no-charts  # summary only
"""

import argparse
import sys
import sqlite3
from pathlib import Path

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import linregress
except ImportError:
    print("ERROR: Missing libraries. Run: pip install pandas numpy matplotlib scipy")
    sys.exit(1)


DB_PATH    = Path("garmin_output") / "garmin.db"
OUTPUT_DIR = Path("garmin_output")
MAX_DURATION_H = 110 / 60  # exclude sessions longer than this (tournaments/anomalies)
OVERRUN_MIN_H  = 60 / 60   # sessions ≥ this were likely not stopped on time

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "figure.facecolor":  "#0f1117",
    "axes.facecolor":    "#171b26",
    "axes.labelcolor":   "#c8cdd8",
    "xtick.color":       "#c8cdd8",
    "ytick.color":       "#c8cdd8",
    "text.color":        "#e8ecf4",
    "grid.color":        "#2a2f3e",
})


def load(conn) -> pd.DataFrame:
    df = pd.read_sql(
        "SELECT * FROM activities WHERE type LIKE '%tennis%' ORDER BY date",
        conn,
    )
    if df.empty:
        print("No tennis activities found in the database.")
        sys.exit(0)
    df["date"] = pd.to_datetime(df["date"])
    df["duration_min"] = df["duration_h"] * 60
    return df


def filter_sessions(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    excluded = df[df["duration_h"] > MAX_DURATION_H]
    included = df[df["duration_h"] <= MAX_DURATION_H].copy().reset_index(drop=True)
    return included, excluded


def correct_overruns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Sessions between OVERRUN_MIN_H and MAX_DURATION_H were likely not stopped on time:
    only the first ~50-55 min are actual tennis. Correct by:
      - Capping duration at the median of clean (<60 min) sessions
      - Scaling calories and distance proportionally
      - Flagging avg HR as underestimated (idle time drags it down)
    """
    df = df.copy()
    clean = df[df["duration_min"] < OVERRUN_MIN_H * 60]
    cap_min = clean["duration_min"].median() if not clean.empty else 52.0

    overrun_mask = df["duration_min"] >= OVERRUN_MIN_H * 60
    df["corrected"] = False

    for idx in df[overrun_mask].index:
        df.at[idx, "duration_min"] = cap_min
        df.at[idx, "corrected"]    = True
        # avg_hr is unreliable for overruns — idle time drags it down
        df.at[idx, "avg_hr_note"]  = "lower bound"

    return df, cap_min


def add_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add linear regression slope annotations per metric."""
    x = np.arange(len(df))
    for col in ["avg_hr", "max_hr", "calories", "duration_min", "distance_km"]:
        if df[col].notna().sum() >= 2:
            slope, intercept, r, p, _ = linregress(x, df[col].fillna(df[col].mean()))
            df[f"{col}_trend"] = intercept + slope * x
            df[f"{col}_slope"] = slope
            df[f"{col}_r"] = r
    return df


def print_summary(df: pd.DataFrame, excluded: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  TENNIS SESSION SUMMARY")
    print("=" * 60)

    if not excluded.empty:
        print(f"\n  ⚠  {len(excluded)} session(s) excluded (>{MAX_DURATION_H*60:.0f} min):")
        for _, row in excluded.iterrows():
            print(f"     {row['date'].date()}  {row['duration_min']:.0f} min")

    print(f"\n  Sessions: {len(df)}  |  "
          f"Period: {df['date'].dt.date.min()} → {df['date'].dt.date.max()}\n")

    print(f"  {'Date':<12} {'Dur':>6} {'Avg HR':>7} {'Max HR':>7} {'Cal':>6} {'Dist':>6}  {'Note'}")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*7} {'-'*6} {'-'*6}  {'-'*15}")
    for _, row in df.iterrows():
        note = "* corrected" if row.get("corrected") else ""
        print(f"  {str(row['date'].date()):<12} "
              f"{row['duration_min']:>5.0f}m "
              f"{row['avg_hr']:>7.0f} "
              f"{row['max_hr']:>7.0f} "
              f"{row['calories']:>6.0f} "
              f"{row['distance_km']:>5.2f}km  {note}")

    if df["corrected"].any():
        cap = df[df["corrected"]]["duration_min"].iloc[0]
        print(f"\n  * Overrun sessions: duration capped at {cap:.0f} min. "
              f"Calories/distance/HR left as recorded.\n"
              f"    Avg HR is a lower bound (idle time dragged it down).")

    print(f"\n  {'Averages':<12} "
          f"{df['duration_min'].mean():>5.0f}m "
          f"{df['avg_hr'].mean():>7.0f} "
          f"{df['max_hr'].mean():>7.0f} "
          f"{df['calories'].mean():>6.0f} "
          f"{df['distance_km'].mean():>5.2f}km")

    # Progression assessment
    early = df.head(max(3, len(df)//3))
    late  = df.tail(max(3, len(df)//3))

    avg_hr_change = late["avg_hr"].mean() - early["avg_hr"].mean()
    max_hr_change = late["max_hr"].mean() - early["max_hr"].mean()

    print("\n" + "=" * 60)
    print("  PROGRESSION ASSESSMENT")
    print("=" * 60)

    print(f"\n  Avg HR:  {early['avg_hr'].mean():.0f} bpm → {late['avg_hr'].mean():.0f} bpm  "
          f"({'↓' if avg_hr_change < 0 else '↑'}{abs(avg_hr_change):.1f})")
    print(f"  Max HR:  {early['max_hr'].mean():.0f} bpm → {late['max_hr'].mean():.0f} bpm  "
          f"({'↓' if max_hr_change < 0 else '↑'}{abs(max_hr_change):.1f})")
    dist_change = late["distance_km"].mean() - early["distance_km"].mean()
    dist_per_min_early = (early["distance_km"] / (early["duration_min"] / 60)).mean()
    dist_per_min_late  = (late["distance_km"]  / (late["duration_min"]  / 60)).mean()

    print(f"  Cal/session: {early['calories'].mean():.0f} → {late['calories'].mean():.0f}")
    print(f"  Distance:    {early['distance_km'].mean():.2f} km → {late['distance_km'].mean():.2f} km  "
          f"({'↑' if dist_change > 0 else '↓'}{abs(dist_change):.2f} km)")
    print(f"  Dist/hour:   {dist_per_min_early:.2f} km/h → {dist_per_min_late:.2f} km/h  "
          f"({'↑' if dist_per_min_late > dist_per_min_early else '↓'}"
          f"{abs(dist_per_min_late - dist_per_min_early):.2f})")

    if avg_hr_change < -5:
        verdict = "✓ Clear fitness improvement — same effort at lower heart rate."
    elif avg_hr_change < 0:
        verdict = "✓ Slight improvement in cardiovascular efficiency."
    else:
        verdict = "→ No clear improvement in avg HR yet. Keep at it!"

    if max_hr_change < -10:
        verdict += "\n  ⚠  Max HR declining — check if sessions are intense enough."

    if dist_change > 0.05:
        verdict += f"\n  ✓ Covering more ground per session (+{dist_change:.2f} km) — more active play."
    elif dist_change < -0.05:
        verdict += f"\n  ↓ Covering less ground lately ({dist_change:.2f} km) — possibly more baseline rallying."

    print(f"\n  {verdict}")
    print("=" * 60 + "\n")


def _save(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    plt.close(fig)


def _fmt_date(ax, df):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xticks(df["date"])
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=8)


def chart_progression(df: pd.DataFrame):
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), facecolor="#0f1117")
    fig.suptitle("Tennis Progression", fontsize=15, fontweight="bold", y=0.99)

    dates = df["date"]

    # ── Panel 1: Heart rate (avg + max)
    ax = axes[0]
    ax.plot(dates, df["avg_hr"], color="#e74c3c", lw=2, marker="o", ms=7, label="Avg HR")
    ax.plot(dates, df["max_hr"], color="#ff9999", lw=1.5, marker="s", ms=5,
            linestyle="--", alpha=0.8, label="Max HR")
    if "avg_hr_trend" in df:
        ax.plot(dates, df["avg_hr_trend"], color="white", lw=1.2, ls=":", alpha=0.5, label="Trend")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.2)
    _fmt_date(ax, df)

    # ── Panel 2: Distance
    ax2 = axes[1]
    colors = ["#e67e22" if r["corrected"] else "#2ecc71" for _, r in df.iterrows()]
    ax2.bar(dates, df["distance_km"], color=colors, alpha=0.8, width=3)
    if "distance_km_trend" in df.columns:
        ax2.plot(dates, df["distance_km_trend"], color="white", lw=1.5, ls=":", alpha=0.6)
    ax2.set_ylabel("km")
    ax2.set_title("Distance Covered  (orange = overrun session, distance as recorded)", fontsize=11)
    _fmt_date(ax2, df)

    # ── Panel 3: Calories
    ax3 = axes[2]
    ax3.bar(dates, df["calories"], color="#f39c12", alpha=0.75, width=3)
    if "calories_trend" in df:
        ax3.plot(dates, df["calories_trend"], color="white", lw=1.5, ls=":", alpha=0.6)
    ax3.set_ylabel("Calories")
    ax3.set_title("Calories Burned", fontsize=11)
    _fmt_date(ax3, df)

    # ── Panel 4: Duration
    ax4 = axes[3]
    ax4.bar(dates, df["duration_min"], color="#3498db", alpha=0.75, width=3)
    ax4.axhline(df["duration_min"].mean(), color="white", lw=1, ls="--",
                alpha=0.5, label=f"Avg {df['duration_min'].mean():.0f} min")
    ax4.set_ylabel("Minutes")
    ax4.set_title("Session Duration (corrected)", fontsize=11)
    ax4.legend(fontsize=8, framealpha=0.2)
    _fmt_date(ax4, df)

    fig.tight_layout()
    _save(fig, "tennis_progression.png")


def chart_hr_efficiency(df: pd.DataFrame):
    """Efficiency metrics per session."""
    df = df.copy()
    df["cal_per_min"] = df["calories"] / df["duration_min"]
    df["dist_per_min"] = df["distance_km"] / df["duration_min"] * 1000  # metres/min
    df["session_n"] = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f1117")
    fig.suptitle("Tennis Efficiency", fontsize=14, fontweight="bold")

    def _plot_metric(ax, y_col, color, ylabel, title):
        vals = df[y_col]
        ax.plot(df["session_n"], vals, color=color, lw=2, marker="o", ms=7)
        slope, intercept, *_ = linregress(df["session_n"], vals)
        ax.plot(df["session_n"], intercept + slope * df["session_n"],
                color="white", lw=1.2, ls=":", alpha=0.6)
        for _, row in df.iterrows():
            ax.text(row["session_n"], row[y_col] + vals.std() * 0.1,
                    str(row["date"].date()), fontsize=6.5, ha="center", color="#c8cdd8")
        ax.set_xlabel("Session #")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    _plot_metric(axes[0], "avg_hr",      "#e74c3c", "Avg HR (bpm)",  "Cardiovascular Load\n(↓ = more efficient)")
    _plot_metric(axes[1], "cal_per_min", "#f39c12", "Cal / min",     "Calorie Intensity per Minute")
    _plot_metric(axes[2], "dist_per_min","#2ecc71", "Metres / min",  "Court Coverage per Minute")

    fig.tight_layout()
    _save(fig, "tennis_efficiency.png")


def main():
    parser = argparse.ArgumentParser(description="Tennis Activity Analyzer")
    parser.add_argument("--no-charts", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df_all = load(conn)
    conn.close()

    df, excluded = filter_sessions(df_all)
    df, cap_min = correct_overruns(df)
    df = add_trends(df)

    print_summary(df, excluded)

    if not args.no_charts:
        print("Generating charts…")
        chart_progression(df)
        chart_hr_efficiency(df)
        print("Done.\n")


if __name__ == "__main__":
    main()
