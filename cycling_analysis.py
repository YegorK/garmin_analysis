#!/usr/bin/env python3
"""
Cycling Activity Analyzer
Summarizes cycling rides and tracks progression over time.

Usage:
    python3 cycling_analysis.py              # full analysis
    python3 cycling_analysis.py --no-charts  # summary only
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
        "SELECT * FROM activities WHERE type LIKE '%cycl%' ORDER BY date",
        conn,
    )
    if df.empty:
        print("No cycling activities found in the database.")
        sys.exit(0)
    df["date"] = pd.to_datetime(df["date"])
    df["duration_min"] = df["duration_h"] * 60
    df["speed_kmh"] = df["distance_km"] / df["duration_h"]
    return df


def add_trends(df: pd.DataFrame) -> pd.DataFrame:
    """Add linear regression slope annotations per metric."""
    x = np.arange(len(df))
    for col in ["avg_hr", "max_hr", "calories", "duration_min", "distance_km", "speed_kmh"]:
        if df[col].notna().sum() >= 2:
            slope, intercept, r, p, _ = linregress(x, df[col].fillna(df[col].mean()))
            df[f"{col}_trend"] = intercept + slope * x
            df[f"{col}_slope"] = slope
            df[f"{col}_r"] = r
    return df


def print_summary(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("  CYCLING SESSION SUMMARY")
    print("=" * 60)

    print(f"\n  Rides: {len(df)}  |  "
          f"Period: {df['date'].dt.date.min()} → {df['date'].dt.date.max()}\n")

    print(f"  {'Date':<12} {'Dur':>6} {'Dist':>6} {'Speed':>7} {'Avg HR':>7} {'Max HR':>7} {'Cal':>6}")
    print(f"  {'-'*12} {'-'*6} {'-'*6} {'-'*7} {'-'*7} {'-'*7} {'-'*6}")
    for _, row in df.iterrows():
        print(f"  {str(row['date'].date()):<12} "
              f"{row['duration_min']:>5.0f}m "
              f"{row['distance_km']:>5.2f}km "
              f"{row['speed_kmh']:>6.1f}km/h "
              f"{row['avg_hr']:>7.0f} "
              f"{row['max_hr']:>7.0f} "
              f"{row['calories']:>6.0f}")

    print(f"\n  {'Averages':<12} "
          f"{df['duration_min'].mean():>5.0f}m "
          f"{df['distance_km'].mean():>5.2f}km "
          f"{df['speed_kmh'].mean():>6.1f}km/h "
          f"{df['avg_hr'].mean():>7.0f} "
          f"{df['max_hr'].mean():>7.0f} "
          f"{df['calories'].mean():>6.0f}")

    # Progression assessment
    early = df.head(max(3, len(df)//3))
    late  = df.tail(max(3, len(df)//3))

    avg_hr_change = late["avg_hr"].mean() - early["avg_hr"].mean()
    max_hr_change = late["max_hr"].mean() - early["max_hr"].mean()
    speed_change  = late["speed_kmh"].mean() - early["speed_kmh"].mean()
    dist_change   = late["distance_km"].mean() - early["distance_km"].mean()

    print("\n" + "=" * 60)
    print("  PROGRESSION ASSESSMENT")
    print("=" * 60)

    print(f"\n  Avg HR:  {early['avg_hr'].mean():.0f} bpm → {late['avg_hr'].mean():.0f} bpm  "
          f"({'↓' if avg_hr_change < 0 else '↑'}{abs(avg_hr_change):.1f})")
    print(f"  Max HR:  {early['max_hr'].mean():.0f} bpm → {late['max_hr'].mean():.0f} bpm  "
          f"({'↓' if max_hr_change < 0 else '↑'}{abs(max_hr_change):.1f})")
    print(f"  Cal/session: {early['calories'].mean():.0f} → {late['calories'].mean():.0f}")
    print(f"  Distance:    {early['distance_km'].mean():.2f} km → {late['distance_km'].mean():.2f} km  "
          f"({'↑' if dist_change > 0 else '↓'}{abs(dist_change):.2f} km)")
    print(f"  Speed:       {early['speed_kmh'].mean():.1f} km/h → {late['speed_kmh'].mean():.1f} km/h  "
          f"({'↑' if speed_change > 0 else '↓'}{abs(speed_change):.1f})")

    if avg_hr_change < -5:
        verdict = "✓ Clear fitness improvement — same effort at lower heart rate."
    elif avg_hr_change < 0:
        verdict = "✓ Slight improvement in cardiovascular efficiency."
    else:
        verdict = "→ No clear improvement in avg HR yet. Keep at it!"

    if max_hr_change < -10:
        verdict += "\n  ⚠  Max HR declining — check if rides are intense enough."

    if speed_change > 0.5:
        verdict += f"\n  ✓ Riding faster on average (+{speed_change:.1f} km/h)."
    elif speed_change < -0.5:
        verdict += f"\n  ↓ Riding slower on average ({speed_change:.1f} km/h) — possibly more casual rides."

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
    fig.suptitle("Cycling Progression", fontsize=15, fontweight="bold", y=0.99)

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

    # ── Panel 2: Speed
    ax2 = axes[1]
    ax2.bar(dates, df["speed_kmh"], color="#2ecc71", alpha=0.8, width=3)
    if "speed_kmh_trend" in df.columns:
        ax2.plot(dates, df["speed_kmh_trend"], color="white", lw=1.5, ls=":", alpha=0.6)
    ax2.set_ylabel("km/h")
    ax2.set_title("Average Speed", fontsize=11)
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
    ax4.set_title("Ride Duration", fontsize=11)
    ax4.legend(fontsize=8, framealpha=0.2)
    _fmt_date(ax4, df)

    fig.tight_layout()
    _save(fig, "cycling_progression.png")


def chart_efficiency(df: pd.DataFrame):
    """Efficiency metrics per session."""
    df = df.copy()
    df["cal_per_min"] = df["calories"] / df["duration_min"]
    df["session_n"] = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="#0f1117")
    fig.suptitle("Cycling Efficiency", fontsize=14, fontweight="bold")

    def _plot_metric(ax, y_col, color, ylabel, title):
        vals = df[y_col]
        ax.plot(df["session_n"], vals, color=color, lw=2, marker="o", ms=7)
        slope, intercept, *_ = linregress(df["session_n"], vals)
        ax.plot(df["session_n"], intercept + slope * df["session_n"],
                color="white", lw=1.2, ls=":", alpha=0.6)
        for _, row in df.iterrows():
            ax.text(row["session_n"], row[y_col] + vals.std() * 0.1,
                    str(row["date"].date()), fontsize=6.5, ha="center", color="#c8cdd8")
        ax.set_xlabel("Ride #")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    _plot_metric(axes[0], "avg_hr",      "#e74c3c", "Avg HR (bpm)", "Cardiovascular Load\n(↓ = more efficient)")
    _plot_metric(axes[1], "cal_per_min", "#f39c12", "Cal / min",    "Calorie Intensity per Minute")
    _plot_metric(axes[2], "speed_kmh",   "#2ecc71", "km/h",         "Average Speed per Ride")

    fig.tight_layout()
    _save(fig, "cycling_efficiency.png")


def main():
    parser = argparse.ArgumentParser(description="Cycling Activity Analyzer")
    parser.add_argument("--no-charts", action="store_true", help="Print summary only")
    args = parser.parse_args()

    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    df = load(conn)
    conn.close()

    df = add_trends(df)

    print_summary(df)

    if not args.no_charts:
        print("Generating charts…")
        chart_progression(df)
        chart_efficiency(df)
        print("Done.\n")


if __name__ == "__main__":
    main()
