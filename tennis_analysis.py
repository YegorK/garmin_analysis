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

try:
    import garmin_common as gc
except ImportError:
    print("ERROR: garmin_common.py not found.")
    print("       It must sit in the same folder as this script — copy it across")
    print("       together with the analysis scripts.")
    sys.exit(1)

gc.apply_theme()


MAX_DURATION_H = 110 / 60  # exclude sessions longer than this (tournaments/anomalies)
OVERRUN_MIN_H  = 60 / 60   # sessions ≥ this were likely not stopped on time

CHARTS = gc.SportCharts(
    title="Tennis",
    unit="Session",
    progression_png="tennis_progression.png",
    efficiency_png="tennis_efficiency.png",
    panel2_col="distance_km",
    panel2_ylabel="km",
    panel2_title="Distance Covered  (orange = overrun session, distance as recorded)",
    highlight_col="corrected",
    duration_title="Session Duration (corrected)",
    eff3_col="dist_per_min",
    eff3_ylabel="Metres / min",
    eff3_title="Court Coverage per Minute",
)


def filter_sessions(df):
    excluded = df[df["duration_h"] > MAX_DURATION_H]
    included = df[df["duration_h"] <= MAX_DURATION_H].copy().reset_index(drop=True)
    return included, excluded


def correct_overruns(df):
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


def print_summary(df, excluded):
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


def main():
    parser = argparse.ArgumentParser(description="Tennis Activity Analyzer")
    parser.add_argument("--no-charts", action="store_true", help="Print summary only")
    args = parser.parse_args()

    conn = gc.connect_db()
    df_all = gc.load_activities(conn, "%tennis%", "tennis")
    conn.close()

    df, excluded = filter_sessions(df_all)
    df, _cap_min = correct_overruns(df)
    df = gc.add_trends(df, gc.TREND_COLS)

    print_summary(df, excluded)

    if not args.no_charts:
        print("Generating charts…")
        gc.chart_progression(df, CHARTS)
        gc.chart_efficiency(df, CHARTS)
        print("Done.\n")


if __name__ == "__main__":
    main()
