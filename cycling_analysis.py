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

try:
    import garmin_common as gc
except ImportError:
    print("ERROR: garmin_common.py not found.")
    print("       It must sit in the same folder as this script — copy it across")
    print("       together with the analysis scripts.")
    sys.exit(1)

gc.apply_theme()


CHARTS = gc.SportCharts(
    title="Cycling",
    unit="Ride",
    progression_png="cycling_progression.png",
    efficiency_png="cycling_efficiency.png",
    panel2_col="speed_kmh",
    panel2_ylabel="km/h",
    panel2_title="Average Speed",
    duration_title="Ride Duration",
    eff3_col="speed_kmh",
    eff3_ylabel="km/h",
    eff3_title="Average Speed per Ride",
)


def print_summary(df):
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


def main():
    parser = argparse.ArgumentParser(description="Cycling Activity Analyzer")
    parser.add_argument("--no-charts", action="store_true", help="Print summary only")
    args = parser.parse_args()

    conn = gc.connect_db()
    df = gc.load_activities(conn, "%cycl%", "cycling")
    conn.close()

    df = gc.add_trends(df, gc.TREND_COLS)

    print_summary(df)

    if not args.no_charts:
        print("Generating charts…")
        gc.chart_progression(df, CHARTS)
        gc.chart_efficiency(df, CHARTS)
        print("Done.\n")


if __name__ == "__main__":
    main()
