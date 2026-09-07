#!/usr/bin/env python3
"""
Sleep & Stress Correlation Analyzer
Analyzes meaningful correlations between sleep quality, sleep duration, and stress levels.
Generates visualizations and a detailed summary report with annotated interesting days.

Usage:
    python sleep_stress_analysis.py           # full analysis
    python sleep_stress_analysis.py --no-charts  # report only
    python sleep_stress_analysis.py --days 30    # limit to last N days
"""

import argparse
import datetime
import sqlite3
import sys
from pathlib import Path
from collections import defaultdict

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import pearsonr, f_oneway
except ImportError:
    print("ERROR: Missing libraries. Run: pip install pandas numpy matplotlib scipy")
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────
DB_PATH = Path("garmin_output") / "garmin.db"
OUTPUT_DIR = Path("garmin_output")
REPORT_PATH = OUTPUT_DIR / "sleep_stress_report.txt"

# Intraday coverage thresholds
INTRADAY_COVERAGE_MIN = 0.70  # 70% of expected data points required

# Colors
SLEEP_SCORE_COLOR = "#5b9bd5"
STRESS_COLOR = "#e74c3c"
HRV_COLOR = "#9b59b6"
BB_COLOR = "#2ecc71"

# Plot theme
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "figure.facecolor": "#0f1117",
    "axes.facecolor": "#171b26",
    "axes.labelcolor": "#c8cdd8",
    "xtick.color": "#c8cdd8",
    "ytick.color": "#c8cdd8",
    "text.color": "#e8ecf4",
    "grid.color": "#2a2f3e",
})


def db_connect() -> sqlite3.Connection:
    if not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)
    return sqlite3.connect(DB_PATH)


def load_raw_data(conn, days_limit: int = None) -> tuple[dict, list]:
    """Load all tables from DB. Return (dataframes_dict, excluded_dates)."""
    excluded = []

    # Load daily aggregates
    daily = pd.read_sql("SELECT * FROM daily ORDER BY date", conn)
    sleep = pd.read_sql("SELECT * FROM sleep ORDER BY date", conn)
    hrv = pd.read_sql("SELECT * FROM hrv ORDER BY date", conn)

    # Load intraday timeseries
    stress_intra = pd.read_sql("SELECT timestamp, date, stress FROM stress ORDER BY timestamp", conn)
    bb_intra = pd.read_sql("SELECT timestamp, date, body_battery FROM body_battery ORDER BY timestamp", conn)
    hr_intra = pd.read_sql("SELECT timestamp, date, heart_rate FROM heart_rate ORDER BY timestamp", conn)

    # Convert timestamps
    for df in [stress_intra, bb_intra, hr_intra]:
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])

    # Convert dates
    for df in [daily, sleep, hrv]:
        if not df.empty and "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"]).dt.date

    # Apply days limit if requested
    if days_limit:
        cutoff = datetime.date.today() - datetime.timedelta(days=days_limit)
        daily = daily[daily["date"] >= cutoff] if not daily.empty else daily
        sleep = sleep[sleep["date"] >= cutoff] if not sleep.empty else sleep
        hrv = hrv[hrv["date"] >= cutoff] if not hrv.empty else hrv
        for df in [stress_intra, bb_intra, hr_intra]:
            if not df.empty:
                df = df[df["date"] >= cutoff.isoformat()]

    return {"daily": daily, "sleep": sleep, "hrv": hrv, "stress_intra": stress_intra, "bb_intra": bb_intra, "hr_intra": hr_intra}, excluded


def filter_incomplete_days(data: dict) -> tuple[pd.DataFrame, list]:
    """
    Filter days with insufficient watch coverage.
    Return merged dataframe and list of excluded dates.
    """
    excluded = []

    # Filter sleep table: exclude duration < 2h (watch not worn during sleep)
    sleep = data["sleep"].copy()
    if not sleep.empty:
        before = len(sleep)
        sleep = sleep[sleep["duration_h"] >= 2.0]
        excluded.extend(data["sleep"][data["sleep"]["duration_h"] < 2.0]["date"].astype(str).tolist())
        print(f"  Sleep: filtered {before - len(sleep)} days with duration < 2h")

    # Filter daily table: exclude NULL stress_avg
    daily = data["daily"].copy()
    if not daily.empty:
        before = len(daily)
        daily = daily[daily["stress_avg"].notna()]
        excluded.extend(data["daily"][data["daily"]["stress_avg"].isna()]["date"].astype(str).tolist())
        print(f"  Daily: filtered {before - len(daily)} days with missing stress_avg")

    # Extract HRV
    hrv = data["hrv"].copy()

    # Check intraday coverage for each day
    intraday_tables = ["stress_intra", "bb_intra", "hr_intra"]
    covered_dates = set()

    for tbl_name in intraday_tables:
        df = data[tbl_name]
        if df.empty:
            continue
        # Count readings per day
        daily_counts = df.groupby("date").size()
        # Estimate expected readings (e.g., stress usually ~1440 per day if hourly, but Garmin may vary)
        # Use median as expected
        expected = daily_counts.median()
        if expected < 10:  # Too sparse to validate
            continue
        threshold = expected * INTRADAY_COVERAGE_MIN
        sparse_dates = daily_counts[daily_counts < threshold].index.tolist()
        excluded.extend([str(d) for d in sparse_dates])
        covered_dates.update(daily_counts[daily_counts >= threshold].index)

    # Merge tables: sleep + daily + hrv
    merged = sleep.copy() if not sleep.empty else pd.DataFrame(columns=["date"])
    if not daily.empty:
        merged = pd.merge(merged, daily, on="date", how="outer") if not merged.empty else daily.copy()
    if not hrv.empty:
        merged = pd.merge(merged, hrv, on="date", how="outer") if not merged.empty else hrv.copy()

    # Filter to only days with good intraday coverage (for stress/HRV analysis)
    if covered_dates:
        # Keep all days for daily aggregates, but mark which have good intraday data
        merged["has_intraday_data"] = merged["date"].isin(covered_dates)
    else:
        merged["has_intraday_data"] = True

    merged = merged.sort_values("date").reset_index(drop=True)

    excluded_unique = sorted(set(excluded))
    if excluded_unique:
        print(f"\n  ⚠ Excluded {len(excluded_unique)} days due to incomplete watch coverage")
        if len(excluded_unique) <= 10:
            for d in excluded_unique:
                print(f"    {d}")
        else:
            for d in excluded_unique[:5]:
                print(f"    {d}")
            print(f"    ... and {len(excluded_unique) - 5} more")

    return merged, excluded_unique


def build_analysis_df(merged: pd.DataFrame, data: dict) -> pd.DataFrame:
    """
    Build analysis dataframe with lagged features for correlation analysis.
    """
    df = merged.copy()

    # Compute lagged stress (previous day's average)
    df["stress_prev"] = df["stress_avg"].shift(1)
    df["stress_next"] = df["stress_avg"].shift(-1)

    # Compute next-day sleep score
    df["sleep_score_next"] = df["score"].shift(-1)

    # Compute sleep percentages
    df["deep_pct"] = (df["deep_h"] / (df["duration_h"] + 0.01)) * 100
    df["rem_pct"] = (df["rem_h"] / (df["duration_h"] + 0.01)) * 100
    df["light_pct"] = (df["light_h"] / (df["duration_h"] + 0.01)) * 100

    # Extract evening (8pm–midnight) and morning (6am–10am) stress from intraday
    stress_intra = data["stress_intra"]
    if not stress_intra.empty:
        stress_intra["hour"] = stress_intra["timestamp"].dt.hour
        evening_stress = stress_intra[stress_intra["hour"].isin([20, 21, 22, 23])].groupby("date")["stress"].mean()
        morning_stress = stress_intra[stress_intra["hour"].isin([6, 7, 8, 9])].groupby("date")["stress"].mean()
        df["evening_stress"] = df["date"].apply(
            lambda d: evening_stress.get(d.isoformat()) if isinstance(d, datetime.date) else None
        )
        df["morning_stress"] = df["date"].apply(
            lambda d: morning_stress.get(d.isoformat()) if isinstance(d, datetime.date) else None
        )

    # Extract body battery peak and drop from intraday
    bb_intra = data["bb_intra"]
    if not bb_intra.empty:
        bb_peak = bb_intra.groupby("date")["body_battery"].max()
        bb_min = bb_intra.groupby("date")["body_battery"].min()
        df["bb_peak"] = df["date"].apply(
            lambda d: bb_peak.get(d.isoformat()) if isinstance(d, datetime.date) else None
        )
        def compute_bb_drop(d):
            if not isinstance(d, datetime.date):
                return None
            pk = bb_peak.get(d.isoformat())
            mn = bb_min.get(d.isoformat())
            if pk is not None and mn is not None:
                return pk - mn
            return None
        df["bb_drop"] = df["date"].apply(compute_bb_drop)

    return df


def compute_correlations(df: pd.DataFrame) -> dict:
    """
    Compute all key correlations.
    Return {name: (r, p, n)} tuples.
    """
    corrs = {}

    pairs = [
        ("stress_prev", "score", "Previous day stress → sleep score"),
        ("evening_stress", "score", "Evening stress (8pm–midnight) → sleep score"),
        ("score", "stress_next", "Sleep score → next day avg stress"),
        ("duration_h", "stress_next", "Sleep duration → next day avg stress"),
        ("deep_pct", "stress_next", "Deep sleep % → next day avg stress"),
        ("last_night", "stress_avg", "HRV last night → daytime stress"),
        ("score", "morning_stress", "Sleep score → morning stress"),
        ("score", "bb_peak", "Sleep score → body battery peak"),
        ("stress_avg", "rhr", "Stress avg → resting heart rate"),
    ]

    for col1, col2, label in pairs:
        if col1 in df.columns and col2 in df.columns:
            subset = df[[col1, col2]].dropna()
            if len(subset) >= 3:
                r, p = pearsonr(subset[col1], subset[col2])
                corrs[label] = (r, p, len(subset))

    return corrs


def compute_weekly_patterns(df: pd.DataFrame) -> dict:
    """
    Compute ANOVA for day-of-week effects on sleep score and stress.
    """
    df_dow = df.copy()
    df_dow["dow"] = pd.to_datetime(df_dow["date"]).dt.dayofweek
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

    results = {}

    # Sleep score by day of week
    if "score" in df.columns:
        groups = [group["score"].dropna().values for _, group in df_dow.groupby("dow")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            results["sleep_score_dow"] = (f_stat, p_val)

    # Stress by day of week
    if "stress_avg" in df.columns:
        groups = [group["stress_avg"].dropna().values for _, group in df_dow.groupby("dow")]
        groups = [g for g in groups if len(g) > 0]
        if len(groups) > 1:
            f_stat, p_val = f_oneway(*groups)
            results["stress_dow"] = (f_stat, p_val)

    return results


def detect_interesting_days(df: pd.DataFrame, corrs: dict) -> list:
    """
    Detect and annotate interesting days.
    Return list of {date, category, reasoning} dicts.
    """
    interesting = []

    # Z-score threshold
    z_thresh = 1.5

    # Standardize key metrics
    for col in ["score", "stress_avg", "last_night", "duration_h"]:
        if col in df.columns and df[col].notna().sum() > 0:
            df[f"{col}_z"] = (df[col] - df[col].mean()) / (df[col].std() + 1e-6)

    # Pattern: worst cascade (3-day stress-sleep loop)
    for i in range(1, len(df) - 1):
        if (
            df.iloc[i - 1]["stress_avg_z"] > z_thresh
            and df.iloc[i]["score_z"] < -z_thresh
            and df.iloc[i + 1]["stress_avg_z"] > z_thresh
        ):
            interesting.append({
                "date": df.iloc[i]["date"],
                "category": "Worst Cascade",
                "reasoning": f"High stress ({df.iloc[i-1]['stress_avg']:.0f}) → poor sleep score ({df.iloc[i]['score']:.0f}) → high stress again ({df.iloc[i+1]['stress_avg']:.0f}). Classic feedback loop.",
            })

    # Pattern: best recovery (high sleep score + low next-day stress + high HRV)
    recovery = df[
        (df["score_z"] > z_thresh)
        & (df["stress_next_z"] if "stress_next_z" in df.columns else False)
        & (df["last_night_z"] > z_thresh if "last_night_z" in df.columns else False)
    ]
    for _, row in recovery.iterrows():
        interesting.append({
            "date": row["date"],
            "category": "Best Recovery",
            "reasoning": f"Excellent sleep score ({row['score']:.0f}), HRV high ({row['last_night']:.0f}ms), and stress low the next day. Ideal recovery day.",
        })

    # Pattern: stress spike didn't hurt sleep (resilience)
    resilience = df[
        (df["stress_avg_z"] > z_thresh)
        & (df["score_z"] > -z_thresh)
        & (df["score"].notna())
    ]
    for _, row in resilience.iterrows():
        interesting.append({
            "date": row["date"],
            "category": "Stress Resilience",
            "reasoning": f"High stress ({row['stress_avg']:.0f}) but sleep score was still decent ({row['score']:.0f}). Shows good stress coping.",
        })

    # Pattern: surprise bad night (no prior stress but poor sleep)
    bad_surprise = df[
        (df["stress_prev_z"] > -z_thresh if "stress_prev_z" in df.columns else False)
        & (df["score_z"] < -z_thresh)
    ]
    for _, row in bad_surprise.iterrows():
        reasoning = f"Poor sleep score ({row['score']:.0f}) despite no elevated stress the day before. Other factors may have contributed."
        if row["evening_stress"] and not pd.isna(row["evening_stress"]):
            reasoning += f" (evening stress was {row['evening_stress']:.0f})"
        interesting.append({
            "date": row["date"],
            "category": "Surprise Bad Night",
            "reasoning": reasoning,
        })

    # Pattern: high-stress cluster (3+ consecutive days > stress 60th percentile)
    if "stress_avg" in df.columns:
        stress_threshold = df["stress_avg"].quantile(0.6)
        high_stress_mask = df["stress_avg"] > stress_threshold
        for i in range(len(df) - 2):
            if high_stress_mask.iloc[i] and high_stress_mask.iloc[i + 1] and high_stress_mask.iloc[i + 2]:
                if i not in [d.get("_idx") for d in interesting]:
                    interesting.append({
                        "date": df.iloc[i]["date"],
                        "category": "High-Stress Cluster",
                        "reasoning": f"Start of 3+ day stress cluster (threshold: {stress_threshold:.0f}). Stress levels: {df.iloc[i]['stress_avg']:.0f} → {df.iloc[i+1]['stress_avg']:.0f} → {df.iloc[i+2]['stress_avg']:.0f}.",
                        "_idx": i,
                    })

    # Deduplicate by date
    seen = set()
    unique = []
    for item in interesting:
        key = str(item["date"])
        if key not in seen:
            seen.add(key)
            unique.append(item)

    return sorted(unique, key=lambda x: x["date"], reverse=True)


def save_report(df: pd.DataFrame, corrs: dict, dow_patterns: dict, interesting: list):
    """Write summary report to file."""
    with open(REPORT_PATH, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("SLEEP & STRESS CORRELATION ANALYSIS REPORT\n")
        f.write("=" * 70 + "\n\n")

        # Overview
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        f.write(f"Overview\n")
        f.write(f"--------\n")
        f.write(f"Date range: {date_range}\n")
        f.write(f"Nights analyzed: {len(df)}\n")
        f.write(f"Days with complete data: {df['has_intraday_data'].sum() if 'has_intraday_data' in df.columns else len(df)}\n\n")

        # Correlations
        f.write(f"Key Correlations (Pearson r, p-value)\n")
        f.write(f"-------------------------------------\n")
        for label, (r, p, n) in sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True):
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "  "
            f.write(f"  {label:<45} r={r:+.3f}  p={p:.4f} {sig}  (n={n})\n")
        f.write("\n")

        # Strongest effect
        if corrs:
            strongest = max(corrs.items(), key=lambda x: abs(x[1][0]))
            f.write(f"Strongest Effect\n")
            f.write(f"----------------\n")
            f.write(f"  {strongest[0]}: r={strongest[1][0]:+.3f}, p={strongest[1][1]:.4f}\n\n")

        # Weekly patterns
        if dow_patterns:
            f.write(f"Weekly Patterns (ANOVA)\n")
            f.write(f"----------------------\n")
            if "sleep_score_dow" in dow_patterns:
                f_stat, p = dow_patterns["sleep_score_dow"]
                f.write(f"  Sleep score varies by day of week: F={f_stat:.2f}, p={p:.4f}\n")
            if "stress_dow" in dow_patterns:
                f_stat, p = dow_patterns["stress_dow"]
                f.write(f"  Stress level varies by day of week: F={f_stat:.2f}, p={p:.4f}\n")
            f.write("\n")

        # Interesting days
        f.write(f"Notable Days ({len(interesting)} total)\n")
        f.write(f"----------------------\n\n")
        for item in interesting:
            f.write(f"  {item['date']} — {item['category']}\n")
            f.write(f"    {item['reasoning']}\n\n")

        # Summary statistics
        f.write(f"Summary Statistics\n")
        f.write(f"------------------\n")
        if "score" in df.columns:
            f.write(f"  Sleep score: mean={df['score'].mean():.1f}, std={df['score'].std():.1f}, range={df['score'].min():.0f}–{df['score'].max():.0f}\n")
        if "stress_avg" in df.columns:
            f.write(f"  Daily stress: mean={df['stress_avg'].mean():.1f}, std={df['stress_avg'].std():.1f}, range={df['stress_avg'].min():.0f}–{df['stress_avg'].max():.0f}\n")
        if "duration_h" in df.columns:
            f.write(f"  Sleep duration: mean={df['duration_h'].mean():.1f}h, std={df['duration_h'].std():.1f}h\n")
        if "last_night" in df.columns and df["last_night"].notna().sum() > 0:
            f.write(f"  HRV (last night): mean={df['last_night'].mean():.0f}ms, std={df['last_night'].std():.0f}ms\n")
        f.write("\n")

        f.write("=" * 70 + "\n")

    print(f"✓ Report saved to {REPORT_PATH}\n")


def _fmt_date(ax):
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right")


def _save(fig, name):
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    plt.close(fig)


def chart_corr_scatter_grid(df: pd.DataFrame, corrs: dict):
    """3×3 scatter plot grid for top correlations."""
    pairs = sorted(corrs.items(), key=lambda x: abs(x[1][0]), reverse=True)[:9]
    if not pairs:
        return

    fig, axes = plt.subplots(3, 3, figsize=(14, 12), facecolor="#0f1117")
    axes = axes.flatten()

    for idx, (label, (r, p, n)) in enumerate(pairs):
        ax = axes[idx]
        col1, col2 = label.split(" → ")
        col1, col2 = col1.strip(), col2.strip()

        if col1 in df.columns and col2 in df.columns:
            subset = df[[col1, col2]].dropna()
            if len(subset) > 1:
                x, y = subset[col1], subset[col2]
                ax.scatter(x, y, alpha=0.5, s=30, color=SLEEP_SCORE_COLOR)
                # Regression line
                z = np.polyfit(x, y, 1)
                p_line = np.poly1d(z)
                x_line = np.linspace(x.min(), x.max(), 50)
                ax.plot(x_line, p_line(x_line), color="white", lw=1.5, alpha=0.7)
                ax.set_title(f"r={r:+.2f}, p={p:.3f}", fontsize=9)
                ax.set_xlabel(col1, fontsize=8)
                ax.set_ylabel(col2, fontsize=8)

    for idx in range(len(pairs), 9):
        axes[idx].axis("off")

    fig.suptitle("Top Correlations — Sleep & Stress", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout()
    _save(fig, "corr_scatter_grid.png")


def chart_corr_heatmap(df: pd.DataFrame):
    """Correlation heatmap of key variables."""
    cols_to_include = [
        "score", "duration_h", "deep_pct", "rem_pct",
        "stress_avg", "stress_prev",
        "evening_stress", "morning_stress",
        "last_night", "rhr", "bb_peak",
    ]
    cols = [c for c in cols_to_include if c in df.columns]
    if len(cols) < 2:
        return

    corr_matrix = df[cols].corr()

    fig, ax = plt.subplots(figsize=(10, 8), facecolor="#0f1117")
    im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(cols, fontsize=8)

    # Annotate with correlation values
    for i in range(len(cols)):
        for j in range(len(cols)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black" if abs(corr_matrix.iloc[i, j]) < 0.5 else "white",
                          fontsize=7)

    plt.colorbar(im, ax=ax, label="Correlation (r)")
    fig.suptitle("Correlation Heatmap", fontsize=14, fontweight="bold")
    fig.tight_layout()
    _save(fig, "corr_heatmap.png")


def chart_corr_timeseries(df: pd.DataFrame):
    """Sleep score vs next-day stress over time."""
    df_plot = df[["date", "score", "stress_next"]].dropna()
    if len(df_plot) < 2:
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), facecolor="#0f1117", gridspec_kw={"height_ratios": [3, 1]})

    # Main plot
    ax1.bar(df_plot["date"], df_plot["score"], color=SLEEP_SCORE_COLOR, alpha=0.6, width=0.8, label="Sleep Score")
    ax1b = ax1.twinx()
    ax1b.plot(df_plot["date"], df_plot["stress_next"], color=STRESS_COLOR, lw=2, label="Next-day Stress", marker="o", ms=3)
    ax1.set_ylabel("Sleep Score", color=SLEEP_SCORE_COLOR)
    ax1b.set_ylabel("Next-day Stress", color=STRESS_COLOR)
    ax1.tick_params(axis="y", labelcolor=SLEEP_SCORE_COLOR)
    ax1b.tick_params(axis="y", labelcolor=STRESS_COLOR)
    ax1.set_title("Sleep Score vs Next-Day Stress", fontsize=12)
    _fmt_date(ax1)

    # Rolling correlation
    rolling_corr = df_plot.set_index("date")[["score", "stress_next"]].rolling(7).corr().unstack()
    rolling_corr = rolling_corr.iloc[::2, 1]  # Every 2nd value to avoid duplication
    ax2.plot(rolling_corr.index, rolling_corr.values, color="white", lw=1.5)
    ax2.axhline(0, color="#666", ls="--", lw=0.8)
    ax2.fill_between(rolling_corr.index, rolling_corr.values, 0, alpha=0.3, color="white")
    ax2.set_ylabel("7-day Rolling Corr")
    ax2.set_ylim(-1, 1)
    _fmt_date(ax2)

    fig.tight_layout()
    _save(fig, "corr_timeseries_overlay.png")


def chart_stress_profile(df: pd.DataFrame, data: dict):
    """Intraday stress for good vs poor sleep nights."""
    stress_intra = data["stress_intra"]
    if stress_intra.empty or "score" not in df.columns:
        return

    good_nights = df[df["score"] >= 75]["date"].astype(str).tolist()
    poor_nights = df[df["score"] < 55]["date"].astype(str).tolist()

    fig, ax = plt.subplots(figsize=(14, 6), facecolor="#0f1117")

    for night_list, label, color, alpha in [
        (good_nights, "Good sleep nights (≥75)", "#2ecc71", 0.7),
        (poor_nights, "Poor sleep nights (<55)", "#e74c3c", 0.7),
    ]:
        if night_list:
            subset = stress_intra[stress_intra["date"].isin(night_list)].copy()
            subset["hour"] = subset["timestamp"].dt.hour
            hourly = subset.groupby("hour")["stress"].mean()
            ax.plot(hourly.index, hourly.values, label=label, lw=2, color=color, marker="o", ms=4, alpha=alpha)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Average Stress Level")
    ax.set_xlim(0, 23)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.suptitle("Intraday Stress Profile: Good vs Poor Sleep Nights", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "corr_stress_profile.png")


def chart_recovery(df: pd.DataFrame):
    """Body battery peak vs sleep score, colored by stress."""
    df_plot = df[["date", "score", "bb_peak", "stress_avg"]].dropna()
    if len(df_plot) < 2:
        return

    fig, ax = plt.subplots(figsize=(10, 7), facecolor="#0f1117")
    scatter = ax.scatter(df_plot["score"], df_plot["bb_peak"], c=df_plot["stress_avg"],
                        cmap="RdYlGn_r", s=100, alpha=0.6, edgecolors="white", linewidth=0.5)
    ax.set_xlabel("Sleep Score")
    ax.set_ylabel("Body Battery Peak (%)")
    ax.set_title("Sleep Quality → Recovery Capacity (colored by stress)", fontsize=12)
    cbar = plt.colorbar(scatter, ax=ax, label="Daily Stress")
    fig.tight_layout()
    _save(fig, "corr_recovery.png")


def chart_weekly(df: pd.DataFrame):
    """Box plots of sleep score and stress by day of week."""
    if "score" not in df.columns or "stress_avg" not in df.columns:
        return

    df_dow = df.copy()
    df_dow["dow"] = pd.to_datetime(df_dow["date"]).dt.day_name()
    dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    df_dow["dow"] = pd.Categorical(df_dow["dow"], categories=dow_order, ordered=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), facecolor="#0f1117")

    # Sleep score
    df_dow.boxplot(column="score", by="dow", ax=ax1)
    ax1.set_title("Sleep Score by Day of Week")
    ax1.set_xlabel("Day")
    ax1.set_ylabel("Score")
    ax1.get_figure().suptitle("")

    # Stress
    df_dow.boxplot(column="stress_avg", by="dow", ax=ax2)
    ax2.set_title("Stress Level by Day of Week")
    ax2.set_xlabel("Day")
    ax2.set_ylabel("Stress Level")
    ax2.get_figure().suptitle("")

    fig.suptitle("Weekly Patterns", fontsize=12, fontweight="bold")
    fig.tight_layout()
    _save(fig, "corr_weekly.png")


def chart_interesting_days(df: pd.DataFrame, interesting: list):
    """Timeline with annotated interesting days."""
    if "score" not in df.columns:
        return

    fig, ax = plt.subplots(figsize=(16, 6), facecolor="#0f1117")

    ax.bar(df["date"], df["score"], color=SLEEP_SCORE_COLOR, alpha=0.5, width=0.8, label="Sleep Score")
    if "stress_avg" in df.columns:
        ax2 = ax.twinx()
        ax2.plot(df["date"], df["stress_avg"], color=STRESS_COLOR, lw=1.5, label="Stress Level", alpha=0.7)
        ax2.set_ylabel("Stress Level", color=STRESS_COLOR)

    # Annotate interesting days
    for item in interesting[:10]:  # Top 10 to avoid clutter
        date = pd.Timestamp(item["date"])
        score = df[df["date"] == item["date"]]["score"].values
        if len(score) > 0:
            y_pos = score[0]
            ax.annotate(
                item["category"],
                xy=(date, y_pos),
                xytext=(0, 10),
                textcoords="offset points",
                fontsize=7,
                ha="center",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#2a2f3e", alpha=0.7),
                arrowprops=dict(arrowstyle="->", lw=0.5, color="#c8cdd8"),
            )

    ax.set_ylabel("Sleep Score", color=SLEEP_SCORE_COLOR)
    ax.set_title("Sleep Score Timeline with Notable Days", fontsize=12)
    _fmt_date(ax)
    fig.tight_layout()
    _save(fig, "corr_interesting_days.png")


def main():
    parser = argparse.ArgumentParser(description="Sleep & Stress Correlation Analyzer")
    parser.add_argument("--no-charts", action="store_true", help="Report only, skip chart generation")
    parser.add_argument("--days", type=int, help="Limit analysis to last N days")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("SLEEP & STRESS CORRELATION ANALYZER")
    print("=" * 70 + "\n")

    conn = db_connect()

    print("Loading data from database…")
    data, _ = load_raw_data(conn, days_limit=args.days)
    print(f"  Sleep: {len(data['sleep'])} nights")
    print(f"  Daily: {len(data['daily'])} days")
    print(f"  HRV: {len(data['hrv'])} days")
    print(f"  Intraday stress: {len(data['stress_intra'])} readings")
    print(f"  Intraday body battery: {len(data['bb_intra'])} readings")

    print("\nFiltering incomplete days…")
    df, excluded = filter_incomplete_days(data)
    print(f"  Final dataset: {len(df)} days with complete data")

    if len(df) < 3:
        print("\nERROR: Insufficient data for correlation analysis (need ≥3 days)")
        return

    print("\nBuilding analysis dataframe with lagged features…")
    df = build_analysis_df(df, data)

    print("\nComputing correlations…")
    corrs = compute_correlations(df)
    print(f"  Found {len(corrs)} significant correlations")

    print("\nAnalyzing weekly patterns…")
    dow_patterns = compute_weekly_patterns(df)

    print("\nDetecting interesting days…")
    interesting = detect_interesting_days(df, corrs)
    print(f"  Found {len(interesting)} interesting day patterns")

    print("\nGenerating report…")
    save_report(df, corrs, dow_patterns, interesting)

    if not args.no_charts:
        print("\nGenerating charts…")
        chart_corr_scatter_grid(df, corrs)
        chart_corr_heatmap(df)
        chart_corr_timeseries(df)
        chart_stress_profile(df, data)
        chart_recovery(df)
        chart_weekly(df)
        chart_interesting_days(df, interesting)

    conn.close()
    print("\n" + "=" * 70)
    print("✓ Analysis complete")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
