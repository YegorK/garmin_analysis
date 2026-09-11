#!/usr/bin/env python3
"""
Shared helpers for the Garmin analysis scripts.

Every other script in this folder imports from here, so keep this file next to
garmin_analysis.py, sleep_stress_analysis.py, tennis_analysis.py and
cycling_analysis.py.
"""

import sqlite3
import sys
from dataclasses import dataclass
from pathlib import Path

# The one place the dependency list is spelled out; every script points here.
INSTALL_HINT = (
    "       Make sure your virtual environment is active (your prompt should\n"
    "       show '(.venv)'), then run:\n"
    "         pip install garminconnect pandas numpy matplotlib scipy"
)

try:
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from scipy.stats import linregress
except ImportError as e:
    print(f"ERROR: missing library '{e.name or 'unknown'}'.")
    print(INSTALL_HINT)
    sys.exit(1)


# ── Paths ─────────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("garmin_output")
DB_PATH    = OUTPUT_DIR / "garmin.db"


# ── Palette ───────────────────────────────────────────────────────────────────
SLEEP_COLORS      = {"deep": "#1a3a5c", "light": "#5b9bd5", "rem": "#8a63d2", "awake": "#e8a838"}
SLEEP_SCORE_COLOR = "#5b9bd5"
STEPS_COLOR       = "#3498db"
STRESS_COLOR      = "#e74c3c"
HRV_COLOR         = "#9b59b6"
HR_COLOR          = "#ff6b6b"
SPO2_COLOR        = "#4fc3f7"
RESP_COLOR        = "#ef9a9a"
BB_COLOR          = "#2ecc71"
CALORIES_COLOR    = "#f39c12"
DISTANCE_COLOR    = "#2ecc71"
DURATION_COLOR    = "#3498db"
HIGHLIGHT_COLOR   = "#e67e22"


# ── Matplotlib theme ──────────────────────────────────────────────────────────
THEME = {
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
}
FIG_BG = THEME["figure.facecolor"]


def apply_theme():
    """Apply the shared dark theme. Call once at import time in each script."""
    plt.rcParams.update(THEME)


# ── DB ────────────────────────────────────────────────────────────────────────
def connect_db(require_exists: bool = True, schema: str = None) -> sqlite3.Connection:
    """
    Open the analysis DB.

    require_exists — exit with a message if the DB isn't there (read-only scripts).
    schema         — DDL to run on connect (the fetcher, which creates the tables).
    """
    OUTPUT_DIR.mkdir(exist_ok=True)
    if require_exists and not DB_PATH.exists():
        print(f"ERROR: Database not found at {DB_PATH}")
        sys.exit(1)

    conn = sqlite3.connect(DB_PATH)
    if schema:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous  = NORMAL")
        conn.executescript(schema)
    return conn


# ── Chart helpers ─────────────────────────────────────────────────────────────
def save_fig(fig, name: str):
    p = OUTPUT_DIR / name
    fig.savefig(p, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"  → {p}")
    plt.close(fig)


def fmt_date_axis(ax, ticks=None, fontsize=None):
    """
    Format an x-axis of dates.

    ticks — pin one tick per value (a handful of sessions). Omit for a long
            daily series, where ticks fall on Mondays instead.
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    if ticks is not None:
        ax.set_xticks(ticks)
    else:
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0))
    extra = {"fontsize": fontsize} if fontsize else {}
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", **extra)


# ── Activity loading / stats ──────────────────────────────────────────────────
def load_activities(conn, type_pattern: str, label: str) -> pd.DataFrame:
    """Load activities whose type matches `type_pattern`, with derived columns."""
    df = pd.read_sql(
        "SELECT * FROM activities WHERE type LIKE ? ORDER BY date",
        conn,
        params=(type_pattern,),
    )
    if df.empty:
        print(f"No {label} activities found in the database.")
        sys.exit(0)
    df["date"] = pd.to_datetime(df["date"])
    df["duration_min"] = df["duration_h"] * 60
    # A zero-duration activity would blow up the division; leave those as NaN.
    df["speed_kmh"] = df["distance_km"] / df["duration_h"].where(df["duration_h"] > 0)
    return df


def add_trends(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Add {col}_trend / _slope / _r from a linear fit over the session index."""
    if len(df) < 2:
        return df
    x = np.arange(len(df))
    for col in cols:
        if col in df.columns and df[col].notna().sum() >= 2:
            slope, intercept, r, _p, _ = linregress(x, df[col].fillna(df[col].mean()))
            df[f"{col}_trend"] = intercept + slope * x
            df[f"{col}_slope"] = slope
            df[f"{col}_r"] = r
    return df


TREND_COLS = ["avg_hr", "max_hr", "calories", "duration_min", "distance_km", "speed_kmh"]


# ── Per-sport chart configuration ─────────────────────────────────────────────
@dataclass
class SportCharts:
    """What differs between the tennis and cycling chart pairs."""
    title: str                  # figure suptitle stem, e.g. "Cycling"
    unit: str                   # x-axis noun in the efficiency chart, e.g. "Ride"
    progression_png: str
    efficiency_png: str

    # Panel 2 of the progression chart (tennis: distance, cycling: speed)
    panel2_col: str
    panel2_ylabel: str
    panel2_title: str
    panel2_color: str = DISTANCE_COLOR

    # Optional boolean column that recolors panel-2 bars (tennis: "corrected")
    highlight_col: str = None

    duration_title: str = "Duration"

    # Third panel of the efficiency chart
    eff3_col: str = "speed_kmh"
    eff3_ylabel: str = "km/h"
    eff3_title: str = "Average Speed per Ride"
    eff3_color: str = DISTANCE_COLOR


def chart_progression(df: pd.DataFrame, cfg: SportCharts):
    """4-panel figure: heart rate, a sport-specific metric, calories, duration."""
    fig, axes = plt.subplots(4, 1, figsize=(12, 14), facecolor=FIG_BG)
    fig.suptitle(f"{cfg.title} Progression", fontsize=15, fontweight="bold", y=0.99)

    dates = df["date"]

    # ── Panel 1: Heart rate (avg + max)
    ax = axes[0]
    ax.plot(dates, df["avg_hr"], color=STRESS_COLOR, lw=2, marker="o", ms=7, label="Avg HR")
    ax.plot(dates, df["max_hr"], color="#ff9999", lw=1.5, marker="s", ms=5,
            linestyle="--", alpha=0.8, label="Max HR")
    if "avg_hr_trend" in df:
        ax.plot(dates, df["avg_hr_trend"], color="white", lw=1.2, ls=":", alpha=0.5, label="Trend")
    ax.set_ylabel("Heart Rate (bpm)")
    ax.set_title("Heart Rate", fontsize=11)
    ax.legend(fontsize=8, framealpha=0.2)
    fmt_date_axis(ax, ticks=dates, fontsize=8)

    # ── Panel 2: sport-specific metric
    ax2 = axes[1]
    if cfg.highlight_col and cfg.highlight_col in df.columns:
        colors = [HIGHLIGHT_COLOR if flag else cfg.panel2_color for flag in df[cfg.highlight_col]]
    else:
        colors = cfg.panel2_color
    ax2.bar(dates, df[cfg.panel2_col], color=colors, alpha=0.8, width=3)
    if f"{cfg.panel2_col}_trend" in df.columns:
        ax2.plot(dates, df[f"{cfg.panel2_col}_trend"], color="white", lw=1.5, ls=":", alpha=0.6)
    ax2.set_ylabel(cfg.panel2_ylabel)
    ax2.set_title(cfg.panel2_title, fontsize=11)
    fmt_date_axis(ax2, ticks=dates, fontsize=8)

    # ── Panel 3: Calories
    ax3 = axes[2]
    ax3.bar(dates, df["calories"], color=CALORIES_COLOR, alpha=0.75, width=3)
    if "calories_trend" in df:
        ax3.plot(dates, df["calories_trend"], color="white", lw=1.5, ls=":", alpha=0.6)
    ax3.set_ylabel("Calories")
    ax3.set_title("Calories Burned", fontsize=11)
    fmt_date_axis(ax3, ticks=dates, fontsize=8)

    # ── Panel 4: Duration
    ax4 = axes[3]
    ax4.bar(dates, df["duration_min"], color=DURATION_COLOR, alpha=0.75, width=3)
    ax4.axhline(df["duration_min"].mean(), color="white", lw=1, ls="--",
                alpha=0.5, label=f"Avg {df['duration_min'].mean():.0f} min")
    ax4.set_ylabel("Minutes")
    ax4.set_title(cfg.duration_title, fontsize=11)
    ax4.legend(fontsize=8, framealpha=0.2)
    fmt_date_axis(ax4, ticks=dates, fontsize=8)

    fig.tight_layout()
    save_fig(fig, cfg.progression_png)


def chart_efficiency(df: pd.DataFrame, cfg: SportCharts):
    """3-panel figure of per-session efficiency metrics."""
    df = df.copy()
    df["cal_per_min"]  = df["calories"] / df["duration_min"]
    df["dist_per_min"] = df["distance_km"] / df["duration_min"] * 1000  # metres/min
    df["session_n"]    = range(1, len(df) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor=FIG_BG)
    fig.suptitle(f"{cfg.title} Efficiency", fontsize=14, fontweight="bold")

    def _plot_metric(ax, y_col, color, ylabel, title):
        vals = df[y_col]
        ax.plot(df["session_n"], vals, color=color, lw=2, marker="o", ms=7)
        # linregress needs at least two points.
        if len(df) >= 2 and vals.notna().sum() >= 2:
            slope, intercept, *_ = linregress(df["session_n"], vals)
            ax.plot(df["session_n"], intercept + slope * df["session_n"],
                    color="white", lw=1.2, ls=":", alpha=0.6)
        for _, row in df.iterrows():
            ax.text(row["session_n"], row[y_col] + vals.std() * 0.1,
                    str(row["date"].date()), fontsize=6.5, ha="center", color="#c8cdd8")
        ax.set_xlabel(f"{cfg.unit} #")
        ax.set_ylabel(ylabel)
        ax.set_title(title, fontsize=11)

    _plot_metric(axes[0], "avg_hr",      STRESS_COLOR,   "Avg HR (bpm)", "Cardiovascular Load\n(↓ = more efficient)")
    _plot_metric(axes[1], "cal_per_min", CALORIES_COLOR, "Cal / min",    "Calorie Intensity per Minute")
    _plot_metric(axes[2], cfg.eff3_col,  cfg.eff3_color, cfg.eff3_ylabel, cfg.eff3_title)

    fig.tight_layout()
    save_fig(fig, cfg.efficiency_png)
