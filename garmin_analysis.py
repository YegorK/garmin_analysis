#!/usr/bin/env python3
"""
Garmin Connect Sleep & Activity Analyzer — with SQLite persistence
Pulls daily aggregates + intraday timeseries from Jan 1, 2026 onward.
Stores everything in a local SQLite DB and skips days already fetched.

Setup:
    pip install garminconnect matplotlib pandas numpy

Usage:
    python garmin_analysis.py            # fetch missing data, save, chart
    python garmin_analysis.py --refetch  # re-pull last 2 days (today might be partial)
    python garmin_analysis.py --charts   # skip fetching, just regenerate charts
    python garmin_analysis.py --status   # show what's in the DB and exit

DB schema (garmin_output/garmin.db):
    sleep, daily, hrv, activities             — one row per date (PK)
    heart_rate, steps, stress,
    spo2, respiration, body_battery           — one row per timestamp (PK)
    fetch_log                                 — tracks which (date, source) pairs are done
"""

import argparse
import datetime
import getpass
import sqlite3
import sys
import time
from pathlib import Path

try:
    import garmin_common as gc
    from garmin_common import (
        OUTPUT_DIR, DB_PATH, INSTALL_HINT,
        SLEEP_COLORS, STEPS_COLOR, STRESS_COLOR, HRV_COLOR,
        HR_COLOR, SPO2_COLOR, RESP_COLOR, BB_COLOR,
    )
except ImportError:
    print("ERROR: garmin_common.py not found.")
    print("       It must sit in the same folder as this script — copy it across")
    print("       together with the analysis scripts.")
    sys.exit(1)

try:
    import garminconnect
except ImportError:
    print("ERROR: missing library 'garminconnect'.")
    print(INSTALL_HINT)
    sys.exit(1)

try:
    import pandas as pd
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    import numpy as np
except ImportError as e:
    print(f"ERROR: missing library '{e.name or 'unknown'}'.")
    print(INSTALL_HINT)
    sys.exit(1)


# ── Config ────────────────────────────────────────────────────────────────────
START_DATE          = datetime.date(2026, 1, 1)
END_DATE            = datetime.date.today()
INTRADAY_CHART_DAYS = 7   # how many recent days to show in the intraday chart
REFETCH_TAIL_DAYS   = 2   # always re-pull the most recent N days (might be partial)

# All "sources" we track per day in fetch_log
DAILY_SOURCES    = ["sleep", "stats", "hrv"]
INTRADAY_SOURCES = ["heart_rate", "steps", "stress", "spo2", "respiration", "body_battery"]
ALL_SOURCES      = DAILY_SOURCES + INTRADAY_SOURCES


# ── DB layer ──────────────────────────────────────────────────────────────────
SCHEMA = """
CREATE TABLE IF NOT EXISTS sleep (
    date TEXT PRIMARY KEY,
    duration_h REAL, deep_h REAL, light_h REAL, rem_h REAL, awake_h REAL,
    score INTEGER, avg_spo2 REAL, avg_respiration REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS daily (
    date TEXT PRIMARY KEY,
    steps INTEGER, calories REAL, active_min INTEGER,
    stress_avg REAL, rhr INTEGER, floors REAL,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS hrv (
    date TEXT PRIMARY KEY,
    weekly_avg REAL, last_night REAL, status TEXT,
    fetched_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS activities (
    activity_id  INTEGER PRIMARY KEY,
    date         TEXT,
    type         TEXT,
    name         TEXT,
    duration_h   REAL,
    distance_km  REAL,
    calories     REAL,
    avg_hr       REAL,
    max_hr       REAL,
    fetched_at   TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS heart_rate   (timestamp TEXT PRIMARY KEY, date TEXT, heart_rate   REAL);
CREATE TABLE IF NOT EXISTS steps        (timestamp TEXT PRIMARY KEY, date TEXT, steps        REAL);
CREATE TABLE IF NOT EXISTS stress       (timestamp TEXT PRIMARY KEY, date TEXT, stress       REAL);
CREATE TABLE IF NOT EXISTS spo2         (timestamp TEXT PRIMARY KEY, date TEXT, spo2         REAL);
CREATE TABLE IF NOT EXISTS respiration  (timestamp TEXT PRIMARY KEY, date TEXT, respiration  REAL);
CREATE TABLE IF NOT EXISTS body_battery (timestamp TEXT PRIMARY KEY, date TEXT, body_battery REAL);

CREATE INDEX IF NOT EXISTS ix_hr_date    ON heart_rate(date);
CREATE INDEX IF NOT EXISTS ix_steps_date ON steps(date);
CREATE INDEX IF NOT EXISTS ix_stress_date ON stress(date);
CREATE INDEX IF NOT EXISTS ix_spo2_date  ON spo2(date);
CREATE INDEX IF NOT EXISTS ix_resp_date  ON respiration(date);
CREATE INDEX IF NOT EXISTS ix_bb_date    ON body_battery(date);

CREATE TABLE IF NOT EXISTS fetch_log (
    date         TEXT NOT NULL,
    source       TEXT NOT NULL,
    fetched_at   TEXT NOT NULL,
    rows_added   INTEGER DEFAULT 0,
    PRIMARY KEY (date, source)
);
"""


def db_connect() -> sqlite3.Connection:
    """Open the DB, creating it and its tables if this is a first run."""
    return gc.connect_db(require_exists=False, schema=SCHEMA)


def get_completed_dates(conn) -> dict[str, set[str]]:
    """Return {source: {date_iso, ...}} of (date, source) pairs already fetched."""
    cur = conn.execute("SELECT date, source FROM fetch_log")
    out: dict[str, set[str]] = {s: set() for s in ALL_SOURCES}
    for d, s in cur.fetchall():
        out.setdefault(s, set()).add(d)
    return out


def mark_done(conn, date_iso: str, source: str, rows: int = 0):
    conn.execute(
        "INSERT OR REPLACE INTO fetch_log(date, source, fetched_at, rows_added) "
        "VALUES (?, ?, ?, ?)",
        (date_iso, source, datetime.datetime.now().isoformat(timespec="seconds"), rows),
    )


def clear_recent(conn, days: int):
    """Drop fetch_log entries for the last N days so they get re-fetched."""
    cutoff = (END_DATE - datetime.timedelta(days=days)).isoformat()
    conn.execute("DELETE FROM fetch_log WHERE date >= ?", (cutoff,))


# ── Auth ──────────────────────────────────────────────────────────────────────
TOKEN_DIR = str(Path.home() / ".garminconnect")


def login() -> garminconnect.Garmin:
    """
    The library handles token caching automatically when you pass a path to
    client.login(). On first run it asks for credentials, then saves tokens.
    On subsequent runs it loads the saved tokens silently — no rate-limit risk.

    Tokens are stored in ~/.garminconnect/ and last about a year.
    """
    # Check if we already have cached tokens
    token_path = Path(TOKEN_DIR)
    has_cached = token_path.exists() and any(token_path.iterdir())

    if has_cached:
        # Resume from cache — no credentials needed
        try:
            client = garminconnect.Garmin()
            client.login(TOKEN_DIR)
            print(f"✓ Resumed session from cached token ({TOKEN_DIR})\n")
            return client
        except Exception as e:
            print(f"  Cached token failed ({e}); falling back to fresh login.")

    # Fresh login — credentials needed once. The library will save tokens
    # to TOKEN_DIR automatically as part of login().
    print("\n── Garmin Connect Login ──")
    print("(token will be cached so you don't need to log in next time)")
    email    = input("Email: ").strip()
    password = getpass.getpass("Password: ")

    try:
        client = garminconnect.Garmin(
            email=email,
            password=password,
            prompt_mfa=lambda: input("MFA code: "),
        )
        # Passing the directory makes the library save tokens after login
        client.login(TOKEN_DIR)
        print(f"✓ Logged in as {email}")
        print(f"  Token cached to {TOKEN_DIR} — future runs won't need credentials.\n")
        return client
    except garminconnect.GarminConnectAuthenticationError:
        print("ERROR: Authentication failed — check your credentials.")
        sys.exit(1)
    except garminconnect.GarminConnectTooManyRequestsError:
        print("ERROR: Garmin is rate-limiting your IP (HTTP 429).")
        print("       Wait 30–60 minutes before trying again.")
        sys.exit(1)
    except Exception as e:
        msg = str(e)
        if "429" in msg or "403" in msg or "rate limit" in msg.lower():
            print("\nERROR: Garmin is rate-limiting your IP (HTTP 429/403).")
            print("       This happens when too many login attempts come from your IP.")
            print("       What to do:")
            print("         1. Wait 30–60 minutes")
            print("         2. Try logging into connect.garmin.com in a browser first")
            print("            (this clears the Cloudflare flag on your IP)")
            print("         3. If you have a VPN, switch to a different IP and try again")
            print("         4. After a successful login, the token is cached for ~1 year")
            sys.exit(1)
        print(f"ERROR: Login failed: {e}")
        sys.exit(1)


# ── Helpers ───────────────────────────────────────────────────────────────────
def daterange(start: datetime.date, end: datetime.date):
    d = start
    while d <= end:
        yield d
        d += datetime.timedelta(days=1)


def parse_ts(value):
    """Convert epoch-ms or ISO string → ISO 8601 UTC string. None on failure."""
    if value is None:
        return None
    try:
        if isinstance(value, (int, float)):
            return pd.Timestamp(int(value), unit="ms", tz="UTC").isoformat()
        ts = pd.Timestamp(value)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts.isoformat()
    except Exception:
        return None


def now_iso() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ── Daily fetchers (one source per call so we can mark each independently) ────
def fetch_sleep(client, conn, ds: str) -> int:
    s = client.get_sleep_data(ds)
    if s and s.get("dailySleepDTO"):
        dto = s["dailySleepDTO"]
        conn.execute(
            "INSERT OR REPLACE INTO sleep VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                ds,
                (dto.get("sleepTimeSeconds")  or 0) / 3600,
                (dto.get("deepSleepSeconds")  or 0) / 3600,
                (dto.get("lightSleepSeconds") or 0) / 3600,
                (dto.get("remSleepSeconds")   or 0) / 3600,
                (dto.get("awakeSleepSeconds") or 0) / 3600,
                (dto.get("sleepScores") or {}).get("overall", {}).get("value"),
                dto.get("averageSpO2Value"),
                dto.get("averageRespirationValue"),
                now_iso(),
            ),
        )
        return 1
    return 0


def fetch_stats(client, conn, ds: str) -> int:
    st = client.get_stats(ds)
    if st:
        conn.execute(
            "INSERT OR REPLACE INTO daily VALUES (?,?,?,?,?,?,?,?)",
            (
                ds,
                st.get("totalSteps", 0),
                st.get("totalKilocalories", 0),
                (st.get("highlyActiveSeconds") or 0) // 60,
                st.get("averageStressLevel"),
                st.get("restingHeartRate"),
                st.get("floorsAscended", 0),
                now_iso(),
            ),
        )
        return 1
    return 0


def fetch_hrv(client, conn, ds: str) -> int:
    h = client.get_hrv_data(ds)
    if h and h.get("hrvSummary"):
        sv = h["hrvSummary"]
        conn.execute(
            "INSERT OR REPLACE INTO hrv VALUES (?,?,?,?,?)",
            (ds, sv.get("weeklyAvg"), sv.get("lastNightAvg"), sv.get("status"), now_iso()),
        )
        return 1
    return 0


def fetch_activities_bulk(client, conn) -> int:
    """Single bulk call covering the entire range. Always overwrites."""
    try:
        acts = client.get_activities_by_date(START_DATE.isoformat(), END_DATE.isoformat())
    except Exception as e:
        print(f"      Warning: activities failed: {e}")
        return 0
    n = 0
    for a in (acts or []):
        ts = a.get("startTimeLocal", "")
        conn.execute(
            "INSERT OR REPLACE INTO activities VALUES (?,?,?,?,?,?,?,?,?,?)",
            (
                a.get("activityId"),
                ts[:10] if ts else None,
                a.get("activityType", {}).get("typeKey", "unknown"),
                a.get("activityName", ""),
                (a.get("duration") or 0) / 3600,
                (a.get("distance") or 0) / 1000,
                a.get("calories", 0),
                a.get("averageHR"),
                a.get("maxHR"),
                now_iso(),
            ),
        )
        n += 1
    return n


# ── Intraday fetchers ─────────────────────────────────────────────────────────
def _insert_many(conn, table: str, rows: list, ds: str) -> int:
    if not rows:
        return 0
    conn.executemany(
        f"INSERT OR REPLACE INTO {table} VALUES (?, ?, ?)",
        [(r[0], ds, r[1]) for r in rows],
    )
    return len(rows)


def fetch_heart_rate(client, conn, ds: str) -> int:
    raw = client.get_heart_rates(ds)
    rows = []
    # raw is usually a dict with 'heartRateValues'; sometimes the list directly
    values = raw.get("heartRateValues") if isinstance(raw, dict) else (raw or [])
    for r in (values or []):
        if isinstance(r, (list, tuple)) and len(r) >= 2 and r[1] is not None:
            ts = parse_ts(r[0])
            if ts:
                rows.append((ts, r[1]))
    return _insert_many(conn, "heart_rate", rows, ds)


def fetch_steps(client, conn, ds: str) -> int:
    raw = client.get_steps_data(ds)
    rows = []
    # raw can be a list of buckets, or a dict wrapping one
    buckets = raw if isinstance(raw, list) else (raw.get("stepsValues") or raw.get("values") or [])
    for b in (buckets or []):
        if not isinstance(b, dict):
            continue
        ts    = parse_ts(b.get("startGMT") or b.get("startTimestamp") or b.get("startTimeGMT"))
        steps = b.get("steps")
        if ts and steps is not None:
            rows.append((ts, steps))
    return _insert_many(conn, "steps", rows, ds)


def fetch_stress(client, conn, ds: str) -> int:
    raw = client.get_all_day_stress(ds)
    rows = []
    values = raw.get("stressValuesArray") if isinstance(raw, dict) else (raw or [])
    for v in (values or []):
        if isinstance(v, (list, tuple)) and len(v) >= 2 and v[1] is not None and v[1] >= 0:
            ts = parse_ts(v[0])
            if ts:
                rows.append((ts, v[1]))
    return _insert_many(conn, "stress", rows, ds)


def fetch_spo2(client, conn, ds: str) -> int:
    raw = client.get_spo2_data(ds)
    # raw can be:
    #   - a list of reading dicts directly
    #   - a dict containing 'spO2HourlyAverages' or 'continuousReadingDTOList'
    if isinstance(raw, list):
        readings = raw
    elif isinstance(raw, dict):
        readings = (
            raw.get("spO2HourlyAverages")
            or raw.get("continuousReadingDTOList")
            or raw.get("spO2DailyValues")
            or []
        )
    else:
        readings = []

    rows = []
    for r in (readings or []):
        if isinstance(r, dict):
            ts  = parse_ts(
                r.get("startGMT") or r.get("readingTimeGMT") or r.get("epochTimestamp")
            )
            val = (
                r.get("averageSpO2")
                or r.get("spO2Reading")
                or r.get("value")
                or r.get("spo2")
            )
            if ts and val:
                rows.append((ts, val))
        elif isinstance(r, (list, tuple)) and len(r) >= 2 and r[1] is not None:
            ts = parse_ts(r[0])
            if ts:
                rows.append((ts, r[1]))
    return _insert_many(conn, "spo2", rows, ds)


def fetch_respiration(client, conn, ds: str) -> int:
    raw = client.get_respiration_data(ds)
    rows = []
    values = raw.get("respirationValuesArray") if isinstance(raw, dict) else (raw or [])
    for r in (values or []):
        if isinstance(r, (list, tuple)) and len(r) >= 2 and r[1] is not None:
            ts = parse_ts(r[0])
            if ts:
                rows.append((ts, r[1]))
    return _insert_many(conn, "respiration", rows, ds)


def fetch_body_battery(client, conn, ds: str) -> int:
    raw = client.get_body_battery(ds, ds)
    rows = []
    items = raw if isinstance(raw, list) else []
    for item in items:
        if not isinstance(item, dict):
            continue
        ts  = parse_ts(item.get("startTimestampGMT") or item.get("date"))
        val = item.get("charged") or item.get("bodyBatteryLevel")
        if ts and val is not None:
            rows.append((ts, val))
    return _insert_many(conn, "body_battery", rows, ds)


# Map source name → fetcher
FETCHERS = {
    "sleep":         fetch_sleep,
    "stats":         fetch_stats,
    "hrv":           fetch_hrv,
    "heart_rate":    fetch_heart_rate,
    "steps":         fetch_steps,
    "stress":        fetch_stress,
    "spo2":          fetch_spo2,
    "respiration":   fetch_respiration,
    "body_battery":  fetch_body_battery,
}


# ── Main fetch loop ───────────────────────────────────────────────────────────
def incremental_fetch(client, conn):
    completed = get_completed_dates(conn)
    all_days  = list(daterange(START_DATE, END_DATE))

    # Plan: list of (date_iso, source) still missing
    plan = [
        (d.isoformat(), src)
        for d in all_days
        for src in ALL_SOURCES
        if d.isoformat() not in completed.get(src, set())
    ]

    if not plan:
        print(f"✓ Database already up-to-date through {END_DATE}.")
        # Refresh activities anyway — it's a single bulk call
        n = fetch_activities_bulk(client, conn)
        conn.commit()
        if n:
            print(f"  (refreshed {n} activities)")
        return

    # Group by source for nicer progress output
    by_source: dict[str, list[str]] = {}
    for d, s in plan:
        by_source.setdefault(s, []).append(d)

    total_calls = len(plan)
    print(f"Fetching {total_calls} missing data points across {len(by_source)} sources…")
    print(f"  (already have data through completed days; estimated {total_calls * 0.3 / 60:.1f} min)\n")

    done_calls = 0
    for source, dates in by_source.items():
        fetcher = FETCHERS[source]
        print(f"  ▸ {source}  ({len(dates)} days)")
        for ds in dates:
            done_calls += 1
            pct = int(done_calls / total_calls * 100)
            print(f"\r    [{pct:3d}%] {ds}                ", end="", flush=True)
            try:
                rows = fetcher(client, conn, ds)
                mark_done(conn, ds, source, rows)
            except Exception as e:
                # Don't mark as done — we'll retry next run
                print(f"\n    ! {ds}/{source}: {e}")
            time.sleep(0.25)
            # commit every 20 calls so a crash doesn't lose much
            if done_calls % 20 == 0:
                conn.commit()
        print()  # newline after each source
        conn.commit()

    # Activities (always bulk)
    print("  ▸ activities (bulk)")
    n = fetch_activities_bulk(client, conn)
    conn.commit()
    print(f"    ✓ {n} activities loaded.\n")


# ── Read DB → DataFrames for charting ─────────────────────────────────────────
def load_dataframes(conn) -> tuple[dict, dict]:
    daily = {
        "sleep":      pd.read_sql("SELECT * FROM sleep      ORDER BY date", conn),
        "daily":      pd.read_sql("SELECT * FROM daily      ORDER BY date", conn),
        "hrv":        pd.read_sql("SELECT * FROM hrv        ORDER BY date", conn),
        "activities": pd.read_sql("SELECT * FROM activities ORDER BY date", conn),
    }
    for k in ("sleep", "daily", "hrv", "activities"):
        if not daily[k].empty and "date" in daily[k].columns:
            daily[k]["date"] = pd.to_datetime(daily[k]["date"]).dt.date

    intra = {}
    for tbl in INTRADAY_SOURCES:
        df = pd.read_sql(f"SELECT timestamp, {tbl} FROM {tbl} ORDER BY timestamp", conn)
        if not df.empty:
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.dropna(subset=["timestamp"])
        intra[tbl] = df
    return daily, intra


# ── Matplotlib theme ──────────────────────────────────────────────────────────
gc.apply_theme()


def _rolling(s, w=7):
    return s.rolling(w, min_periods=1).mean()


# ── Daily charts ──────────────────────────────────────────────────────────────
def chart_sleep(df):
    if df.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f1117")
    fig.suptitle("Sleep Analysis — Jan 2026 onwards", fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.bar(df["date"], df["deep_h"],  label="Deep",  color=SLEEP_COLORS["deep"],  width=0.8)
    ax.bar(df["date"], df["rem_h"],   label="REM",   color=SLEEP_COLORS["rem"],   width=0.8, bottom=df["deep_h"])
    ax.bar(df["date"], df["light_h"], label="Light", color=SLEEP_COLORS["light"], width=0.8, bottom=df["deep_h"]+df["rem_h"])
    ax.bar(df["date"], df["awake_h"], label="Awake", color=SLEEP_COLORS["awake"], width=0.8, bottom=df["deep_h"]+df["rem_h"]+df["light_h"])
    ax.plot(df["date"], _rolling(df["duration_h"]), color="white", lw=1.5, label="7d avg", zorder=5)
    ax.set_ylabel("Hours"); ax.set_title("Sleep Duration & Stages", fontsize=12)
    ax.legend(loc="upper right", framealpha=0.2, fontsize=8); gc.fmt_date_axis(ax)

    ax2 = axes[1]
    sd = df.dropna(subset=["score"])
    if not sd.empty:
        ax2.scatter(sd["date"], sd["score"], c=sd["score"], cmap="RdYlGn", vmin=0, vmax=100, s=40, zorder=5)
        ax2.plot(sd["date"], _rolling(sd["score"]), color="#f0c040", lw=1.5, label="7d avg")
        ax2.set_ylim(0, 100); ax2.axhline(70, color="#aaa", ls="--", lw=0.8, alpha=0.5)
        ax2.set_ylabel("Score"); ax2.set_title("Sleep Score", fontsize=12)
        ax2.legend(framealpha=0.2, fontsize=8); gc.fmt_date_axis(ax2)

    ax3 = axes[2]
    s2 = df.dropna(subset=["avg_spo2"])
    if not s2.empty:
        ax3.plot(s2["date"], s2["avg_spo2"], color=SPO2_COLOR, lw=1.2, label="Avg SpO₂ (%)")
        ax3.set_ylabel("SpO₂ (%)", color=SPO2_COLOR); ax3.set_ylim(90, 100)
    r2 = df.dropna(subset=["avg_respiration"])
    if not r2.empty:
        ax3b = ax3.twinx()
        ax3b.plot(r2["date"], r2["avg_respiration"], color=RESP_COLOR, lw=1.2)
        ax3b.set_ylabel("Breaths/min", color=RESP_COLOR); ax3b.tick_params(colors=RESP_COLOR)
        ax3b.spines["right"].set_visible(True); ax3b.spines["right"].set_color("#2a2f3e")
    ax3.set_title("SpO₂ & Respiration Rate", fontsize=12); gc.fmt_date_axis(ax3)

    fig.tight_layout(); gc.save_fig(fig, "sleep_analysis.png")


def chart_activity(df_daily, df_acts):
    if df_daily.empty:
        return
    fig, axes = plt.subplots(3, 1, figsize=(14, 12), facecolor="#0f1117")
    fig.suptitle("Activity Analysis — Jan 2026 onwards", fontsize=16, fontweight="bold", y=0.98)

    ax = axes[0]
    ax.bar(df_daily["date"], df_daily["steps"], color=STEPS_COLOR, alpha=0.7, width=0.8)
    ax.plot(df_daily["date"], _rolling(df_daily["steps"]), color="white", lw=1.5, label="7d avg")
    ax.axhline(10000, color="#f0c040", ls="--", lw=0.8, alpha=0.7, label="10k goal")
    ax.set_ylabel("Steps"); ax.set_title("Daily Steps", fontsize=12)
    ax.legend(framealpha=0.2, fontsize=8); gc.fmt_date_axis(ax)

    ax2 = axes[1]
    sd = df_daily.dropna(subset=["stress_avg"])
    if not sd.empty:
        ax2.fill_between(sd["date"], sd["stress_avg"], alpha=0.4, color=STRESS_COLOR)
        ax2.plot(sd["date"], _rolling(sd["stress_avg"]), color=STRESS_COLOR, lw=1.5, label="Avg Stress")
        ax2.set_ylabel("Stress Level", color=STRESS_COLOR); ax2.set_ylim(0, 100)
    rd = df_daily.dropna(subset=["rhr"])
    if not rd.empty:
        ax2b = ax2.twinx()
        ax2b.plot(rd["date"], rd["rhr"], color="#80deea", lw=1.5, marker="o", ms=2, label="RHR (bpm)")
        ax2b.set_ylabel("RHR (bpm)", color="#80deea"); ax2b.tick_params(colors="#80deea")
        ax2b.spines["right"].set_visible(True); ax2b.spines["right"].set_color("#2a2f3e")
    ax2.set_title("Stress & Resting Heart Rate", fontsize=12); gc.fmt_date_axis(ax2)

    ax3 = axes[2]
    if not df_acts.empty:
        ac = df_acts.groupby("type").size().sort_values(ascending=True)
        ax3.barh(ac.index, ac.values, color=plt.cm.Set2(np.linspace(0, 1, len(ac))))
        ax3.set_xlabel("# of Activities"); ax3.set_title("Activity Types", fontsize=12)
        for j, v in enumerate(ac.values):
            ax3.text(v + 0.1, j, str(v), va="center", fontsize=9)
    else:
        ax3.text(0.5, 0.5, "No activity data", ha="center", va="center", transform=ax3.transAxes)

    fig.tight_layout(); gc.save_fig(fig, "activity_analysis.png")


def chart_hrv(df):
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f1117")
    fig.suptitle("HRV — Jan 2026 onwards", fontsize=16, fontweight="bold")
    ln = df.dropna(subset=["last_night"])
    wa = df.dropna(subset=["weekly_avg"])
    if not ln.empty:
        ax.bar(ln["date"], ln["last_night"], color=HRV_COLOR, alpha=0.5, width=0.8, label="Last Night")
    if not wa.empty:
        ax.plot(wa["date"], wa["weekly_avg"], color="white", lw=2, label="Weekly Avg")
    ax.set_ylabel("HRV (ms)"); ax.legend(framealpha=0.2); gc.fmt_date_axis(ax)
    fig.tight_layout(); gc.save_fig(fig, "hrv_analysis.png")


# ── Intraday chart ────────────────────────────────────────────────────────────
def chart_intraday(intra: dict):
    cutoff = pd.Timestamp(END_DATE - datetime.timedelta(days=INTRADAY_CHART_DAYS), tz="UTC")
    signals = [
        ("heart_rate",   "heart_rate",   HR_COLOR,     "bpm",         "💓 Heart Rate"),
        ("steps",        "steps",        STEPS_COLOR,  "steps",       "👟 Steps (15-min buckets)"),
        ("stress",       "stress",       STRESS_COLOR, "0–100",       "😰 Stress Level"),
        ("body_battery", "body_battery", BB_COLOR,     "0–100",       "🔋 Body Battery"),
        ("spo2",         "spo2",         SPO2_COLOR,   "%",           "🫁 Blood Oxygen (SpO₂)"),
        ("respiration",  "respiration",  RESP_COLOR,   "breaths/min", "💨 Respiration Rate"),
    ]
    available = [(k, col, c, yl, t) for k, col, c, yl, t in signals
                 if k in intra and not intra[k].empty]
    if not available:
        print("  No intraday data to chart."); return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(16, 3.8 * n), facecolor="#0f1117")
    if n == 1:
        axes = [axes]

    window_start = (END_DATE - datetime.timedelta(days=INTRADAY_CHART_DAYS)).isoformat()
    fig.suptitle(
        f"Intraday Signals — last {INTRADAY_CHART_DAYS} days  ({window_start} → {END_DATE})",
        fontsize=14, fontweight="bold", y=1.005,
    )

    for ax, (key, col, color, ylabel, title) in zip(axes, available):
        df = intra[key].copy()
        df = df[df["timestamp"] >= cutoff]
        if df.empty:
            ax.text(0.5, 0.5, "No data in window", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title, fontsize=11); continue

        t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
        day0 = t_min.normalize()

        # Night shading midnight–7am
        d = day0
        while d <= t_max:
            ax.axvspan(d, d + pd.Timedelta(hours=7), color="#ffffff", alpha=0.03, linewidth=0)
            d += pd.Timedelta(days=1)
        # Day separators
        for d in pd.date_range(day0, t_max.normalize() + pd.Timedelta(days=1), freq="D"):
            ax.axvline(d, color="#2a2f3e", lw=0.8, ls="--", zorder=1)

        if key == "steps":
            ax.bar(df["timestamp"], df[col], width=pd.Timedelta(minutes=13),
                   color=color, alpha=0.75, zorder=2)
        else:
            ax.plot(df["timestamp"], df[col], color=color, lw=0.7, alpha=0.7, zorder=2)
            if len(df) > 15:
                smooth = df[col].rolling(15, min_periods=1, center=True).mean()
                ax.plot(df["timestamp"], smooth, color="white", lw=1.6, alpha=0.55, zorder=3)

        if key == "spo2":
            ax.set_ylim(88, 100)
        elif key in ("stress", "body_battery"):
            ax.set_ylim(0, 100)

        ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(title, fontsize=11)
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_minor_locator(mdates.HourLocator(byhour=[6, 12, 18]))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
        ax.tick_params(axis="x", which="minor", length=3, color="#3a4050")
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=0, ha="center", fontsize=8)

    fig.tight_layout()
    gc.save_fig(fig, "intraday_sample.png")


# ── Status report ─────────────────────────────────────────────────────────────
def show_status(conn):
    print("\n" + "═"*60)
    print(f"  📊  GARMIN DB STATUS  |  {DB_PATH}")
    print("═"*60)

    counts = {}
    for tbl in ["sleep", "daily", "hrv", "activities"] + INTRADAY_SOURCES:
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        counts[tbl] = n

    print("\nDaily aggregates:")
    for tbl in ["sleep", "daily", "hrv", "activities"]:
        rng = conn.execute(f"SELECT MIN(date), MAX(date) FROM {tbl}").fetchone()
        rng_str = f"{rng[0]} → {rng[1]}" if rng[0] else "—"
        print(f"  {tbl:<14} {counts[tbl]:>6,} rows   {rng_str}")

    print("\nIntraday timeseries:")
    for tbl in INTRADAY_SOURCES:
        rng = conn.execute(
            f"SELECT MIN(date), MAX(date) FROM {tbl}"
        ).fetchone()
        rng_str = f"{rng[0]} → {rng[1]}" if rng[0] else "—"
        print(f"  {tbl:<14} {counts[tbl]:>7,} rows   {rng_str}")

    completed = get_completed_dates(conn)
    total_target  = (END_DATE - START_DATE).days + 1
    print(f"\nFetch completion (out of {total_target} days, {START_DATE} → {END_DATE}):")
    for src in ALL_SOURCES:
        done = len(completed.get(src, set()))
        pct  = done / total_target * 100
        bar  = "█" * int(pct / 5) + "·" * (20 - int(pct / 5))
        print(f"  {src:<14} [{bar}] {done:>3}/{total_target}  {pct:5.1f}%")

    print("\n" + "═"*60 + "\n")


# ── Summary printout ──────────────────────────────────────────────────────────
def print_summary(daily, intra):
    df_sleep, df_daily, df_hrv, df_acts = daily["sleep"], daily["daily"], daily["hrv"], daily["activities"]

    print("\n" + "═"*60)
    print(f"  📊  GARMIN SUMMARY  |  {START_DATE} → {END_DATE}")
    print("═"*60)

    if not df_sleep.empty:
        avg_sleep = df_sleep["duration_h"].mean()
        avg_deep  = df_sleep["deep_h"].mean()
        avg_rem   = df_sleep["rem_h"].mean()
        avg_score = df_sleep["score"].dropna().mean() if "score" in df_sleep else None
        print(f"\n💤 SLEEP  ({len(df_sleep)} nights)")
        print(f"   Avg total  : {avg_sleep:.1f} h")
        print(f"   Avg deep   : {avg_deep:.1f} h  ({avg_deep/avg_sleep*100:.0f}%)")
        print(f"   Avg REM    : {avg_rem:.1f} h  ({avg_rem/avg_sleep*100:.0f}%)")
        if avg_score:
            print(f"   Avg score  : {avg_score:.0f}/100")

    if not df_daily.empty:
        avg_steps  = df_daily["steps"].mean()
        days_10k   = (df_daily["steps"] >= 10000).sum()
        avg_stress = df_daily["stress_avg"].dropna().mean() if "stress_avg" in df_daily else None
        avg_rhr    = df_daily["rhr"].dropna().mean() if "rhr" in df_daily else None
        print(f"\n🏃 ACTIVITY  ({len(df_daily)} days)")
        print(f"   Avg steps/day   : {avg_steps:,.0f}")
        print(f"   Days ≥ 10k steps: {days_10k}")
        if avg_stress: print(f"   Avg stress      : {avg_stress:.0f}/100")
        if avg_rhr:    print(f"   Avg resting HR  : {avg_rhr:.0f} bpm")

    if not df_hrv.empty:
        avg_hrv = df_hrv["last_night"].dropna().mean()
        if avg_hrv: print(f"\n💓 HRV\n   Avg nightly HRV : {avg_hrv:.0f} ms")

    if not df_acts.empty:
        print(f"\n🏋️  ACTIVITIES  ({len(df_acts)} total)")
        for t, cnt in df_acts["type"].value_counts().head(5).items():
            print(f"   {t:<26}: {cnt}")

    print("\n📈 INTRADAY TIMESERIES")
    for key, df in intra.items():
        if not df.empty:
            span = (df["timestamp"].max() - df["timestamp"].min()).days
            res  = (df["timestamp"].diff().dropna().median().total_seconds() / 60)
            print(f"   {key:<14}: {len(df):>7,} pts  ~{res:.0f}-min res  ({span}d span)")
        else:
            print(f"   {key:<14}: no data")

    print("\n" + "═"*60)
    print(f"  DB:     ./{DB_PATH}")
    print(f"  Charts: ./{OUTPUT_DIR}/*.png")
    print("═"*60 + "\n")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Garmin Connect → SQLite analyzer")
    parser.add_argument("--status",  action="store_true",
                        help="show DB contents and exit (no fetching, no charts)")
    parser.add_argument("--charts",  action="store_true",
                        help="skip fetching, just regenerate charts from existing DB")
    parser.add_argument("--refetch", type=int, metavar="N", default=REFETCH_TAIL_DAYS,
                        help=f"re-pull last N days even if marked done (default: {REFETCH_TAIL_DAYS})")
    parser.add_argument("--full-refetch", action="store_true",
                        help="wipe fetch_log and re-pull EVERYTHING (data is upserted)")
    args = parser.parse_args()

    conn = db_connect()

    if args.status:
        show_status(conn)
        return

    if args.full_refetch:
        print("⚠  Wiping fetch_log — every day will be re-pulled.")
        conn.execute("DELETE FROM fetch_log")
        conn.commit()
    elif args.refetch and not args.charts:
        clear_recent(conn, args.refetch)
        conn.commit()
        print(f"  (will re-pull last {args.refetch} days)")

    if not args.charts:
        client = login()
        incremental_fetch(client, conn)

    # Always: load → chart → summarize
    daily, intra = load_dataframes(conn)
    print("Generating charts…")
    chart_sleep(daily["sleep"])
    chart_activity(daily["daily"], daily["activities"])
    chart_hrv(daily["hrv"])
    chart_intraday(intra)
    print_summary(daily, intra)

    conn.close()


if __name__ == "__main__":
    main()

