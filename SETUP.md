# Garmin Analysis — Setup Guide

These scripts pull your health and activity data from Garmin Connect and generate charts and reports. Here's how to get everything running on your Windows PC.

---

## What you need before starting

- A Windows 10 or 11 PC
- A Garmin Connect account (the same one you use in the Garmin Connect app)
- A Garmin device that syncs to Garmin Connect (Forerunner, Fenix, Venu, etc.)
- Basic comfort with typing commands

---

## Step 1 — Open PowerShell

Press **Win + S**, type `PowerShell`, and press Enter.

All commands below are typed into PowerShell. Do not use Command Prompt — PowerShell is required.

---

## Step 2 — Install Python 3

Open your browser and go to **python.org/downloads**. Download the latest Python 3.12 installer (the big yellow button).

Run the installer. **Important:** On the first screen, tick the box that says **"Add Python to PATH"** before clicking Install Now. Without this, nothing will work.

After installation, verify it worked:

```powershell
python --version
```

You should see `Python 3.12.x`. If you get "not recognised", restart PowerShell and try again.

---

## Step 3 — Get the scripts

Copy the script files Yegor sent you (`garmin_analysis.py`, `tennis_analysis.py`, `sleep_stress_analysis.py`) into a folder. For example, create a folder on your Desktop:

```powershell
mkdir "$env:USERPROFILE\Desktop\garmin_analysis"
```

Then move the files there (drag them in File Explorer, or copy-paste). All commands from here on assume you're working in that folder:

```powershell
cd "$env:USERPROFILE\Desktop\garmin_analysis"
```

---

## Step 4 — Allow PowerShell scripts to run

Windows blocks scripts by default. Run this once to allow them:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Type `Y` and press Enter when prompted.

---

## Step 5 — Create a virtual environment

A virtual environment keeps these scripts' dependencies separate from everything else on your PC:

```powershell
python -m venv .venv
```

Then activate it:

```powershell
.venv\Scripts\Activate.ps1
```

Your prompt will change to show `(.venv)` at the start — this means the environment is active. **You'll need to run this activation command every time you open a new PowerShell window before running the scripts.**

---

## Step 6 — Install dependencies

With the virtual environment active, install all required libraries:

```powershell
pip install garminconnect pandas numpy matplotlib scipy
```

This will take a minute or two. You should see a list of packages being installed.

---

## Step 7 — Run the main analysis script

```powershell
python garmin_analysis.py
```

**First run:** The script will ask for your Garmin Connect email and password. This is the same login you use in the Garmin Connect app.

```
── Garmin Connect Login ──
Email: you@example.com
Password:          ← won't show as you type, that's normal
```

If you have two-factor authentication (2FA) enabled on your Garmin account, it will also ask for your MFA code.

> **Note:** Your login token is saved automatically to `C:\Users\YourName\.garminconnect\` and lasts about a year — you won't be asked for credentials again on future runs.

**First-run fetch time:** The script fetches data from January 1, 2026 onwards. Depending on how much data you have, this can take **20–60 minutes**. It will show progress as it goes. Subsequent runs only fetch new days and are much faster.

When done, charts are saved to a `garmin_output\` folder in the same directory as the scripts.

---

## Step 8 — Run the tennis analysis script

This script analyses your tennis sessions specifically. Run it after `garmin_analysis.py` has fetched your data at least once:

```powershell
python tennis_analysis.py
```

It reads from the same local database — no additional Garmin login needed.

---

## Common commands

```powershell
# Fetch new data and regenerate all charts (normal daily use)
python garmin_analysis.py

# Regenerate charts only, without fetching new data
python garmin_analysis.py --charts

# Show what data is in the database
python garmin_analysis.py --status

# Tennis analysis — full with charts
python tennis_analysis.py

# Tennis analysis — summary only, no charts
python tennis_analysis.py --no-charts

# Sleep and stress correlation analysis
python sleep_stress_analysis.py
```

---

## Where to find your charts

All output files are in the `garmin_output\` folder (inside your scripts folder):

| File | What it shows |
|------|--------------|
| `sleep_analysis.png` | Sleep duration, stages, score, SpO₂ |
| `activity_analysis.png` | Steps, stress, resting heart rate |
| `hrv_analysis.png` | Heart rate variability over time |
| `intraday_sample.png` | Last 7 days of intraday signals |
| `tennis_progression.png` | Heart rate, distance, calories across sessions |
| `tennis_efficiency.png` | Fitness trends per tennis session |
| `sleep_stress_report.txt` | Written correlation analysis report |

---

## Troubleshooting

**"python is not recognized as a command"**
You likely didn't tick "Add Python to PATH" during installation. Uninstall Python, run the installer again, and make sure to tick that box on the first screen.

**"running scripts is disabled on this system"**
Run this in PowerShell and try again:
```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**"No module named pandas" or similar**
Make sure the virtual environment is active — your prompt should show `(.venv)`. If not, run:
```powershell
.venv\Scripts\Activate.ps1
```
Then re-run the pip install step.

**Garmin rate limit error (HTTP 429)**
Garmin temporarily blocks too many login attempts. Wait 30–60 minutes, then log into connect.garmin.com in your browser first (this clears the block), then run the script again.

**Script stops partway through fetching**
Just run it again — it saves progress as it goes and will resume from where it left off.

**Charts open briefly then disappear**
That's normal — charts are saved as image files, not displayed on screen. Open the `garmin_output\` folder in File Explorer to view them.

---

## Every time you want to run the scripts

```powershell
cd "$env:USERPROFILE\Desktop\garmin_analysis"   # go to the folder
.venv\Scripts\Activate.ps1                       # activate the environment
python garmin_analysis.py                         # run
```
