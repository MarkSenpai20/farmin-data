# FarmIn — Forecast Data Service

This is a tiny **public** companion repo for the FarmIn thesis app. A scheduled
GitHub Action runs the ARIMA pipeline and commits two static files:

- **`forecast.json`** — next 6 months of predicted farmgate prices
- **`history.json`** — the historical farmgate price series

The FarmIn app reads these files directly over their raw URL, so there is **no
always-on server and no cold start** — and it's completely free.

> Only the public PSA price dataset and the forecasting script live here. The
> main thesis app code stays in its own (private) repo.

## One-time setup

1. Create a new **public** GitHub repo (suggested name: `farmin-data`).
2. Push the contents of this folder to it:
   ```bash
   cd farmin-data-service
   git init -b main
   git add .
   git commit -m "FarmIn forecast data service"
   git remote add origin https://github.com/<your-username>/farmin-data.git
   git push -u origin main
   ```
3. In the new repo: **Settings → Actions → General → Workflow permissions** →
   select **Read and write permissions** → Save. (Lets the Action commit the
   refreshed JSON.)
4. Open the **Actions** tab → **Update FarmIn forecast** → **Run workflow** to
   generate the first refresh now (otherwise it waits for the daily schedule).
5. In the app, set `kForecastDataBase` (top of `lib/main.dart`) to:
   ```
   https://raw.githubusercontent.com/<your-username>/farmin-data/main
   ```

## Refresh schedule

The Action runs **daily at 21:00 UTC (05:00 PH)** and also on demand via
**Run workflow**. Edit the `cron:` line in
`.github/workflows/update-forecast.yml` to change the cadence.

## Run it locally

```bash
pip install -r requirements.txt
python generate_static_json.py     # writes forecast.json + history.json
```

## How it works

`generate_static_json.py` mirrors `ARIMA_MODEL/api.py`: it loads the baseline
PSA CSV, tries to scrape the latest Bantay Presyo price (falling back to a local
mock, then to CSV-only), trains `auto_arima` in memory, and writes the two JSON
files in the exact shape the app already parses.
