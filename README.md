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

`generate_static_json.py` mirrors `ARIMA_MODEL/api.py`. Each run:

1. **Pulls the farmgate series from the PSA OpenSTAT API** (`openstat.py`) —
   Palay, Ilocos Norte, monthly. Falls back to the bundled CSV only if the API
   is unreachable. The CSV is a frozen snapshot and goes stale; the API does not.
2. **Trains `auto_arima` in memory**, seasonal with `m=12`.
3. **Scrapes DA Bantay Presyo** for today's retail prices — every grade, every
   reporting market in Region I.
4. **Pulls the wholesale series** from OpenSTAT for the middle of the chain.
5. Writes `forecast.json` + `history.json`.

### The three price bases

Only the first trains the model. They are different measures and are never mixed:

| Basis | Source | Meaning |
|---|---|---|
| **Farmgate** | PSA OpenSTAT, Ilocos Norte, monthly | what a trader pays the farmer at the field |
| **Wholesale** | PSA OpenSTAT, Region I, monthly | what a trader sells milled rice for |
| **Retail** | DA Bantay Presyo, 16 markets, daily | what a shopper pays |

### Why the model is seasonal

`seasonal=True, m=12` is deliberate. On the full Ilocos Norte series a seasonal
search selects **(2,1,2)(0,0,2,12)** at **AIC 639.8**, against **650.2** for the
best non-seasonal fit. More importantly, the non-seasonal search collapses to
**(0,1,0)** — a random walk that predicts the last observed price flat across
all six months. A flat forecast has no peak, so "best month to sell" degenerates
and the app would tell every farmer to sell immediately, every day.

## Data sources

```bash
python openstat.py     # inspect what OpenSTAT currently holds
```

Tables used (browse at <https://openstat.psa.gov.ph>):

| Table | Contents |
|---|---|
| `2M/NFG/0032M4AFN01.px` | Cereals: **Farmgate** prices, monthly, province level |
| `2M/NWSNEW/0052M4AWB01.px` | Cereals: **Wholesale** prices, monthly, region level |
| `2E/CS/0012E4EVCP0.px` | Palay production by ecosystem, quarterly, province level |

⚠️ **Response format:** this PXWeb instance returns a malformed `value` array
for `json-stat2` — the array is truncated to one element while `size` still
claims the full extent. Use `format: "json"`, which returns explicit key/value
pairs. Also note the geolocation codes differ between tables: the farmgate table
uses PSGC-style codes (`012800000`), production and wholesale use ordinal
indices (`9`). `openstat.py` records the right code per table.
