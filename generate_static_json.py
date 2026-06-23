"""
FarmIn — Static forecast generator.

Runs the SAME pipeline as ARIMA_MODEL/api.py, but instead of serving an HTTP
endpoint it writes two static files next to this script:

    forecast.json   -> {"forecast": [{"month": "YYYY-MM-DD", "predicted_price": ..}, ..], ..}
    history.json    -> {"history":  [{"month": "YYYY-MM-DD", "price": ..}, ..], ..}

A GitHub Action runs this on a schedule and commits the refreshed JSON, so the
Flutter app reads plain static files over a raw URL (no always-on server, no
cold start). The JSON shapes are byte-compatible with what the app already
parses from /forecast and /history.

Run locally with:  python generate_static_json.py
"""

import json
import os
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Baseline PSA farmgate price series (sits next to this script in the data repo).
BASELINE_CSV = os.path.join(BASE_DIR, "clean_psa_farmgate_prices.csv")

# Live Ilocos Region rice price table (Bantay Presyo).
BANTAY_PRESYO_URL = "http://www.bantaypresyo.da.gov.ph/tbl_rice.php"

# Table targets on the Bantay Presyo page.
PRIMARY_MARKET = "LAOAG"          # preferred price column
FALLBACK_MARKET = "BATAC"         # used when Laoag is N/A
TARGET_COMMODITY = "Regular Milled"
NA_TOKENS = {"", "N/A", "NA", "-", "--"}

FORECAST_MONTHS = 6               # matches the app's default /forecast?months=6

OUT_FORECAST = os.path.join(BASE_DIR, "forecast.json")
OUT_HISTORY = os.path.join(BASE_DIR, "history.json")


# =============================================================================
# 1. SCRAPER  (identical selectors to ARIMA_MODEL/api.py)
# =============================================================================
def parse_rice_price(html):
    """Return the Regular Milled price for Laoag (fallback Batac), or None."""
    soup = BeautifulSoup(html, "html.parser")

    target_table = None
    for table in soup.find_all("table"):
        if TARGET_COMMODITY.lower() in table.get_text(" ", strip=True).lower():
            target_table = table
            break
    if target_table is None:
        return None

    rows = target_table.find_all("tr")
    if not rows:
        return None

    laoag_idx = None
    batac_idx = None
    for row in rows:
        headers = [c.get_text(" ", strip=True).upper() for c in row.find_all(["th", "td"])]
        for i, text in enumerate(headers):
            if PRIMARY_MARKET in text:
                laoag_idx = i
            if FALLBACK_MARKET in text:
                batac_idx = i
        if laoag_idx is not None or batac_idx is not None:
            break
    if laoag_idx is None and batac_idx is None:
        return None

    target_cells = None
    for row in rows:
        cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
        if cells and TARGET_COMMODITY.lower() in cells[0].lower():
            target_cells = cells
            break
    if target_cells is None:
        return None

    def value_at(idx):
        if idx is None or idx >= len(target_cells):
            return ""
        return target_cells[idx].strip()

    raw = value_at(laoag_idx)
    if raw.upper() in NA_TOKENS:
        raw = value_at(batac_idx)

    cleaned = raw.replace("₱", "").replace("PHP", "").replace(",", "").strip()
    if cleaned.upper() in NA_TOKENS:
        return None
    try:
        return float(cleaned)
    except ValueError:
        return None


def scrape_da_bulletin():
    """
    Attempt the LIVE Bantay Presyo scrape only. Returns a fresh row on success,
    or None when the site is down / has no usable price — in which case the
    caller trains on the baseline CSV alone, so the forecast is anchored to the
    LAST REAL RECORDED price rather than a fabricated mock value.
    """
    try:
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        resp = requests.get(BANTAY_PRESYO_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        price = parse_rice_price(resp.text)
        if price is not None:
            record_date = pd.Timestamp.today().normalize().replace(day=1)
            print(f"[scraper] Scraped {price} for {record_date.date()} (live Bantay Presyo).", flush=True)
            return pd.DataFrame({"record_date": [record_date], "price": [price]})
        print("[scraper] Live site connected, but couldn't find the rice data.", flush=True)
    except Exception as e:
        print(f"[scraper] Live fetch failed: {e}", flush=True)

    print("[scraper] No live price — anchoring to the last recorded CSV price.", flush=True)
    return None


def load_baseline_dataframe():
    """Load the baseline PSA farmgate CSV into a ['record_date','price'] frame."""
    if not os.path.exists(BASELINE_CSV):
        raise FileNotFoundError(f"Baseline CSV not found: {BASELINE_CSV}")
    df = pd.read_csv(BASELINE_CSV)
    df["record_date"] = pd.to_datetime(df["record_date"])
    print(f"[data] Loaded baseline CSV ({len(df)} rows).", flush=True)
    return df[["record_date", "price"]]


# =============================================================================
# 2. BUILD + WRITE
# =============================================================================
def main():
    from pmdarima import auto_arima  # heavy import kept local to main()

    df = load_baseline_dataframe()
    data_source = "PSA baseline CSV"

    scraped = scrape_da_bulletin()
    if scraped is not None and not scraped.empty:
        df = pd.concat([df, scraped], ignore_index=True)
        df = df.drop_duplicates(subset="record_date", keep="last")
        data_source = "PSA baseline CSV + live DA scrape"
        print("[data] Merged scraped row(s) into the training set.", flush=True)

    df = (
        df.dropna(subset=["price"])
        .sort_values("record_date")
        .reset_index(drop=True)
        .set_index("record_date")
    )

    training_rows = len(df)
    last_date = df.index.max()

    history_points = [
        {"month": idx.strftime("%Y-%m-%d"), "price": round(float(val), 2)}
        for idx, val in df["price"].items()
    ]

    print(f"[train] auto_arima on {training_rows} rows (last {last_date.date()})...", flush=True)
    model = auto_arima(
        df["price"],
        seasonal=False,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    print(f"[train] Model ready: {model.order}.", flush=True)

    predictions = [float(p) for p in model.predict(n_periods=FORECAST_MONTHS)]
    forecast = []
    for i in range(1, FORECAST_MONTHS + 1):
        future_date = (last_date + relativedelta(months=i)).replace(day=1)
        forecast.append({
            "month": future_date.strftime("%Y-%m-%d"),
            "predicted_price": round(predictions[i - 1], 2),
        })

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    forecast_payload = {
        "status": "success",
        "data_source": data_source,
        "last_trained_date": last_date.strftime("%Y-%m-%d"),
        "training_rows": training_rows,
        "generated_at": generated_at,
        "forecast": forecast,
    }
    history_payload = {
        "status": "success",
        "count": len(history_points),
        "generated_at": generated_at,
        "history": history_points,
    }

    with open(OUT_FORECAST, "w", encoding="utf-8") as fh:
        json.dump(forecast_payload, fh, indent=2)
    with open(OUT_HISTORY, "w", encoding="utf-8") as fh:
        json.dump(history_payload, fh, indent=2)

    print(f"[done] Wrote {OUT_FORECAST} ({len(forecast)} points).", flush=True)
    print(f"[done] Wrote {OUT_HISTORY} ({len(history_points)} points).", flush=True)


if __name__ == "__main__":
    main()
