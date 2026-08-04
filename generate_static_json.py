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

PRICE BASIS — the model is trained on PSA FARMGATE prices, and `forecast`
predicts farmgate prices. The live DA Bantay Presyo scrape returns RETAIL
prices for Region I markets, which are a different basis entirely; it is
reported separately under `market_price` and never enters the training data.

Run locally with:  python generate_static_json.py
"""

import json
import os
import re
from datetime import datetime, timezone

import pandas as pd
import requests
from bs4 import BeautifulSoup
from dateutil.relativedelta import relativedelta

import openstat

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Baseline PSA farmgate price series (sits next to this script in the data repo).
BASELINE_CSV = os.path.join(BASE_DIR, "clean_psa_farmgate_prices.csv")

# Live Bantay Presyo rice price service (DA). The public page is a thin shell:
# its table is populated by three POST endpoints, which is what we call here.
# Requesting tbl_rice.php directly only ever returns the empty NCR shell.
BANTAY_BASE = "http://www.bantaypresyo.da.gov.ph/"
EP_DATE = "tbl_rice.php"                          # action=get_latest_date
EP_HEADER = "tbl_price_get_comm_header_rice.php"  # market column names
EP_PRICES = "tbl_price_get_comm_price_rice.php"   # commodity rows

REGION_CODE = "010000000"   # REGION I (ILOCOS REGION)
REGION_NAME = "Region I (Ilocos Region)"
COMMODITY_CODE = "1"        # RICE

# EVERY grade published in the rice table is captured — Local and Imported,
# across Premium / Regular Milled / Special / Well Milled, plus NFA. A farmer
# deciding when to sell needs the whole price ladder, not one line of it.
#
# One grade is still nominated as PRIMARY: it drives the headline figure on the
# dashboard card and is the closest retail counterpart to the farmgate series
# the model forecasts. Note the row is matched EXACTLY — the table carries both
# an IMPORTED and a LOCAL "Regular Milled", and a bare substring match hits the
# imported row first, which is almost always N/A in this region. That was why
# the original scraper never returned a price.
PRIMARY_COMMODITY = "COMMERCIAL (LOCAL) Regular Milled"

# Rows carrying no price anywhere in the region are dropped from the payload.
SKIP_EMPTY_GRADES = True

# The province this study is about. Its markets drive the headline figure; the
# rest of Region I supplies context.
FOCUS_PROVINCE = "Ilocos Norte"

# "0"/"0.00" mean "not reported" in this table, not a real price of zero.
NA_TOKENS = {"", "N/A", "NA", "-", "--", "0", "0.00"}

# Region I market -> province. Matched on keyword because the published market
# names vary in wording between refreshes.
PROVINCE_BY_KEYWORD = [
    ("Ilocos Norte", ("BATAC", "LAOAG", "BACARRA", "BADOC", "PASUQUIN", "DINGRAS")),
    ("Ilocos Sur", ("CANDON", "VIGAN", "NARVACAN", "TAGUDIN", "SANTA MARIA")),
    ("La Union", ("AGOO", "BALAOAN", "SAN FERNANDO", "BAUANG", "NAGUILIAN")),
    ("Pangasinan", ("ALAMINOS", "DAGUPAN", "MALIMGAS", "URDANETA", "SAN CARLOS", "LINGAYEN")),
]

FORECAST_MONTHS = 6               # matches the app's default /forecast?months=6

OUT_FORECAST = os.path.join(BASE_DIR, "forecast.json")
OUT_HISTORY = os.path.join(BASE_DIR, "history.json")


# =============================================================================
# 1. SCRAPER — DA Bantay Presyo, Region I (Ilocos Region)
# =============================================================================
def province_of(market_name):
    """Map a published market name to its Region I province."""
    up = market_name.upper()
    for province, keywords in PROVINCE_BY_KEYWORD:
        if any(k in up for k in keywords):
            return province
    return "Region I (other)"


def parse_price(raw):
    """Convert one table cell to a price, or None when it was not reported."""
    cleaned = str(raw).replace("₱", "").replace("PHP", "").replace(",", "").strip()
    if cleaned.upper() in NA_TOKENS:
        return None
    try:
        value = float(cleaned)
    except ValueError:
        return None
    return value if value > 0 else None


def _post(path, data, timeout=20):
    resp = requests.post(
        BANTAY_BASE + path,
        data=data,
        timeout=timeout,
        headers={"User-Agent": "Mozilla/5.0", "X-Requested-With": "XMLHttpRequest"},
    )
    resp.raise_for_status()
    return resp.text


def _summarise(prices):
    """Mean/min/max over the markets that actually reported a price."""
    if not prices:
        return None
    return {
        "average": round(sum(prices) / len(prices), 2),
        "min": round(min(prices), 2),
        "max": round(max(prices), 2),
        "markets_reporting": len(prices),
    }


def split_grade(commodity):
    """
    Split a published row label into (origin, short label).

        "COMMERCIAL (LOCAL) Regular Milled"  -> ("Local",    "Regular Milled")
        "COMMERCIAL (IMPORTED) Special ..."  -> ("Imported", "Special (Blue tagged)")
        "NFA"                                -> ("NFA",      "NFA")
    """
    m = re.match(r"^COMMERCIAL\s*\((LOCAL|IMPORTED)\)\s*(.+)$", commodity.strip(), re.I)
    if m:
        return m.group(1).capitalize(), m.group(2).strip()
    return commodity.strip(), commodity.strip()


def summarise_grade(commodity, specification, markets, cells):
    """Build one grade's payload from its table row, or None if nothing reported."""
    # Columns 0 and 1 are COMMODITY and SPECIFICATIONS; markets start at index 2.
    readings = []
    for i in range(2, min(len(markets), len(cells))):
        price = parse_price(cells[i])
        if price is not None:
            readings.append(
                {"market": markets[i], "province": province_of(markets[i]), "price": price}
            )
    if not readings and SKIP_EMPTY_GRADES:
        return None

    all_prices = [r["price"] for r in readings]
    focus_prices = [r["price"] for r in readings if r["province"] == FOCUS_PROVINCE]

    by_province = {}
    for r in readings:
        by_province.setdefault(r["province"], []).append(r["price"])

    # Headline: the study province when any of its markets report, otherwise the
    # whole region. `headline_scope` tells the app which of the two it is looking
    # at, so a Pangasinan average is never presented as a local reading.
    headline_scope = FOCUS_PROVINCE if focus_prices else REGION_NAME
    origin, label = split_grade(commodity)

    return {
        "commodity": commodity,
        "label": label,
        "origin": origin,
        "specification": specification,
        "headline_scope": headline_scope,
        "headline": _summarise(focus_prices or all_prices),
        "region_summary": _summarise(all_prices),
        "by_province": {p: _summarise(v) for p, v in sorted(by_province.items())},
        "markets": sorted(readings, key=lambda r: (r["province"], r["market"])),
    }


def parse_all_grades(header_html, rows_html):
    """
    Parse EVERY rice grade in the table, not just one.

    Pure function over the two HTML fragments, so it can be exercised offline
    against a saved copy of the page without touching the network.
    Returns (markets, grades) with grades ordered as published.
    """
    markets = [
        th.get_text(" ", strip=True)
        for th in BeautifulSoup(
            "<table><tr>" + header_html + "</tr></table>", "html.parser"
        ).find_all("th")
    ]

    soup = BeautifulSoup("<table>" + rows_html + "</table>", "html.parser")
    grades = []
    for tr in soup.find_all("tr"):
        cells = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if len(cells) < 3 or not cells[0].strip():
            continue
        grade = summarise_grade(cells[0].strip(), cells[1].strip(), markets, cells)
        if grade is not None:
            grades.append(grade)
    return markets, grades


def scrape_da_bulletin():
    """
    Retrieve today's RETAIL rice prices for every reporting market in Region I.

    IMPORTANT — this is a reference value, NOT training data. Bantay Presyo
    publishes RETAIL prices (roughly PHP 35-50/kg) while the PSA baseline series
    is FARMGATE (roughly PHP 21/kg). They are different price bases; appending a
    retail observation to a farmgate series injects a step change at the last
    point the model sees and skews every prediction after it.

    Returns the regional payload, or None when the service is unreachable or no
    market reported a price. Every failure is non-fatal: the forecast is built
    from the farmgate CSV regardless.
    """
    payload = {"commodity": COMMODITY_CODE, "region": REGION_CODE}
    try:
        raw_date = re.sub(
            r"<[^>]+>", "", _post(EP_DATE, dict(payload, action="get_latest_date"))
        ).strip()
        header_html = _post(EP_HEADER, payload)
        rows_html = _post(EP_PRICES, payload)
    except Exception as e:
        print(f"[scraper] Bantay Presyo unreachable: {e}", flush=True)
        return None

    try:
        markets, grades = parse_all_grades(header_html, rows_html)
    except Exception as e:
        print(f"[scraper] Could not parse the price table: {e}", flush=True)
        return None

    if not grades:
        print("[scraper] No Region I market reported any rice price today.", flush=True)
        return None

    try:
        observed_on = datetime.strptime(raw_date, "%B %d, %Y").strftime("%Y-%m-%d")
    except ValueError:
        observed_on = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    # Nominate the primary grade for the dashboard headline. If it happens not to
    # be reported today, fall back to the local grade with the widest coverage so
    # the card still shows something meaningful rather than going blank.
    primary = next((g for g in grades if g["commodity"].upper() == PRIMARY_COMMODITY.upper()), None)
    if primary is None:
        local = [g for g in grades if g["origin"].lower() == "local"] or grades
        primary = max(local, key=lambda g: g["region_summary"]["markets_reporting"])
        print(
            f"[scraper] {PRIMARY_COMMODITY!r} not reported; "
            f"headline falls back to {primary['commodity']!r}.",
            flush=True,
        )

    total_readings = sum(len(g["markets"]) for g in grades)
    print(
        f"[scraper] {raw_date}: {len(grades)} grade(s), {total_readings} market readings "
        f"across {max(len(markets) - 2, 0)} Region I markets. Headline "
        f"{primary['headline']['average']:.2f} "
        f"({primary['label']}, {primary['headline_scope']}).",
        flush=True,
    )
    for g in grades:
        r = g["region_summary"]
        print(
            f"           {g['origin']:<8} {g['label']:<26} "
            f"avg {r['average']:>6.2f}  ({r['markets_reporting']:>2} mkt, "
            f"{r['min']:.0f}-{r['max']:.0f})",
            flush=True,
        )

    return {
        "basis": "retail",
        "region": REGION_NAME,
        "observed_on": observed_on,
        "source": "DA Bantay Presyo",
        "markets_total": max(len(markets) - 2, 0),
        "grades_reporting": len(grades),
        "readings_total": total_readings,
        # --- headline: the primary grade, promoted to the top level so the
        #     dashboard card can render without walking the grade list ---
        "commodity": primary["commodity"],
        "headline_scope": primary["headline_scope"],
        "headline": primary["headline"],
        "region_summary": primary["region_summary"],
        "by_province": primary["by_province"],
        "markets": primary["markets"],
        # --- every grade published today, in table order ---
        "grades": grades,
    }


def load_baseline_dataframe():
    """
    The farmgate series the model trains on.

    Preferred source is the PSA OpenSTAT API, so the series stays current on its
    own. The bundled CSV is the fallback for when the API is unreachable — it is
    a frozen snapshot and goes stale, which is exactly the failure this ordering
    is meant to avoid.

    Returns (frame, source_label).
    """
    series = openstat.fetch_farmgate()
    if series:
        df = pd.DataFrame(series)
        df["record_date"] = pd.to_datetime(df["record_date"])
        newest = df["record_date"].max().strftime("%Y-%m")
        print(f"[data] OpenSTAT farmgate: {len(df)} months, newest {newest}.", flush=True)
        return df[["record_date", "price"]], "PSA OpenSTAT — Palay farmgate, Ilocos Norte (monthly)"

    if not os.path.exists(BASELINE_CSV):
        raise FileNotFoundError(
            f"OpenSTAT unavailable and no fallback CSV at {BASELINE_CSV}"
        )
    df = pd.read_csv(BASELINE_CSV)
    df["record_date"] = pd.to_datetime(df["record_date"])
    newest = df["record_date"].max().strftime("%Y-%m")
    print(
        f"[data] OpenSTAT unavailable — falling back to the bundled CSV "
        f"({len(df)} rows, newest {newest}). This snapshot does not update itself.",
        flush=True,
    )
    return df[["record_date", "price"]], "PSA farmgate CSV snapshot (OpenSTAT unavailable)"


# =============================================================================
# 2. BUILD + WRITE
# =============================================================================
def main():
    from pmdarima import auto_arima  # heavy import kept local to main()

    df, data_source = load_baseline_dataframe()

    # Neither of these is training data — they are different points in the price
    # chain, reported alongside the forecast so the farmer can see the margin
    # between what they are paid and what the same rice sells for.
    #   farmgate (trains the model) -> wholesale (PSA) -> retail (Bantay Presyo)
    market_price = scrape_da_bulletin()
    wholesale_series = openstat.fetch_wholesale()

    # Demand side and cost side. Neither trains the model either.
    stocks = openstat.fetch_stocks()
    costs = openstat.fetch_costs()

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
    # SEASONAL (m=12) is deliberate, on two grounds.
    #
    # Statistically: on the full Ilocos Norte series a seasonal search selects
    # (2,1,2)(0,0,2,12) at AIC 639.8, against 650.2 for the best non-seasonal
    # fit — the annual cycle is real and worth modelling.
    #
    # Functionally: the non-seasonal search collapses to (0,1,0), a pure random
    # walk, which predicts the last observed price flat across all six months.
    # A flat forecast has no peak, so "best month to sell" degenerates — every
    # month ties, the nearest month wins by position, and the app would tell
    # every farmer to sell immediately, every day. Rice farmgate prices move
    # with the harvest calendar; the model has to be allowed to say so.
    model = auto_arima(
        df["price"],
        seasonal=True,
        m=12,
        error_action="ignore",
        suppress_warnings=True,
        stepwise=True,
    )
    seasonal_order = getattr(model, "seasonal_order", None)
    print(f"[train] Model ready: {model.order} seasonal {seasonal_order}, "
          f"AIC {model.aic():.1f}.", flush=True)

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
        "price_basis": "farmgate",          # what `forecast` predicts
        "last_trained_date": last_date.strftime("%Y-%m-%d"),
        "training_rows": training_rows,
        # The fitted specification, so the thesis can cite the exact model
        # rather than describing it only as "an ARIMA".
        "model": {
            "order": list(model.order),
            "seasonal_order": list(seasonal_order) if seasonal_order else None,
            "aic": round(float(model.aic()), 2),
        },
        "generated_at": generated_at,
        "forecast": forecast,
        # Separate live reference, null when the scrape found nothing. Never
        # feeds the model — a different price basis to everything above.
        "market_price": market_price,
        # The middle link in the chain: what a trader sells milled rice for.
        # Also not training data.
        "wholesale": {
            "basis": "wholesale",
            "commodity": "Regular Milled Rice",
            "region": openstat.T_WHOLESALE["geo_label"],
            "source": "PSA OpenSTAT",
            "latest": wholesale_series[-1] if wholesale_series else None,
            "series": wholesale_series or [],
        } if wholesale_series else None,
        # Demand side: regional rice stocks. The drawdown between months is the
        # only measured demand signal available at this geography.
        "stocks": stocks,
        # Cost side: what a hectare of palay actually costs to grow in Region I.
        # `cost_per_kg` against the forecast price is the farmer's margin.
        "costs": costs,
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
