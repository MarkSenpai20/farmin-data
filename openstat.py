"""
FarmIn — PSA OpenSTAT API client.

Replaces the hand-cleaned CSVs with live pulls from the Philippine Statistics
Authority's PXWeb API, so the series the model trains on is reproducible and
stays current instead of drifting out of date between manual refreshes.

    https://openstat.psa.gov.ph/API-Documentation

Three series are exposed:

    fetch_farmgate()    Palay farmgate price, Ilocos Norte, monthly   <- trains the model
    fetch_wholesale()   Milled rice wholesale, Region I, monthly      <- price chain
    fetch_production()  Palay production by ecosystem, quarterly      <- supply side

Together with the DA Bantay Presyo retail scrape these give the full chain a
farmer sits at the bottom of:

    FARMGATE (what the farmer is paid)
        -> WHOLESALE (what the trader sells for)
            -> RETAIL (what the shopper pays)

NOTE ON RESPONSE FORMATS: this PXWeb instance returns a malformed `value` array
for `json-stat2` (the array is truncated to a single element while `size` still
claims the full extent). Use `format: "json"`, which returns explicit
key/values pairs and is what every function here requests.

Run standalone to inspect what the API currently holds:
    python openstat.py
"""

import json
import urllib.request
from datetime import datetime

API_ROOT = "https://openstat.psa.gov.ph/PXWeb/api/v1/en/DB"

# --- Table identifiers -------------------------------------------------------
# Each entry records the geolocation code for the level we want, because the
# codes are NOT consistent between tables: the farmgate table uses PSGC-style
# codes ("012800000") while production and wholesale use ordinal indices.
T_FARMGATE = {
    "path": "/2M/NFG/0032M4AFN01.px",
    "title": "Cereals: Farmgate Prices",
    "geo": "012800000",                 # ....Ilocos Norte
    "geo_label": "Ilocos Norte",
    "commodity": "1",                   # Palay [Paddy] Other Variety, dry (14% mc)
}
T_WHOLESALE = {
    "path": "/2M/NWSNEW/0052M4AWB01.px",
    "title": "Cereals: Wholesale Selling Prices",
    # Province rows are published empty for this table — region is the finest
    # level actually populated, so asking for Ilocos Norte returns nothing.
    "geo": "9",                         # ..REGION I (ILOCOS REGION)
    "geo_label": "Region I (Ilocos Region)",
    "commodity": "3",                   # Regular Milled Rice (RMR)
}
T_PRODUCTION = {
    "path": "/2E/CS/0012E4EVCP0.px",
    "title": "Palay and Corn: Volume of Production",
    "geo": "9",                         # ....Ilocos Norte
    "geo_label": "Ilocos Norte",
}
T_STOCKS = {
    "path": "/2E/CS/0048E4ECNV1.px",
    "title": "Rice and Corn: Monthly Total Stocks Inventory by Sector and Region",
    "geo": "3",                         # ..REGION I (ILOCOS REGION)
    "geo_label": "Region I (Ilocos Region)",
}
T_COSTS = {
    "path": "/2B/AA/CR/0012B5EAPC0.px",
    "title": "Palay: Average Production Costs and Returns",
    "geo": "10000000",                  # ..Region I (Ilocos Region)
    "geo_label": "Region I (Ilocos Region)",
}

# Stock sectors worth carrying. Household + commercial + NFA sum to total; the
# split matters because a drawdown in household stock is farmers and consumers
# eating through supply, while commercial stock is traders holding inventory.
STOCK_SECTORS = {
    "Rice: Total Stocks": "total",
    "Rice: Household Stock": "household",
    "Rice: Commercial Stock": "commercial",
    "Rice: NFA Stock": "nfa",
}

# Cost/return lines to publish. Everything is per hectare except the ratio and
# the per-kilogram figure.
COST_ITEMS = {
    "CASH COSTS": "cash_costs",
    "..Seeds": "seeds",
    "..Fertilizer": "fertilizer",
    "..Pesticides": "pesticides",
    "..Hired labor": "hired_labor",
    "TOTAL COSTS": "total_costs",
    "GROSS RETURNS": "gross_returns",
    "NET RETURNS": "net_returns",
    "NET PROFIT - COST RATIO": "net_profit_cost_ratio",
    "Cost per kilogram (in PhP)": "cost_per_kg",
}

MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
MONTH_NUMBER = {name: i + 1 for i, name in enumerate(MONTHS)}

# PXWeb marks a missing observation with any of these.
BLANK = {"..", "...", "-", "", ":", "*"}

TIMEOUT = 60


# =============================================================================
# Low-level API access
# =============================================================================
def _request(url, payload=None):
    data = json.dumps(payload).encode() if payload is not None else None
    headers = {"User-Agent": "FarmIn/1.0 (thesis project)"}
    if data:
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
        return resp.read().decode("utf-8-sig")


def get_metadata(table):
    """Variable codes and value labels for a table."""
    return json.loads(_request(API_ROOT + table["path"]))


def query(table, selections):
    """
    POST a query and return (rows, meta) where rows are {"key": [...],
    "values": [...]}. `selections` maps variable code -> list of value codes.
    """
    payload = {
        "query": [
            {"code": code, "selection": {"filter": "item", "values": values}}
            for code, values in selections.items()
        ],
        "response": {"format": "json"},
    }
    body = json.loads(_request(API_ROOT + table["path"], payload))
    return body.get("data", [])


def _label_maps(meta):
    """code -> label for every variable, keyed by variable code."""
    return {
        v["code"]: dict(zip(v["values"], v["valueTexts"]))
        for v in meta["variables"]
    }


def _codes_for(meta, var_code, wanted=None, exclude=None):
    """Value codes for a variable, optionally filtered by their labels."""
    for v in meta["variables"]:
        if v["code"] != var_code:
            continue
        out = []
        for code, label in zip(v["values"], v["valueTexts"]):
            if wanted is not None and label not in wanted:
                continue
            if exclude is not None and label in exclude:
                continue
            out.append(code)
        return out
    raise KeyError(f"variable {var_code!r} not in table")


def _is_blank(raw):
    return raw is None or str(raw).strip() in BLANK


# =============================================================================
# Monthly price series
# =============================================================================
def _monthly_series(table, years=None):
    """
    Shared shape for the two monthly price tables: returns
    [{"record_date": "YYYY-MM-01", "price": float}, ...] oldest first.
    """
    meta = get_metadata(table)
    labels = _label_maps(meta)

    year_codes = _codes_for(meta, "Year", wanted=years)
    month_codes = _codes_for(meta, "Period", exclude={"Annual"})

    rows = query(table, {
        "Geolocation": [table["geo"]],
        "Commodity": [table["commodity"]],
        "Year": year_codes,
        "Period": month_codes,
    })

    # Key order follows the table's own variable order, not the query's.
    order = [v["code"] for v in meta["variables"]]
    idx = {code: i for i, code in enumerate(order)}

    points = {}
    for row in rows:
        raw = row["values"][0]
        if _is_blank(raw):
            continue
        year = labels["Year"][row["key"][idx["Year"]]]
        month = labels["Period"][row["key"][idx["Period"]]]
        if month not in MONTH_NUMBER:
            continue
        try:
            points[f"{year}-{MONTH_NUMBER[month]:02d}-01"] = float(raw)
        except ValueError:
            continue

    return [{"record_date": d, "price": points[d]} for d in sorted(points)]


def fetch_farmgate(years=None):
    """
    Palay farmgate price for Ilocos Norte, monthly, in PHP/kg.

    This is the series the ARIMA model trains on — what a trader actually pays
    the farmer at the field. Returns oldest-first, or None on any failure so the
    caller can fall back to the bundled CSV.
    """
    try:
        series = _monthly_series(T_FARMGATE, years)
    except Exception as e:
        print(f"[openstat] farmgate fetch failed: {e}", flush=True)
        return None
    if not series:
        print("[openstat] farmgate returned no observations.", flush=True)
        return None
    print(
        f"[openstat] farmgate: {len(series)} months "
        f"({series[0]['record_date']} .. {series[-1]['record_date']}), "
        f"{T_FARMGATE['geo_label']}.",
        flush=True,
    )
    return series


def fetch_wholesale(years=None):
    """
    Regular Milled Rice wholesale price for Region I, monthly, in PHP/kg.

    Not training data — a different point in the chain. Reported alongside the
    forecast so the farmer can see the margin between what they are paid and
    what the same rice trades for.
    """
    try:
        series = _monthly_series(T_WHOLESALE, years)
    except Exception as e:
        print(f"[openstat] wholesale fetch failed: {e}", flush=True)
        return None
    if not series:
        print("[openstat] wholesale returned no observations.", flush=True)
        return None
    print(
        f"[openstat] wholesale: {len(series)} months "
        f"({series[0]['record_date']} .. {series[-1]['record_date']}), "
        f"{T_WHOLESALE['geo_label']}.",
        flush=True,
    )
    return series


# =============================================================================
# Quarterly production
# =============================================================================
def fetch_production(years=None):
    """
    Palay production for Ilocos Norte in metric tons, by quarter and ecosystem.

    Returns [{"year": int, "quarter": "Q1", "ecosystem": "Palay"|"Irrigated
    Palay"|"Rainfed Palay", "volume_mt": float}, ...]. "Palay" is the total; the
    irrigated/rainfed split is what makes a drought or an irrigation failure
    visible in the supply signal rather than just a smaller total.
    """
    table = T_PRODUCTION
    try:
        meta = get_metadata(table)
        labels = _label_maps(meta)

        eco_codes = _codes_for(
            meta, "Ecosystem/Croptype",
            wanted={"Irrigated Palay", "Rainfed Palay", "Palay"},
        )
        year_codes = _codes_for(meta, "Year", wanted=years)
        quarter_codes = _codes_for(
            meta, "Period", wanted={"Quarter 1", "Quarter 2", "Quarter 3", "Quarter 4"},
        )

        rows = query(table, {
            "Ecosystem/Croptype": eco_codes,
            "Geolocation": [table["geo"]],
            "Year": year_codes,
            "Period": quarter_codes,
        })
    except Exception as e:
        print(f"[openstat] production fetch failed: {e}", flush=True)
        return None

    order = [v["code"] for v in meta["variables"]]
    idx = {code: i for i, code in enumerate(order)}

    out = []
    for row in rows:
        raw = row["values"][0]
        if _is_blank(raw):
            continue
        try:
            volume = float(raw)
        except ValueError:
            continue
        out.append({
            "year": int(labels["Year"][row["key"][idx["Year"]]]),
            "quarter": labels["Period"][row["key"][idx["Period"]]].replace("Quarter ", "Q"),
            "ecosystem": labels["Ecosystem/Croptype"][row["key"][idx["Ecosystem/Croptype"]]],
            "volume_mt": volume,
        })

    out.sort(key=lambda r: (r["ecosystem"], r["year"], r["quarter"]))
    if not out:
        print("[openstat] production returned no observations.", flush=True)
        return None
    years_seen = sorted({r["year"] for r in out})
    print(
        f"[openstat] production: {len(out)} rows, {years_seen[0]}-{years_seen[-1]}, "
        f"{table['geo_label']}.",
        flush=True,
    )
    return out


# =============================================================================
# Rice stocks — the demand side
# =============================================================================
def fetch_stocks(years=None):
    """
    Monthly rice stocks inventory for Region I, in metric tons, split by sector.

    This is the closest thing to a measured DEMAND signal available at regional
    level. Stocks are not consumption, but the month-on-month drawdown shows
    supply being eaten through, which is a genuine observation rather than the
    supply-deviation proxy the study otherwise relies on.

    Returns {"latest": {...}, "series": [{"month", "total", "household",
    "commercial", "nfa"}, ...]} oldest first, or None.
    """
    table = T_STOCKS
    try:
        meta = get_metadata(table)
        labels = _label_maps(meta)

        sector_codes = _codes_for(meta, "Sector", wanted=set(STOCK_SECTORS))
        year_codes = _codes_for(meta, "Year", wanted=years)
        month_codes = _codes_for(meta, "Month", exclude={"Annual"})

        rows = query(table, {
            "Sector": sector_codes,
            "Geolocation": [table["geo"]],
            "Year": year_codes,
            "Month": month_codes,
        })
    except Exception as e:
        print(f"[openstat] stocks fetch failed: {e}", flush=True)
        return None

    order = [v["code"] for v in meta["variables"]]
    idx = {code: i for i, code in enumerate(order)}

    by_month = {}
    for row in rows:
        raw = row["values"][0]
        if _is_blank(raw):
            continue
        try:
            volume = float(raw)
        except ValueError:
            continue
        year = labels["Year"][row["key"][idx["Year"]]]
        month = labels["Month"][row["key"][idx["Month"]]]
        if month not in MONTH_NUMBER:
            continue
        sector = STOCK_SECTORS.get(labels["Sector"][row["key"][idx["Sector"]]])
        if sector is None:
            continue
        key = f"{year}-{MONTH_NUMBER[month]:02d}"
        by_month.setdefault(key, {"month": key})[sector] = volume

    # Only months with a total are useful; the sector split may be partial.
    series = [by_month[k] for k in sorted(by_month) if "total" in by_month[k]]
    if not series:
        print("[openstat] stocks returned no observations.", flush=True)
        return None

    latest = series[-1]
    previous = series[-2] if len(series) > 1 else None
    change_pct = None
    if previous and previous.get("total"):
        change_pct = round(
            (latest["total"] - previous["total"]) / previous["total"] * 100, 1
        )

    print(
        f"[openstat] stocks: {len(series)} months "
        f"({series[0]['month']} .. {latest['month']}), {table['geo_label']}, "
        f"latest total {latest['total']:,.0f} MT"
        + (f" ({change_pct:+.1f}% MoM)" if change_pct is not None else ""),
        flush=True,
    )

    return {
        "region": table["geo_label"],
        "unit": "metric tons",
        "source": "PSA OpenSTAT",
        "latest": latest,
        "change_pct_mom": change_pct,
        "series": series,
    }


# =============================================================================
# Production costs and returns — what a hectare actually costs
# =============================================================================
def fetch_costs(years=None):
    """
    Average palay production costs and returns for Region I, per hectare.

    Gives the Profit Calculator a real benchmark instead of asking the farmer to
    guess their own input costs, and yields `cost_per_kg` — which, set against
    the forecast farmgate price, is the farmer's actual margin per kilogram.

    Returns the most recent year available with the full breakdown, or None.
    """
    table = T_COSTS
    try:
        meta = get_metadata(table)
        labels = _label_maps(meta)

        item_codes = _codes_for(meta, "Item", wanted=set(COST_ITEMS))
        year_codes = _codes_for(meta, "Year", wanted=years)
        # "Average" spans both cropping seasons.
        season_codes = _codes_for(meta, "Cropping Season", wanted={"Average"})
        type_codes = _codes_for(meta, "Type", wanted={"Palay"})

        rows = query(table, {
            "Type": type_codes,
            "Geolocation": [table["geo"]],
            "Item": item_codes,
            "Cropping Season": season_codes,
            "Year": year_codes,
        })
    except Exception as e:
        print(f"[openstat] costs fetch failed: {e}", flush=True)
        return None

    order = [v["code"] for v in meta["variables"]]
    idx = {code: i for i, code in enumerate(order)}

    by_year = {}
    for row in rows:
        raw = row["values"][0]
        if _is_blank(raw):
            continue
        try:
            value = float(raw)
        except ValueError:
            continue
        year = labels["Year"][row["key"][idx["Year"]]]
        field = COST_ITEMS.get(labels["Item"][row["key"][idx["Item"]]])
        if field is None:
            continue
        by_year.setdefault(year, {})[field] = value

    # Take the newest year that actually carries the headline figures.
    usable = [
        y for y in sorted(by_year)
        if "cash_costs" in by_year[y] and "cost_per_kg" in by_year[y]
    ]
    if not usable:
        print("[openstat] costs returned no usable year.", flush=True)
        return None

    year = usable[-1]
    data = by_year[year]
    print(
        f"[openstat] costs: {year}, {table['geo_label']} — cash "
        f"{data['cash_costs']:,.0f}/ha, cost/kg {data['cost_per_kg']:.2f}.",
        flush=True,
    )

    return {
        "year": int(year),
        "region": table["geo_label"],
        "crop": "Palay",
        "season": "Average of both cropping seasons",
        "unit": "PHP per hectare (cost_per_kg in PHP per kilogram)",
        "source": "PSA OpenSTAT — Costs and Returns",
        "history_years": [int(y) for y in usable],
        **data,
    }


# =============================================================================
# Standalone inspection
# =============================================================================
def _describe(series, label, unit="PHP/kg"):
    if not series:
        print(f"  {label}: unavailable")
        return
    latest = series[-1]
    print(f"  {label}: {len(series)} points, "
          f"{series[0]['record_date']} .. {latest['record_date']}, "
          f"latest {latest['price']:.2f} {unit}")


if __name__ == "__main__":
    print(f"PSA OpenSTAT snapshot — {datetime.now():%Y-%m-%d %H:%M}\n")

    recent = {str(y) for y in range(2021, 2027)}

    farmgate = fetch_farmgate(recent)
    wholesale = fetch_wholesale(recent)
    production = fetch_production(recent)

    print("\nSeries:")
    _describe(farmgate, "farmgate (Ilocos Norte)")
    _describe(wholesale, "wholesale (Region I)")

    if farmgate and wholesale:
        fg = {p["record_date"]: p["price"] for p in farmgate}
        ws = {p["record_date"]: p["price"] for p in wholesale}
        shared = sorted(set(fg) & set(ws))[-6:]
        if shared:
            print("\n  Farmgate vs wholesale, last 6 shared months:")
            for d in shared:
                gap = ws[d] - fg[d]
                print(f"    {d}   farmgate {fg[d]:>6.2f}   wholesale {ws[d]:>6.2f}   "
                      f"margin {gap:>6.2f}  ({gap / fg[d] * 100:5.1f}%)")

    if production:
        totals, quarters = {}, {}
        for r in production:
            if r["ecosystem"] == "Palay":
                totals[r["year"]] = totals.get(r["year"], 0) + r["volume_mt"]
                quarters.setdefault(r["year"], set()).add(r["quarter"])
        print("\n  Annual palay production (Ilocos Norte, MT):")
        for y in sorted(totals):
            partial = "  <- PARTIAL YEAR" if len(quarters[y]) < 4 else ""
            print(f"    {y}  {totals[y]:>12,.2f}  ({len(quarters[y])}/4 quarters){partial}")

    stocks = fetch_stocks({"2026"})
    if stocks:
        print(f"\n  Rice stocks, {stocks['region']} (MT):")
        for row in stocks["series"]:
            print(f"    {row['month']}  total {row.get('total', 0):>12,.0f}   "
                  f"household {row.get('household', 0):>11,.0f}   "
                  f"commercial {row.get('commercial', 0):>11,.0f}")

    costs = fetch_costs()
    if costs:
        print(f"\n  Palay costs & returns, {costs['region']}, {costs['year']} (per hectare):")
        for field in ("cash_costs", "seeds", "fertilizer", "hired_labor",
                      "total_costs", "gross_returns", "net_returns"):
            if field in costs:
                print(f"    {field:<22} {costs[field]:>12,.2f}")
        print(f"    {'cost_per_kg':<22} {costs['cost_per_kg']:>12,.2f}  PHP/kg")
        if farmgate:
            margin = farmgate[-1]["price"] - costs["cost_per_kg"]
            print(f"\n    Latest farmgate {farmgate[-1]['price']:.2f} - cost/kg "
                  f"{costs['cost_per_kg']:.2f} = margin {margin:.2f} PHP/kg")
