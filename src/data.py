"""Fetch + cache the India GDP series used across the whole curriculum.

Three real series, all internally consistent (do NOT compare levels across them —
different base years / price bases):

  1. annual_real_gdp   World Bank  NY.GDP.MKTP.KN  GDP, constant LCU (INR), 1960-
  2. q_real_gdp_nsa     OECD QNA    real GDP, *N*ot seasonally adjusted, quarterly
  3. q_real_gdp_sa      OECD QNA    real GDP, seasonally+calendar adjusted, quarterly

Design notes
------------
* Uses only the standard library (urllib, json) + pandas, so it runs the moment
  pandas is installed and needs no API keys.
* Network fetch is separated from disk loading. `build()` writes CSVs once;
  notebooks call the fast `load_*()` functions. Re-run `python -m src.data`
  to refresh the cache.

Run:  python -m src.data
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import pandas as pd

# --- paths -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"

# --- source identifiers ------------------------------------------------------
WB_URL = (
    "https://api.worldbank.org/v2/country/IND/indicator/"
    "NY.GDP.MKTP.KN?format=json&per_page=20000"
)
# OECD Quarterly National Accounts (expenditure, national currency) via DBnomics.
# B1GQ = GDP; XDC.Q = chained-volume (real); the 2nd code field is the
# ADJUSTMENT flag:  N = neither SA nor calendar adj,  Y = SA + calendar adj.
DBNOMICS = (
    "https://api.db.nomics.world/v22/series/OECD/"
    "DSD_NAMAIN1@DF_QNA_EXPENDITURE_NATIO_CURR/{code}?observations=1"
)
OECD_NSA = "Q.N.IND.S1.S1.B1GQ._Z._Z._Z.XDC.Q.N.T0102"
OECD_SA = "Q.Y.IND.S1.S1.B1GQ._Z._Z._Z.XDC.Q.N.T0102"

_HEADERS = {"User-Agent": "ts-forecasting-curriculum/1.0 (educational)"}


def _get(url: str, timeout: int = 60) -> bytes:
    req = urllib.request.Request(url, headers=_HEADERS)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# --- fetchers (network) ------------------------------------------------------
def fetch_worldbank_annual() -> pd.DataFrame:
    """Annual real GDP (constant LCU). Returns columns: year, gdp."""
    payload = json.loads(_get(WB_URL))
    records = payload[1]  # payload[0] is pagination metadata
    rows = [
        (int(r["date"]), float(r["value"]))
        for r in records
        if r["value"] is not None
    ]
    df = pd.DataFrame(rows, columns=["year", "gdp"]).sort_values("year")
    return df.reset_index(drop=True)


def _fetch_dbnomics_quarterly(code: str) -> pd.Series:
    """One OECD quarterly series -> pandas Series indexed by quarterly Period."""
    payload = json.loads(_get(DBNOMICS.format(code=code)))
    doc = payload["series"]["docs"][0]
    periods = doc["period"]            # e.g. '2004-Q2'
    values = doc["value"]
    pairs = [(p, v) for p, v in zip(periods, values) if v is not None]
    idx = pd.PeriodIndex([p.replace("-", "") for p in (p for p, _ in pairs)], freq="Q")
    return pd.Series([v for _, v in pairs], index=idx, name="gdp").sort_index()


def fetch_oecd_quarterly() -> pd.DataFrame:
    """Quarterly real GDP, both NSA and SA. Index: quarterly Period."""
    nsa = _fetch_dbnomics_quarterly(OECD_NSA).rename("gdp_nsa")
    sa = _fetch_dbnomics_quarterly(OECD_SA).rename("gdp_sa")
    return pd.concat([nsa, sa], axis=1)


# --- build (fetch -> disk) ---------------------------------------------------
def build() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    PROC.mkdir(parents=True, exist_ok=True)

    print("Fetching World Bank annual real GDP ...")
    annual = fetch_worldbank_annual()
    annual.to_csv(RAW / "worldbank_annual_real_gdp.csv", index=False)
    annual.to_csv(PROC / "annual_real_gdp.csv", index=False)
    print(f"  -> {len(annual)} years, {annual.year.min()}-{annual.year.max()}")

    print("Fetching OECD quarterly real GDP (NSA + SA) ...")
    q = fetch_oecd_quarterly()
    raw_q = q.copy()
    raw_q.index = raw_q.index.astype(str)
    raw_q.to_csv(RAW / "oecd_quarterly_real_gdp.csv")
    # processed: store the Period as an ISO-ish string + a quarter-start timestamp
    out = q.copy()
    out.insert(0, "date", q.index.to_timestamp(how="start"))
    out.insert(1, "quarter", q.index.astype(str))
    out.to_csv(PROC / "quarterly_real_gdp.csv", index=False)
    print(f"  -> {len(q)} quarters, {q.index.min()}-{q.index.max()}")
    print("Done. Cached to data/processed/.")


# --- loaders (disk) ----------------------------------------------------------
def load_annual() -> pd.Series:
    """Annual real GDP as a Series indexed by an integer year."""
    df = pd.read_csv(PROC / "annual_real_gdp.csv")
    return df.set_index("year")["gdp"]


def load_quarterly() -> pd.DataFrame:
    """Quarterly real GDP (gdp_nsa, gdp_sa) indexed by a quarterly PeriodIndex."""
    df = pd.read_csv(PROC / "quarterly_real_gdp.csv")
    # the 'quarter' column is already in pandas Period form, e.g. '2004Q2'
    idx = pd.PeriodIndex(df["quarter"], freq="Q")
    return df.set_index(idx)[["gdp_nsa", "gdp_sa"]]


if __name__ == "__main__":
    build()
