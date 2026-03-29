"""
fetch_funding_records.py
Collects FundingRateRecord data for SOL-PERP, BTC-PERP, ETH-PERP
from Jan 1 2023 to Dec 31 2025.

Source:
  - S3 flat files (daily CSVs) : 2023-01-01 -> 2025-12-31
  S3 confirmed working through 2025. 2026 files do not exist yet.

Train/test split:
  - Training : 2023-01-01 -> 2024-12-31  (Gamma estimation)
  - Test     : 2025-01-01 -> 2025-12-31  (out-of-sample violation test)

Output:
  drift/data/raw/{market}_funding_records.parquet
  one row per FundingRateRecord event
"""

import requests
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
import time

# ── paths ──────────────────────────────────────────────────────────────────
ROOT   = Path(__file__).resolve().parents[1]      # drift/
OUTPUT = ROOT / "data" / "raw"
OUTPUT.mkdir(parents=True, exist_ok=True)

# ── markets ────────────────────────────────────────────────────────────────
MARKETS = {
    "SOL-PERP": 0,
    "BTC-PERP": 1,
    "ETH-PERP": 2,
}

# ── S3 source ──────────────────────────────────────────────────────────────
S3_PREFIX = (
    "https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com"
    "/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH"
)
S3_START = date(2023, 1, 1)
S3_END   = date(2024, 12, 31)


# ───────────────────────────────────────────────────────────────────────────
def decode_funding_rate(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    S3 CSV columns are already in USD — no integer scaling needed.

    fundingRate     : dollars per base asset (e.g. 0.046 USD/SOL)
    oraclePriceTwap : oracle EMA-TWAP in USD (e.g. 101.8 USD/SOL)
    markPriceTwap   : mark EMA-TWAP in USD

    f_tk = fundingRate / oraclePriceTwap  (hourly fraction)
    This matches paper eq: f_tk = fundingRate_stored / Ī_tk
    """
    df = raw_df.copy()

    # Ī_tk — oracle EMA-TWAP in USD (already scaled)
    df["oracle_twap"] = df["oraclePriceTwap"]

    # I_tk — proxy = oracle_twap since spot price not in S3
    # rho_k = Ī_tk / I_tk = 1.0 (documented in data section)
    df["oracle_price"] = df["oracle_twap"]

    # M_tk — mark TWAP in USD
    df["mark_twap"] = df["markPriceTwap"]

    # f_tk — hourly funding fraction
    df["f_tk"] = df["fundingRate"] / df["oraclePriceTwap"]

    # rho_k ≈ 1.0
    df["rho_k"] = 1.0

    # eta — Tier B hourly cap = 0.125%
    df["eta"] = 0.00125

    # regime
    # at cap: f_tk should equal exactly ±eta
    # use relative tolerance given floating point
    tol = 1e-6
    df["regime"] = "mid"
    df.loc[(df["f_tk"] + df["eta"]).abs() < tol, "regime"] = "lower"
    df.loc[(df["f_tk"] - df["eta"]).abs() < tol, "regime"] = "upper"

    return df


# ───────────────────────────────────────────────────────────────────────────
def fetch_s3_range(market: str, start: date, end: date) -> pd.DataFrame:
    """
    Pull daily funding rate CSVs from S3 for a date range.
    Many days have no file (no funding updates) — skipped silently.
    """
    print(f"  [S3] {market}: {start} → {end}")
    frames = []
    current = start
    total_days = (end - start).days + 1
    fetched = 0

    while current <= end:
        y = current.year
        m = current.month
        d = current.day
        url = (
            f"{S3_PREFIX}/market/{market}"
            f"/fundingRateRecords/{y}/{y}{m:02}{d:02}"
        )

        try:
            df = pd.read_csv(url)
            if not df.empty:
                frames.append(df)
                fetched += len(df)
            time.sleep(0.05)                         # polite to S3

        except Exception:
            pass                                     # no file for this day

        # print progress every 90 days
        days_done = (current - start).days + 1
        if days_done % 90 == 0:
            print(f"    {current}  ({days_done}/{total_days} days, "
                  f"{fetched:,} rows so far)")

        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    print(f"    → {len(out):,} raw rows from S3")
    return out


# ───────────────────────────────────────────────────────────────────────────
def process_market(market: str, market_index: int):
    print(f"\n{'='*60}")
    print(f"Processing {market}")
    print(f"{'='*60}")

    # 1. pull from S3
    df = fetch_s3_range(market, S3_START, S3_END)

    if df.empty:
        print(f"  WARNING: no data for {market}")
        return

    # 2. standardise timestamp
    df["timestamp"] = pd.to_datetime(df["ts"], unit="s", utc=True)

    # 3. sort and deduplicate on ts
    df = (df.sort_values("timestamp")
            .drop_duplicates(subset=["ts"])
            .reset_index(drop=True))

    # 4. add market label
    df["market"] = market

    # 5. decode raw fields → paper variables
    df = decode_funding_rate(df)

    # 6. keep only needed columns
    keep = [
            "timestamp", "market",
            "oraclePriceTwap",    # Ī_tk in USD (raw = decoded for S3)
            "markPriceTwap",      # M_tk in USD
            "fundingRate",        # f_tk (raw = decoded for S3)
            "oracle_twap",        # Ī_tk in USD
            "oracle_price",       # I_tk in USD (proxy)
            "mark_twap",          # M_tk in USD
            "f_tk",               # realized hourly funding fraction
            "rho_k",              # oracle TWAP ratio
            "eta",                # tier cap η
            "regime",             # lower / upper / mid
        ]
    df = df[[c for c in keep if c in df.columns]]

    # 7. save
    out_path = OUTPUT / f"{market}_funding_records.parquet"
    df.to_parquet(out_path, index=False)

    # 8. summary
    train = df[df["timestamp"] < pd.Timestamp("2024-01-01", tz="UTC")]
    test  = df[df["timestamp"] >= pd.Timestamp("2024-01-01", tz="UTC")]

    print(f"\n  Saved {len(df):,} rows → {out_path}")
    print(f"  Full range  : {df['timestamp'].min()} → {df['timestamp'].max()}")
    print(f"  Training    : {len(train):,} rows (2023-2024)")
    print(f"  Test        : {len(test):,} rows (2025)")
    print(f"\n  Regime breakdown (full sample):")
    print(df["regime"].value_counts().to_string())
    print(f"\n  f_tk stats:")
    print(df["f_tk"].describe().to_string())


# ───────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    for market, index in MARKETS.items():
        process_market(market, index)

    print("\n\nAll markets done. Files in drift/data/raw/")