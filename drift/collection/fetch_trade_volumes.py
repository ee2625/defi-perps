"""
Compute Drift cumulative notional volume for BTC-PERP, ETH-PERP, SOL-PERP
over the sample window (2023-01-01 to 2024-12-31), summing
`quoteAssetAmountFilled` from the public S3 tradeRecords archive.

Streaming sum design: each daily CSV is downloaded, parsed, and summed
in-process; only the per-day totals are retained. The full trade tape
is never written to disk.

Output:
  drift/data/raw/cumulative_volume_summary.csv  (per-market totals)
  Stdout:                                       per-market and combined totals.

Run:
  python drift/collection/fetch_trade_volumes.py
"""

from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from io import BytesIO
from pathlib import Path
import sys
import time
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "data" / "raw"
OUTPUT.mkdir(parents=True, exist_ok=True)

S3_PREFIX = (
    "https://drift-historical-data-v2.s3.eu-west-1.amazonaws.com"
    "/program/dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH"
)
MARKETS = ["BTC-PERP", "ETH-PERP", "SOL-PERP"]
START = date(2023, 1, 1)
END = date(2024, 12, 31)
MAX_WORKERS = 16
SESSION = requests.Session()


def fetch_day_volume(market: str, d: date):
    """Return (date, market, notional_USD, n_trades) for one day."""
    url = f"{S3_PREFIX}/market/{market}/tradeRecords/{d.year}/{d.year}{d.month:02}{d.day:02}"
    try:
        r = SESSION.get(url, timeout=30)
        if r.status_code == 404:
            return (d, market, 0.0, 0)
        r.raise_for_status()
        df = pd.read_csv(BytesIO(r.content))
        if df.empty or "quoteAssetAmountFilled" not in df.columns:
            return (d, market, 0.0, 0)
        # Defensive: only count rows that look like real fills.
        if "action" in df.columns:
            df = df[df["action"].astype(str).str.contains("fill", case=False, na=False)]
        notional = float(df["quoteAssetAmountFilled"].sum())
        return (d, market, notional, len(df))
    except Exception as e:
        return (d, market, None, str(e))


def date_range(a: date, b: date):
    cur = a
    while cur <= b:
        yield cur
        cur += timedelta(days=1)


def main():
    days = list(date_range(START, END))
    print(f"Fetching {len(days)} days x {len(MARKETS)} markets "
          f"= {len(days) * len(MARKETS):,} S3 objects...")
    print(f"Sample window: {START} -- {END}")
    print(f"Workers: {MAX_WORKERS}\n")
    t0 = time.time()

    results = []  # list of (date, market, notional_usd, n)
    errors = []

    tasks = [(m, d) for m in MARKETS for d in days]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(fetch_day_volume, m, d): (m, d) for m, d in tasks}
        done = 0
        for fut in as_completed(futures):
            res = fut.result()
            d, m, notional, n_or_err = res
            if notional is None:
                errors.append((m, d, n_or_err))
            else:
                results.append((d, m, notional, n_or_err))
            done += 1
            if done % 500 == 0 or done == len(tasks):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed else 0
                eta = (len(tasks) - done) / rate if rate else 0
                print(f"  {done:>5d}/{len(tasks):,}  "
                      f"({rate:.0f}/s, eta {eta:.0f}s)")

    if errors:
        print(f"\n{len(errors)} fetch errors (first 5):")
        for m, d, e in errors[:5]:
            print(f"  {m} {d}: {e}")

    df = pd.DataFrame(results, columns=["date", "market", "notional_usd", "n_trades"])
    df = df.sort_values(["market", "date"]).reset_index(drop=True)

    print("\n" + "=" * 70)
    print(f"Drift cumulative notional volume, {START} -- {END}")
    print("=" * 70)
    summary = (
        df.groupby("market")
          .agg(notional_usd=("notional_usd", "sum"),
               n_trades=("n_trades", "sum"),
               days_covered=("date", "nunique"),
               days_with_data=("notional_usd", lambda s: int((s > 0).sum())))
          .reset_index()
    )
    print(summary.to_string(index=False))
    total = summary["notional_usd"].sum()
    print(f"\nCombined BTC + ETH + SOL: ${total/1e9:.2f}B "
          f"({total:,.0f} USD)")
    print(f"Total trades: {summary['n_trades'].sum():,}")
    print(f"\nElapsed: {time.time() - t0:.0f}s")

    out = OUTPUT / "cumulative_volume_summary.csv"
    df.to_csv(out, index=False)
    print(f"\nPer-day series saved to {out}")


if __name__ == "__main__":
    main()
