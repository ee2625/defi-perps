"""
Verify the introduction's volume statistics against public APIs.

The draft introduction cites:
  (1) Monthly perp notional volume ~$7.24T in early 2026 vs ~$4.14T two years
      earlier (CoinGecko Research, January 2026).
  (2) Perpetuals account for ~78% of crypto derivatives activity (CoinLaw).
  (3) DEX perp share grew from ~2.0% (Jan 2024) to ~10.2% (Jan 2026)
      (CoinGecko Research).
  (4) Drift cumulative trading volume ~$150B since 2021 (Drift comms).
  (5) dYdX cumulative trading volume ~$1.47T across v3+v4 (Bitget Research).

What this script can verify directly:
  - A snapshot of CURRENT 24h derivatives volume across all exchanges
    listed on CoinGecko (free /derivatives/exchanges endpoint). Annualizing
    to monthly gives an order-of-magnitude check on (1).
  - Current DEX share by aggregating exchanges flagged as decentralized.
    Spot-checks the 10.2% figure in (3).
  - Whether Drift / dYdX are listed in CoinGecko's derivatives roster and
    their current OI / 24h notional.

What this script CANNOT verify (research-report figures, no free API):
  - The 78% perpetuals/derivatives split (CoinLaw).
  - Historical Jan 2024 DEX share comparison point.
  - Cumulative volumes since 2021 for Drift and dYdX (DefiLlama derivatives
    overview moved behind the Pro paywall in 2026).

Run:
    python analysis/verify_intro_stats.py
"""

from __future__ import annotations
import sys
from datetime import datetime, timezone

try:
    import requests
except ImportError:
    print("Install requests: pip install requests", file=sys.stderr)
    sys.exit(1)

CG = "https://api.coingecko.com/api/v3"
TIMEOUT = 30


def fetch(url: str):
    r = requests.get(url, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()


def fmt_usd(x):
    if x is None:
        return "N/A"
    try:
        x = float(x)
    except (TypeError, ValueError):
        return "N/A"
    if x >= 1e12:
        return f"${x/1e12:.3f}T"
    if x >= 1e9:
        return f"${x/1e9:.2f}B"
    if x >= 1e6:
        return f"${x/1e6:.2f}M"
    return f"${x:,.0f}"


# Known DEX derivatives venues (CoinGecko name match).  Used because the
# CoinGecko exchange object does not carry an explicit "decentralized" flag.
DEX_NAMES = {
    "dydx", "dydx_chain", "gmx", "hyperliquid", "vertex", "aevo",
    "drift", "drift_protocol", "synfutures", "kwenta", "perpetual",
    "rabbitx", "ostium", "myx", "edgex", "lighter", "extended",
    "paradex", "satori", "apollox", "intentx", "pacifica", "merkle",
    "orderly", "level", "krav", "gains",
}


def is_dex(name_or_id: str) -> bool:
    n = (name_or_id or "").lower()
    return any(d in n for d in DEX_NAMES)


def pull_btc_usd():
    data = fetch(f"{CG}/simple/price?ids=bitcoin&vs_currencies=usd")
    return data["bitcoin"]["usd"]


def pull_all_derivatives_exchanges():
    """Page through CoinGecko's derivatives exchanges list."""
    rows = []
    for page in range(1, 6):  # 5 pages × 100 = 500 ceiling
        data = fetch(
            f"{CG}/derivatives/exchanges?per_page=100&page={page}&order=open_interest_btc_desc"
        )
        if not data:
            break
        rows.extend(data)
        if len(data) < 100:
            break
    return rows


def main():
    ts = datetime.now(timezone.utc).isoformat()
    print("=" * 76)
    print(f"Volume verification — fetched {ts}")
    print("=" * 76)

    btc_usd = pull_btc_usd()
    print(f"\nBTC/USD spot (CoinGecko): ${btc_usd:,.0f}")

    rows = pull_all_derivatives_exchanges()
    print(f"Derivatives exchanges retrieved: {len(rows)}\n")

    # Compute totals
    sum_24h_btc = sum(float(r.get("trade_volume_24h_btc") or 0) for r in rows)
    sum_oi_btc = sum(float(r.get("open_interest_btc") or 0) for r in rows)
    sum_24h_usd = sum_24h_btc * btc_usd
    sum_oi_usd = sum_oi_btc * btc_usd
    monthly_proxy = sum_24h_usd * 30
    annual_proxy = sum_24h_usd * 365

    dex_24h_btc = sum(
        float(r.get("trade_volume_24h_btc") or 0)
        for r in rows
        if is_dex(r.get("name") or r.get("id"))
    )
    dex_share = dex_24h_btc / sum_24h_btc if sum_24h_btc else 0.0

    print("All derivatives exchanges (CoinGecko, current snapshot)")
    print(f"  Aggregate 24h volume: {fmt_usd(sum_24h_usd)}")
    print(f"  Aggregate open interest: {fmt_usd(sum_oi_usd)}")
    print(f"  30 x 24h volume (monthly proxy): {fmt_usd(monthly_proxy)}")
    print(f"  365 x 24h volume (annual proxy): {fmt_usd(annual_proxy)}")
    print()
    print(f"  DEX share of 24h volume: {dex_share*100:.2f}%")
    print(f"  DEX 24h volume (USD):    {fmt_usd(dex_24h_btc * btc_usd)}")

    # Top 15 by OI
    rows_by_oi = sorted(
        rows, key=lambda r: float(r.get("open_interest_btc") or 0), reverse=True
    )[:15]
    print("\nTop 15 derivatives exchanges by open interest:")
    print(f"{'Name':<28s}  {'24h vol (USD)':>14s}  {'OI (USD)':>14s}  DEX?")
    for r in rows_by_oi:
        name = r.get("name", "?")
        v24 = float(r.get("trade_volume_24h_btc") or 0) * btc_usd
        oi = float(r.get("open_interest_btc") or 0) * btc_usd
        flag = "DEX" if is_dex(r.get("name") or r.get("id")) else ""
        print(f"  {name:<26s}  {fmt_usd(v24):>14s}  {fmt_usd(oi):>14s}  {flag}")

    # Drift / dYdX lookup
    print("\nDrift / dYdX entries:")
    for r in rows:
        n = (r.get("name") or "").lower()
        if "drift" in n or "dydx" in n:
            v24 = float(r.get("trade_volume_24h_btc") or 0) * btc_usd
            oi = float(r.get("open_interest_btc") or 0) * btc_usd
            print(
                f"  {r.get('name'):<30s} id={r.get('id'):<22s} "
                f"24h={fmt_usd(v24)}  OI={fmt_usd(oi)}"
            )

    # Per-claim verdict
    print("\n" + "=" * 76)
    print("Per-claim verdict")
    print("=" * 76)
    print(
        f"""
(1) Monthly aggregate ~$7.24T claim:
    Direct check: 30 x current 24h = {fmt_usd(monthly_proxy)}.
    The CoinGecko Research figure is a January 2026 monthly-traded number
    and may differ from a current snapshot * 30. If our snapshot * 30 is
    within ~30% of $7.24T it is consistent. Treat as cross-checked, not
    audited.

(2) 78% perpetuals/derivatives share:
    NOT verifiable from this script. Comes from CoinLaw research.
    Cross-check: number_of_perpetual_pairs vs number_of_futures_pairs in
    the CoinGecko data is suggestive of perp dominance but is a count, not
    volume-weighted.

(3) DEX share 2.0% (Jan 2024) to 10.2% (Jan 2026):
    Current snapshot DEX share: {dex_share*100:.2f}%.
    The 10.2% Jan 2026 endpoint is consistent if our snapshot lands in
    the high single-digits to low double-digits. Historical 2024 Jan
    point requires a paid time-series.

(4) Drift cumulative ~$150B since 2021:
    NOT verifiable here. Drift is not consistently in CoinGecko's
    derivatives exchanges roster, and DefiLlama's derivatives summary
    endpoint is now Pro-only. Source: Drift Protocol public comms (Apr
    2026); cross-check on defillama.com web UI.

(5) dYdX cumulative ~$1.47T (v3 + v4):
    Same situation as (4); v3 and v4 each have a DefiLlama page but
    cumulative-volume access is Pro-gated. Source: Bitget Research
    (Jan 2025) for the v3+v4 sum. The dYdX Chain entry above only shows
    the current 24h / OI snapshot, not cumulative.
"""
    )


if __name__ == "__main__":
    main()
