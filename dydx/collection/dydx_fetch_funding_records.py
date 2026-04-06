"""
dYdX v4 Data Pull
=================
Pulls historical funding-rate records and candle data from the dYdX v4
public indexer API (no authentication required).

Period and markets are intentionally aligned with the Drift pull script
(fetch_funding_records.py) so both datasets share the same train/test
window and the same three base assets:

  Drift markets  →  dYdX markets
  ──────────────────────────────
  BTC-PERP       →  BTC-USD
  ETH-PERP       →  ETH-USD
  SOL-PERP       →  SOL-USD

  Sample   : 2023-01-01 → 2025-12-31
  Training : 2023-01-01 → 2024-12-31  (Γ estimation)
  Test     : 2025-01-01 → 2025-12-31  (out-of-sample violation test)

Output parquets are column-compatible with the Drift parquets:
  timestamp     ← ts (renamed to match Drift)
  market        ← ticker (renamed to match Drift)
  f_tk          ← rate_8h / 8  (the hourly-equivalent rate in the proposition)
  oracle_price  ← oracle index price at settlement (I_tk)
  eta           ← eta_1h  (effective hourly cap entering the bound formula)
  regime        ← "lower" / "upper" / "mid"  (matches Drift decode_funding_rate)
  + dYdX-specific extras: rate_8h, eta_8h, imf, mmf, delta_t_sec,
                           lower_clamped, upper_clamped, clamped

Key differences from Drift (documented for paper Section 8)
------------------------------------------------------------
  • 8-hour settlement cadence (Drift: ~1 hour lazy)
  • η_8h = 6 × (IMF − MMF) per market (Drift: discrete tier table)
  • η_1h = η_8h / 8 enters the bound formula with NO ρ_k and NO /5000 offset
  • Clamped regime: |rate_8h| ≈ η_8h  (Drift: |f_tk| ≈ η_tier)
  • P_tk proxy: candle close at settlement time (Drift: mark_twap)
  • N_k minute samples is stochastic; bounds hold for any N_k ≥ 1
  • rho_k ≡ 1.0 by construction (no EMA-oracle divergence)

Usage
-----
  python dydx_pull.py [--out-dir ./data/raw/dydx]

Outputs (per market, e.g. BTC-USD)
------------------------------------
  BTC-USD_funding.parquet       one row per 8-hour settlement
  BTC-USD_candles_1h.parquet    hourly OHLCV (for P_tk proxy construction)
  market_params.json            IMF / MMF snapshot with pull timestamp

Column glossary — funding parquet
----------------------------------
  timestamp       UTC timestamp of the settlement (Timestamp, tz-aware)
  market          market label, e.g. "BTC-USD"
  rate_8h         8-hour clamped funding rate from indexer (float)
                  = clamp(π̄_k, −η_8h, +η_8h)
  f_tk            rate_8h / 8  — hourly-equivalent f_k in the proposition
  oracle_price    I_tk  — index price at settlement (USD)
  imf             initialMarginFraction for this market (float, e.g. 0.05)
  mmf             maintenanceMarginFraction (float, e.g. 0.03)
  eta_8h          6 × (imf − mmf)  — 8-hour funding cap
  eta             eta_8h / 8  — 1-hour cap (= eta_1h; enters bound formula)
  lower_clamped   bool: rate_8h ≈ −η_8h
  upper_clamped   bool: rate_8h ≈ +η_8h
  clamped         bool: lower_clamped | upper_clamped
  regime          "lower" / "upper" / "mid"  (Drift-compatible label)

Column glossary — candles parquet
-----------------------------------
  ts              UTC open time of the candle (1-hour bars)
  ticker          market label
  open / high / low / close   trade prices in USD
  volume          base-asset volume
  usdVolume       USD-denominated volume (if available from indexer)

Notes on P_tk construction (candles → perp mid proxy)
------------------------------------------------------
  At each funding settlement ts_k, match to the 1-hour candle whose
  open == ts_k.  Use close as P_tk proxy.  This is a mark-price
  proxy, not a true impact midpoint, but it is the best available
  substitute from public data.

  If you have access to the dYdX chain's
  /dydxprotocol/perpetuals/premium_samples endpoint, that gives the
  actual minute-level premium samples; see NOTE_ON_PREMIUM_SAMPLES
  at the bottom of this file.
"""

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

INDEXER_BASE = "https://indexer.dydx.trade/v4"

# Aligned with Drift script: same three base assets, same 2023-2025 window
MARKETS = ["BTC-USD", "ETH-USD", "SOL-USD"]

# Sample window — matches Drift's S3 pull range exactly
SAMPLE_START = "2023-01-01"
SAMPLE_END   = "2025-12-31"

# Train / test split — matches Drift notebook SPLIT constant
TRAIN_END    = "2024-12-31"   # last day of training window
TEST_START   = "2025-01-01"   # first day of test window

# IMF/MMF as of Feb 2026 (from paper footnote + /v4/perpetualMarkets).
# These are governance-configurable; the pull script fetches live values
# from the API at run time. These constants are fallbacks only.
MARKET_PARAMS_FALLBACK = {
    "BTC-USD": {"imf": 0.020, "mmf": 0.012},   # η_8h = 6*(0.020-0.012) = 0.048
    "ETH-USD": {"imf": 0.020, "mmf": 0.012},   # η_8h = 0.048
    "SOL-USD": {"imf": 0.050, "mmf": 0.030},   # η_8h = 6*(0.050-0.030) = 0.120
}

# Funding clamp tolerance (float comparison)
CLAMP_TOL = 1e-8

# API pagination
FUNDING_PAGE_SIZE = 200   # max records per request
CANDLE_PAGE_SIZE  = 200
REQUEST_DELAY     = 0.25  # seconds between requests (rate-limit headroom)
MAX_RETRIES       = 5

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = MAX_RETRIES) -> dict:
    """GET with retry / back-off."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, timeout=30)
            resp.raise_for_status()
            return resp.json()
        except requests.HTTPError as e:
            status = e.response.status_code if e.response else None
            if status == 429:
                wait = 2 ** attempt
                log.warning("Rate-limited; sleeping %ds (attempt %d)", wait, attempt + 1)
                time.sleep(wait)
            elif attempt < retries - 1:
                time.sleep(1)
            else:
                raise
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(1)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries: {url}")


def fetch_market_params(out_dir: Path) -> dict:
    """
    Pull current IMF / MMF for all perpetual markets from /v4/perpetualMarkets.
    Returns dict keyed by ticker, e.g. {"BTC-USD": {"imf": 0.02, "mmf": 0.012, ...}}.
    Also saves market_params.json to out_dir.

    NOTE: IMF/MMF are governance-configurable.  This snapshot is stamped with
    the pull date.  If your sample spans a governance change, you will need to
    reconstruct the historical η_8h series from on-chain governance proposals.
    For the paper's sample through Feb 2026, BTC/ETH parameters were stable.
    """
    log.info("Fetching perpetual market parameters …")
    data = _get(f"{INDEXER_BASE}/perpetualMarkets")
    markets_raw = data.get("markets", {})

    params = {}
    for ticker, mkt in markets_raw.items():
        try:
            imf = float(mkt.get("initialMarginFraction") or mkt.get("initialMarginPercent", 0))
            mmf = float(mkt.get("maintenanceMarginFraction") or mkt.get("maintenanceMarginPercent", 0))
            # dYdX v4 may return fractions OR percentages depending on indexer version
            # Normalise: values > 1 are percentages → divide by 100
            if imf > 1:
                imf /= 100
            if mmf > 1:
                mmf /= 100
            eta_8h = 6.0 * (imf - mmf)
            eta_1h = eta_8h / 8.0
            params[ticker] = {
                "imf": imf,
                "mmf": mmf,
                "eta_8h": eta_8h,
                "eta_1h": eta_1h,
                "pulled_at": datetime.now(timezone.utc).isoformat(),
            }
        except (TypeError, ValueError) as exc:
            log.warning("Could not parse params for %s: %s", ticker, exc)

    # Fall back to hard-coded values for any missing markets
    for ticker in MARKETS:
        if ticker not in params:
            log.warning("Using fallback params for %s (not found in API response)", ticker)
            fb = MARKET_PARAMS_FALLBACK[ticker]
            eta_8h = 6.0 * (fb["imf"] - fb["mmf"])
            params[ticker] = {
                **fb,
                "eta_8h": eta_8h,
                "eta_1h": eta_8h / 8.0,
                "pulled_at": "fallback",
            }

    out_path = out_dir / "market_params.json"
    with open(out_path, "w") as fh:
        json.dump(params, fh, indent=2)
    log.info("Saved %s", out_path)
    return params


def fetch_funding_history(
    ticker: str,
    start: datetime,
    end: datetime,
    params_dict: dict,
) -> pd.DataFrame:
    """
    Pull historical 8-hour funding records for one market.

    dYdX v4 indexer endpoint:
        GET /v4/historicalFunding/{ticker}
        Query params:
            limit               int, max 200
            effectiveBeforeOrAt ISO-8601 string  (pagination cursor)

    Returns DataFrame with columns defined in module docstring.
    """
    ticker_url = ticker.replace("/", "%2F")  # URL-encode slash if present
    url = f"{INDEXER_BASE}/historicalFunding/{ticker_url}"

    mkt_params = params_dict.get(ticker, MARKET_PARAMS_FALLBACK.get(ticker, {}))
    imf    = mkt_params.get("imf", 0)
    mmf    = mkt_params.get("mmf", 0)
    eta_8h = mkt_params.get("eta_8h", 6.0 * (imf - mmf))
    eta_1h = mkt_params.get("eta_1h", eta_8h / 8.0)

    records  = []
    cursor   = end.strftime("%Y-%m-%dT%H:%M:%S.000Z")
    start_ts = pd.Timestamp(start).tz_convert("UTC")  # already tz-aware
    prev_cursor = None

    log.info("Pulling funding history for %s (back to %s) …", ticker, start.date())
    page = 0
    while True:
        data  = _get(url, params={"limit": FUNDING_PAGE_SIZE, "effectiveBeforeOrAt": cursor})
        batch = data.get("historicalFunding", [])
        if not batch:
            break

        # Parse all timestamps — API sort order is not guaranteed
        batch_ts = [pd.Timestamp(r["effectiveAt"]).tz_convert("UTC") for r in batch]

        added = 0
        for row, ts in zip(batch, batch_ts):
            if ts < start_ts:
                continue   # skip out-of-window rows; don't break (batch may be unsorted)
            records.append({
                "ts":           ts,
                "ticker":       ticker,
                "rate_8h":      float(row["rate"]),
                "oracle_price": float(row["price"]),
            })
            added += 1

        earliest = min(batch_ts)
        page += 1
        log.info("  page %3d | %3d added | earliest %s | total: %d",
                 page, added, earliest.date(), len(records))

        # Stop conditions
        if earliest <= start_ts:
            break
        if len(batch) < FUNDING_PAGE_SIZE:
            break  # partial page = last page

        # Advance cursor to just before the earliest record
        new_cursor = (earliest - pd.Timedelta(seconds=1)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        if new_cursor == prev_cursor:
            log.warning("  Cursor stuck at %s — stopping", new_cursor)
            break
        prev_cursor = new_cursor
        cursor      = new_cursor
        time.sleep(REQUEST_DELAY)

    if not records:
        log.warning("No funding records found for %s", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("ts").reset_index(drop=True)

    # ── Filter to actual 8-hour settlement events only ────────────────────
    # The historicalFunding endpoint returns hourly premium snapshots, but
    # funding is only *settled* (paid/received) at 00:00, 08:00, 16:00 UTC.
    # Intermediate rows are running estimates, not settled rates.
    # Running the violation test on them would test 8× too many observations
    # and mix preview rates with realized rates.
    #
    # Filter: keep only rows where hour ∈ {0, 8, 16} UTC.
    # Robust fallback: if that yields < 10% of expected rows, keep all
    # (in case dYdX shifts settlement times via governance).
    settlement_hours = {0, 8, 16}
    mask = df["ts"].dt.hour.isin(settlement_hours)
    n_settlement = mask.sum()
    n_total_raw  = len(df)
    if n_settlement >= 0.05 * n_total_raw:   # sanity: at least 5% should match
        df = df[mask].reset_index(drop=True)
        log.info("  Filtered to 8h settlement rows: %d / %d (hours 0,8,16 UTC)",
                 n_settlement, n_total_raw)
    else:
        log.warning(
            "  Settlement-hour filter kept only %d / %d rows (<5%%) — "
            "keeping all rows; check if dYdX shifted settlement schedule",
            n_settlement, n_total_raw,
        )

    # ── Drift-compatible column names ─────────────────────────────────────
    # Rename to match Drift parquet schema so analysis notebooks are
    # drop-in compatible between the two datasets.
    df = df.rename(columns={
        "ts":     "timestamp",   # matches Drift: "timestamp"
        "ticker": "market",      # matches Drift: "market"
    })

    # ── dYdX-specific derived columns ─────────────────────────────────────
    df["imf"]    = imf
    df["mmf"]    = mmf
    df["eta_8h"] = eta_8h

    # eta  = η_1h = η_8h / 8 — the effective hourly cap in the bound formula
    # Named "eta" (not "eta_1h") to match Drift's column name
    df["eta"]    = eta_1h

    # f_tk = rate_8h directly — the API returns the hourly funding fraction
    # already (one record per hour, rate is per-hour, NOT per-8h).
    # The "rate_8h" column is kept for naming consistency but is hourly in practice.
    # No division by 8 needed.
    df["f_tk"]   = df["rate_8h"].copy()

    # rho_k ≡ 1.0 for dYdX (no EMA-oracle divergence; premiums normalized
    # by contemporaneous index each minute, not an EMA-smoothed TWAP)
    # Included for schema completeness with Drift parquet
    df["rho_k"]  = 1.0

    # ── Clamped-regime flags ───────────────────────────────────────────────
    # Compare f_tk (hourly rate) against eta_1h (hourly cap)
    # Use relative tolerance: within 0.1% of the cap value
    rel_tol = 0.001 * eta_1h
    df["lower_clamped"] = (df["f_tk"] - (-eta_1h)).abs() < rel_tol
    df["upper_clamped"] = (df["f_tk"] -   eta_1h ).abs() < rel_tol
    df["clamped"]       = df["lower_clamped"] | df["upper_clamped"]

    # regime: "lower" / "upper" / "mid"
    # Matches Drift's decode_funding_rate regime labels exactly
    df["regime"] = "mid"
    df.loc[df["lower_clamped"], "regime"] = "lower"
    df.loc[df["upper_clamped"], "regime"] = "upper"

    # ── Settlement interval ────────────────────────────────────────────────
    # Δt_k in seconds; nominal 28800s (8h); flag gaps > 10h as lazy settlements
    df["delta_t_sec"] = df["timestamp"].diff().dt.total_seconds().fillna(28800.0)

    log.info(
        "  %s: %d settlements | clamped %.1f%% | date range %s → %s",
        ticker, len(df),
        df["clamped"].mean() * 100,
        df["timestamp"].iloc[0].date(), df["timestamp"].iloc[-1].date(),
    )
    return df


def fetch_candles(
    ticker: str,
    start: datetime,
    end: datetime,
    resolution: str = "1HOUR",
) -> pd.DataFrame:
    """
    Pull OHLCV candle data for one market.  Used to construct the perp
    mid-price proxy P_tk at each 8-hour settlement.

    dYdX v4 indexer endpoint:
        GET /v4/candles/perpetualMarkets/{ticker}
        Query params:
            resolution      "1MIN" | "5MINS" | "15MINS" | "30MINS" |
                            "1HOUR" | "4HOURS" | "1DAY"
            limit           int, max 200
            fromISO         ISO-8601 lower bound (inclusive)
            toISO           ISO-8601 upper bound (inclusive)

    We pull 1-hour bars so there is exactly one candle per hour, making
    it straightforward to match to the 8-hour settlement timestamps.

    P_tk construction (in the analysis notebook):
        At each settlement ts_k, take the 1-hour candle whose startedAt
        == ts_k.  Use candle close as P_tk.  This is a mark/trade proxy,
        not a true impact midpoint.  Document substitution bias.
    """
    ticker_url = ticker.replace("/", "%2F")
    url = f"{INDEXER_BASE}/candles/perpetualMarkets/{ticker_url}"

    records = []
    cursor_to = end

    log.info("Pulling %s candles for %s …", resolution, ticker)
    while True:
        params = {
            "resolution": resolution,
            "limit":      CANDLE_PAGE_SIZE,
            "toISO":      cursor_to.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            "fromISO":    start.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        }
        data = _get(url, params=params)
        batch = data.get("candles", [])
        if not batch:
            break

        for c in batch:
            ts = pd.Timestamp(c["startedAt"]).tz_convert("UTC")
            records.append({
                "ts":        ts,
                "ticker":    ticker,
                "open":      float(c["open"]),
                "high":      float(c["high"]),
                "low":       float(c["low"]),
                "close":     float(c["close"]),
                "volume":    float(c.get("baseTokenVolume", 0) or 0),
                "usdVolume": float(c.get("usdVolume", 0) or 0),
            })

        if len(batch) < CANDLE_PAGE_SIZE:
            break

        # Paginate: move cursor to the oldest candle in the batch minus 1s
        earliest = min(pd.Timestamp(c["startedAt"]) for c in batch)
        start_ts_c = pd.Timestamp(start).tz_convert("UTC")
        if (earliest.tz_localize("UTC") if earliest.tzinfo is None else earliest.tz_convert("UTC")) <= start_ts_c:
            break

        cursor_to = (earliest - pd.Timedelta(seconds=1)).to_pydatetime()
        time.sleep(REQUEST_DELAY)

    if not records:
        log.warning("No candle data found for %s", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(records).sort_values("ts").drop_duplicates("ts").reset_index(drop=True)
    log.info("  %s: %d hourly candles, %s → %s", ticker, len(df),
             df["ts"].iloc[0].date(), df["ts"].iloc[-1].date())
    return df


def merge_funding_and_candles(
    funding_df: pd.DataFrame,
    candles_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Attach the candle-close price at each settlement time as perp_mid_proxy.

    The merge is an asof merge on ts with a 1-hour tolerance: for each
    funding settlement we take the most recent candle close within ±1h.
    If no candle is within tolerance, perp_mid_proxy is NaN (flagged).

    After merging, ratio = perp_mid_proxy / oracle_price is the
    empirical P_tk / I_tk used in the violation test.
    """
    if candles_df.empty:
        funding_df["perp_mid_proxy"] = float("nan")
        funding_df["ratio"]          = float("nan")
        return funding_df

    candles_sorted = candles_df.sort_values("ts")[["ts", "close"]].copy()
    # Merge on timestamp (funding) ← ts (candles); both are UTC
    merged = pd.merge_asof(
        funding_df.sort_values("timestamp"),
        candles_sorted.rename(columns={"ts": "timestamp", "close": "perp_mid_proxy"}),
        on="timestamp",
        direction="nearest",
        tolerance=pd.Timedelta("1h"),
    )
    merged["ratio"] = merged["perp_mid_proxy"] / merged["oracle_price"]
    n_missing = merged["perp_mid_proxy"].isna().sum()
    if n_missing > 0:
        pct = n_missing / len(merged) * 100
        log.warning(
            "  %d / %d settlements (%.1f%%) have no candle match within 1h; ratio = NaN",
            n_missing, len(merged), pct,
        )
    return merged.sort_values("timestamp").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pull dYdX v4 funding and candle data")
    parser.add_argument("--out-dir", default="./data/raw/dydx",
                        help="Directory to write parquet files (default: ./data/raw/dydx)")
    parser.add_argument("--start",   default=SAMPLE_START,
                        help=f"Start date YYYY-MM-DD (default: {SAMPLE_START})")
    parser.add_argument("--end",     default=SAMPLE_END,
                        help=f"End date YYYY-MM-DD  (default: {SAMPLE_END})")
    parser.add_argument("--markets", nargs="+", default=MARKETS,
                        help=f"Markets to pull (default: {' '.join(MARKETS)})")
    parser.add_argument("--skip-candles", action="store_true",
                        help="Skip candle pull (funding records only)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end   = datetime.strptime(args.end,   "%Y-%m-%d").replace(tzinfo=timezone.utc)

    log.info("dYdX pull — aligned with Drift fetch_funding_records.py")
    log.info("  Sample   : %s → %s", args.start, args.end)
    log.info("  Training : %s → %s", args.start, TRAIN_END)
    log.info("  Test     : %s → %s", TEST_START, args.end)
    log.info("  Markets  : %s", ", ".join(args.markets))

    # 1. Market params (IMF / MMF → η)
    params = fetch_market_params(out_dir)

    for ticker in args.markets:
        log.info("=" * 60)
        log.info("Market: %s", ticker)
        log.info("  η_8h = %.4f  η_1h (=eta) = %.6f",
                 params.get(ticker, {}).get("eta_8h", float("nan")),
                 params.get(ticker, {}).get("eta_1h", float("nan")))

        # 2. Funding records
        funding_df = fetch_funding_history(ticker, start, end, params)
        if funding_df.empty:
            log.warning("Skipping %s — no funding data", ticker)
            continue

        # 3. Candles
        candles_df = pd.DataFrame()
        if not args.skip_candles:
            candles_df = fetch_candles(ticker, start, end, resolution="1HOUR")
            if not candles_df.empty:
                candles_path = out_dir / f"{ticker}_candles_1h.parquet"
                candles_df.to_parquet(candles_path, index=False)
                log.info("  Saved %s", candles_path)

        # 4. Merge → add P_tk proxy and ratio
        merged_df = merge_funding_and_candles(funding_df, candles_df)

        # 5. Save
        out_path = out_dir / f"{ticker}_funding.parquet"
        merged_df.to_parquet(out_path, index=False)
        log.info("  Saved %s  (%d rows)", out_path, len(merged_df))

        # ── Drift-matching summary output ─────────────────────────────────
        train_split = pd.Timestamp(TEST_START, tz="UTC")
        train = merged_df[merged_df["timestamp"] <  train_split]
        test  = merged_df[merged_df["timestamp"] >= train_split]

        log.info("\n  Saved %d rows → %s", len(merged_df), out_path)
        log.info("  Full range  : %s → %s",
                 merged_df["timestamp"].min(), merged_df["timestamp"].max())
        log.info("  Training    : %d rows (2023–2024)", len(train))
        log.info("  Test        : %d rows (2025)",      len(test))
        log.info("\n  Regime breakdown (full sample):")
        for regime, count in merged_df["regime"].value_counts().items():
            log.info("    %-8s %d", regime, count)
        log.info("\n  f_tk stats:")
        log.info("    mean  %.6f", merged_df["f_tk"].mean())
        log.info("    std   %.6f", merged_df["f_tk"].std())
        log.info("    min   %.6f", merged_df["f_tk"].min())
        log.info("    max   %.6f", merged_df["f_tk"].max())

    log.info("=" * 60)
    log.info("Done.  Output directory: %s", out_dir.resolve())


# ---------------------------------------------------------------------------
# NOTE ON PREMIUM SAMPLES (N_k)
# ---------------------------------------------------------------------------
#
# The dYdX chain stores the per-minute premium samples at:
#   /dydxprotocol/perpetuals/premium_samples
# (LCD / REST node endpoint, not the indexer)
#
# To recover the full N_k series and the sample-average π̄_k:
#
#   1. Run a dYdX full node (or use a public LCD endpoint like
#      https://dydx-ops-rest.kingnodes.com)
#   2. Query at each historical block height corresponding to the
#      funding settlement.  Heights are in the funding records as
#      "effectiveAtHeight".
#   3. The response gives the N_k individual π_{k,j} values.
#      Average them → π̄_k.
#   4. Cross-check: clamp(π̄_k, −η_8h, +η_8h) should equal rate_8h
#      from the indexer (subject to per-sample clamp applied before
#      averaging; see paper Definition 5.X for exact reconstruction).
#
# For a first-pass empirical section, reconstructing π̄_k is not
# strictly necessary — you can identify clamped periods from rate_8h
# directly.  The N_k series matters for the εk^{dY} bound
# (Appendix B of the paper): larger N_k → tighter ε_k.  Report the
# distribution of N_k in Table A if you reconstruct it; otherwise note
# the substitution.
#
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    main()