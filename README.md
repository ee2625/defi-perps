# DeFi Perpetuals — No-Arbitrage Bounds

Empirical analysis accompanying *Arbitrage Bounds for DeFi Perpetual Contracts*. The paper adapts the no-arbitrage framework of Dai, Li, and Yang (2025) to on-chain decentralized perpetual exchanges — Drift Protocol and dYdX v4 — where oracle-based funding mechanisms introduce a tracking error layer absent from centralized venues.

---

## Repository structure

```
defi-perps/
├── drift/
│   ├── collection/
│   │   ├── fetch_funding_records.py      # pulls FundingRateRecord data from Drift S3
│   │   └── fetch_trade_volumes.py        # streams Drift S3 tradeRecords; sums quoteAssetAmountFilled
│   ├── analysis/
│   │   ├── test1_bounds.ipynb            # main empirical notebook (Drift)
│   │   ├── robustness_checks.py          # bootstrap + placebo + rolling-Γ
│   │   ├── robustness_checks.ipynb       # cell-by-cell mirror of the script
│   │   ├── figure1_bounds_timeseries.png
│   │   ├── figure2_violation_magnitudes.png
│   │   ├── figure3_violation_durations.png
│   │   ├── figure4_band_width.png
│   │   ├── figure5_strategy_pnl.png
│   │   └── figure6_placebo_distributions.png
│   └── data/                             # gitignored — regenerate with collection scripts
│       ├── raw/
│       └── processed/
├── dydx/
│   ├── collection/
│   ├── analysis/
│   │   ├── dydx_test1_bounds.ipynb       # main empirical notebook (dYdX)
│   │   ├── figure1_bounds_timeseries.png
│   │   ├── figure2_violation_magnitudes.png
│   │   ├── figure3_violation_durations.png
│   │   ├── figure4_band_width.png
│   │   └── figure5_pct_of_cap.png
│   └── data/                             # gitignored
└── analysis/
    └── verify_intro_stats.py             # cross-checks introduction volume figures
```

---

## Data

### Drift Protocol

**Source:** Drift Protocol S3 historical data repository (`fundingRateHistory` account store, `tradeRecords` for cumulative volume)
**Markets:** BTC-PERP, ETH-PERP, SOL-PERP
**Period:** January 1, 2023 — December 31, 2024
**Frequency:** Hourly (one row per `FundingRateRecord` event)
**Observations:** 17,417 / 17,396 / 17,494 per market (~52,307 total)

Each record carries: `markPriceTwap`, `oraclePriceTwap`, `fundingRate`, `oraclePrice`, `ts` (Unix timestamp in microseconds). All price fields are in Drift's scaled integer format; conversion uses the precision conventions in Section 2 of the paper.

Raw data files are gitignored. To regenerate funding records:

```bash
python drift/collection/fetch_funding_records.py
```

This pulls daily CSV files from the Drift S3 bucket and saves three parquet files to `drift/data/raw/`. Takes approximately 25–30 minutes.

To compute cumulative trade volume from the same archive (for the §1 motivation figure):

```bash
python drift/collection/fetch_trade_volumes.py
```

Streams daily `tradeRecords` CSVs in parallel and sums `quoteAssetAmountFilled` per market. Takes ~85 seconds. Note: Drift's S3 archive cuts off in early January 2025, so this fetcher is bounded to the funding-records sample window.

### dYdX v4

**Source:** dYdX v4 public indexer API (`/v4/historicalFunding/{ticker}`), candle data from `/v4/candles/perpetualMarkets/{ticker}`
**Markets:** BTC-USD, ETH-USD, SOL-USD
**Period:** October 27, 2023 (mainnet launch) — December 30, 2025
**Frequency:** 8-hourly (one row per funding settlement)
**Observations:** 2,395 per market (7,185 total); 161 records excluded due to missing candle match within 1 hour

Each record carries: `rate` (realized 8-hour clamped funding rate), `price` (oracle index price at settlement). The hourly candle close is used as a proxy for the perpetual mid-price $P_{t_k}$; see Section 8.2 of the paper for discussion of proxy noise.

---

## Empirical design

### Drift — key variables

| Symbol | Column | Description |
|--------|--------|-------------|
| $\bar{I}_{t_k}$ | `oracle_twap` | Oracle EMA-TWAP (USD) |
| $M_{t_k}$ | `mark_twap` | Mark EMA-TWAP — midpoint of bid/ask TWAPs (USD) |
| $f_{t_k}$ | `f_tk` | Realized hourly funding fraction |
| $\eta_{t_k}$ | `eta` | Tier-dependent hourly cap (0.00125 for Tier B = 12.5 bps) |
| $\rho_k$ | `rho_k` | Oracle TWAP ratio $\bar{I}_{t_k}/I_{t_k}$ (set to 1.0; see note below) |

**Note on $\rho_k$:** The Drift S3 records expose only the EMA-TWAP $\bar{I}_{t_k}$, not the instantaneous oracle price $I_{t_k}$. We set $\rho_k = 1.0$ throughout — defensible for liquid hourly markets where EMA lag is small. The ratio tested is therefore $M_{t_k}/\bar{I}_{t_k}$ rather than $P_{t_k}/I_{t_k}$.

### dYdX — key variables

| Symbol | Column | Description |
|--------|--------|-------------|
| $f_k$ | `f_tk` | Realized 8-hour funding fraction (= `rate` from API) |
| $\eta_{1h}$ | `eta` | Implied hourly cap: $\tfrac{1}{8}\eta_{8h}$ where $\eta_{8h} = 6(\mathrm{IMF}-\mathrm{MMF})$ |
| $P_{t_k}/I_{t_k}$ | `ratio` | Candle-close proxy for perpetual-to-oracle ratio |

Funding caps: BTC-USD and ETH-USD use IMF = 2.0%, MMF = 1.2%, giving $\eta_{8h} = 4.8\%$ and $\eta_{1h} = 60$ bps/hour. SOL-USD uses IMF = 5.0%, MMF = 3.0%, giving $\eta_{8h} = 12.0\%$ and $\eta_{1h} = 150$ bps/hour. These are 5–12× wider than Drift's Tier B cap of 12.5 bps/hour.

### Train/test split

| Protocol | Training window | Test window | Purpose |
|----------|----------------|-------------|---------|
| Drift | Jan 2023 — Dec 2023 | Jan 2024 — Dec 2024 | Calibrate $\hat\Gamma$, evaluate violations |
| dYdX | Oct 2023 — Dec 2024 | Jan 2025 — Dec 2025 | Calibrate $\hat\Gamma$, evaluate cap proximity |

$\hat\Gamma$ is the 99th percentile of $|M_{t_{k+1}} - \bar{I}_{t_{k+1}}|$ (Drift) or $|P_{t_{k+1}} - I_{t_{k+1}}|$ (dYdX) on the training window, held fixed for out-of-sample evaluation.

### No-arbitrage bounds (Drift, simplified: $\kappa_F = r = r^C = 0$)

$$\bar{L}_{t_k}^{\mathrm{emp}} = (1 - \kappa_S) + |f_{t_k}|\rho_k - \frac{\hat\Gamma}{\bar{I}_{t_k}}, \qquad \bar{U}_{t_k}^{\mathrm{emp}} = (1 + \kappa_S) - |f_{t_k}|\rho_k + \frac{\hat\Gamma}{\bar{I}_{t_k}}$$

Baseline: $\kappa_S = 1.8$ bps (two-sided maker spot fee). Conservative robustness check: $\kappa_S = 4.5$ bps (full taker fee).

---

## Results

### Drift — Test 1: Violation rates (2024 test window)

| Market | Obs | Violations | Violation rate | Mean magnitude | Mean duration |
|--------|-----|------------|----------------|----------------|---------------|
| BTC-PERP | 8,774 | 399 | 4.55% | 10.0 bps | 6.2 hours |
| ETH-PERP | 8,773 | 282 | 3.21% | 9.7 bps | 5.6 hours |
| SOL-PERP | 8,773 | 440 | 5.02% | 9.2 bps | 4.9 hours |

The ratio $M_{t_k}/\bar{I}_{t_k}$ stays inside the no-arbitrage bounds 95–97% of the time. Violations are brief (mean 5–6 hours) and small (~10 bps), consistent with transaction cost friction rather than persistent exploitable dislocations. Two violation clusters — January–February 2024 and March–April 2024 — coincide with periods of elevated directional momentum.

### Drift — Robustness: $\hat\Gamma$ percentile

| $\hat\Gamma$ percentile | BTC | ETH | SOL |
|-------------------------|-----|-----|-----|
| 99th (baseline) | 4.55% | 3.21% | 5.02% |
| 95th | 14.47% | 9.96% | 23.38% |
| 90th | 24.71% | 19.35% | 32.06% |

### Drift — Robustness: fee assumption ($\hat\Gamma$ at 99th percentile)

| Fee assumption | BTC | ETH | SOL |
|----------------|-----|-----|-----|
| Baseline ($\kappa_S = 1.8$ bps) | 4.55% | 3.21% | 5.02% |
| Conservative ($\kappa_S = 4.5$ bps) | 3.42% | 2.37% | 3.83% |

### Drift — Robustness: rolling-window $\hat\Gamma$

| $\hat\Gamma$ specification | BTC | ETH | SOL |
|----------------------------|-----|-----|-----|
| Static 2023 (baseline) | 4.55% | 3.21% | 5.02% |
| Expanding window | 1.79% | 1.86% | 1.87% |
| Trailing 365 days | 1.44% | 1.72% | 1.39% |

The expanding and trailing-365d specifications incorporate 2024 information into the calibrated $\hat\Gamma$, mechanically widening the band. All three specifications produce positive violation rates.

### Drift — Within-test stability (H1 vs H2 of 2024, static 2023 $\hat\Gamma$)

| Market | H1 rate | H2 rate |
|--------|---------|---------|
| BTC-PERP | 7.56% | 1.56% |
| ETH-PERP | 5.91% | 0.54% |
| SOL-PERP | 7.17% | 2.88% |

Violations concentrate in H1, reflecting the March–April 2024 momentum episode. H2 rates remain positive across all markets.

### Drift — Reduced-form reversion test

A mean-reversion strategy — enter long (short) ratio at each lower (upper) bound violation, exit when $|r_{t_k} - 1| < 10$ bps or after 48 settlement periods — generates positive cumulative PnL across all configurations.

| Scenario | Market | Trades | Win rate | Mean PnL (bps) | Total PnL (bps) | Sharpe |
|----------|--------|--------|----------|----------------|-----------------|--------|
| Frictionless, price only | BTC-PERP | 30 | 86.7% | +15.75 | +472.4 | 1.12 |
| Frictionless, price only | ETH-PERP | 32 | 93.8% | +16.80 | +537.6 | 1.76 |
| Frictionless, price only | SOL-PERP | 43 | 93.0% | +21.44 | +922.1 | 0.99 |
| Net of 3.6 bps r/t, price only | BTC-PERP | 30 | 83.3% | +12.15 | +364.4 | 0.86 |
| Net of 3.6 bps r/t, price only | ETH-PERP | 32 | 90.6% | +13.20 | +422.4 | 1.38 |
| Net of 3.6 bps r/t, price only | SOL-PERP | 43 | 93.0% | +17.84 | +767.3 | 0.82 |
| Net of 3.6 bps r/t, with funding | BTC-PERP | 30 | 100.0% | +44.95 | +1,348.6 | 1.92 |
| Net of 3.6 bps r/t, with funding | ETH-PERP | 32 | 100.0% | +32.13 | +1,028.0 | 1.68 |
| Net of 3.6 bps r/t, with funding | SOL-PERP | 43 | 100.0% | +42.17 | +1,813.4 | 1.60 |

Sharpe is per-trade (mean net PnL / std net PnL), not annualized. Including the per-period funding income accumulated over the holding window approximately triples net total PnL: bound violations occur precisely in regimes where $|f_{t_k}|$ is large and signed in the position's favor. The primary drag on the price-only series is the forced-exit population (9–30% of trades, concentrated in the March–April 2024 persistence episode).

### Drift — Statistical robustness of the strategy

Bootstrap (1,000 resamples) and placebo (1,000 random-entry simulations with the actual exit rule) on the funding-augmented strategy:

| Market | Total PnL [95% CI] | Sharpe [95% CI] | Placebo mean | Placebo SD | Empirical $p$-value |
|--------|--------------------|-----------------|--------------|------------|---------------------|
| BTC-PERP | +1,349 [+1,107, +1,587] | 1.92 [1.55, 2.54] | −110 bps | 104 bps | < 0.001 |
| ETH-PERP | +1,028 [+830, +1,253] | 1.68 [1.40, 2.26] | −116 bps | 80 bps | < 0.001 |
| SOL-PERP | +1,813 [+1,522, +2,177] | 1.60 [1.37, 2.11] | −150 bps | 110 bps | < 0.001 |

The placebo means cluster near $-3.6 \times N_{\text{trades}}$ basis points (the round-trip cost of $N$ random trades). The actual realized PnL lies 14–18 standard deviations above the placebo mean.

### dYdX — Cap proximity (2025 test window)

| Market | Max hourly $|f_k|$ | Cap $\eta_{1h}$ | Max as % of cap |
|--------|--------------------|-----------------|-----------------|
| BTC-USD | 0.12 bps | 60 bps | 0.20% |
| ETH-USD | 0.15 bps | 60 bps | 0.25% |
| SOL-USD | 0.26 bps | 150 bps | 0.18% |

The funding cap never comes close to binding. The dYdX no-arbitrage bounds are theoretically valid but empirically vacuous: the mechanism designed to enforce them never activates. This is a direct consequence of the cap being calibrated to margin parameters ($\eta_{8h} = 6(\mathrm{IMF} - \mathrm{MMF})$) rather than to oracle-tracking objectives, producing caps 5–12× wider than Drift's.

---

## Setup

**Requirements:** Python 3.11+

```bash
# clone the repo
git clone https://github.com/ee2625/defi-perps.git
cd defi-perps

# create virtual environment
python3.11 -m venv venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

# regenerate Drift funding records (~30 mins)
python drift/collection/fetch_funding_records.py

# (optional) compute Drift cumulative volume (~85 seconds)
python drift/collection/fetch_trade_volumes.py

# (optional) re-run Drift robustness checks (bootstrap + placebo + rolling Γ)
python drift/analysis/robustness_checks.py

# open notebooks
jupyter notebook drift/analysis/test1_bounds.ipynb
jupyter notebook drift/analysis/robustness_checks.ipynb
jupyter notebook dydx/analysis/dydx_test1_bounds.ipynb
```

---

## References

- Dai, M., Li, L., and Yang, C. (2025). *Arbitrage in Perpetual Contracts*. SSRN working paper.
- Ackerer, D., Hugonnier, J., and Jermann, U. (2025). *Perpetual Futures Pricing*. *Mathematical Finance*, Early View.
- Drift Protocol. *Funding Rates*. https://docs.drift.trade/protocol/trading/perpetuals-trading/funding-rates. Accessed February 5, 2026.
- Drift Protocol. *Protocol v2 Source Code: Funding Controller*. https://github.com/drift-labs/protocol-v2/blob/master/programs/drift/src/controller/funding.rs. Accessed February 16, 2026.
- dYdX. *Funding*. https://docs.dydx.xyz/concepts/trading/funding. Accessed February 5, 2026.
