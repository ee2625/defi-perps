# DeFi Perpetuals — No-Arbitrage Bounds

Empirical analysis accompanying the thesis *Arbitrage in DeFi Perpetual Futures*, supervised by Professor Smirnov. The paper adapts the no-arbitrage framework of Dai, Li, and Yang (2024) to on-chain decentralized perpetual exchanges — Drift Protocol and dYdX v4 — where oracle-based funding mechanisms introduce tracking error absent from centralized venues.

---

## Repository structure

```
defi-perps/
├── drift/
│   ├── collection/
│   │   └── fetch_funding_records.py   # pulls FundingRateRecord data from Drift S3
│   ├── processing/                    # (forthcoming) panel construction
│   ├── analysis/
│   │   ├── test1_bounds.ipynb         # main empirical notebook
│   │   ├── figure1_bounds_timeseries.png
│   │   ├── figure2_violation_magnitudes.png
│   │   ├── figure3_violation_durations.png
│   │   ├── figure4_band_width.png
│   │   └── figure5_strategy_pnl.png
│   └── data/                          # gitignored — regenerate with collection script
│       ├── raw/
│       └── processed/
└── dydx/
    ├── collection/                    # (forthcoming)
    ├── processing/
    ├── analysis/
    └── data/                          # gitignored
```

---

## Data

**Source:** Drift Protocol S3 historical data repository  
**Markets:** SOL-PERP, BTC-PERP, ETH-PERP  
**Period:** January 1 2023 — December 31 2024  
**Frequency:** Hourly (one row per FundingRateRecord event)  
**Observations:** ~17,400–17,500 per market (~52,300 total)

The raw data files are gitignored and not stored in this repository. To regenerate them, run the collection script from the repo root:

```bash
python drift/collection/fetch_funding_records.py
```

This will pull daily CSV files from the Drift S3 bucket and save three parquet files to `drift/data/raw/`. The script takes approximately 25–30 minutes to complete for all three markets.

---

## Empirical design

### Variables

| Symbol | Column | Description |
|--------|--------|-------------|
| $\bar{I}_{t_k}$ | `oracle_twap` | Oracle EMA-TWAP in USD |
| $M_{t_k}$ | `mark_twap` | Mark EMA-TWAP in USD (bid/ask midpoint TWAP) |
| $f_{t_k}$ | `f_tk` | Realized hourly funding fraction |
| $\eta_{t_k}$ | `eta` | Tier-dependent hourly funding cap (0.00125 for Tier B) |
| $\rho_k$ | `rho_k` | Oracle TWAP ratio $\bar{I}_{t_k}/I_{t_k}$ (set to 1.0, see note) |

**Note on $\rho_k$:** The Drift S3 records do not contain the instantaneous oracle price $I_{t_k}$ — only the EMA-TWAP $\bar{I}_{t_k}$. We set $\rho_k = 1.0$ throughout, which is defensible for liquid hourly markets where EMA lag is small. The ratio tested is therefore $M_{t_k}/\bar{I}_{t_k}$ rather than $P_{t_k}/I_{t_k}$.

### Train/test split

| Window | Period | Rows per market | Purpose |
|--------|--------|-----------------|---------|
| Training | Jan 2023 — Dec 2023 | ~8,700 | Estimate $\hat\Gamma$ |
| Test | Jan 2024 — Dec 2024 | ~8,700 | Out-of-sample violation test |

$\hat\Gamma$ is the 99th percentile of $|M_{t_{k+1}} - \bar{I}_{t_{k+1}}|$ estimated on the training window and fixed for the test window.

### No-arbitrage bounds (simplified, $\kappa_F = r = r^C = 0$)

$$\bar{L}_{t_k} = (1 - \kappa_S) + |f_{t_k}|\rho_k - \frac{\hat\Gamma}{\bar{I}_{t_k}}$$

$$\bar{U}_{t_k} = (1 + \kappa_S) - |f_{t_k}|\rho_k + \frac{\hat\Gamma}{\bar{I}_{t_k}}$$

where $\kappa_S = 1.8$ bps (spot round-trip fee).

---

## Results

### Test 1 — Violation rates (test window 2024)

| Market | Obs | Violations | Violation rate | Mean magnitude | Mean duration |
|--------|-----|------------|----------------|----------------|---------------|
| SOL-PERP | 8,773 | 440 | 5.02% | 9.2 bps | 4.9 hours |
| BTC-PERP | 8,774 | 399 | 4.55% | 10.0 bps | 6.2 hours |
| ETH-PERP | 8,773 | 282 | 3.21% | 9.7 bps | 5.6 hours |

The ratio $M_{t_k}/\bar{I}_{t_k}$ stays inside the no-arbitrage bounds **95–97% of the time** across all three markets. Violations are brief (mean 5–6 hours) and small in magnitude (~10 bps), consistent with transaction cost friction rather than persistent exploitable dislocations.

### Robustness — sensitivity to $\hat\Gamma$ choice

| $\hat\Gamma$ percentile | SOL violation rate | BTC violation rate | ETH violation rate |
|------------------------|-------------------|-------------------|-------------------|
| 99th | 5.0% | 4.6% | 3.2% |
| 95th | 23.4% | 14.5% | 10.0% |
| 90th | 32.1% | 24.7% | 19.4% |

Results are sensitive to the Gamma calibration choice. The 99th percentile is the most conservative empirical estimate. An alternative theory-consistent calibration ($\Gamma = 24\eta_{t_k}\bar{I}_{t_k}$) is under consideration.

### Trading strategy

A simple mean-reversion strategy — enter on bound violation, exit when ratio reverts to within 10 bps of 1.0, 48-hour max hold — earns negative returns after a 1.8 bps round-trip cost across all three markets:

| Market | Trades | Win rate | Mean PnL (net) | Total PnL |
|--------|--------|----------|----------------|-----------|
| SOL-PERP | 43 | 7.0% | −25.0 bps | −1,077 bps |
| BTC-PERP | 30 | 10.0% | −19.4 bps | −580 bps |
| ETH-PERP | 32 | 6.2% | −20.4 bps | −653 bps |

This confirms that observed violations do not represent exploitable arbitrage opportunities after transaction costs.

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

# regenerate data (takes ~30 mins)
python drift/collection/fetch_funding_records.py

# open analysis notebook
jupyter notebook drift/analysis/test1_bounds.ipynb
```

---

## References

- Dai, M., Li, L., and Yang, C. (2024). *Arbitrage in Perpetual Contracts*. Working paper.
- Drift Protocol. *Funding Rates*. https://docs.drift.trade/protocol/trading/perpetuals-trading/funding-rates
- Drift Protocol. *Protocol v2 Source Code*. https://github.com/drift-labs/protocol-v2