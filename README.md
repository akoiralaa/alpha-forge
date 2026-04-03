# One Brain Fund — Quantitative Trading Platform

A Renaissance-style systematic trading engine that ranks stocks cross-sectionally across the S&P 500 using multi-factor signals, constructs sector-neutral portfolios, and manages risk dynamically through volatility regimes and drawdown circuit breakers.

## Backtest Results vs SPY (2007-2026, $10M starting NAV)

### Canonical Version Comparison (v1-v10)

Canonical source of truth: `data/reports/version_comparison_v1_v10.csv`

| Version | CAGR | Sharpe | Max DD | Final NAV ($M) | Turnover (x/yr) | Tx Costs (bps/yr) | Source |
|---|---:|---:|---:|---:|---:|---:|---|
| v1 | +2.00% | 0.20 | -40.00% | n/a | n/a | n/a | readme_baseline |
| v2 | +4.01% | 0.41 | -34.20% | $22.10M | 11x | 33 | readme_baseline |
| v3 | +8.39% | 0.57 | -32.00% | $55.20M | 22x | 66 | readme_baseline |
| v4 | +10.97% | 0.71 | -29.79% | $70.63M | 21x | 62 | current_run |
| v5 | +10.73% | 0.67 | -33.96% | $85.60M | 21x | 62 | devlog_calibrated |
| v6 | +4.61% | 0.66 | -15.75% | $25.80M | 7x | 21 | devlog_calibrated |
| v7 | +11.09% | 0.72 | -29.45% | $72.05M | 21x | 62 | current_run |
| v8 | +11.03% | 0.78 | -20.57% | $71.36M | 17x | 48 | current_run |
| v9 | +10.97% | 0.78 | -20.42% | $70.64M | 17x | 48 | current_run |
| v10 | +10.94% | 0.76 | -20.45% | $73.93M | 18x | 52 | current_run |

Notes:
- `v1` is approximate historical baseline (`~2% CAGR`, `~0.2 Sharpe`, `-40%+ DD`).
- `v2-v6` are from stable published/calibrated records.
- `v7-v10` are current calibrated runs.
- Consistency check: `./.venv/bin/python scripts/check_version_comparison_table.py`

### Version Notes (v1-v10)

- `v1`: baseline single-factor engine with simple environment scoring, high turnover, and weak crash control.
- `v2`: introduced risk budgets by asset class, lower turnover, and smoother execution; improved survivability at the cost of slower growth.
- `v3`: moved to cross-sectional multi-factor ranking (6 factors, residual momentum lead), added adaptive factor weighting and stronger drawdown controls.
- `v4`: promoted multi-strategy stacking as core engine (8 sleeves, risk-parity blend, volatility targeting, crisis overlays); became the main production-style baseline.
- `v5`: added central allocator/capacity logic to re-route risk across sleeves based on live-quality/performance diagnostics.
- `v6`: conservative deployment branch; tightened risk and governance for lower drawdown/time-under-water, sacrificing upside.
- `v7`: added event-driven PIT alpha (SEC fundamentals, estimates/revisions, sentiment quality gating) on top of the v4-style core.
- `v8`: expanded to multi-asset sleeves (ETF, futures, FX) plus options hedge path, execution realism, and strict walk-forward/no-lookahead validation tooling.
- `v9`: `v8` plus lightweight free-data macro overlay (FRED curve/unemployment/rates) as an optional risk-scaling de-risk layer.
- `v10`: implemented the consistency stack (router/convexity/whipsaw/yearly budget), then locked to router-only for alpha recovery after ablation; removed always-on hedge floor and kept strict no-lookahead walk-forward validation.

### v4 Year-by-Year vs SPY

```
Year     v4 Ret      SPY    Alpha    v4 DD   SPY DD
────────────────────────────────────────────────────
2007    +1.35%  -4.11%  +5.46%   -11.74%  -9.95%
2008   -29.40% -37.71%  +8.31%   -31.60% -47.92%   ← crash protection
2009    -2.40% +19.88% -22.28%    -9.41% -27.13%
2010   +15.99% +10.99%  +5.01%    -9.30% -16.09%
2011    -5.55%  -1.19%  -4.36%   -22.31% -19.49%
2012   +20.56% +11.79%  +8.77%   -13.04%  -9.66%
2013   +48.56% +26.37% +22.19%   -10.97%  -6.06%   ← +22% alpha
2014   +13.09% +12.37%  +0.72%   -12.65%  -7.70%
2015    +7.10%  -0.76%  +7.85%   -13.82% -12.29%
2016    +3.89% +11.20%  -7.31%   -18.15%  -9.19%
2017   +79.84% +19.38% +60.45%    -7.27%  -3.03%   ← +60% alpha
2018    -8.91%  -7.01%  -1.90%   -26.66% -20.18%
2019   +38.67% +28.65% +10.02%   -11.58%  -6.62%   ← +10% alpha
2020   +20.09% +15.09%  +5.00%   -20.45% -34.10%   ← half SPY DD
2021   +23.79% +28.79%  -5.00%   -13.85%  -5.42%
2022   -20.72% -19.95%  -0.78%   -25.40% -25.36%
2023    +5.93% +24.81% -18.88%   -16.59% -10.29%
2024   +26.77% +24.00%  +2.77%   -24.66%  -8.41%
2025   +16.94% +16.64%  +0.31%   -23.66% -19.00%
2026    +2.01%  -4.81%  +6.82%   -19.60%  -9.13%

Beat SPY return:  13/20 years
Loss years:        5/20
```

### Monte Carlo (10K simulations, 1-year forward)

```
Median return:  +10.16%
5th percentile: -21.81%
95th percentile:+53.01%
Prob of loss:    31.4%
Prob of >20%:    33.7%
Median max DD:  -16.82%
```

## Strategy Evolution

### v1 — Baseline (`backtest.py`)
Single-factor signal engine with environment scoring. Basic vol-targeting and Kelly sizing. No cross-sectional ranking.

### v2 — Risk-Budgeted (`backtest_v2.py`)
- Per-asset-class risk budgets (equities 50%, futures 12%, commodities 10%, bonds 8%, FX 10%)
- Position concentration (top N by signal strength)
- Turnover dampening via weight blending
- Parquet data caching to `~/.one_brain_fund/cache/bars/`
- Monthly attribution and per-trade logging

### v3 — Cross-Sectional Alpha Engine (`backtest_v3.py`)
The core innovation. Instead of asking "will AAPL go up?", asks "will AAPL outperform MSFT?"

**6-Factor Model:**
1. **Residual Momentum** (Jegadeesh-Titman, beta-hedged) — 12-1 month returns with market and sector beta stripped out
2. **Quality** (Asness) — high idiosyncratic return / low vol proxy
3. **Short-term Reversal** — 5-day mean reversion
4. **Trend** (Moskowitz-Ooi-Pedersen) — SMA50/200 crossover + slope
5. **Earnings Drift Proxy** — post-gap continuation without earnings data
6. **52-Week High Proximity** (George & Hwang) — anchoring bias

**Key Features:**
- Adaptive IC-weighted factor combination (trailing 63-day Information Coefficient)
- Conditional shorts (only in bearish environments, env < 0.75)
- Drawdown circuit breaker (-12% mild, -20% caution, -30% emergency)
- Market breadth overlay (% stocks above 200-day SMA)
- Sector-neutral portfolio construction

### v4 — Multi-Strategy Engine (`backtest_v4.py`)
8 independent strategy signals combined with risk-parity weighting:

| Strategy | Weight | Description |
|----------|--------|-------------|
| Residual Momentum | 25% | Beta-hedged 12-1m cross-sectional |
| Quality | 25% | Stable idiosyncratic returns / low vol |
| 52-Week High | 25% | Anchoring bias (George & Hwang 2004) |
| Carry | 15% | Dividend yield proxy from price patterns |
| Sector Rotation | 10% | 3-1 month sector momentum |
| Mean Reversion | 0% | Disabled (needs daily rebal, incompatible with 15d freq) |
| BAB | 0% | Disabled (growth decade killed low-beta premium) |
| Earnings Drift | 0% | Disabled (noisy signal) |

**Strategy Correlation Matrix:**
```
              Mom   MRev   Qual    BAB  Carry  SecRt EDrift   52wH
       Mom  +1.00  +0.19  -0.10  +0.35  -0.19  +0.24  +0.05  +0.54
      MRev  +0.19  +1.00  -0.45  +0.04  -0.09  -0.02  -0.09  -0.20
      Qual  -0.10  -0.45  +1.00  -0.44  +0.18  +0.03  +0.01  -0.10
       BAB  +0.35  +0.04  -0.44  +1.00  -0.09  +0.16  +0.29  +0.72
     Carry  -0.19  -0.09  +0.18  -0.09  +1.00  +0.11  +0.09  +0.02
     SecRt  +0.24  -0.02  +0.03  +0.16  +0.11  +1.00  +0.18  +0.33
    EDrift  +0.05  -0.09  +0.01  +0.29  +0.09  +0.18  +1.00  +0.33
      52wH  +0.54  -0.20  -0.10  +0.72  +0.02  +0.33  +0.33  +1.00

Avg pairwise correlation: +0.075 (excellent diversification)
```

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────────┐
│  Data Layer  │───>│Feature Engine│───>│Signal Models │───>│  Portfolio   │
│  (ArcticDB)  │    │  (C++20)     │    │ (Tier 1-3)  │    │  & Risk      │
└─────────────┘    └──────────────┘    └─────────────┘    └──────┬───────┘
       │                                                         │
       │           ┌──────────────┐    ┌─────────────┐          │
       └──────────>│   Regime     │───>│  Execution   │<────────┘
                   │  Detection   │    │   Engine     │
                   │  (GMM+HMM)  │    │  (C++/FIX)   │
                   └──────────────┘    └──────────────┘
```

## Stack

| Component | Technology |
|-----------|-----------|
| Tick Store | ArcticDB (Man Group) |
| Feature Engine | C++20, pybind11 |
| Data + Broker | Interactive Brokers TWS API |
| Historical Backfill | Polygon.io, Alpaca Markets |
| Signals & Research | Python 3.12+, NumPy, pandas |
| Data Caching | Parquet files (~407 instruments cached locally) |
| Regime Detection | Volatility regimes + market breadth |
| Monitoring | Prometheus + Grafana |

## Universe

~410 instruments from `config/sp500_universe.yaml`:

- **Equities**: ~370 S&P 500 stocks across 11 GICS sectors
- **Sector ETFs**: SPY, QQQ, XLK, XLV, XLE, XLF, XLI, XLB, XLU, XLRE, XLC, XLP, XLY
- **Equity Futures**: ES, NQ, RTY, YM
- **Commodities**: CL, NG, GC, SI, HG, ZC, ZW, ZS
- **Fixed Income**: ZN, ZB, ZF, ZT
- **FX**: EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD, USD/CHF, NZD/USD

## Data Provider Routing

```
Historical equities/ETFs:  Polygon → Alpaca → IBKR (fallback chain)
Historical futures/FX:     IBKR → Polygon
Live streaming:            IBKR only
All data cached as Parquet: ~/.one_brain_fund/cache/bars/
```

## Setup

### Prerequisites

- Python 3.12+
- Interactive Brokers TWS or IB Gateway (paper account is free)
- Polygon.io API key (free tier works for daily bars)
- Alpaca Markets API key (free)

### Installation

```bash
git clone https://github.com/akoiralaa/algorithmic-trading-platform.git
cd algorithmic-trading-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Add your API keys:
#   IBKR_HOST, IBKR_PORT, IBKR_CLIENT_ID
#   POLYGON_API_KEY
#   ALPHAVANTAGE_API_KEY   (optional, enables richer event/revision research)
#   ALPACA_API_KEY, ALPACA_API_SECRET
```

### Optional: Alpha Vantage PIT Enrichment

```bash
# Backfill event/revision/news data into local PIT stores
./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --max-symbols 50 --requests-per-minute 10

# Full-universe local cache hydrate (run once; saves raw API payloads on disk)
./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --requests-per-minute 20 --chunk-size 25

# Offline verification (no API calls; reads only local cache)
./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --cache-only --requests-per-minute 20
```

### Optional: FinBERT Sentiment Enrichment (Hugging Face)

```bash
# One-time NLP deps for FinBERT scoring
./.venv/bin/pip install -e ".[nlp]"

# Enrich existing PIT events with FINBERT_ENRICHED records (cached on disk)
./.venv/bin/python scripts/backfill_finbert_pit.py --since 2016-01-01T00:00:00Z

# Offline-only rerun (uses local cache, no model/API calls)
./.venv/bin/python scripts/backfill_finbert_pit.py --cache-only

# Fill missing VX cache for vix sleeve (real providers only)
./.venv/bin/python scripts/backfill_daily_bars.py --symbols VX --config config/sp500_universe.yaml
```

Raw payload cache location:
- `data/cache/pit/alpha_vantage/{json,news,csv}`
- Backtests read event/fundamental PIT data from local SQLite files:
  - `data/events.db`
  - `data/fundamentals.db`

### Optional: Lightweight Macro Cache (Free FRED API)

```bash
# Backfill a small macro panel and persist locally (daily forward-filled parquet)
./.venv/bin/python scripts/backfill_macro_fred.py --start-date 1990-01-01

# Offline-only run (uses raw cache + existing parquet only)
./.venv/bin/python scripts/backfill_macro_fred.py --cache-only
```

Macro cache output:
- per-series: `~/.one_brain_fund/cache/macro/<SERIES>.parquet`
- merged panel: `~/.one_brain_fund/cache/macro/fred_daily.parquet`

### Running Backtests

```bash
# v3 — Best risk-adjusted (Sharpe 0.57, lowest DD)
python3 backtest_v3.py --target-gross 1.3 --n-long 40 --n-short 12 \
  --rebal-freq 21 --target-vol 0.15 --max-pos 0.06

# v4 — Highest absolute returns (CAGR 9.76%, $70.9M final NAV)
python3 backtest_v4.py --target-gross 1.5 --n-long 35 --n-short 12 \
  --rebal-freq 21 --target-vol 0.20 --max-pos 0.08

# Use --no-cache to force fresh data download from providers
python3 backtest_v4.py --no-cache

# v7 — Equity core + PIT event/sentiment alpha (current calibrated baseline)
./.venv/bin/python backtest_v7.py --force-event-weight 0.05

# v8 — v7 core + ETF/Futures/FX sleeves + VIX + options-hedge execution path
# (options layer is a conservative synthetic put-overlay in backtest mode)
./.venv/bin/python backtest_v8.py --force-event-weight 0.05

# v8 strict no-lookahead run on cache-complete universe only
./.venv/bin/python backtest_v8.py --force-event-weight 0.05 --cache-complete-only --enforce-no-lookahead

# v8 + optional lightweight macro overlay (risk scaling only; multi-strategy remains primary)
./.venv/bin/python backtest_v8.py --force-event-weight 0.05 --enable-macro-overlay

# v8 constrained sweep (Path #2): optimize for LP returns under DD/turnover caps
./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 24

# v8 walk-forward (strict out-of-sample, no lookahead)
./.venv/bin/python scripts/run_v8_walk_forward.py --min-train-years 5 --test-years 1 --step-years 1

# v8 walk-forward using locked production config
./.venv/bin/python scripts/run_v8_walk_forward.py --params-file config/v8_production_locked.yaml \
  --output-csv data/reports/v8_walk_forward_locked.csv

# v8 regime split validation (pre-2008, 2008, 2010s, 2020+, recent)
./.venv/bin/python scripts/run_v8_regime_splits.py --train-years 5
```

## Design Principles

1. **One Brain** — Single feature engine, single Market State Vector, all asset classes
2. **Cross-Sectional** — Rank stocks against each other, not against absolute thresholds
3. **Survival > Returns** — Drawdown circuit breakers are hard constraints
4. **No Lookahead** — Walk-forward only, all signals use past data
5. **Institutional Data** — IBKR + Polygon + Alpaca only. Never yfinance.
