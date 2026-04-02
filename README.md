# One Brain Fund — Quantitative Trading Platform

A Renaissance-style systematic trading engine that ranks stocks cross-sectionally across the S&P 500 using multi-factor signals, constructs sector-neutral portfolios, and manages risk dynamically through volatility regimes and drawdown circuit breakers.

## Backtest Results vs SPY (2007-2026, $10M starting NAV)

| Metric | v2 | v3 | v4 | SPY (Buy & Hold) |
|--------|------|------|------|-------------------|
| **CAGR** | +4.01% | **+8.39%** | +9.76% | +7.10% |
| **Sharpe** | 0.41 | **0.57** | 0.56 | 0.46 |
| **Sortino** | 0.47 | **0.69** | 0.66 | 0.54 |
| **Max Drawdown** | -34.2% | **-32.0%** | -37.4% | -56.5% |
| **Final NAV** | $22.1M | **$55.2M** | $70.9M | $43.0M |
| **Win Years** | 12/20 | 15/20 | **15/20** | 16/20 |
| **Loss Years** | 8/20 | **5/20** | 5/20 | 4/20 |
| **Beat SPY Return** | 6/20 | 10/20 | **13/20** | — |
| **Beat SPY Sharpe** | 4/20 | 9/20 | **7/20** | — |
| **Turnover** | 11x/yr | **22x/yr** | 28x/yr | 0 |
| **Tx Costs** | 33 bps | **66 bps** | 83 bps | 0 |

**v3** is the best risk-adjusted engine (highest Sharpe, lowest max DD, lowest costs).
**v4** produces the highest absolute returns ($70.9M vs SPY $43.0M) at higher volatility.

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

Raw payload cache location:
- `data/cache/pit/alpha_vantage/{json,news,csv}`
- Backtests read event/fundamental PIT data from local SQLite files:
  - `data/events.db`
  - `data/fundamentals.db`

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
```

## Design Principles

1. **One Brain** — Single feature engine, single Market State Vector, all asset classes
2. **Cross-Sectional** — Rank stocks against each other, not against absolute thresholds
3. **Survival > Returns** — Drawdown circuit breakers are hard constraints
4. **No Lookahead** — Walk-forward only, all signals use past data
5. **Institutional Data** — IBKR + Polygon + Alpaca only. Never yfinance.
