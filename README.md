# Cross-Asset Algorithmic Trading Platform

A systematic trading engine that detects statistical edges across equities, futures, FX, commodities, and fixed income — sizes positions using Kelly-optimal risk budgets, and executes with sub-millisecond latency tracking.

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
| Historical Backfill | Polygon.io |
| Signals & Research | Python 3.12+, scikit-learn, LightGBM |
| Regime Detection | Gaussian Mixture Models + Hidden Markov Models |
| Monitoring | Prometheus + Grafana |

## Asset Coverage

- **Equities**: NYSE, NASDAQ — all stocks passing liquidity filters
- **Sector ETFs**: SPY, QQQ, XLK, XLV, XLE, XLF, and more
- **Equity Futures**: ES, NQ, RTY, YM
- **Commodities**: CL, NG, GC, SI, HG, ZC, ZW, ZS
- **Fixed Income**: ZN, ZB, ZF, ZT, GE
- **FX**: EUR/USD, USD/JPY, GBP/USD, AUD/USD, USD/CAD, USD/CHF, NZD/USD
- **Volatility**: VIX futures

## Build Phases

The system is built and validated in strict sequential phases. Each phase has a validation gate that must pass every assertion before the next phase begins.

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Data Layer (ArcticDB, symbol master, quality, universe) | Done |
| 2 | Feature Engine (C++20 ring buffers, pybind11) | Done |
| 3 | Deterministic Backtester (tick-by-tick, walk-forward) | Done |
| 4 | Signal Models (Tier 1-3, combiner, decay tracking) | Done |
| 5 | Portfolio & Risk (HRP, Kelly sizing, P&L attribution) | Done |
| 6 | Regime Detection (5-regime GMM, HMM, smoothed posterior) | Done |
| 7 | Execution Engine (WAL, kill switch, reconciliation, self-trade prevention) | Done |
| 8 | Monitoring (Prometheus metrics, health checks, HFT alerts) | Done |
| 9 | Paper Trading & Pre-Live (infra Sharpe, disaster drills, staged deployment) | Done |

## Setup

### Prerequisites

- Python 3.12+
- Interactive Brokers TWS or IB Gateway (paper account is free)
- C++20 compiler (for the feature engine)

### Installation

```bash
git clone https://github.com/akoiralaa/algorithmic-trading-platform.git && cd algorithmic-trading-platform
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# add your IB and Polygon credentials
```

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Validation Gates

Each phase has its own validation script that must pass before moving on:

```bash
python scripts/validate_phase1.py
# ...
python scripts/validate_phase9.py
```

## Design Principles

1. **One Brain** — Single feature engine, single Market State Vector, all asset classes
2. **Zero Logic Drift** — Same C++ binary in backtest and live
3. **Survival > Returns** — Every risk control is a hard constraint, not a suggestion
4. **No Assumptions** — Every backtest parameter must correspond to something measurable in reality
