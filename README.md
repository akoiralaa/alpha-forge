# One Brain Quantitative Execution Factory

A unified quantitative trading system that finds statistical edges across every liquid asset class — equities, futures, FX, commodities, and bonds — sizes each bet correctly relative to its risk, and executes at institutional speed.

One brain. Not one hundred.

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

The system is built and validated in strict sequential phases. Each phase has a validation gate that must pass ALL assertions before the next phase begins.

| Phase | Component | Status |
|-------|-----------|--------|
| 1 | Data Layer | In Progress |
| 2 | Feature Engine (C++20) | Pending |
| 3 | Deterministic Backtester | Pending |
| 4 | Signal Models | Pending |
| 5 | Portfolio & Risk | Pending |
| 6 | Regime Detection | Pending |
| 7 | Execution Engine | Pending |
| 8 | Operational Monitoring | Pending |
| 9 | Paper Trading & Pre-Live | Pending |

## Setup

### Prerequisites

- Python 3.12+
- Interactive Brokers TWS or IB Gateway (paper account is free)
- C++20 compiler (for Phase 2+)

### Installation

```bash
git clone <repo-url> && cd one_brain_fund
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

### Configuration

```bash
cp .env.example .env
# Edit .env with your IB and Polygon credentials
```

### Running Tests

```bash
source .venv/bin/activate
python -m pytest tests/ -v
```

### Running Phase Validation

```bash
python scripts/validate_phase1.py
```

## Principles

1. **One Brain** — Single feature engine, single Market State Vector, all asset classes
2. **Zero Logic Drift** — Same C++ binary in backtest and live. No Python reimplementations.
3. **Survival > Returns** — Every risk control is a hard constraint, not a suggestion
4. **No Assumptions** — Every backtest parameter must correspond to something measurable in reality
