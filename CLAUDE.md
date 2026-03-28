# One Brain Quantitative Execution Factory

## Project Overview
Renaissance-style quantitative trading system. One unified signal engine across all liquid asset classes.

## Build Protocol
- 9 sequential phases, each with a validation gate
- NO phase N+1 until phase N passes ALL_PASS
- Production-grade from Phase 1. No stubs, no TODOs.
- Every validation report archived to `validation_reports/` with git SHA

## Stack
- **Database**: ArcticDB (columnar time-series, Man Group)
- **Data + Broker**: Interactive Brokers (all asset classes, paper + live)
- **Supplementary Data**: Polygon.io (deep US equity tick history)
- **Feature Engine**: C++20, pybind11
- **Research/Signals/Portfolio**: Python 3.12+
- **Monitoring**: Prometheus + Grafana

## Architecture Principles
- One Brain: single feature engine, single Market State Vector, all asset classes
- Zero Logic Drift: same C++ binary in backtest and live
- Survival > Returns: every risk control is a hard constraint

## Directory Layout
```
src/data/          Phase 1: Data layer (ArcticDB, symbol master, quality, universe)
src/cpp/           Phase 2: C++20 feature engine
src/engine/        Phase 2: Python-side engine interface (pybind11)
src/backtester/    Phase 3: Deterministic tick-by-tick backtester
src/signals/       Phase 4: Signal models (Tier 1-3, combiner)
src/portfolio/     Phase 5: Portfolio optimization, risk, P&L attribution
src/regime/        Phase 6: GMM + HMM regime detection
src/execution/     Phase 7: Broker interface, kill switch, WAL, reconciliation
src/monitoring/    Phase 8: Prometheus metrics, Grafana, alerting
config/            Version-controlled parameters (every change = git commit with reason)
validation_reports/ Phase gate reports (archived with git SHA)
tests/             Per-phase test suites
scripts/           Validation gate runners
```

## Config Version Control
Every parameter change is a git commit. No exceptions.
