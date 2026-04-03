# One Brain Fund Devlog

## 2026-04-02

### Focus
- Start hardening `v8` for live deployment quality by executing roadmap items `#2 -> #5`:
  - constrained sweep runner
  - sleeve governance / kill switches
  - more realistic execution-cost model
  - stricter event alpha selectivity

### What Changed

#### #2 Constrained `v8` sweep harness
- Added `scripts/run_v8_constraint_sweep.py`.
- Runs a deterministic parameter grid subset and writes machine-readable output to:
  - `data/reports/v8_constraint_sweep.csv`
- Adds a constrained objective that prioritizes LP net CAGR while penalizing runs that violate:
  - max DD cap
  - turnover cap

#### #3 Sleeve governance / kill-switch layer
- Added sleeve-level governance in `backtest_v8.py`:
  - rolling Sharpe and hit-rate scoring
  - rolling drawdown penalties
  - cooldown kill-switch when sleeve quality collapses
- Governance now scales these sleeves independently:
  - core
  - etf
  - futures
  - fx
  - vix
- Added CLI controls (`--gov-*`) and optional disable flag (`--disable-sleeve-governance`).

#### #4 Execution realism upgrade
- Replaced static turnover-only transaction costs with a dynamic model:
  - volatility regime scaling
  - liquidity/ADV stress scaling
  - jump-day turnover penalty
- Added execution controls via CLI (`--exec-*`) and diagnostics in output.

#### #5 Event alpha quality upgrade
- Strengthened `EventAlphaConfig` and sentiment ingestion filters in `src/signals/event_alpha.py`:
  - minimum relevance and novelty gates
  - event-type-specific minimum absolute sentiment thresholds
  - per-symbol/day event cap
  - burst-event decay (deconflicts noisy same-day headlines)
- Extended `backtest_v8.py` CLI to expose these event selectivity controls.

#### Validation updates
- Added/updated tests:
  - `tests/phase8/test_backtest_v8_helpers.py`
    - sleeve governance cooldown behavior
    - dynamic execution-cost stress behavior
  - `tests/phase4/test_event_alpha.py`
    - daily event cap selectivity

#### Walk-forward / no-lookahead hardening
- `backtest_v8.py` now supports strict out-of-sample windows:
  - `--start-date`, `--end-date`
  - `--eval-start`, `--eval-end`
  - `--enforce-no-lookahead`
- Added `--cache-complete-only` to skip any symbol missing local daily-bar cache (VX remains optional, non-blocking).
- Added `scripts/run_v8_walk_forward.py`:
  - builds cache-complete universe automatically
  - runs rolling walk-forward folds using strict OOS evaluation windows
  - writes canonical fold metrics to `data/reports/v8_walk_forward.csv`

#### Constrained sweep + walk-forward run (strict)
- Ran full constrained sweep:
  - `./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 24`
  - output: `data/reports/v8_constraint_sweep.csv`
- Top constrained config selected:
  - `force_event_weight=0.03`
  - `overlay_min_signal=0.08`
  - `etf_gross=0.20`
  - `futures_gross=0.20`
  - `fx_gross=0.08`
  - `option_max_notional=0.04`
  - `total_gross_cap=1.8`
  - `gov_min_mult=0.30`
  - `exec_liq_cost_mult=0.80`
- Sweep best-row outcomes:
  - Gross CAGR `+10.96%`
  - Gross Sharpe `0.77`
  - Gross Max DD `-20.39%`
  - LP Net CAGR `+7.08%`
  - LP Net Sharpe `0.53`
  - LP Net Max DD `-25.20%`
  - Turnover `16.0x/yr`
  - Tx costs `46 bps/yr`
- Ran strict rolling walk-forward:
  - `./.venv/bin/python scripts/run_v8_walk_forward.py --min-train-years 5 --test-years 1 --step-years 1`
  - output: `data/reports/v8_walk_forward.csv`
  - folds: `14` (all completed)
  - mean LP Net CAGR: `+6.77%`
  - mean LP Net Sharpe: `0.38`
  - median LP Net Max DD: `-12.72%`
  - fold passes at target (15% CAGR + 1.0 Sharpe + DD <= 30%): `4/14`

#### Locked production config + regime splits
- Locked production config created:
  - `config/v8_production_locked.yaml`
  - source: top constrained row from `data/reports/v8_constraint_sweep.csv`
- Locked walk-forward rerun (same strict settings):
  - output: `data/reports/v8_walk_forward_locked.csv`
  - result: unchanged from strict run (`4/14` fold passes at fund hurdle target)
- Added and ran `scripts/run_v8_regime_splits.py` with strict no-lookahead train/test windows:
  - requested splits: `pre_2008`, `crisis_2008`, `decade_2010s`, `regime_2020_plus`, `recent`
  - output: `data/reports/v8_regime_splits.csv`
  - all splits completed (`5/5 ok`)
  - early-era validation used `--warmup-days 180 --min-rows 360` to make pre-2008 windows evaluable on available history while keeping no-lookahead policy intact
  - LP outcomes:
    - `pre_2008`: CAGR `-2.44%`, Sharpe `-0.13`, Max DD `-10.56%`
    - `crisis_2008`: CAGR `-16.77%`, Sharpe `-1.56`, Max DD `-18.24%`
    - `2010s`: CAGR `+9.25%`, Sharpe `0.68`, Max DD `-19.15%`
    - `2020+`: CAGR `+6.28%`, Sharpe `0.52`, Max DD `-22.20%`
    - `recent`: CAGR `+4.22%`, Sharpe `0.36`, Max DD `-13.88%`
  - split passes at locked hedge-fund target (`15% CAGR`, `1.0 Sharpe`, DD <= `30%`): `0/5`

## 2026-04-01

### Focus
- Move the project from "single backtest engine" toward a hedge-fund-style research and deployment stack.
- Improve medium-horizon returns without losing sight of live deployability, crowding, capacity, and strategy governance.
- Add a `v5` layer that can dynamically reallocate capital across sleeves instead of relying on mostly static weights.

### What Changed

#### v3 work
- Tightened pruning so stale positions do not linger in the book.
- Reduced short drag by gating shorts more selectively.
- Fixed carry/live-weight behavior so drawdown logic responds to the actual book instead of stale rebalance targets.
- Result from the stronger `v3` branch was roughly:
  - Sharpe `0.71`
  - CAGR `10.34%`
  - Max DD `-30.11%`
  - Turnover `13x/yr`
  - Tx costs `38 bps/yr`

#### v4 work
- Added regime-aware strategy tilts and more conservative short activation.
- Corrected standalone strategy diagnostics so they no longer rely on same-day lookahead-style attribution.
- Added LP-style reporting after standard `2% / 20%` fees and annual hurdle counts.
- Brought over cleaner pruning / live-book handling.
- Stronger `v4` result landed around:
  - Gross CAGR `12.78%`
  - Gross Sharpe `0.73`
  - Gross Max DD `-35.63%`
  - LP Net CAGR `8.57%`
  - LP Net Sharpe `0.53`
- Conclusion: better than prior versions, but still below the target mandate for a real hedge fund.

#### Live / paper deployment plumbing
- Fixed paper engine state sync so broker positions flow back into internal portfolio state.
- Added fund-style live metrics:
  - gross exposure
  - net exposure
  - peak NAV
  - daily PnL
  - kill-switch level
  - critical alert inputs
- Added staged promotion logic for capital deployment:
  - live return vs paper return
  - live Sharpe vs expected Sharpe
  - drawdown limits
  - infra/reconciliation health
  - minimum live days

#### Intraday alpha groundwork
- Added `src/signals/intraday_alpha.py`.
- New sleeve combines:
  - short-horizon mean reversion
  - cross-asset lead/lag
  - spread reversion
  - regime-aware sleeve weighting
- Hooked latest tick retention into the paper engine so event timing is preserved in live/paper scoring.
- Important limitation:
  - local "tick" storage is still daily-snapshot-like, not true high-frequency history
  - this means intraday/HFT alpha is scaffolded, but not honestly proven yet

#### Capacity and central allocation layer
- Added `src/portfolio/capacity.py`:
  - per-symbol liquidity snapshots
  - impact estimation
  - max safe order notional
  - strategy-level NAV capacity estimate
- Added `src/portfolio/allocator.py`:
  - tracks realized vs expected strategy performance
  - dynamically tilts weights toward outperformers
  - penalizes sleeves with drawdowns or capacity pressure
  - preserves exploration floor so sleeves are not shut off too early
- Wired allocator + capacity observations into the live paper engine.
- Added strategy-level monitoring metrics for:
  - weight
  - realized Sharpe
  - expected Sharpe
  - performance gap
  - capacity utilization
  - average impact

### v5 Backtest

Command run:

```bash
./.venv/bin/python backtest_v5.py
```

`v5` is the first pass at a capacity-aware, allocator-driven fund brain. It combines five medium-horizon sleeves:
- momentum
- quality
- carry
- sector rotation
- 52-week-high proximity

It does **not** yet include:
- NLP / FinBERT / Hugging Face event alpha
- true intraday/HFT alpha validated on real sub-minute history

#### Aggregate result
- Gross CAGR: `+10.73%`
- Gross Sharpe: `0.67`
- Gross Sortino: `0.84`
- Gross Max DD: `-33.96%`
- LP Net CAGR after `2% / 20%`: `+6.94%`
- LP Net Sharpe: `0.47`
- LP Net Max DD: `-35.72%`
- Final NAV: `$85.6M`
- LP NAV: `$41.0M`
- Average gross exposure: `1.36x`
- Turnover: `21x/yr`
- Transaction costs: `62 bps/yr`

#### Strategy mix observed
- `carry`: avg weight `41.9%`, sleeve Sharpe `0.67`, capacity util `118.63`
- `momentum`: avg weight `19.2%`, sleeve Sharpe `0.69`, capacity util `194.84`
- `high_52w`: avg weight `17.3%`, sleeve Sharpe `0.64`, capacity util `144.90`
- `quality`: avg weight `15.3%`, sleeve Sharpe `0.67`, capacity util `215.48`
- `sector_rot`: avg weight `6.2%`, sleeve Sharpe `0.57`, capacity util `3511.42`

#### Annual hurdle check
- Years with `>= 15%` LP net return: `6 / 20`
- Years with `>= 1.0` annual Sharpe: `5 / 20`
- Years hitting both hurdles: `5 / 20`

### What v5 Tells Us
- The central allocator/capacity framework works mechanically and produces a sensible dynamic weight mix.
- `v5` outperforms SPY on full-period CAGR and drawdown, but it still falls far short of the stated hedge-fund mandate.
- The current daily signal stack is still not enough to deliver something like:
  - `~20%` average net annual return
  - `~1.0+` annual Sharpe consistently
  - crisis resilience close to the Medallion-style bar

### Biggest Bottlenecks Right Now
- Missing alpha:
  - no real earnings-surprise / estimate-revision sleeve yet
  - no real NLP/news sentiment sleeve yet
  - no validated intraday/HFT sleeve on real sub-minute history
- Capacity pressure:
  - every major sleeve is already showing utilization above `100%`
  - `sector_rot` is extreme and should not be trusted at current sizing
- Data limitations:
  - the current local intraday store is not good enough for true HFT research
  - fundamentals coverage is too thin for a robust point-in-time event alpha model

### Key Engineering Note From This Run
- First `v5` run failed because the capacity model received `NaN` liquidity inputs and attempted to convert them into order-share integers.
- Hardened both:
  - `src/portfolio/capacity.py`
  - `backtest_v5.py`
- Behavior now degrades safely when liquidity inputs are sparse instead of crashing the backtest.

### Next Steps
- Add hard capacity clipping so the allocator cannot keep promoting sleeves already beyond feasible scale.
- Add strategy-level NAV throttles and participation caps directly into portfolio construction.
- Build a real point-in-time event sleeve:
  - earnings surprise
  - analyst revisions
  - guidance drift
- Add NLP / sentiment as the `v5+` sleeve once event data plumbing is in place.
- Ingest real intraday data for a small starter set:
  - `SPY`
  - `QQQ`
  - `AAPL`
  - major FX pairs
  - lead futures
- Treat current `v5` as a control system improvement, not a finished alpha breakthrough.

### v6 Backtest

Command run:

```bash
./.venv/bin/python backtest_v6.py
```

`v6` keeps the stronger `v4` alpha core but adds:
- hard capacity clipping at the sleeve level
- symbol-level rebalance delta clipping
- mild allocator tilts instead of full `v5` rotation
- a sparse point-in-time event sleeve using the local fundamentals store

Important limitation:
- the local point-in-time fundamentals data is still extremely thin
- the event sleeve only found real support for `1` ticker, so it stays effectively dormant

#### Aggregate result
- Gross CAGR: `+4.61%`
- Gross Sharpe: `0.66`
- Gross Sortino: `0.81`
- Gross Max DD: `-15.75%`
- LP Net CAGR after `2% / 20%`: `+2.02%`
- LP Net Sharpe: `0.31`
- LP Net Max DD: `-17.86%`
- Final NAV: `$25.8M`
- LP NAV: `$15.2M`
- Average gross exposure: `0.47x`
- Turnover: `7x/yr`
- Transaction costs: `21 bps/yr`
- Average portfolio capacity scale: `0.45`
- Average clipped symbols per rebalance day: `19.42`

#### Strategy mix observed
- `high_52w`: avg weight `26.7%`, sleeve Sharpe `0.64`, capacity util `55.50`
- `carry`: avg weight `25.0%`, sleeve Sharpe `0.67`, capacity util `53.50`
- `momentum`: avg weight `24.1%`, sleeve Sharpe `0.69`, capacity util `75.34`
- `quality`: avg weight `19.2%`, sleeve Sharpe `0.67`, capacity util `90.12`
- `sector_rot`: avg weight `5.0%`, sleeve Sharpe `0.57`, capacity util `1236.71`
- `pit_event`: avg weight `0.0%`, sleeve Sharpe `0.00`, capacity util `0.00`

#### Annual hurdle check
- Years with `>= 15%` LP net return: `0 / 20`
- Years with `>= 1.0` annual Sharpe: `4 / 20`
- Years hitting both hurdles: `0 / 20`

### What v6 Tells Us
- `v6` is more realistic from a live-deployment and scalability perspective than `v4` or `v5`.
- It is not a better flagship strategy. Hard investability constraints cut exposure too aggressively and crush returns.
- The capacity model is clearly binding:
  - portfolio-level scaling averages only `0.45x`
  - roughly `19` symbols per rebalance day are being clipped
- This means the control layer is useful, but the current alpha stack is still too weak and too crowded once we force realistic sizing.

### Revised Direction
- Keep `v4` as the alpha core for now.
- Reuse `v5` and `v6` components as live overlays:
  - allocator
  - capacity throttles
  - symbol-level order clipping
- The next real breakthrough must come from new alpha sources with better data, not from another reweighting pass.

### v7 Event Alpha Foundation

New research and data plumbing added:
- `src/data/events.py`
  - SQLite-backed point-in-time event store for timestamped news, guidance, earnings, and revision events
- `src/signals/sentiment.py`
  - optional Hugging Face sentiment scoring with finance-lexicon fallback
- `src/signals/event_alpha.py`
  - point-in-time event sleeve combining:
    - earnings surprise
    - analyst estimate / revision drift
    - event/news sentiment
    - sparse-aware cross-sectional ranking
    - coverage-aware sleeve sizing
- `scripts/ingest_news_sentiment.py`
  - CSV ingestion path for timestamped news headlines into the event store

Engineering cleanup:
- `backtest_v4.py`
  - multi-strategy combiner now skips disabled zero-weight sleeves in the risk-parity / IC loop
  - avoids bogus zero-variance correlations when a sleeve is intentionally turned off
- `backtest_v7.py`
  - new backtest variant that keeps the `v4` winners as the alpha core and uses the `earnings_drift` slot for the new event sleeve
  - event weight automatically collapses toward zero when coverage is too sparse

Validation:
- `./.venv/bin/python -m pytest tests/phase1/test_event_store.py tests/phase4/test_event_alpha.py -q`
  - `4 passed`
- `./.venv/bin/python -m py_compile backtest_v4.py backtest_v7.py src/data/events.py src/signals/sentiment.py src/signals/event_alpha.py scripts/ingest_news_sentiment.py`
  - passed

### v7 Backtest

Command run:

```bash
./.venv/bin/python backtest_v7.py
```

`v7` is designed to be honest about missing data. It will only allocate to the event sleeve when there is enough point-in-time event coverage to justify it.

Observed coverage on the local dataset:
- fundamentals coverage: `1` ticker
- sentiment coverage: `0`
- total scored events: `0`
- live event sleeve weight used in backtest: `0.0%`

#### Aggregate result
- Gross CAGR: `+9.66%`
- Gross Sharpe: `0.66`
- Gross Max DD: `-29.81%`
- LP Net CAGR after `2% / 20%`: `+6.07%`
- LP Net Sharpe: `0.45`
- LP Net Max DD: `-31.64%`
- Final NAV: `$69.7M`
- LP NAV: `$34.5M`
- Average gross exposure: `1.33x`
- Turnover: `16x/yr`
- Transaction costs: `47 bps/yr`

#### Annual hurdle check
- Years with `>= 15%` LP net return: `6 / 20`
- Years with `>= 1.0` annual Sharpe: `6 / 20`
- Years hitting both hurdles: `5 / 20`

### What v7 Tells Us
- The new event alpha infrastructure is in place and works mechanically in both research and paper/live paths.
- It did **not** improve returns yet because the local point-in-time event dataset is still effectively empty.
- That is actually a useful result:
  - the sleeve stayed dormant instead of hallucinating alpha from sparse data
  - the backtest reverted cleanly toward the `v4` baseline

### Immediate Next Step
- Fill `data/events.db` with real timestamped events:
  - earnings headlines
  - guidance updates
  - estimate revision/news flow
  - transcript/news sentiment
- Expand point-in-time fundamentals coverage beyond the current tiny sample.
- Once coverage is real, rerun `v7` before adding more portfolio complexity.

### SEC PIT Backfill and `v7` Recalibration

The next pass focused on turning the `v7` event sleeve from scaffolding into a real, data-backed signal path.

#### New ingestion paths
- Added `src/data/ingest/polygon_event_backfill.py`
  - maps Benzinga-style earnings, guidance, analyst, and news endpoints into:
    - `data/events.db`
    - `data/fundamentals.db`
  - includes sentiment scoring and duplicate-safe inserts
- Added `scripts/backfill_polygon_pit.py`
  - CLI wrapper for Polygon/Benzinga point-in-time backfills
- Added `src/data/ingest/sec_companyfacts_backfill.py`
  - free-data fallback using SEC company facts
  - extracts:
    - EPS actuals
    - revenue actuals
    - filing-timestamped earnings events
- Added `scripts/backfill_sec_companyfacts.py`
  - CLI wrapper for the SEC fallback path
- Added symbol-master synchronization so the equity universe is actually represented in `data/symbol_master.db`

#### What happened with Polygon
- The local Polygon key worked for price history, but the premium Benzinga/financial endpoints required for:
  - earnings surprise
  - analyst revisions
  - guidance/news backfill
  were not entitled.
- Result: no premium PIT event dataset could be built from Polygon on the current plan.

#### What worked with SEC
- The SEC fallback path now backfills filing-based actuals and event timestamps across the equity universe.
- During the process, `358` missing equity symbols were inserted into symbol master.
- Latest observed local store counts after the full SEC pass:
  - `data/fundamentals.db`
    - total rows: `94,621`
    - canonical ids: `428`
    - `EPS`: `64,143` rows across `360` ids
    - `REVENUE`: `30,206` rows across `360` ids
  - `data/events.db`
    - total rows: `21,787`
    - event type: `earnings` only
    - canonical ids with events: `362`
- This is enough to activate the event sleeve with real coverage, but it is still weaker than a premium estimate/news dataset because:
  - analyst estimates are still almost absent
  - revision history is effectively absent
  - sentiment/news coverage is still zero

#### Event alpha weighting fix
- `src/signals/event_alpha.py` now distinguishes between:
  - raw coverage
  - data quality
- Broad SEC actual coverage no longer gets the same weight as rich:
  - surprise
  - revision
  - sentiment
  coverage.
- Added quality-aware sizing:
  - actual-only fundamentals get a limited base quality score
  - analyst estimate/revision support increases quality
  - sentiment/news support increases quality
- This prevents `v7` from auto-allocating the full event sleeve weight to low-richness filing data.

Validation:
- `./.venv/bin/python -m pytest tests/phase1/test_polygon_event_backfill.py tests/phase1/test_event_store.py tests/phase4/test_event_alpha.py -q`
  - `9 passed`
- `./.venv/bin/python -m py_compile backtest_v7.py src/signals/event_alpha.py`
  - passed

### Updated `v7` Backtest

Command run:

```bash
./.venv/bin/python backtest_v7.py
```

Observed event diagnostics on the enriched SEC dataset:
- fundamentals coverage: `360`
- estimate coverage: `1`
- revision support coverage: `1`
- sentiment coverage: `0`
- total scored events: `0`
- event data quality scale: `0.35`
- live event sleeve weight used in backtest: `4.9%`

This is a deliberate change from the earlier SEC-enriched run where the sleeve floated up to `14%` and hurt performance. The new default keeps the sleeve active, but sizes it more like a weak-but-real alpha source instead of a fully trusted premium event book.

#### Aggregate result
- Gross CAGR: `+9.85%`
- Gross Sharpe: `0.67`
- Gross Max DD: `-29.86%`
- LP Net CAGR after `2% / 20%`: `+6.22%`
- LP Net Sharpe: `0.46`
- LP Net Max DD: `-31.69%`
- Final NAV: `$72.2M`
- LP NAV: `$35.6M`
- Average gross exposure: `1.33x`
- Turnover: `16x/yr`
- Transaction costs: `47 bps/yr`

#### Annual hurdle check
- Years with `>= 15%` LP net return: `7 / 20`
- Years with `>= 1.0` annual Sharpe: `6 / 20`
- Years hitting both hurdles: `6 / 20`

### What This Changes
- `v7` is now better than the no-event baseline, but only modestly:
  - previous baseline-like `v7`: roughly `+9.66% CAGR / 0.66 Sharpe / $69.7M final NAV`
  - current quality-aware `v7`: `+9.85% CAGR / 0.67 Sharpe / $72.2M final NAV`
- The event sleeve is no longer fake or dormant:
  - it is using real SEC filing data
  - it is just not yet strong enough to transform the whole fund on its own
- The next real upside still requires richer data:
  - analyst estimates and revisions
  - news / transcript sentiment
  - more than earnings-only event coverage

### Revised Next Step
- Keep the quality-aware event sizing logic as the new default.
- Continue using SEC as the free fallback spine for actuals and filing timestamps.
- Add richer timestamped event data on top of that spine:
  - guidance
  - revisions
  - headlines
  - transcript/news sentiment
- Once that data lands, rerun `v7` before moving on to the next intraday/HFT pass.

### Raw Data Persistence Layer

The next improvement was about iteration speed rather than alpha math: all new SEC / Polygon point-in-time backfills now persist raw upstream payloads locally instead of only storing the normalized rows.

#### What changed
- Added `src/data/ingest/raw_cache.py`
  - gzip-compressed JSON cache for raw API payloads
  - shared by both SEC and Polygon/Benzinga backfills
- Updated `src/data/ingest/sec_companyfacts_backfill.py`
  - caches:
    - ticker-to-CIK map
    - per-ticker company facts payloads
- Updated `src/data/ingest/polygon_event_backfill.py`
  - caches chunked raw payloads for:
    - Benzinga earnings
    - guidance
    - analyst insights
    - news
    - Polygon income statements
- Updated CLI wrappers:
  - `scripts/backfill_sec_companyfacts.py`
  - `scripts/backfill_polygon_pit.py`

#### Default cache location
- `data/cache/pit/`

#### New CLI controls
- `--cache-dir`
  - override cache root
- `--refresh-cache`
  - force a fresh download and overwrite the local raw cache
- `--cache-only`
  - do not hit the network; fail over to existing local raw cache only

#### Why this matters
- Repeated backfill iterations no longer need to re-download the same raw SEC / Polygon payloads.
- Offline reruns are now possible once the cache is warm.
- Research loops get faster because:
  - daily bars are already in parquet
  - normalized events/fundamentals are already in SQLite
  - raw upstream payloads are now cached on disk too

Validation:
- `./.venv/bin/python -m pytest tests/phase1/test_polygon_event_backfill.py tests/phase1/test_sec_companyfacts_backfill.py tests/phase1/test_event_store.py tests/phase4/test_event_alpha.py -q`
  - `11 passed`
- `./.venv/bin/python -m py_compile src/data/ingest/raw_cache.py src/data/ingest/polygon_event_backfill.py src/data/ingest/sec_companyfacts_backfill.py scripts/backfill_polygon_pit.py scripts/backfill_sec_companyfacts.py`
  - passed

### SEC Submissions Event Upgrade

To push `v7` beyond earnings timestamps, the SEC backfill path now also ingests filing/submissions events.

#### What changed
- `src/data/ingest/sec_companyfacts_backfill.py`
  - now pulls SEC submissions history in addition to company facts
  - walks both:
    - `filings.recent`
    - historical submission files listed under `filings.files`
  - creates filing-based events for forms such as:
    - `8-K`
    - `10-Q`
    - `10-K`
    - `6-K`
    - `20-F`
  - maps form/items into event types such as:
    - `earnings`
    - `guidance`
    - `m&a`
    - `news`
  - adds sentiment scoring on filing headlines/descriptions using the existing sentiment stack
- `scripts/backfill_sec_companyfacts.py`
  - now supports optional Hugging Face scoring flags:
    - `--prefer-hf`
    - `--model-name`

#### Why this matters
- The free SEC path is no longer just:
  - EPS actuals
  - revenue actuals
  - generic earnings timestamps
- It now produces a richer event surface from filings themselves, which is the best realistic free-data upgrade available before moving to a paid revisions/news feed.

Validation:
- `./.venv/bin/python -m pytest tests/phase1/test_daily_bar_cache.py tests/phase1/test_polygon_event_backfill.py tests/phase1/test_sec_companyfacts_backfill.py tests/phase1/test_event_store.py tests/phase4/test_event_alpha.py -q`
  - `14 passed`
- `./.venv/bin/python -m py_compile src/data/ingest/daily_bar_cache.py src/data/ingest/raw_cache.py src/data/ingest/polygon_event_backfill.py src/data/ingest/sec_companyfacts_backfill.py scripts/backfill_daily_bars.py scripts/backfill_polygon_pit.py scripts/backfill_sec_companyfacts.py`
  - passed

### Overnight Daily-Bar Prefetch

Backtests no longer need to be the thing that fills bar cache lazily.

#### New workflow
- Added `src/data/ingest/daily_bar_cache.py`
  - reusable helpers for:
    - universe loading
    - cache staleness checks
    - deduping / merging old and new bar frames
    - per-symbol metadata writes
- Added `scripts/backfill_daily_bars.py`
  - dedicated full-universe daily bar prefetcher
  - writes parquet bars directly into:
    - `~/.one_brain_fund/cache/bars/`
  - writes companion metadata JSON per symbol

#### Bar backfill controls
- `--stale-days`
  - skip symbols whose cache is already fresh enough
- `--incremental-buffer-days`
  - refetch a small overlap window before the last cached date and merge safely
- `--full-refresh`
  - ignore existing parquet and rewrite from maximum available history
- `--cache-only`
  - validate local cache coverage without hitting providers
- `--provider`
  - force a specific provider when needed

#### Operational note
- The first live launch of `scripts/backfill_daily_bars.py` revealed that the script was not loading `.env`, so it came up with zero providers.
- Fixed that immediately by adding dotenv loading.
- The corrected run now connects to Polygon and is able to write local parquet bars directly without needing a backtest to trigger the downloads.

### `v7` Rerun With SEC Submission Events

After the SEC submissions events started landing, the first `v7` rerun showed a new problem:
- sentiment coverage jumped
- event sleeve weight jumped with it
- but returns got worse

Observed on the first rerun:
- fundamentals coverage: `360`
- sentiment coverage: `87`
- total scored events: `14,104`
- event weight: `9.8%`
- result: roughly `+9.43% CAGR / 0.65 Sharpe / $66.6M final NAV`

That exposed another weighting issue: the model was treating SEC filing-derived sentiment as if it were the same quality as premium timestamped news.

#### Source-aware sentiment quality fix
- `src/signals/event_alpha.py`
  - sentiment quality is now source-aware, not just coverage-aware
  - high-quality sources get more credit
  - SEC filing-derived events get partial credit
- `backtest_v7.py`
  - now prints sentiment-source-quality diagnostics alongside coverage and final sleeve weight

Validation:
- `./.venv/bin/python -m pytest tests/phase4/test_event_alpha.py tests/phase1/test_sec_companyfacts_backfill.py tests/phase1/test_daily_bar_cache.py -q`
  - `7 passed`
- `./.venv/bin/python -m py_compile src/signals/event_alpha.py backtest_v7.py`
  - passed

### Latest `v7` Snapshot

Command run:

```bash
./.venv/bin/python backtest_v7.py
```

Observed diagnostics on the richer local SEC dataset:
- fundamentals coverage: `360`
- estimate coverage: `1`
- revision support coverage: `1`
- sentiment coverage: `126`
- total scored events: `20,084`
- sentiment source quality: `0.45`
- final event quality scale: `0.51`
- live event sleeve weight used in backtest: `7.1%`

#### Aggregate result
- Gross CAGR: `+9.50%`
- Gross Sharpe: `0.65`
- Gross Max DD: `-30.15%`
- LP Net CAGR after `2% / 20%`: `+5.94%`
- LP Net Sharpe: `0.44`
- LP Net Max DD: `-32.04%`
- Final NAV: `$67.6M`
- LP NAV: `$33.7M`
- Average gross exposure: `1.34x`
- Turnover: `16x/yr`
- Transaction costs: `47 bps/yr`

#### What this means
- The richer free-data event layer is real now.
- It increases event coverage materially.
- It still does **not** beat the earlier quality-capped SEC-only event setup.
- Conclusion:
  - local free-data event coverage is useful infrastructure
  - it is still not equivalent to a premium revisions/news feed
  - the event sleeve should stay quality-capped until better sources are added

### Priority-Order Stabilization Run (CAGR Fix + Fast MR Retune)

Date: 2026-04-02

Completed in priority order:

1) Fixed annualization/reporting inconsistency
- `backtest_v4.py`
  - added `elapsed_years_from_index(...)` to annualize by elapsed calendar time instead of raw row count
  - aligned benchmark returns to strategy index before metrics:
    - `spy_ret = spy_ret.reindex(net_ret.index).fillna(0.0)`
  - added explicit sample window print:
    - `Sample window: <start> -> <end> (<years> years)`
- `backtest_v7.py`
  - switched annualization to `v4.elapsed_years_from_index(equity_curve.index)`
  - aligned SPY benchmark index to strategy returns
  - added explicit sample window print

2) Retuned fast mean-reversion overlay to reduce cost drag
- `backtest_v4.py` constants:
  - `FAST_MREV_REBAL_FREQ: 3 -> 5`
  - `FAST_MREV_TARGET_GROSS: 0.18 -> 0.08`
  - `FAST_MREV_MAX_POS: 0.018 -> 0.010`

3) Re-ran both engines under same cache/hygiene state
- Universe config: `config/sp500_universe.yaml` (expected 410)
- Effective symbols used: `396` (same short-history hygiene drop as previous run)
- Residual missing cache/provider symbols during run: `PEAK`, `DISH`, `VX`
- Sample window: `2007-06-18 -> 2026-04-01 (18.79 years)`

#### Corrected baseline before this retune
- `v4`: Gross CAGR `+10.32%`, Sharpe `0.68`, Max DD `-29.55%`, Final NAV `$63.19M`, LP Net CAGR `+6.57%`, LP NAV `$32.99M`, Turnover `27x/yr`, Tx `82 bps/yr`
- `v7`: Gross CAGR `+10.15%`, Sharpe `0.67`, Max DD `-29.57%`, Final NAV `$61.48M`, LP Net CAGR `+6.43%`, LP NAV `$32.24M`, Turnover `28x/yr`, Tx `83 bps/yr`

#### Post-retune result
- `v4`: Gross CAGR `+10.80%`, Sharpe `0.71`, Max DD `-29.12%`, Final NAV `$68.57M`, LP Net CAGR `+6.94%`, LP Sharpe `0.49`, LP NAV `$35.26M`, Turnover `20x/yr`, Tx `61 bps/yr`
- `v7`: Gross CAGR `+10.62%`, Sharpe `0.70`, Max DD `-29.14%`, Final NAV `$66.55M`, LP Net CAGR `+6.80%`, LP Sharpe `0.48`, LP NAV `$34.39M`, Turnover `20x/yr`, Tx `61 bps/yr`

#### Decision
- Keep this retune. It improved return, Sharpe, drawdown, LP outcomes, and materially cut turnover/costs.
- `v4` remains the better base engine vs `v7` under current data quality by roughly:
  - `+0.18%` gross CAGR
  - `+0.01` Sharpe
  - `+$2.02M` final NAV

Validation:
- `./.venv/bin/python -m py_compile backtest_v4.py backtest_v7.py src/signals/event_alpha.py` (pass)
- `./.venv/bin/python -m pytest tests/phase4/test_backtest_v4_helpers.py tests/phase4/test_event_alpha.py -q`
  - `6 passed`

### Live Deployment Hardening: Import Cycle Fix

Date: 2026-04-02

During live-stack validation (`phase5` + `phase9` tests), test collection failed with a circular import:
- `src.data.symbol_master -> src.data.ingest.base -> src.data.ingest.__init__ -> polygon_event_backfill -> src.data.symbol_master`

Fix:
- Refactored `src/data/ingest/__init__.py` to lazy-load heavy modules via `__getattr__` instead of importing all providers/backfillers eagerly at module import time.
- This keeps public imports stable (`from src.data.ingest import ...`) while removing eager import side-effects that created the cycle.

Validation:
- `./.venv/bin/python -m py_compile src/data/ingest/__init__.py src/data/symbol_master.py src/data/ingest/polygon_event_backfill.py` (pass)
- `./.venv/bin/python -m pytest tests/phase5/test_allocator.py tests/phase9/test_strategy_allocator_live.py tests/phase9/test_intraday_alpha_paper.py tests/phase9/test_paper_trading.py -q`
  - `33 passed`

### Priority Follow-On: v4 Leverage Frontier Sweep

Date: 2026-04-02

Goal:
- Recover/beat old v4 terminal NAV while keeping drawdown materially below the old `~ -37%` profile.

Sweep runs (same cache state, same hygiene):
- Baseline retune (`--target-vol 0.15 --target-gross 1.5`)
  - Gross CAGR `+10.80%`
  - Gross Sharpe `0.71`
  - Gross Max DD `-29.12%`
  - Final NAV `$68.57M`
- Moderate step-up (`--target-vol 0.16 --target-gross 1.6`)
  - Gross CAGR `+11.20%`
  - Gross Sharpe `0.70`
  - Gross Max DD `-32.81%`
  - Final NAV `$73.34M`
  - LP Net CAGR `+7.27%`
  - LP NAV `$37.32M`
- Aggressive step-up (`--target-vol 0.17 --target-gross 1.7`)
  - Gross CAGR `+11.60%`
  - Gross Sharpe `0.69`
  - Gross Max DD `-33.68%`
  - Final NAV `$78.48M`
  - LP Net CAGR `+7.59%`
  - LP NAV `$39.50M`

Interpretation:
- Both step-up settings beat old v4-level terminal NAV while preserving clearly better drawdown and Sharpe than the earlier high-DD profile.
- `0.16 / 1.6` is the cleaner production compromise:
  - already over `$70M` terminal NAV
  - only modest Sharpe giveback vs `0.15 / 1.5`
  - keeps drawdown buffer stronger than `0.17 / 1.7`

### v4/v7 Risk-Control Integration + Cache-State Compare

Date: 2026-04-02

Implemented in `backtest_v4.py` and inherited by `backtest_v7.py`:
- Dynamic strategy allocator scaling in the blending engine.
- Explicit high-52w kill-switch multiplier based on crash/underperformance regime.
- Capacity-aware trade delta clipping at rebalance time using:
  - `src/portfolio/allocator.py`
  - `src/portfolio/capacity.py`

Allocator calibration update (to avoid over-defensive behavior):
- `ewma_decay`: `0.98`
- `min_observations`: `126`
- `exploration_floor`: `0.60`
- `score_temperature`: `0.55`
- allocator scale clip: `[0.70, 1.30]`

Validation:
- `./.venv/bin/python -m py_compile backtest_v4.py backtest_v7.py src/portfolio/allocator.py src/portfolio/capacity.py` (pass)
- `./.venv/bin/python -m pytest tests/phase4/test_backtest_v4_helpers.py tests/phase5/test_allocator.py -q`
  - `9 passed`

#### Cache audit + targeted repair attempt
- Cache audit command:
  - `./.venv/bin/python scripts/backfill_daily_bars.py --config config/sp500_universe.yaml --cache-only`
- Result:
  - `407/410` present
  - missing: `PEAK_EQUITY`, `DISH_EQUITY`, `VX_VOLATILITY`
- Targeted repair attempts:
  - `./.venv/bin/python scripts/backfill_daily_bars.py --config config/sp500_universe.yaml --symbols PEAK,DISH,VX`
  - `./.venv/bin/python scripts/backfill_daily_bars.py --config config/sp500_universe.yaml --symbols PEAK,DISH,VX --provider polygon`
- Both attempts completed but returned no bars for all 3 symbols.
- Therefore this comparison is a documented degraded-cache run (`407/410`), not full-universe complete.

#### Side-by-side run on same cache state (`407/410`)

| Metric | v4 | v7 |
|---|---:|---:|
| Dynamic allocator | on | on |
| Capacity clamps | 35 | 38 |
| Gross CAGR | `+10.97%` | `+10.88%` |
| Gross Sharpe | `0.71` | `0.71` |
| Gross Max DD | `-29.79%` | `-29.66%` |
| LP Net CAGR (`2/20`) | `+7.08%` | `+7.01%` |
| LP Net Sharpe | `0.49` | `0.49` |
| Final NAV | `$70.63M` | `$69.57M` |
| Turnover | `21x/yr` | `21x/yr` |
| Tx costs | `62 bps/yr` | `62 bps/yr` |
| Years `>=15%` LP net | `5/20` | `5/20` |
| Years `>=1.0` Sharpe | `5/20` | `5/20` |
| Years hitting both | `4/20` | `4/20` |

Decision:
- `v4` remains the better flagship under the current data stack (higher terminal NAV with equal Sharpe/hurdle profile).
- `v7` remains close, but event/richer-PIT edge is still data-limited and not yet producing a durable uplift.

### Richer Source Trial (Event / Revision / Intraday) — Pre-Deployment Gate

Date: 2026-04-02

Goal:
- Add richer event/revision sources and real intraday cache paths, then test before any live deployment.

#### New source adapters and scripts
- Added `src/data/ingest/yahoo_event_backfill.py`
  - quote-summary parser for:
    - earnings history
    - upgrade/downgrade history
    - earnings trend revisions
  - intraday chart backfill into parquet cache
- Added `scripts/backfill_yahoo_pit.py`
  - CLI for Yahoo PIT backfill + optional intraday cache fill
- Added `src/data/ingest/alpha_vantage_event_backfill.py`
  - maps Alpha Vantage into PIT stores:
    - `EARNINGS`
    - `EARNINGS_ESTIMATES` (revision-aware fields)
    - optional `NEWS_SENTIMENT`
    - optional `EARNINGS_CALENDAR`
- Added `scripts/backfill_alpha_vantage_pit.py`
  - CLI for Alpha Vantage PIT backfill
- Added `scripts/backfill_intraday_bars.py`
  - provider-agnostic intraday parquet cache prefetch (`ibkr/polygon/alpaca`)
  - works across equities/futures/fx/commodities
- Added `scripts/evaluate_intraday_alpha_from_cache.py`
  - replay utility to test the intraday sleeve against cached bars before deployment

#### Event-quality model update
- `src/signals/event_alpha.py`
  - added source quality mappings for:
    - Alpha Vantage sources
    - Yahoo sources

Validation:
- `./.venv/bin/python -m py_compile src/data/ingest/yahoo_event_backfill.py scripts/backfill_yahoo_pit.py scripts/backfill_intraday_bars.py src/data/ingest/alpha_vantage_event_backfill.py scripts/backfill_alpha_vantage_pit.py scripts/evaluate_intraday_alpha_from_cache.py src/signals/event_alpha.py` (pass)
- `./.venv/bin/python -m pytest tests/phase1/test_yahoo_event_backfill.py tests/phase1/test_alpha_vantage_event_backfill.py tests/phase4/test_event_alpha.py tests/phase9/test_intraday_alpha_paper.py -q`
  - passed

#### Trial run outcomes

1) Yahoo PIT trial (`15` liquid tickers)
- Command:
  - `./.venv/bin/python scripts/backfill_yahoo_pit.py --symbols AAPL,MSFT,NVDA,AMZN,GOOGL,META,TSLA,JPM,XOM,UNH,LLY,AVGO,BRK.B,PG,MA --with-intraday --intraday-interval 5m --intraday-range 60d`
- Outcome:
  - quote-summary path returned `401` (unauthorized) for all tested symbols
  - intraday chart path worked for `14/15` symbols (`BRK.B` failed due symbol format)
  - inserts:
    - events `0`
    - fundamentals `0`
    - intraday files `14`

2) Provider-agnostic intraday cache trial (Polygon, cross-asset)
- Command:
  - `./.venv/bin/python scripts/backfill_intraday_bars.py --symbols SPY,QQQ,AAPL,MSFT,NVDA,ES,NQ,CL,GC,EURUSD,USDJPY --interval 5m --days-back 60 --provider polygon`
- Outcome:
  - wrote `9/11` symbol files
  - failed: `NQ`, `GC` on current provider entitlement/symbol mapping
  - produced multi-asset 5-minute local cache in `data/cache/intraday`

3) Alpha Vantage PIT trial (`IBM,AAPL,MSFT,NVDA`, demo key constraints)
- Command:
  - `./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --symbols IBM,AAPL,MSFT,NVDA --requests-per-minute 20`
- Outcome:
  - inserted:
    - events `156`
    - fundamentals `434`
  - symbols skipped `3` (demo-key restricted; mostly IBM coverage)

#### v7 backtest impact after enrichment

`v7` default (quality-scaled event weight):
- Event coverage:
  - fundamentals `358`
  - estimates `2`
  - revisions `2`
  - sentiment `360`
  - total events `57,965`
  - event live weight `3.0%`
- Aggregate:
  - Gross CAGR `+10.87%`
  - Gross Sharpe `0.71`
  - Gross Max DD `-29.65%`
  - LP Net CAGR `+7.00%`
  - LP Net Sharpe `0.49`
  - Final NAV `$69.45M`

Event ablations:
- `v7 --force-event-weight 0.0`
  - Gross CAGR `+10.97%`, Sharpe `0.71`, Final NAV `$70.63M` (same as current v4 baseline)
- `v7 --force-event-weight 0.10`
  - Gross CAGR `+10.26%`, Sharpe `0.68`, Final NAV `$62.52M`

Interpretation:
- Current richer event/revision data does not yet add positive alpha at production-relevant weight.
- Forcing higher event weight is currently value-destructive.

#### Intraday sleeve replay gate (from local 5m cache)

Command:
- `./.venv/bin/python scripts/evaluate_intraday_alpha_from_cache.py --cache-dir data/cache/intraday --symbols QQQ,AAPL --interval 5m`
  - Final NAV `$984,756` (`-1.52%` over window)
  - Max DD `-1.86%`
- `./.venv/bin/python scripts/evaluate_intraday_alpha_from_cache.py --cache-dir data/cache/intraday --symbols SPY,QQQ,AAPL,MSFT,NVDA --interval 5m`
  - Final NAV `$966,356.86` (`-3.36%` over window)
  - Max DD `-4.59%`

Interpretation:
- Intraday sleeve is not deployment-ready on current feature calibration and replay setup.

### Pre-Deployment Decision
- Do **not** promote richer event/revision or intraday sleeves to live capital yet.
- Keep `v4` as flagship production core.
- Keep `v7` event sleeve behind quality-aware low-weight gating until:
  - non-demo revision/news coverage is broad (not single-name concentrated),
  - forced-weight ablations are non-destructive,
  - intraday sleeve replay is at least non-negative with stable drawdown under transaction costs.

### Step-1 Launch (Requested): Full PIT Ingestion + Coverage Audit

Date: 2026-04-02

User request:
- "start it" for the priority sequence:
  1) scale revision/news coverage,
  2) run `v7` event-weight grid before deployment.

#### What was run immediately

1) Polygon PIT large tranche
- Command:
  - `./.venv/bin/python scripts/backfill_polygon_pit.py --max-tickers 250 --chunk-size 25 --start-date 2016-01-01`
- Outcome:
  - all Benzinga/news/analyst/guidance + Polygon financial endpoints returned `NOT_AUTHORIZED` on current plan
  - inserts:
    - `events_inserted: 0`
    - `fundamentals_inserted: 0`

2) Post-tranche coverage audit
- `data/events.db`:
  - total rows: `184,539`
  - dominant sources:
    - `SEC_SUBMISSIONS`: `162,596`
    - `SEC_COMPANYFACTS`: `21,787`
    - `ALPHA_VANTAGE_EARNINGS`: `120`
    - `ALPHA_VANTAGE_EARNINGS_EST`: `35`
  - event-type coverage (distinct ids):
    - `earnings`: `362`
    - `news`: `362` (mostly SEC submission classification, not premium newswire)
    - `m&a`: `358`
    - `estimate_revision`: `1`
- `data/fundamentals.db`:
  - total rows: `95,055`
  - `ANALYST_EST_EPS` coverage: `2` ids
  - `ANALYST_REVISION` coverage: `1` id

Interpretation:
- coverage remains dominated by SEC filings.
- true analyst revision/news richness is still missing at scale.

#### Step-2 launched and completed: `v7` event-weight ablation grid

Same cache/data state (`407/410` daily cache; missing `PEAK`, `DISH`, `VX`):

| Forced event weight | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV | >=15% net years | >=1.0 Sharpe years | Both |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `0%` | `+10.97%` | `0.71` | `-29.79%` | `+7.08%` | `0.49` | `$70.63M` | `5/20` | `5/20` | `4/20` |
| `3%` | `+10.85%` | `0.71` | `-29.65%` | `+6.98%` | `0.49` | `$69.12M` | `5/20` | `5/20` | `4/20` |
| `5%` | `+10.63%` | `0.70` | `-29.70%` | `+6.81%` | `0.48` | `$66.60M` | `5/20` | `5/20` | `4/20` |
| `8%` | `+10.53%` | `0.69` | `-29.76%` | `+6.73%` | `0.47` | `$65.47M` | `4/20` | `5/20` | `4/20` |

Decision from grid:
- higher event weight remains destructive under current data quality.
- keep event sleeve near-zero/strictly quality-gated until richer revision/news feeds are available.

### Alpha Vantage Full-Key Run + Re-Ablation (Post-Key)

Date: 2026-04-02

After user added a real `ALPHAVANTAGE_API_KEY` to `.env`, a full-universe Alpha Vantage PIT run was completed and then `v7` was re-tested.

#### Full Alpha Vantage PIT backfill run

Command:
- `./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --requests-per-minute 20 --chunk-size 25`

Result summary:
- `events_inserted: 1283`
- `fundamentals_inserted: 2693`
- `event_duplicates: 121`
- `fundamental_duplicates: 360`
- `raw_cache_hits: 16`
- `raw_cache_writes: 1475`
- `symbols_skipped: 365`

Coverage after full run:
- `events_total: 185,822`
- major AV event sources:
  - `ALPHA_VANTAGE_EARNINGS: 854`
  - `ALPHA_VANTAGE_NEWS: 299`
  - `ALPHA_VANTAGE_EARNINGS_EST: 278`
  - `ALPHA_VANTAGE_EARNINGS_CALENDAR: 8`
- `fund_total: 97,748`
- major AV fundamentals:
  - `ALPHA_VANTAGE_EARNINGS: 2,536`
  - `ALPHA_VANTAGE_EARNINGS_EST: 591`
- distinct cross-id coverage:
  - `ANALYST_EST_EPS: 9`
  - `ANALYST_REVISION: 8`

#### Post-key v7 vs v4 and event-weight grid

Same run state: daily cache remains `407/410` (`PEAK`, `DISH`, `VX` unavailable in this environment).

| Engine | Event wt | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV | >=15% net years | >=1.0 Sharpe years | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v4` baseline | n/a | `+10.97%` | `0.71` | `-29.79%` | `+7.08%` | `0.49` | `$70.63M` | `5/20` | `5/20` | `4/20` |
| `v7` | `0%` | `+10.97%` | `0.71` | `-29.79%` | `+7.08%` | `0.49` | `$70.63M` | `5/20` | `5/20` | `4/20` |
| `v7` | `3%` | `+10.81%` | `0.71` | `-29.46%` | `+6.95%` | `0.49` | `$68.67M` | `5/20` | `5/20` | `4/20` |
| `v7` | `5%` | `+11.00%` | `0.72` | `-29.49%` | `+7.10%` | `0.49` | `$70.98M` | `5/20` | `5/20` | `4/20` |
| `v7` | `8%` | `+10.82%` | `0.71` | `-29.34%` | `+6.96%` | `0.49` | `$68.82M` | `4/20` | `5/20` | `4/20` |

Updated decision:
- event sleeve is no longer uniformly destructive after full-key ingestion.
- current best point in this grid is `v7` with forced `5%` event weight.
- improvement is modest but positive versus `v4`:
  - CAGR `+0.03%` (11.00 vs 10.97)
  - Sharpe `+0.01` (0.72 vs 0.71)
  - Final NAV `+$0.35M` (70.98 vs 70.63)
- remaining hard gap: the strategy still misses hedge-fund hurdles (`>=15% net` and `>=1.0 annual Sharpe`) in most years.

### Alpha Vantage Local-Only Cache Completion

Date: 2026-04-02

Goal:
- ensure Alpha Vantage payloads are fully persisted on device so reruns/backtests do not wait on API calls.

Code update:
- `src/data/ingest/alpha_vantage_event_backfill.py`
  - cache-aware pacing: only sleeps after real network requests
  - added cache telemetry in summary:
    - `raw_cache_misses`
    - `network_requests`

Verification run:
- `./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --cache-only --requests-per-minute 20`
  - initially showed `raw_cache_misses: 1`
  - missing key identified: `EARNINGS:MCD`
- hydrated missing key:
  - `./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --symbols MCD --requests-per-minute 20`
  - result: `network_requests: 1`, `raw_cache_writes: 1`
- re-verified full cache:
  - `./.venv/bin/python scripts/backfill_alpha_vantage_pit.py --cache-only --requests-per-minute 20`
  - result:
    - `raw_cache_hits: 1492`
    - `raw_cache_misses: 0`
    - `network_requests: 0`

Current local Alpha Vantage cache footprint:
- `data/cache/pit/alpha_vantage` total files: `1492`
- namespace split:
  - `json`: `746`
  - `news`: `373`
  - `csv`: `373`

Operational note:
- Backtests (`v7`) read from local `data/events.db` and `data/fundamentals.db`.
- Alpha Vantage API calls are now only needed for explicit refresh/backfill runs, not for normal backtest iteration.

### v7 Richer-Signal Iteration (Revision-Flow)

Date: 2026-04-02

User request:
- "use richer signal iteration for v7"

Implementation:
- `src/signals/event_alpha.py`
  - added dedicated **revision-flow panel** from PIT analyst revision/update history:
    - explicit `ANALYST_REVISION` pulses
    - implicit estimate-delta pulses from `ANALYST_EST_EPS` history
  - new config controls:
    - `revision_flow_weight` (default `0.18`)
    - `revision_flow_half_life_days` (default `18`)
    - `min_revision_flow_updates` (default `1`)
    - `revision_flow_quality_bonus` (default `0.08`)
  - richer composite now blends:
    - fundamental surprise/revision signal
    - revision-flow signal
    - sentiment/event signal
  - added `revision_flow_coverage` to build diagnostics.
- `backtest_v7.py`
  - wired new args through to event-alpha config:
    - `--revision-flow-weight`
    - `--revision-flow-half-life-days`
  - print diagnostics now include `revision_flow=<count>`.
- tests:
  - updated `tests/phase4/test_event_alpha.py` to assert revision-flow coverage/panel behavior.

Validation:
- `./.venv/bin/python -m pytest -q tests/phase4/test_event_alpha.py tests/phase9/test_intraday_alpha_paper.py tests/phase1/test_alpha_vantage_event_backfill.py`
  - `7 passed`

Backtest results on same cache state (`407/410` daily cache):

| Engine | Event wt | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV | >=15% net years | >=1.0 Sharpe years | Both |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `v4` baseline | n/a | `+10.97%` | `0.71` | `-29.79%` | `+7.08%` | `0.49` | `$70.63M` | `5/20` | `5/20` | `4/20` |
| `v7` richer (auto) | `3.5%` | `+10.85%` | `0.71` | `-29.34%` | `+6.98%` | `0.49` | `$69.12M` | `5/20` | `5/20` | `4/20` |
| `v7` richer (forced) | `5.0%` | `+11.01%` | `0.72` | `-29.45%` | `+7.11%` | `0.50` | `$71.01M` | `5/20` | `5/20` | `4/20` |

Interpretation:
- richer iteration improved the best `v7` point slightly vs prior run and vs `v4`:
  - `+0.04%` gross CAGR vs `v4`
  - `+0.01` gross Sharpe vs `v4`
  - `+$0.37M` final NAV vs `v4`
- hurdle profile remains unchanged (still misses 15% net and 1.0 yearly Sharpe in most years).

### v7 Kelly + Sentiment Overlay (NLP-driven risk scaling)

Date: 2026-04-02

Question addressed:
- can we apply Kelly-style position/risk scaling conditioned on market sentiment?

Implementation:
- `src/portfolio/sizing.py`
  - added `build_fractional_kelly_overlay(...)`
  - computes a rolling, no-lookahead fractional Kelly multiplier from historical event-sleeve edge
  - optionally modulates that multiplier with world sentiment (from event/NLP signal stream).
- `backtest_v7.py`
  - added optional overlay integration after portfolio construction
  - new CLI controls:
    - `--enable-kelly-sentiment-overlay` (default off)
    - `--kelly-lookback`
    - `--kelly-min-obs`
    - `--kelly-fraction`
    - `--kelly-max-abs`
    - `--kelly-min-scale`
    - `--kelly-max-scale`
    - `--kelly-sentiment-sensitivity`
  - prints overlay diagnostics (avg scale, range, kelly mean, positive-day share).
- tests:
  - new: `tests/phase5/test_sizing.py`
  - validates overlay directionality for positive/negative edge and sentiment shift.

Validation:
- `./.venv/bin/python -m pytest -q tests/phase5/test_sizing.py tests/phase4/test_event_alpha.py`
  - `6 passed`

Backtest A/B (`v7 --force-event-weight 0.05`, same cache/data state):

| Config | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | Final NAV |
|---|---:|---:|---:|---:|---:|
| Kelly overlay **off** (default) | `+11.01%` | `0.72` | `-29.45%` | `+7.11%` | `$71.01M` |
| Kelly overlay **on** | `+10.93%` | `0.73` | `-30.20%` | `+7.04%` | `$70.06M` |

Overlay diagnostics (on):
- avg scale `0.980`, range `[0.920, 1.081]`
- kelly avg `-0.2527`, positive days `36.9%`

Decision:
- keep Kelly overlay **opt-in** (default off) for now.
- it improved Sharpe slightly but hurt CAGR/NAV and worsened drawdown in this run.
- next step before enabling by default: calibrate kelly from cleaner alpha sources (e.g., richer intraday/event edge + stronger sentiment quality).

### Hugging Face FinBERT Enrichment Pipeline (Implemented + Run)

Date: 2026-04-02

User request:
- "do it" for Hugging Face / NLP integration into `v7`.

Implementation:
- Added `src/data/ingest/finbert_event_backfill.py`
  - scores existing PIT events with FinBERT
  - writes enriched events back to `event_pit` with source `FINBERT_ENRICHED`
  - uses local raw-cache (`RawDataCache`) for deterministic offline reruns
  - supports `--cache-only`, `--since`, source/event filters, and dedupe.
- Added CLI `scripts/backfill_finbert_pit.py`
  - end-to-end FinBERT enrichment runner.
- Added tests:
  - `tests/phase1/test_finbert_event_backfill.py`
- Updated signal source quality:
  - `FINBERT_ENRICHED` added to `EVENT_SOURCE_QUALITY` in `src/signals/event_alpha.py`.
- Added NLP optional deps:
  - `pyproject.toml` optional extra `nlp = ["transformers>=4.45", "torch>=2.3"]`.
- Updated docs:
  - `README.md` includes FinBERT install/backfill/cache commands.

Validation:
- `./.venv/bin/python -m pytest -q tests/phase1/test_finbert_event_backfill.py tests/phase4/test_event_alpha.py tests/phase5/test_sizing.py`
  - `8 passed`

FinBERT backfill run (Hugging Face model):
- command:
  - `./.venv/bin/python scripts/backfill_finbert_pit.py --since 2016-01-01T00:00:00Z --sources ALPHA_VANTAGE_NEWS,ALPHA_VANTAGE_EARNINGS,ALPHA_VANTAGE_EARNINGS_EST,POLYGON_BENZINGA_NEWS,POLYGON_BENZINGA_ANALYST,POLYGON_BENZINGA_GUIDANCE,csv_ingest --event-types 'news,guidance,earnings,estimate_revision,m&a,product'`
- output:
  - `processed: 1391`
  - `inserted: 1391`
  - `raw_cache_writes: 1391`
  - `model: ProsusAI/finbert`
- offline verification:
  - same command with `--cache-only`
  - `raw_cache_hits: 1391`
  - `raw_cache_misses: 0`
  - `inserted: 0` (all duplicates, as expected)

Post-enrichment event state:
- `event_pit` total rows: `187,713`
- `FINBERT_ENRICHED`: `1,391` rows
- distinct tickers in `FINBERT_ENRICHED`: `8`

`v7` post-FinBERT backtests:

| Config | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV |
|---|---:|---:|---:|---:|---:|---:|
| `v7` auto (Kelly off) | `+10.85%` | `0.71` | `-29.34%` | `+6.98%` | `0.49` | `$69.08M` |
| `v7` force 5% (Kelly off) | `+11.01%` | `0.72` | `-29.45%` | `+7.11%` | `0.50` | `$71.01M` |
| `v7` force 5% + Kelly on | `+10.93%` | `0.73` | `-30.20%` | `+7.04%` | `0.50` | `$70.06M` |

Interpretation:
- FinBERT integration is live and cached locally.
- Performance impact is minimal so far because FinBERT coverage is still narrow (`8` tickers); this is a data coverage bottleneck, not a model wiring issue.

### Free SEC-NLP Expansion (No Paid Data) — Completed

Date: 2026-04-02

Goal:
- expand NLP sentiment coverage without paying for premium news feeds.

What was implemented:
- Added SEC-scale FinBERT enrichment pipeline:
  - `src/data/ingest/finbert_event_backfill.py`
  - `scripts/backfill_finbert_pit.py`
- Added optional NLP dependencies:
  - `pyproject.toml` optional extra `nlp` (`transformers`, `torch`)
- Added tests:
  - `tests/phase1/test_finbert_event_backfill.py`
- Updated event source quality mapping:
  - `FINBERT_ENRICHED` in `src/signals/event_alpha.py`
- Exposed in docs:
  - `README.md` FinBERT install and backfill commands

Validation:
- `./.venv/bin/python -m pytest -q tests/phase1/test_finbert_event_backfill.py tests/phase4/test_event_alpha.py tests/phase5/test_sizing.py`
  - `8 passed`

Full SEC FinBERT backfill run:
- command:
  - `./.venv/bin/python scripts/backfill_finbert_pit.py --sources SEC_SUBMISSIONS,SEC_COMPANYFACTS --event-types 'news,guidance,earnings,m&a,product' --min-confidence 0.05`
- output:
  - `processed: 184,383`
  - `inserted: 182,383`
  - `duplicates: 2,000`
  - `raw_cache_misses/raw_cache_writes: 182,383`
  - `model: ProsusAI/finbert`
- cache-only verification slice:
  - `./.venv/bin/python scripts/backfill_finbert_pit.py --cache-only ... --limit 5000`
  - `raw_cache_hits: 5000`, `raw_cache_misses: 0`

Coverage after SEC expansion:
- `event_pit` total: `372,096`
- `FINBERT_ENRICHED`: `185,774`
- `SEC_SUBMISSIONS`: `162,596`
- `SEC_COMPANYFACTS`: `21,787`
- distinct tickers in `FINBERT_ENRICHED`: `362`

Post-expansion `v7` results:

| Config | Event quality | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v7` auto, Kelly off | sentiment quality `0.49`, quality scale `0.53`, event wt `3.5%` | `+10.85%` | `0.71` | `-29.41%` | `+6.98%` | `0.49` | `$69.12M` |
| `v7` force 5%, Kelly off | sentiment quality `0.49`, quality scale `0.53`, event wt `5.0%` | `+10.87%` | `0.71` | `-29.44%` | `+7.00%` | `0.49` | `$69.37M` |
| `v7` force 5%, Kelly on | overlay avg `0.981` range `[0.920,1.081]` | `+10.85%` | `0.72` | `-30.19%` | `+6.98%` | `0.50` | `$69.16M` |

Interpretation:
- Free SEC-based NLP coverage was successfully scaled to almost full equity universe.
- Raw coverage quality improved (`0.46 -> 0.49` sentiment quality scale), but portfolio alpha did **not** improve; final NAV declined vs prior best `v7` checkpoint.
- Current bottleneck is **signal selectivity**, not data availability: most additional SEC-NLP events are low edge/noisy.

### Event Selectivity Calibration (FINBERT/SEC filters) — Completed

Date: 2026-04-02

Goal:
- verify whether filtering low-conviction FinBERT/SEC events improves `v7` vs fully unfiltered ingestion.

Implementation:
- Added conviction gates in `src/signals/event_alpha.py`:
  - `finbert_min_abs_score`, `finbert_min_confidence`
  - `sec_min_abs_score`, `sec_min_confidence`
- Exposed flags in `backtest_v7.py`:
  - `--finbert-min-abs-score`
  - `--finbert-min-confidence`
  - `--sec-min-abs-score`
  - `--sec-min-confidence`
- Added tests in `tests/phase4/test_event_alpha.py`:
  - FinBERT high-conviction filtering
  - SEC low-signal filtering

Validation:
- `./.venv/bin/python -m pytest -q tests/phase4/test_event_alpha.py`
  - pass

Head-to-head calibration (`--force-event-weight 0.05`, Kelly off):

| Config | Events | Sentiment Quality | Gross CAGR | Gross Sharpe | Gross Max DD | LP Net CAGR | LP Net Sharpe | Final NAV |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Filtered default thresholds | `59,872` | `0.48` | `+11.09%` | `0.72` | `-29.45%` | `+7.17%` | `0.50` | `$72.05M` |
| Unfiltered (`all mins = 0`) | `59,966` | `0.49` | `+10.87%` | `0.71` | `-29.44%` | `+7.00%` | `0.49` | `$69.37M` |

Decision:
- Keep filtered defaults as current `v7` baseline.
- Removing filters increased event count slightly but hurt CAGR/Sharpe and reduced final NAV by ~`$2.69M`.
- Next alpha step should focus on richer event types (analyst estimate deltas/transcript surprises) and intraday execution edge, not simply adding more low-conviction daily events.

### v8 Multi-Asset Expansion + Options Execution Path (Implemented + Backtested)

Date: 2026-04-02

Goal:
- expand the tradable book beyond equities and add an explicit options hedge path while keeping the v7 core.

What was added:
- New runner: `backtest_v8.py`
  - keeps v7 equity/event core (same `EventAlphaConfig` wiring)
  - adds alpha sleeves for:
    - ETFs
    - futures/commodities/bonds
    - FX
  - adds dedicated VIX sleeve (long-vol hedge when stress rises)
  - adds options execution path via synthetic rolled-put overlay:
    - notional scales only in high-crisis regimes
    - strike/theta/payout are configurable via CLI
    - emits a roll plan (`date`, `SPY`, put strike, target notional) for live order-routing handoff
- New robustness controls in v8:
  - cross-asset return sanitization to zero-out impossible jumps from futures roll/data artifacts
  - per-asset transaction cost model (ETF/FX/futures/vol differentiated)
  - global gross cap across combined sleeves.
- New tests:
  - `tests/phase8/test_backtest_v8_helpers.py`
    - return sanitization clipping
    - option overlay calm-regime off behavior
    - option overlay crisis-regime activation behavior.

Validation:
- `./.venv/bin/python -m pytest -q tests/phase8/test_backtest_v8_helpers.py`
  - `3 passed`
- `./.venv/bin/python -m pytest -q tests/phase4/test_event_alpha.py tests/phase5/test_sizing.py`
  - `8 passed`

Backtest observations (`v8` default):
- Universe actually traded in this run:
  - equities `360`, ETFs `13`, futures/commod/bond `16`, FX `7`, VIX `0` (`VX` cache missing)
- Option overlay:
  - avg notional `0.011x`, max `0.040x`, active days `47.6%`
- Performance:
  - Gross CAGR `+11.51%`
  - Gross Sharpe `0.77`
  - Gross Max DD `-26.04%`
  - LP Net CAGR `+7.49%` (2/20)
  - LP Net Sharpe `0.53`
  - Final NAV `$77.23M`
  - LP NAV `$38.79M`
  - Avg gross `1.74x`
  - Tx costs `49 bps/yr`
  - Hurdles: `Years >=15% net 6/20`, `Years >=1.0 Sharpe 6/20`, both `5/20`

Sanity baseline (`v8` with all new sleeves off):
- command:
  - `./.venv/bin/python backtest_v8.py --etf-gross 0 --futures-gross 0 --fx-gross 0 --vix-hedge-max 0 --option-max-notional 0`
- result:
  - Gross CAGR `+10.66%`
  - Gross Sharpe `0.74`
  - Gross Max DD `-23.90%`
  - Final NAV `$66.97M`

Interpretation:
- v8 implementation is working and materially improves over its own “core-only” mode.
- return uplift is moderate and now plausible after artifact sanitization; no longer relies on impossible futures jumps.
- largest remaining blocker for full multi-asset hedge behavior is missing `VX` cache coverage.

Extra calibration:
- `v8 --overlay-shorts` was tested and did **not** improve headline returns:
  - Gross CAGR `+10.91%`, Gross Sharpe `0.77`, Final NAV `$69.84M`
- default long-bias cross-asset overlays remain the better current baseline.

### VIX Sleeve Block Fix (`VX` cache gap) — Implemented

Date: 2026-04-02

Issue:
- `VX_VOLATILITY.parquet` was missing because IBKR was unavailable and Polygon/Alpaca do not return `VX`.

Fix:
- Updated `scripts/backfill_daily_bars.py` with two-level fallback for `VOLATILITY` symbols (`VX`, `VIX`, `^VIX`):
  1. Alpha Vantage daily VIX proxy (`TIME_SERIES_DAILY`, symbol `VIX`) when DNS/API is available.
  2. Synthetic VIX proxy generated locally from cached `SPY_ETF.parquet` when external fetch fails.
- Synthetic proxy writes the same cache target:
  - `~/.one_brain_fund/cache/bars/VX_VOLATILITY.parquet`

Validation:
- `./.venv/bin/python scripts/backfill_daily_bars.py --symbols VX --config config/sp500_universe.yaml`
  - fallback used: `synthetic_vix_from_spy`
  - wrote `5034` rows (`2006-04-06 -> 2026-04-01`)
- `v8` now reports `vix=1` trade bucket and non-zero vix sleeve gross (`~0.05x`).

Impact:
- Technical block is resolved (sleeve runs), but this synthetic proxy reduced overall returns in current calibration.
- Recommended policy:
  - use real `VX` (IBKR/CFE) for production if possible,
  - keep synthetic fallback for continuity/testing and gate vix sleeve size conservatively.

### Synthetic Data Policy Update — Enforced

Date: 2026-04-02

User directive:
- no synthetic data, only real verified data.

Changes made:
- removed synthetic VX generation fallback from `scripts/backfill_daily_bars.py`.
- retained only real-provider pathways for `VX`:
  - IBKR / Polygon / Alpaca (native paths)
  - Alpha Vantage VIX proxy fallback (real external feed) when reachable
- deleted synthetic cache artifacts:
  - `~/.one_brain_fund/cache/bars/VX_VOLATILITY.parquet`
  - `~/.one_brain_fund/cache/bars/VX_VOLATILITY.meta.json`

Validation:
- reran `./.venv/bin/python scripts/backfill_daily_bars.py --symbols VX --config config/sp500_universe.yaml`
  - result: no write when real providers unavailable (`written=0`, `failed=1`)
  - confirms synthetic fallback is no longer in effect.

### v8 Hardening Pass (Kelly Lock + OOS Revalidation)

Date: 2026-04-03

Goal:
- improve robustness without sacrificing alpha before live deployment.

Code changes:
- `backtest_v8.py`
  - added optional global regime-risk scaler (`--enable-regime-risk-scaler`) with bounds/smoothing knobs:
    - `--risk-floor`, `--risk-ceiling`, `--risk-smooth-days`
  - added diagnostics to metrics:
    - `option_active_days`
    - `regime_risk_scaler_enabled`
    - `regime_risk_scaler_diag`
- `scripts/run_v8_walk_forward.py`
  - params ingestion now supports:
    - `target_vol`
    - `enable_kelly_sentiment_overlay`
    - `enable_regime_risk_scaler`
    - `risk_floor`, `risk_ceiling`, `risk_smooth_days`
- `scripts/run_v8_regime_splits.py`
  - same param extensions as walk-forward runner.

Production lock update:
- `config/v8_production_locked.yaml` updated to Kelly-on lock:
  - `enable_kelly_sentiment_overlay: 1`
  - `enable_regime_risk_scaler: 0`
  - retained constrained core params (`event=0.03`, `overlay_min_signal=0.08`, gross cap `1.80`, etc.).

Full-history A/B (cache-complete, no-lookahead):
- baseline lock:
  - LP CAGR `+7.075%`, LP Sharpe `0.527`, LP Max DD `-25.20%`, Final NAV `$70.52M`
- +Kelly (selected lock):
  - LP CAGR `+7.066%`, LP Sharpe `0.530`, LP Max DD `-23.23%`, Final NAV `$70.41M`
- regime-scaler variants were tested but not promoted:
  - conservative scaler reduced CAGR/NAV
  - aggressive scaler improved CAGR/NAV but worsened drawdown.

Walk-forward revalidation (`14` folds):
- command:
  - `./.venv/bin/python scripts/run_v8_walk_forward.py --params-file config/v8_production_locked.yaml --output-csv data/reports/v8_walk_forward_locked_kelly.csv --min-train-years 5 --test-years 1 --step-years 1`
- summary:
  - mean LP CAGR: `0.06776` (vs `0.06769` prior lock)
  - mean LP Sharpe: `0.40746` (vs `0.38464`)
  - mean LP Max DD: `-0.12598` (vs `-0.12859`)
  - pass count: `4/14` (unchanged)

Regime-split revalidation (`5` splits):
- command:
  - `./.venv/bin/python scripts/run_v8_regime_splits.py --params-file config/v8_production_locked.yaml --output-csv data/reports/v8_regime_splits_kelly.csv --train-years 5 --recent-years 2 --warmup-days 180 --min-rows 360`
- highlights vs prior lock:
  - improved `2020+` and `recent` LP CAGR/Sharpe and shallower DD
  - `2008` split slightly worse CAGR/DD (still within expected crisis stress behavior)
  - all splits remain below the strict `15% / Sharpe 1.0` hurdle.

Validation:
- `./.venv/bin/python -m pytest tests/phase8/test_backtest_v8_helpers.py tests/phase4/test_event_alpha.py -q`
- result: `11 passed`.

### v8 Crash-Hedge Upgrade: Adaptive Put-Spread Overlay

Date: 2026-04-03

Objective:
- improve crisis robustness (especially 2008-like tape) without giving up core alpha.

Implemented:
- `backtest_v8.py`
  - upgraded synthetic option overlay to adaptive put-spread logic:
    - long put leg always active when hedge is on
    - short farther-OTM leg used in moderate stress to reduce carry bleed
    - short-leg coverage auto-reduced in severe stress to restore convexity
  - added stress-trigger knobs:
    - `--option-short-strike-daily` (default `0.07`)
    - `--option-short-credit-bps-daily` (default `0.9`)
    - `--option-activation-score` (default `0.38`)
    - `--option-severe-score` (default `0.78`)
  - fixed overlay timing to avoid same-bar lookahead:
    - roll decisions made at bar `t` apply to bar `t+1` returns
  - extended diagnostics:
    - `option_avg_short_coverage`
    - enhanced option roll plan fields (`short_strike`, `short_coverage`, `PUT_SPREAD` tag)
- runner plumbing:
  - `scripts/run_v8_walk_forward.py` reads/passes new option params.
  - `scripts/run_v8_regime_splits.py` reads/passes new option params.
- lock update:
  - `config/v8_production_locked.yaml` now explicitly includes new option params.

Validation (full-history, same locked settings, cache-complete + no-lookahead):
- baseline Kelly lock (before put-spread):
  - Gross CAGR `10.955%`, Gross Sharpe `0.777`, Gross Max DD `-21.10%`
  - LP CAGR `7.066%`, LP Sharpe `0.530`, LP Max DD `-23.23%`
  - Final NAV `$70.41M`, LP NAV `$36.02M`
- adaptive put-spread lock (after upgrade):
  - Gross CAGR `11.034%`, Gross Sharpe `0.782`, Gross Max DD `-20.57%`
  - LP CAGR `7.130%`, LP Sharpe `0.534`, LP Max DD `-22.71%`
  - Final NAV `$71.36M`, LP NAV `$36.42M`
- delta:
  - LP CAGR `+0.063 pp`
  - LP Sharpe `+0.004`
  - LP Max DD improved by `0.52 pp`
  - Final NAV `+$0.95M`

Walk-forward revalidation (14 folds):
- output: `data/reports/v8_walk_forward_locked_putspread.csv`
- vs prior Kelly lock:
  - mean LP CAGR `0.06776 -> 0.06787`
  - mean LP Sharpe `0.40746 -> 0.40820`
  - mean LP Max DD `-0.12598 -> -0.12594`
  - pass count unchanged: `4/14`

Regime split revalidation (5 splits):
- output: `data/reports/v8_regime_splits_putspread.csv`
- notable changes vs prior Kelly lock:
  - `crisis_2008` improved materially:
    - LP CAGR `-0.1747 -> -0.1686`
    - LP Sharpe `-1.5613 -> -1.5049`
    - LP Max DD `-0.1901 -> -0.1848`
  - `2020+` and `recent` also improved slightly
  - `2010s` mostly flat/slightly softer DD.

Tests:
- updated phase8 helper tests for new overlay API + short-leg coverage assertions.
- `./.venv/bin/python -m pytest tests/phase8/test_backtest_v8_helpers.py tests/phase4/test_event_alpha.py -q`
- result: `11 passed`.

### Annual Hurdle Instrumentation + Strict 2010+ Check

Date: 2026-04-03

Objective update from PM:
- optimize for annual **gross** hurdles:
  - yearly gross return `>= 15%`
  - yearly gross Sharpe `>= 1.0`
- also minimize drawdown **time** (not only depth).

Implementation:
- `backtest_v8.py`
  - added annual gross hurdle accounting:
    - `years_ge_gross_return_target`
    - `years_ge_gross_sharpe_target`
    - `years_ge_gross_both_targets`
  - retained LP hurdle accounting for parallel tracking.
  - added drawdown-duration metrics:
    - `max_underwater_days_gross`
    - `max_underwater_days_lp`
  - added CLI targets:
    - `--gross-target-return` (default `0.15`)
    - `--gross-target-sharpe` (default `1.0`)
- `scripts/run_v8_constraint_sweep.py`
  - scoring now rewards annual gross hurdle hit counts and penalizes LP underwater duration.
  - strict constraint gate added:
    - `--min-gross-both-ratio` (default `1.0` for “all years”)
  - added optional date controls for focused regime sweeps:
    - `--start-date`, `--eval-start`, `--eval-end`, `--warmup-days`, `--min-rows`

Validation run (2010+ strict):
- command:
  - `./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 8 --start-date 2010-01-01 --eval-start 2010-01-01 --warmup-days 1 --min-rows 360 --min-gross-both-ratio 1.0 --output-csv data/reports/v8_constraint_sweep_2010_strict.csv`
- result:
  - `0/8` configs satisfied strict all-year gross-hurdle constraint.
  - best observed `years_ge_gross_both_targets` in this grid: `6/17` years (`ratio 0.353`).

Reference single-run (current lock, 2010+ full eval):
- gross CAGR `11.35%`, gross Sharpe `0.83`, gross max DD `-17.02%`
- annual gross hurdles:
  - gross return `>=15%`: `6/17`
  - gross Sharpe `>=1.0`: `8/17`
  - both: `6/17`
- max underwater days:
  - gross `445`
  - LP `547`

### Full-History Annual Gross Hurdle Push (15% + Sharpe 1.0 Every Year)

Date: 2026-04-03

Objective:
- treat this as primary hard target on full history:
  - yearly gross return `>= 15%`
  - yearly gross Sharpe `>= 1.0`
  - all years (no averaging across decades)

Implementation:
- `backtest_v8.py`
  - added per-year diagnostics payload in metrics json:
    - `annual_summary` (year, gross/lp return, gross/lp Sharpe, and hit flags)
  - added minimum annual stats:
    - `min_gross_year_return`
    - `min_gross_year_sharpe`
    - `min_lp_year_return`
    - `min_lp_year_sharpe`
  - `total_years` now counts only valid evaluated years in the annual loop.
- `scripts/run_v8_constraint_sweep.py`
  - expanded search space to include yearly-consistency levers:
    - `target_vol`
    - option stress controls (`option_short_strike_daily`, `option_activation_score`, `option_severe_score`)
    - Kelly and regime-risk toggles
    - risk scaler bounds (`risk_floor`, `risk_ceiling`)
    - governance drawdown thresholds (`gov_soft_dd`, `gov_hard_dd`)
  - objective now penalizes worst-year gaps directly:
    - penalty on `gross_target_return - min_gross_year_return`
    - penalty on `gross_target_sharpe - min_gross_year_sharpe`
  - added explicit CLI targets:
    - `--gross-target-return`
    - `--gross-target-sharpe`
  - replaced full cartesian materialization with deterministic sampled candidate generation
    (keeps runtime/memory stable after expanding search dimensions).

Validation / runs:
- probe sweep (expanded space):
  - `./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 6 --min-gross-both-ratio 0.0 --output-csv data/reports/v8_constraint_sweep_yearly_probe.csv`
  - best observed annual gross-both hit count: `6/20` years.
- strict full-history sweep:
  - `./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 4 --min-gross-both-ratio 1.0 --output-csv data/reports/v8_constraint_sweep_yearly_strict_full.csv`
  - result: `0/4` feasible at all-year hard constraint.
  - top row still only `6/20` years with both gross hurdles.
- sampler smoke test:
  - `./.venv/bin/python scripts/run_v8_constraint_sweep.py --max-runs 2 --min-gross-both-ratio 0.0 --output-csv /tmp/v8_sweep_smoke.csv`
  - confirms deterministic low/high anchors execute correctly in the new sampled-candidate mode.
- single-run deep diagnostic (best strict candidate):
  - gross CAGR `10.23%`, gross Sharpe `0.756`, max DD `-18.96%`, final NAV `$62.23M`
  - annual gross both: `6/20`
  - worst annual gross return: `-12.33%`
  - worst annual gross Sharpe: `-1.175`
  - worst years by gross Sharpe: `2008`, `2022`, `2011`, `2018`.
- current production lock diagnostic:
  - gross CAGR `11.03%`, gross Sharpe `0.782`, max DD `-20.57%`, final NAV `$71.36M`
  - annual gross both: `6/20`
  - worst annual gross return: `-14.83%`
  - worst annual gross Sharpe: `-1.311`

Takeaway:
- under current daily-bar, medium-frequency architecture, “every year >=15% gross and >=1.0 gross Sharpe” is not reachable in sampled realistic parameter space.
- bottleneck is not just aggregate Sharpe; it is specific crisis/whipsaw years where annual Sharpe turns negative.

Quick sanity test:
- `./.venv/bin/python -m pytest tests/phase8/test_backtest_v8_helpers.py -q`
- result: `5 passed`.

### Macro Signal Reality Check + Walk-Forward Hardening (v7/v8)

Date: 2026-04-03

Objective:
- test whether adding macro improves OOS performance.
- tighten no-lookahead walk-forward enforcement beyond v8.

What we found on macro data:
- `data/events.db` currently has `0` rows with `event_type='macro'`.
- Current event mix in `event_pit` is dominated by `earnings`, `news`, and `m&a`.
- implication: “macro via event alpha” is currently inactive until dedicated macro event backfill is added.

Macro proxy experiment (v8):
- tested macro-style regime control by enabling `--enable-regime-risk-scaler` on top of production lock.
- full-history compare (same cache-complete + no-lookahead guard):
  - base lock:
    - gross CAGR `11.03%`, gross Sharpe `0.782`, gross max DD `-20.57%`, final NAV `$71.36M`
  - macro-proxy on:
    - gross CAGR `10.62%`, gross Sharpe `0.762`, gross max DD `-21.29%`, final NAV `$66.50M`
  - annual gross-both hits unchanged: `6/20`
- walk-forward compare (14 folds, strict OOS windows):
  - base (`data/reports/v8_walk_forward_locked_putspread.csv`)
    - mean LP CAGR `0.06787`
    - mean LP Sharpe `0.40820`
    - median LP max DD `-0.12590`
    - pass count `4/14`
  - macro-proxy (`data/reports/v8_walk_forward_macroproxy.csv`)
    - mean LP CAGR `0.06461`
    - mean LP Sharpe `0.37987`
    - median LP max DD `-0.12424`
    - pass count `4/14`
- verdict:
  - macro-proxy reduced return/Sharpe and did not improve hurdle hit count.
  - no evidence (yet) that macro improves this stack with current data.

Walk-forward/no-lookahead hardening:
- `backtest_v7.py` now supports strict OOS controls:
  - `--start-date`, `--end-date`
  - `--eval-start`, `--eval-end`
  - `--warmup-days`, `--min-rows`
  - `--enforce-no-lookahead`
  - `--metrics-json`
- added `scripts/run_v7_walk_forward.py`:
  - cache-complete fold generation
  - strict test windows (end-date + eval-start/eval-end)
  - fold metrics + hurdle pass summary
- smoke validation:
  - `./.venv/bin/python backtest_v7.py --cache-complete-only --enforce-no-lookahead --start-date 2015-01-01 --end-date 2021-12-31 --eval-start 2020-01-01 --eval-end 2021-12-31 --warmup-days 120 --min-rows 120 --force-event-weight 0.03 --metrics-json /tmp/v7_smoke_metrics.json`
  - produced metrics successfully with `lookahead_guard=true`.
  - `gross Sharpe ~1.01`, `gross CAGR ~16.99%` on that bounded sample.
  - `scripts/run_v7_walk_forward.py --max-folds 2` smoke run completed and wrote `/tmp/v7_wf_smoke.csv`.

Repository audit helper:
- added `scripts/audit_walk_forward_backtests.py` to print per-version support matrix for:
  - walk-forward banner
  - lagged execution (`shift(1)`)
  - strict OOS CLI controls (`start/end`, `eval`, `enforce-no-lookahead`, `metrics-json`)
- current matrix confirms strict window controls are now present in `v7` and `v8`; earlier versions still need explicit strict-window CLI standardization.

### Lightweight Macro (Optional) + Multi-Strategy First

Date: 2026-04-03

Decision:
- keep macro as an optional overlay inside v8, not a separate complex subsystem.
- preserve multi-strategy architecture (core + ETF + futures + FX + options) as primary alpha engine.

What was validated:
- two-fold strict walk-forward A/B with identical settings except macro overlay toggle:
  - off: `/tmp/v8_wf_off_2fold.csv`
  - on: `/tmp/v8_wf_macro_2fold.csv`
  - result: metrics were identical on this environment because macro cache was missing.
- direct v8 run with `--enable-macro-overlay` confirmed safe no-op behavior:
  - macro status prints `no_macro_cache`
  - scaler stats stayed `avg=1.000 min=1.000 max=1.000`
  - run completed normally and preserved lookahead guard.

Current macro data status:
- `~/.one_brain_fund/cache/macro` is not present yet.
- `FRED_API_KEY` is not set in current shell session.
- implication: macro overlay is production-safe but inactive until free FRED cache is backfilled.

Why this is useful:
- we can continue improving multi-strategy sleeves without adding fragility.
- when macro cache is ready, overlay can be activated with one flag/config toggle and immediately measured in walk-forward.

### Macro Backtest Run (with real FRED cache) + Alignment Fix

Date: 2026-04-03

What was done:
- populated free macro cache from FRED after adding `FRED_API_KEY`:
  - command: `./.venv/bin/python scripts/backfill_macro_fred.py --refresh-cache --start-date 1990-01-01 --end-date 2026-04-03`
  - result: `6` series written, merged panel at `~/.one_brain_fund/cache/macro/fred_daily.parquet` (`13241` rows, `6` cols).

Issue found and fixed:
- macro overlay initially showed no impact despite cache present.
- root cause: timezone/intraday index mismatch in `load_macro_panel(...)` produced all-NaN macro features after reindex.
- fix:
  - normalize both macro index and target index to calendar dates.
  - drop timezone before alignment.
  - align using normalized dates, then restore caller index shape.
  - collapse duplicate normalized dates with `groupby(level=0).last()`.
- regression test added:
  - `tests/phase8/test_backtest_v8_helpers.py::test_macro_panel_aligns_on_calendar_dates_for_tz_aware_index`
  - phase-8 helpers now `8 passed`.

Backtest compare (full-history, cache-complete, no-lookahead):
- baseline (macro off): `/tmp/v8_macro_off_full.json`
- macro on (fixed): `/tmp/v8_macro_on_full_fix.json`

Observed deltas (macro on minus off):
- gross CAGR: `-0.06 pts` (11.03% -> 10.97%)
- gross Sharpe: `+0.0003` (effectively flat)
- gross max DD: improved by `+0.15 pts` (less negative)
- final NAV: `- $0.73M`
- LP net CAGR: `-0.05 pts`
- LP net Sharpe: `-0.0005` (flat)
- LP net max DD: improved by `+0.15 pts`
- LP annual hurdles:
  - years with net Sharpe >= 1.0: `5 -> 6`
  - years hitting both net hurdles: `4 -> 5`

Macro overlay diagnostics (fixed run):
- status: `ok`
- components used: `3`
- scale avg/min/max: `0.983 / 0.913 / 1.000`
- avg macro score: `-0.009`

Interpretation:
- lightweight macro de-risking is now active and real.
- it slightly improves downside profile / LP hurdle consistency, but trims total return and terminal NAV in this calibration.

### v10 Priority Stack Implemented (Router -> Convexity -> Whipsaw -> Yearly Budget -> Worst-Year Objective)

Date: 2026-04-03

Implemented in order of priority:
- `backtest_v8.py`
  - added Regime Router 2.0 (`risk_on/risk_off/crash`) with lagged inputs only (`shift(1)` style usage).
  - upgraded crash convexity overlay to keep hedge always defined and scale by crash probability.
  - added whipsaw control layer (entry persistence + exit hysteresis).
  - added yearly risk budget controller (monthly-adjusted scaler from realized YTD vol/DD, lagged).
  - exposed new CLI flags for all v10 controls.
- sweep objective updated to worst-year-first:
  - `scripts/run_v8_constraint_sweep.py` now optimizes primary constraints first
    (min annual return/sharpe gaps + both-hurdle gap), then secondary quality terms.
- walk-forward/regime runners updated to accept/pass v10 controls:
  - `scripts/run_v8_walk_forward.py`
  - `scripts/run_v8_regime_splits.py`
- locked config added:
  - `config/v10_production_locked.yaml`
- tests expanded:
  - `tests/phase8/test_backtest_v8_helpers.py` includes v10 helper coverage.

Validation runs:
- phase-8 helper tests:
  - `./.venv/bin/python -m pytest tests/phase8/test_backtest_v8_helpers.py -q`
  - result: `11 passed`
- strict no-lookahead v10 smoke:
  - sample: `2020-01-02 -> 2021-12-31` (cache-complete mode, `396` tradable instruments)
  - gross: CAGR `18.04%`, Sharpe `1.07`, max DD `-13.26%`, final NAV `$14.02M`
  - LP net: CAGR `12.49%`, Sharpe `0.78`, max DD `-13.42%`, LP NAV `$12.74M`
  - annual gross hurdles (sample): both `1/2`
  - diagnostics:
    - router: `risk_on=1608, risk_off=108, crash=89, avg_crash_prob=0.101`
    - whipsaw turnover reduction: `0.00236`
    - yearly risk budget scale avg/min/max: `1.004 / 0.838 / 1.054`
- one-fold walk-forward smoke (`/tmp/v10_wf_smoke.csv`):
  - fold test `2011-04-06..2012-04-05`
  - LP net CAGR `-10.08%`, LP net Sharpe `-0.69`, LP max DD `-20.85%`, pass `False`
- regime split smoke (`/tmp/v10_regime_smoke.csv`):
  - all splits executed successfully (`5/5` status `ok`)
  - LP results still below strict 15%/1.0 gates in all current splits.
- worst-year-first constrained sweep smoke (`/tmp/v10_constraint_smoke.csv`):
  - ran `6` candidates on bounded strict window (`2020-2021` eval).
  - all candidates remained `constraint_ok=False`.
  - top candidate snapshot: LP net CAGR `11.08%`, LP net Sharpe `0.71`, LP max DD `-12.45%`,
    min gross year return `9.82%`, min gross year Sharpe `0.75`.
  - confirms objective plumbing is active and ranking by worst-year gaps, but gates are still not met.

### v10 Alpha-Recovery Calibration (full-history, strict no-lookahead)

Date: 2026-04-03

Prompted by NAV drop concern (`~$70M` -> `~$65M`), ran full-history layer ablation to identify v10 drag sources.

Artifacts:
- `data/reports/v10_tuning_ablation_2026-04-03.json`
- `data/reports/v10_layer_ablation_2026-04-03.json`

Key full-history results (all from same base settings, cache-complete, no-lookahead):
- `v10_full_current` (router + whipsaw + yearly + always-on min hedge):
  - gross CAGR `10.21%`, gross Sharpe `0.720`, gross max DD `-21.69%`, final NAV `$65.14M`
- `v8_like_no_v10_layers`:
  - gross CAGR `10.78%`, gross Sharpe `0.773`, gross max DD `-20.57%`, final NAV `$71.86M`
- `v10_router_only` (best tradeoff):
  - gross CAGR `10.94%`, gross Sharpe `0.757`, gross max DD `-20.45%`, final NAV `$73.93M`

Attribution takeaway:
- Router layer adds return and improves DD control.
- Whipsaw + yearly budget stack introduces most of the alpha drag in this calibration.
- Always-on minimum hedge notional (`0.004`) did not help enough to justify carry drag.

Action taken:
- Updated `config/v10_production_locked.yaml` to router-only alpha-recovery lock:
  - `option_always_on_min_notional: 0.0`
  - `enable_regime_router_v10: 1`
  - `enable_whipsaw_control_v10: 0`
  - `enable_yearly_risk_budget_v10: 0`

Quick OOS smoke on new lock:
- `scripts/run_v8_walk_forward.py --params-file config/v10_production_locked.yaml --max-folds 3`
- output: `data/reports/v10_router_only_wf3_2026-04-03.csv`
- summary:
  - folds ok: `3`
  - mean LP net CAGR: `10.30%`
  - mean LP net Sharpe: `0.53`
  - median LP max DD: `-9.95%`
  - pass count: `1/3`

Takeaway:
- v10 plumbing is in place with strict walk-forward/no-lookahead support.
- short-sample risk-adjusted behavior improved, but full OOS gate consistency is not yet achieved;
  next step is targeted calibration on weak regimes (not architecture gaps).

### Live Toggle Policy (Pre-committed ON/OFF rules)

Date: 2026-04-03

Purpose:
- avoid discretionary toggle changes in live trading.
- only keep layers ON when they win in OOS and live shadow stats after costs.

Current champion lock:
- `enable_regime_router_v10: 1`
- `enable_whipsaw_control_v10: 0`
- `enable_yearly_risk_budget_v10: 0`
- `option_always_on_min_notional: 0.0`

Promotion rule (OFF -> ON):
- challenger must beat champion in walk-forward + regime-split aggregate with no-lookahead,
  and improve either annual hurdle hit-rate or downside metrics without degrading CAGR/NAV beyond tolerance.
- require two consecutive evaluation cycles before promotion.

Demotion rule (ON -> OFF):
- if live shadow underperforms expected distribution for 2-3 monthly reviews and worsens
  drawdown/sharpe diagnostics versus champion bounds, demote to OFF at next scheduled release window.

Execution discipline:
- config changes only on scheduled cadence (monthly/quarterly) unless hard risk controls fire.
- every toggle change must include artifact links and before/after metrics in DEVLOG + reports.
