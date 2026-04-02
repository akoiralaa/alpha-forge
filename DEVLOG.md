# One Brain Fund Devlog

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
