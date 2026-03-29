#!/usr/bin/env python3

from __future__ import annotations

import argparse
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.arctic_store import TickStore
from src.data.fundamentals import FundamentalsStore
from src.data.ingest.base import AssetClass, Tick, date_to_ns
from src.data.quality_pipeline import DataQualityPipeline
from src.data.symbol_master import SymbolMaster
from src.data.universe import UniverseManager

@dataclass
class AssertionResult:
    name: str
    observed: str
    target: str
    result: str  # "PASS" or "FAIL"
    notes: str = ""

@dataclass
class ValidationReport:
    phase: int
    phase_name: str
    git_sha: str
    assertions: list[AssertionResult] = field(default_factory=list)

    @property
    def gate_result(self) -> str:
        if any(a.result == "FAIL" for a in self.assertions):
            return "ANY_FAIL"
        return "ALL_PASS"

    def format(self) -> str:
        timestamp = datetime.now(timezone.utc).isoformat()
        lines = []
        lines.append("=" * 72)
        lines.append(f"  PHASE [{self.phase}] VALIDATION REPORT")
        lines.append(f"  Phase name:     {self.phase_name}")
        lines.append(f"  Executed at:    {timestamp}")
        lines.append(f"  Git commit:     {self.git_sha}")
        lines.append(f"  Gate result:    {self.gate_result}")
        lines.append("=" * 72)
        lines.append("")
        lines.append("ASSERTION RESULTS")
        lines.append("-" * 72)

        for a in self.assertions:
            lines.append(f"[{a.name}]")
            lines.append(f"  Observed : {a.observed}")
            lines.append(f"  Target   : {a.target}")
            lines.append(f"  Result   : {a.result}")
            if a.notes:
                lines.append(f"  Notes    : {a.notes}")
            lines.append("")

        lines.append("-" * 72)
        lines.append("")
        lines.append("SUMMARY")
        total = len(self.assertions)
        passed = sum(1 for a in self.assertions if a.result == "PASS")
        failed = total - passed
        lines.append(f"  Total assertions : {total}")
        lines.append(f"  Passed           : {passed}")
        lines.append(f"  Failed           : {failed}")
        lines.append(f"  Gate result      : {self.gate_result}")
        lines.append("")

        if self.gate_result == "ALL_PASS":
            lines.append(f"  Cleared to proceed to Phase [{self.phase + 1}].")
            lines.append(
                f"  Report archived to: validation_reports/phase_{self.phase}_{timestamp}.log"
            )
        else:
            lines.append("  --- HALT ---")
            lines.append(f"  Phase [{self.phase + 1}] is LOCKED. Do not proceed.")
            lines.append("")
            for a in self.assertions:
                if a.result == "FAIL":
                    lines.append(f"  FAILED: [{a.name}]")
                    lines.append(f"    Observed: {a.observed}")
                    lines.append(f"    Target:   {a.target}")
                    if a.notes:
                        lines.append(f"    Notes:    {a.notes}")
                    lines.append("")

        return "\n".join(lines)

def get_git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).resolve().parent.parent
        )
        return result.stdout.strip() or "NO_GIT_REPO"
    except Exception:
        return "NO_GIT_REPO"

def assert_survivorship_bias(sm: SymbolMaster) -> AssertionResult:
    existed_at = date_to_ns(2010, 1, 1)
    delisted_before = date_to_ns(2023, 1, 1)
    delisted = sm.get_delisted_instruments(existed_at, delisted_before)
    count = len(delisted)

    return AssertionResult(
        name="survivorship_bias_check",
        observed=f"{count} delisted instruments found in 2010 universe",
        target="count >= 50",
        result="PASS" if count >= 50 else "FAIL",
        notes="Instruments that existed in 2010 but were later delisted must be present",
    )

def assert_dual_timestamp_coverage(ts: TickStore) -> AssertionResult:
    sample = ts.sample_random_records(n=10_000)
    if sample.empty:
        return AssertionResult(
            name="dual_timestamp_coverage",
            observed="0 records in store",
            target="10000 records with both timestamps != 0",
            result="FAIL",
            notes="No data in tick store",
        )

    total = len(sample)
    valid = sample[
        (sample["exchange_time_ns"] != 0) & (sample["capture_time_ns"] != 0)
    ]
    valid_count = len(valid)
    pct = valid_count / total * 100

    return AssertionResult(
        name="dual_timestamp_coverage",
        observed=f"{valid_count}/{total} records valid ({pct:.1f}%)",
        target="100.0% have both exchange_time_ns != 0 and capture_time_ns != 0",
        result="PASS" if valid_count == total else "FAIL",
    )

def assert_capture_time_ordering(ts: TickStore) -> AssertionResult:
    symbols = ts.list_symbols()
    if not symbols:
        return AssertionResult(
            name="capture_time_ordering",
            observed="No symbols in store",
            target="Strictly monotonically increasing capture_time_ns",
            result="FAIL",
        )

    violations = []
    tested = 0
    for sid in symbols[:50]:  # Test up to 50 symbols
        df = ts.read_ticks_raw(sid)
        if len(df) < 2:
            continue
        tested += 1
        sample = df.head(1000)
        ct = sample["capture_time_ns"].values
        non_mono = (ct[1:] <= ct[:-1]).sum()
        if non_mono > 0:
            violations.append((sid, int(non_mono)))

    if violations:
        detail = "; ".join(f"sym={s} violations={v}" for s, v in violations[:5])
        return AssertionResult(
            name="capture_time_ordering",
            observed=f"{len(violations)} symbols with non-monotonic timestamps",
            target="0 non-monotonic pairs",
            result="FAIL",
            notes=detail,
        )

    return AssertionResult(
        name="capture_time_ordering",
        observed=f"{tested} symbols tested, all strictly monotonic",
        target="Strictly monotonically increasing capture_time_ns for all symbols",
        result="PASS",
    )

def assert_pit_fundamental_integrity(fund: FundamentalsStore) -> AssertionResult:
    as_of = date_to_ns(2019, 6, 1)

    # Find any symbol with EPS data
    df = fund.to_dataframe()
    eps_records = df[df["metric_name"] == "EPS"]
    if eps_records.empty:
        return AssertionResult(
            name="pit_fundamental_integrity",
            observed="No EPS records in store",
            target="Returned value has published_at_ns <= 2019-06-01",
            result="FAIL",
            notes="No fundamental data loaded",
        )

    # Test with the first symbol that has EPS data
    test_id = int(eps_records["canonical_id"].iloc[0])
    record = fund.get_as_of(test_id, "EPS", as_of)

    if record is None:
        return AssertionResult(
            name="pit_fundamental_integrity",
            observed=f"No EPS found for symbol {test_id} as-of 2019-06-01",
            target="Returned value has published_at_ns <= 2019-06-01",
            result="FAIL",
        )

    passes = record.published_at_ns <= as_of
    return AssertionResult(
        name="pit_fundamental_integrity",
        observed=f"published_at_ns={record.published_at_ns} (as_of={as_of})",
        target="published_at_ns <= as_of_ns",
        result="PASS" if passes else "FAIL",
        notes=f"Symbol {test_id}, value={record.value}",
    )

def assert_split_adjustment_correctness(
    sm: SymbolMaster, ts: TickStore
) -> AssertionResult:
    from src.data.adjustments import PriceAdjuster

    # Find any symbol with a split
    df = sm.to_dataframe()
    splits = df[df["action_type"] == "SPLIT"]
    if splits.empty:
        return AssertionResult(
            name="split_adjustment_correctness",
            observed="No splits in symbol master",
            target="Adjusted price continuous across split (within 0.01%)",
            result="FAIL",
            notes="No split data available for testing",
        )

    test_id = int(splits["canonical_id"].iloc[0])
    adjuster = PriceAdjuster(sm)

    raw_df = ts.read_ticks_raw(test_id)
    if raw_df.empty:
        return AssertionResult(
            name="split_adjustment_correctness",
            observed=f"No tick data for symbol {test_id}",
            target="Adjusted price continuous across split (within 0.01%)",
            result="FAIL",
        )

    adjusted_df = adjuster.adjust_dataframe(test_id, raw_df)
    passes = adjuster.verify_adjustment(test_id, raw_df, adjusted_df, tolerance_pct=0.0001)

    return AssertionResult(
        name="split_adjustment_correctness",
        observed="Adjusted prices continuous across split" if passes else "Discontinuity found",
        target="adjusted_price before split == adjusted_price after split (within 0.01%)",
        result="PASS" if passes else "FAIL",
        notes=f"Tested symbol {test_id}",
    )

def assert_quality_pipeline_rejection() -> AssertionResult:
    pipeline = DataQualityPipeline()

    # First, feed some good ticks to build rolling state
    base_time = date_to_ns(2024, 1, 2)
    for i in range(200):
        good_tick = Tick(
            exchange_time_ns=base_time + i * 1_000_000,
            capture_time_ns=base_time + i * 1_000_000 + 100_000,
            symbol_id=1,
            bid=100.0 + (i % 10) * 0.01,
            ask=100.05 + (i % 10) * 0.01,
            bid_size=100,
            ask_size=100,
            last_price=100.02 + (i % 10) * 0.01,
            last_size=50,
        )
        pipeline.check_tick(good_tick)

    # Now inject 100 bad ticks with known violations
    bad_ticks = []
    t = base_time + 300_000_000

    for i in range(20):  # zero price
        bad_ticks.append(Tick(
            exchange_time_ns=t + i * 1_000_000,
            capture_time_ns=t + i * 1_000_000 + 100_000,
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=0.0, last_size=50,
        ))

    for i in range(20):  # crossed book (bid >= ask)
        bad_ticks.append(Tick(
            exchange_time_ns=t + (20 + i) * 1_000_000,
            capture_time_ns=t + (20 + i) * 1_000_000 + 100_000,
            symbol_id=1, bid=101.0, ask=100.0, bid_size=100, ask_size=100,
            last_price=100.5, last_size=50,
        ))

    for i in range(20):  # time travel (capture < exchange)
        bad_ticks.append(Tick(
            exchange_time_ns=t + (40 + i) * 1_000_000 + 200_000,
            capture_time_ns=t + (40 + i) * 1_000_000,
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=50,
        ))

    for i in range(20):  # negative bid
        bad_ticks.append(Tick(
            exchange_time_ns=t + (60 + i) * 1_000_000,
            capture_time_ns=t + (60 + i) * 1_000_000 + 100_000,
            symbol_id=1, bid=-1.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=50,
        ))

    for i in range(20):  # stale data (> 10s lag)
        bad_ticks.append(Tick(
            exchange_time_ns=t + (80 + i) * 1_000_000,
            capture_time_ns=t + (80 + i) * 1_000_000 + 15_000_000_000,
            symbol_id=1, bid=100.0, ask=100.05, bid_size=100, ask_size=100,
            last_price=100.02, last_size=50,
        ))

    rejected_count = 0
    for tick in bad_ticks:
        result = pipeline.check_tick(tick)
        if not result.accepted:
            rejected_count += 1

    total_bad = len(bad_ticks)
    pct = rejected_count / total_bad * 100

    return AssertionResult(
        name="quality_pipeline_rejection",
        observed=f"{rejected_count}/{total_bad} bad records rejected ({pct:.1f}%)",
        target="100% of injected bad records rejected",
        result="PASS" if rejected_count == total_bad else "FAIL",
        notes="Tested: zero price, crossed book, time travel, negative bid, stale data",
    )

def assert_universe_point_in_time(
    sm: SymbolMaster, fund: FundamentalsStore, ts: TickStore
) -> AssertionResult:
    um = UniverseManager(sm, fund, ts)

    date_2015 = date_to_ns(2015, 1, 1)
    date_today = date_to_ns(2026, 3, 28)

    universe_2015 = um.compute_universe(date_2015)
    universe_today = um.compute_universe(date_today)

    ids_2015 = set(universe_2015.symbol_ids)
    ids_today = set(universe_today.symbol_ids)
    differ = ids_2015 != ids_today

    return AssertionResult(
        name="universe_point_in_time",
        observed=f"2015 universe: {len(ids_2015)} instruments, today: {len(ids_today)} instruments, differ={differ}",
        target="Two universes must differ (universe changes over time)",
        result="PASS" if differ else "FAIL",
        notes="Identical universes indicate survivorship bias in construction"
        if not differ else "",
    )

def run_validation(data_dir: str) -> ValidationReport:
    data_path = Path(data_dir).expanduser().resolve()

    ts = TickStore(data_path / "arcticdb")
    sm = SymbolMaster(data_path / "symbol_master.db")
    fund = FundamentalsStore(data_path / "fundamentals.db")

    report = ValidationReport(
        phase=1,
        phase_name="Data Layer",
        git_sha=get_git_sha(),
    )

    # Run all assertions
    report.assertions.append(assert_survivorship_bias(sm))
    report.assertions.append(assert_dual_timestamp_coverage(ts))
    report.assertions.append(assert_capture_time_ordering(ts))
    report.assertions.append(assert_pit_fundamental_integrity(fund))
    report.assertions.append(assert_split_adjustment_correctness(sm, ts))
    report.assertions.append(assert_quality_pipeline_rejection())
    report.assertions.append(assert_universe_point_in_time(sm, fund, ts))

    sm.close()
    fund.close()

    return report

def main():
    parser = argparse.ArgumentParser(description="Phase 1 Validation Gate")
    parser.add_argument(
        "--data-dir",
        default="~/one_brain_fund/data",
        help="Path to data directory",
    )
    args = parser.parse_args()

    report = run_validation(args.data_dir)
    report_text = report.format()
    print(report_text)

    # Archive the report
    reports_dir = Path(__file__).resolve().parent.parent / "validation_reports"
    reports_dir.mkdir(exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    status = "PASS" if report.gate_result == "ALL_PASS" else "FAIL"
    report_path = reports_dir / f"phase_1_{status}_{timestamp}.log"
    report_path.write_text(report_text)
    print(f"\nReport archived to: {report_path}")

    sys.exit(0 if report.gate_result == "ALL_PASS" else 1)

if __name__ == "__main__":
    main()
