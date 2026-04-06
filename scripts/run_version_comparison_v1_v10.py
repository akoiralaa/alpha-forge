#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from datetime import date
from pathlib import Path

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = Path(os.path.expanduser("~/.one_brain_fund/cache/bars"))
CSV_OUT = REPO_ROOT / "data" / "reports" / "version_comparison_v1_v10.csv"
LOG_DIR = REPO_ROOT / "data" / "reports" / "run_logs"

REQUIRED_COLUMNS = [
    "version",
    "gross_cagr_pct",
    "gross_sharpe",
    "gross_max_dd_pct",
    "final_nav_usd_m",
    "lp_net_cagr_pct",
    "lp_net_sharpe",
    "lp_net_max_dd_pct",
    "turnover_x_per_year",
    "tx_cost_bps_per_year",
    "years_ge_15pct_net",
    "years_ge_1_sharpe",
    "years_both_hurdles",
    "source_tag",
    "notes",
]


def _as_float(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text.replace(",", "").strip())


def _as_pct(text: str | None) -> float | None:
    if text is None or text == "":
        return None
    return float(text.replace("%", "").replace(",", "").strip()) / 100.0


def _m(pattern: str, text: str) -> str | None:
    m = re.search(pattern, text, flags=re.MULTILINE)
    return m.group(1) if m else None


def _window_from_text(text: str) -> tuple[str | None, str | None, float | None]:
    m = re.search(
        r"Sample window:\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*->\s*([0-9]{4}-[0-9]{2}-[0-9]{2})\s*\(([0-9.]+)\s*years\)",
        text,
    )
    if m:
        return m.group(1), m.group(2), float(m.group(3))
    m = re.search(
        r"Period:\s*([0-9\-: ]+)\s*to\s*([0-9\-: ]+)\s*\([0-9,]+\s*days,\s*([0-9.]+)\s*years\)",
        text,
    )
    if m:
        return m.group(1).strip(), m.group(2).strip(), float(m.group(3))
    return None, None, None


def _parse_legacy_stdout(text: str) -> dict[str, float | int | None]:
    cagr = _as_pct(_m(r"^\s*(?:Gross\s+)?CAGR:\s*([+\-]?[0-9.,]+%)", text))
    sharpe = _as_float(_m(r"^\s*(?:Gross\s+)?Sharpe:\s*([+\-]?[0-9.]+)", text))
    max_dd = _as_pct(
        _m(r"^\s*(?:Gross\s+)?Max DD:\s*([+\-]?[0-9.,]+%)", text)
        or _m(r"^\s*Max drawdown:\s*([+\-]?[0-9.,]+%)", text)
    )
    final_nav = _as_float(_m(r"^\s*Final NAV:\s*\$([0-9,]+(?:\.[0-9]+)?)", text))
    lp_cagr = _as_pct(_m(r"^\s*LP Net CAGR:\s*([+\-]?[0-9.,]+%)", text))
    lp_sharpe = _as_float(_m(r"^\s*LP Net Sharpe:\s*([+\-]?[0-9.]+)", text))
    lp_max_dd = _as_pct(_m(r"^\s*LP Net Max DD:\s*([+\-]?[0-9.,]+%)", text))
    turnover = _as_float(_m(r"^\s*Turnover:\s*([0-9.]+)x/yr", text))
    tx = _as_float(_m(r"^\s*Tx costs:\s*([0-9.]+)\s*bps/yr", text))

    years_15 = _m(r"Years >= 15% net:\s*([0-9]+)\/[0-9]+", text)
    years_1 = _m(r"Years >= 1\.0 Sharpe:\s*([0-9]+)\/[0-9]+", text)
    years_both = _m(r"Years hitting both hurdles:\s*([0-9]+)\/[0-9]+", text)

    start, end, years = _window_from_text(text)

    return {
        "gross_cagr": cagr,
        "gross_sharpe": sharpe,
        "gross_max_dd": max_dd,
        "final_nav": final_nav,
        "lp_net_cagr": lp_cagr,
        "lp_net_sharpe": lp_sharpe,
        "lp_net_max_dd": lp_max_dd,
        "turnover_x_per_year": turnover,
        "tx_cost_bps_per_year": tx,
        "years_ge_15pct_net": int(years_15) if years_15 is not None else None,
        "years_ge_1_sharpe": int(years_1) if years_1 is not None else None,
        "years_both_hurdles": int(years_both) if years_both is not None else None,
        "sample_start": start,
        "sample_end": end,
        "sample_years": years,
    }


def _build_cache_complete_config(config_path: Path) -> tuple[Path, list[str], int]:
    cfg = yaml.safe_load(config_path.read_text())
    instruments = cfg.get("instruments", {})
    key_to_class = {
        "sector_etfs": "ETF",
        "equities": "EQUITY",
        "equity_index_futures": "FUTURE",
        "commodity_futures": "COMMODITY",
        "fixed_income_futures": "BOND",
        "fx_pairs": "FX",
        "vix_futures": "VOLATILITY",
    }
    dropped: list[str] = []
    kept = 0
    for key, cls in key_to_class.items():
        syms = list(instruments.get(key, []) or [])
        keep = []
        for sym in syms:
            cache_file = CACHE_DIR / f"{sym}_{cls}.parquet"
            if cache_file.exists():
                keep.append(sym)
                kept += 1
            else:
                dropped.append(f"{sym}_{cls}")
        instruments[key] = keep
    cfg["instruments"] = instruments
    fd, tmp_path = tempfile.mkstemp(prefix="sp500_cache_complete_", suffix=".yaml")
    os.close(fd)
    out = Path(tmp_path)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return out, dropped, kept


def _run(cmd: list[str], log_path: Path) -> tuple[int, str]:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    text = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    log_path.write_text(text)
    return proc.returncode, text


def _row_from_metrics(version: str, m: dict, notes: str, source_tag: str = "current_run") -> dict:
    return {
        "version": version,
        "gross_cagr_pct": (m.get("gross_cagr") * 100.0) if m.get("gross_cagr") is not None else None,
        "gross_sharpe": m.get("gross_sharpe"),
        "gross_max_dd_pct": (m.get("gross_max_dd") * 100.0) if m.get("gross_max_dd") is not None else None,
        "final_nav_usd_m": (m.get("final_nav") / 1_000_000.0) if m.get("final_nav") is not None else None,
        "lp_net_cagr_pct": (m.get("lp_net_cagr") * 100.0) if m.get("lp_net_cagr") is not None else None,
        "lp_net_sharpe": m.get("lp_net_sharpe"),
        "lp_net_max_dd_pct": (m.get("lp_net_max_dd") * 100.0) if m.get("lp_net_max_dd") is not None else None,
        "turnover_x_per_year": m.get("turnover_x_per_year"),
        "tx_cost_bps_per_year": m.get("tx_cost_bps_per_year"),
        "years_ge_15pct_net": m.get("years_ge_15pct_net"),
        "years_ge_1_sharpe": m.get("years_ge_1_sharpe"),
        "years_both_hurdles": m.get("years_both_hurdles"),
        "source_tag": source_tag,
        "notes": notes,
        "sample_start": m.get("sample_start"),
        "sample_end": m.get("sample_end"),
        "sample_years": m.get("sample_years"),
    }


def main() -> int:
    p = argparse.ArgumentParser(description="Run canonical v1-v10 comparison and refresh CSV.")
    p.add_argument("--python", default=str(REPO_ROOT / ".venv" / "bin" / "python"))
    p.add_argument("--config", default=str(REPO_ROOT / "config" / "sp500_universe.yaml"))
    args = p.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (REPO_ROOT / "data" / "reports").mkdir(parents=True, exist_ok=True)

    cfg_path, dropped, kept = _build_cache_complete_config(Path(args.config))
    run_date = date.today().isoformat()

    print(f"Using cache-complete config: {cfg_path} (kept={kept}, dropped={len(dropped)})")
    if dropped:
        print("Dropped:", ", ".join(dropped))

    rows: list[dict] = []

    def run_legacy(version: str, cmd: list[str]) -> None:
        log_path = LOG_DIR / f"{run_date}_{version}.log"
        print(f"[run] {version}: {' '.join(cmd)}")
        code, text = _run(cmd, log_path)
        if code != 0:
            raise RuntimeError(f"{version} failed (exit {code}); see {log_path}")
        m = _parse_legacy_stdout(text)
        notes = f"current_run {run_date}; cache-complete={kept}; log={log_path.name}"
        rows.append(_row_from_metrics(version, m, notes))

    def run_v7plus(version: str, cmd: list[str], metrics_path: Path) -> None:
        log_path = LOG_DIR / f"{run_date}_{version}.log"
        print(f"[run] {version}: {' '.join(cmd)}")
        code, text = _run(cmd, log_path)
        if code != 0:
            raise RuntimeError(f"{version} failed (exit {code}); see {log_path}")
        m = json.loads(metrics_path.read_text())
        notes = f"current_run {run_date}; cache-complete={kept}; log={log_path.name}"
        rows.append(_row_from_metrics(version, m, notes))

    py = args.python
    base_cfg = str(cfg_path)

    run_legacy("v1", [py, "backtest.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15"])
    run_legacy(
        "v2",
        [py, "backtest_v2.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15", "--no-fundamentals"],
    )
    run_legacy("v3", [py, "backtest_v3.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15"])
    run_legacy("v4", [py, "backtest_v4.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15"])
    run_legacy("v5", [py, "backtest_v5.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15"])
    run_legacy("v6", [py, "backtest_v6.py", "--config", base_cfg, "--nav", "10000000", "--target-vol", "0.15"])

    v7_metrics = Path(tempfile.mkstemp(prefix="v7_metrics_", suffix=".json")[1])
    run_v7plus(
        "v7",
        [
            py,
            "backtest_v7.py",
            "--config",
            base_cfg,
            "--nav",
            "10000000",
            "--target-vol",
            "0.15",
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--metrics-json",
            str(v7_metrics),
        ],
        v7_metrics,
    )

    v8_metrics = Path(tempfile.mkstemp(prefix="v8_metrics_", suffix=".json")[1])
    run_v7plus(
        "v8",
        [
            py,
            "backtest_v8.py",
            "--config",
            base_cfg,
            "--nav",
            "10000000",
            "--target-vol",
            "0.15",
            "--force-event-weight",
            "0.03",
            "--overlay-min-signal",
            "0.08",
            "--etf-gross",
            "0.20",
            "--futures-gross",
            "0.20",
            "--fx-gross",
            "0.08",
            "--option-max-notional",
            "0.04",
            "--option-short-strike-daily",
            "0.07",
            "--option-short-credit-bps-daily",
            "0.9",
            "--option-activation-score",
            "0.38",
            "--option-severe-score",
            "0.78",
            "--total-gross-cap",
            "1.80",
            "--gov-min-mult",
            "0.30",
            "--exec-liq-cost-mult",
            "0.80",
            "--enable-kelly-sentiment-overlay",
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--metrics-json",
            str(v8_metrics),
        ],
        v8_metrics,
    )

    v9_metrics = Path(tempfile.mkstemp(prefix="v9_metrics_", suffix=".json")[1])
    run_v7plus(
        "v9",
        [
            py,
            "backtest_v8.py",
            "--config",
            base_cfg,
            "--nav",
            "10000000",
            "--target-vol",
            "0.15",
            "--force-event-weight",
            "0.03",
            "--overlay-min-signal",
            "0.08",
            "--etf-gross",
            "0.20",
            "--futures-gross",
            "0.20",
            "--fx-gross",
            "0.08",
            "--option-max-notional",
            "0.04",
            "--option-short-strike-daily",
            "0.07",
            "--option-short-credit-bps-daily",
            "0.9",
            "--option-activation-score",
            "0.38",
            "--option-severe-score",
            "0.78",
            "--total-gross-cap",
            "1.80",
            "--gov-min-mult",
            "0.30",
            "--exec-liq-cost-mult",
            "0.80",
            "--enable-kelly-sentiment-overlay",
            "--enable-macro-overlay",
            "--macro-max-de-risk",
            "0.12",
            "--macro-min-scale",
            "0.85",
            "--macro-smooth-days",
            "5",
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--metrics-json",
            str(v9_metrics),
        ],
        v9_metrics,
    )

    v10_metrics = Path(tempfile.mkstemp(prefix="v10_metrics_", suffix=".json")[1])
    run_v7plus(
        "v10",
        [
            py,
            "backtest_v8.py",
            "--config",
            base_cfg,
            "--nav",
            "10000000",
            "--target-vol",
            "0.15",
            "--force-event-weight",
            "0.03",
            "--overlay-min-signal",
            "0.08",
            "--etf-gross",
            "0.20",
            "--futures-gross",
            "0.20",
            "--fx-gross",
            "0.08",
            "--option-max-notional",
            "0.04",
            "--option-always-on-min-notional",
            "0.0",
            "--option-short-strike-daily",
            "0.07",
            "--option-short-credit-bps-daily",
            "0.9",
            "--option-activation-score",
            "0.38",
            "--option-severe-score",
            "0.78",
            "--total-gross-cap",
            "1.80",
            "--gov-min-mult",
            "0.30",
            "--exec-liq-cost-mult",
            "0.80",
            "--enable-kelly-sentiment-overlay",
            "--enable-regime-router-v10",
            "--router-vol-window",
            "21",
            "--router-trend-window",
            "200",
            "--router-crash-threshold",
            "0.62",
            "--router-risk-off-threshold",
            "0.50",
            "--router-smooth-days",
            "3",
            "--cache-complete-only",
            "--enforce-no-lookahead",
            "--metrics-json",
            str(v10_metrics),
        ],
        v10_metrics,
    )

    by_ver = {r["version"]: r for r in rows}
    ordered = [by_ver[f"v{i}"] for i in range(1, 11)]
    df = pd.DataFrame(ordered)
    for c in REQUIRED_COLUMNS:
        if c not in df.columns:
            df[c] = None
    front = REQUIRED_COLUMNS + ["sample_start", "sample_end", "sample_years"]
    df = df[front]
    df.to_csv(CSV_OUT, index=False)
    print(f"\nWrote canonical CSV: {CSV_OUT}")
    print(df[["version", "gross_cagr_pct", "gross_sharpe", "gross_max_dd_pct", "final_nav_usd_m"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
