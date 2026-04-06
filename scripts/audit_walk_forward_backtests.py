#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import pandas as pd


FILES = [
    "backtest.py",
    "backtest_v2.py",
    "backtest_v3.py",
    "backtest_v4.py",
    "backtest_v5.py",
    "backtest_v6.py",
    "backtest_v7.py",
    "backtest_v8.py",
]


def _has(text: str, pattern: str) -> bool:
    return pattern in text


def main() -> int:
    rows: list[dict[str, object]] = []
    for rel in FILES:
        p = Path(rel)
        text = p.read_text(encoding="utf-8")
        rows.append(
            {
                "file": rel,
                "walk_forward_banner": _has(text, "WALK-FORWARD BACKTEST"),
                "lagged_exec_shift1": _has(text, "shift(1)"),
                "cli_start_end": _has(text, "--start-date") and _has(text, "--end-date"),
                "cli_eval_window": _has(text, "--eval-start") and _has(text, "--eval-end"),
                "cli_no_lookahead_flag": _has(text, "--enforce-no-lookahead"),
                "cli_metrics_json": _has(text, "--metrics-json"),
            }
        )
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
