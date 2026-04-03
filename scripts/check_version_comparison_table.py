#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parent.parent
CSV_PATH = REPO_ROOT / "data" / "reports" / "version_comparison_v1_v10.csv"
README_PATH = REPO_ROOT / "README.md"
CANONICAL_HEADING = "### Canonical Version Comparison (v1-v10)"

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

NUMERIC_COLUMNS = [
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
]

DISPLAY_MAP = {
    "Version": "version",
    "CAGR": "gross_cagr_pct",
    "Sharpe": "gross_sharpe",
    "Max DD": "gross_max_dd_pct",
    "Final NAV ($M)": "final_nav_usd_m",
    "Turnover (x/yr)": "turnover_x_per_year",
    "Tx Costs (bps/yr)": "tx_cost_bps_per_year",
    "Source": "source_tag",
}


def is_missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def format_for_readme(column: str, value: object) -> str:
    if is_missing(value):
        return "n/a"

    if column in {"version", "source_tag"}:
        return str(value)

    v = float(value)
    if column in {"gross_cagr_pct", "gross_max_dd_pct", "lp_net_cagr_pct", "lp_net_max_dd_pct"}:
        return f"{v:+.2f}%"
    if column in {"gross_sharpe", "lp_net_sharpe"}:
        return f"{v:.2f}"
    if column == "final_nav_usd_m":
        return f"${v:.2f}M"
    if column == "turnover_x_per_year":
        return f"{int(round(v))}x"
    if column == "tx_cost_bps_per_year":
        return str(int(round(v)))
    if column in {"years_ge_15pct_net", "years_ge_1_sharpe", "years_both_hurdles"}:
        return str(int(round(v)))
    return str(value)


def parse_markdown_table(lines: list[str]) -> tuple[list[str], list[list[str]]]:
    rows = [ln.strip() for ln in lines if ln.strip().startswith("|")]
    if len(rows) < 3:
        raise ValueError("Canonical README table not found or incomplete.")
    header = [c.strip() for c in rows[0].strip("|").split("|")]
    data_rows = []
    for row in rows[2:]:
        cells = [c.strip() for c in row.strip("|").split("|")]
        if len(cells) == len(header):
            data_rows.append(cells)
    return header, data_rows


def main() -> int:
    if not CSV_PATH.exists():
        print(f"ERROR: missing CSV: {CSV_PATH}")
        return 1
    if not README_PATH.exists():
        print(f"ERROR: missing README: {README_PATH}")
        return 1

    df = pd.read_csv(CSV_PATH, keep_default_na=False)
    missing_cols = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing_cols:
        print(f"ERROR: CSV missing required columns: {missing_cols}")
        return 1

    expected_versions = [f"v{i}" for i in range(1, 11)]
    csv_versions = df["version"].tolist()
    if csv_versions != expected_versions:
        print(f"ERROR: versions mismatch. expected {expected_versions}, got {csv_versions}")
        return 1

    for col in NUMERIC_COLUMNS:
        for i, raw in enumerate(df[col].tolist(), start=1):
            if is_missing(raw):
                continue
            try:
                float(raw)
            except ValueError:
                print(f"ERROR: non-numeric value in {col} row {i}: {raw!r}")
                return 1

    text_lines = README_PATH.read_text().splitlines()
    try:
        start_idx = next(i for i, ln in enumerate(text_lines) if ln.strip() == CANONICAL_HEADING)
    except StopIteration:
        print("ERROR: canonical section heading not found in README.")
        return 1

    section_lines = text_lines[start_idx + 1 : start_idx + 40]
    header, rows = parse_markdown_table(section_lines)
    if header != list(DISPLAY_MAP.keys()):
        print("ERROR: README canonical table header drift detected.")
        print(f"Expected: {list(DISPLAY_MAP.keys())}")
        print(f"Got:      {header}")
        return 1

    readme_by_version = {r[0]: r for r in rows}
    if set(readme_by_version.keys()) != set(expected_versions):
        print(f"ERROR: README versions mismatch. got {sorted(readme_by_version.keys())}")
        return 1

    for _, csv_row in df.iterrows():
        version = csv_row["version"]
        readme_row = readme_by_version[version]
        for idx, display_col in enumerate(header):
            schema_col = DISPLAY_MAP[display_col]
            expected = format_for_readme(schema_col, csv_row[schema_col])
            got = readme_row[idx]
            if got != expected:
                print(
                    f"ERROR: README drift for {version} / {display_col}: "
                    f"expected {expected!r}, got {got!r}"
                )
                return 1

    print("PASS: canonical version table is consistent (CSV schema + README render).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
