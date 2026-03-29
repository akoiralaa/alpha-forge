
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class FundamentalRecord:
    canonical_id: int
    metric_name: str        # EPS, FLOAT, SHORT_INTEREST, ANALYST_EST, etc.
    value: float
    published_at_ns: int    # when this value became publicly known
    period_end_ns: int      # reporting period this value covers
    source: str             # data provider identifier

class FundamentalsStore:

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS fundamentals_pit (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_id      INTEGER NOT NULL,
        metric_name       TEXT NOT NULL,
        value             REAL NOT NULL,
        published_at_ns   INTEGER NOT NULL,
        period_end_ns     INTEGER NOT NULL,
        source            TEXT NOT NULL
    );

    CREATE INDEX IF NOT EXISTS idx_fund_lookup
        ON fundamentals_pit(canonical_id, metric_name, published_at_ns);

    CREATE INDEX IF NOT EXISTS idx_fund_published
        ON fundamentals_pit(published_at_ns);
    """

    # Standard metric names for consistency
    METRIC_EPS = "EPS"
    METRIC_REVENUE = "REVENUE"
    METRIC_FLOAT = "FLOAT"
    METRIC_SHORT_INTEREST = "SHORT_INTEREST"
    METRIC_ANALYST_EST_EPS = "ANALYST_EST_EPS"
    METRIC_ANALYST_REVISION = "ANALYST_REVISION"
    METRIC_MARKET_CAP = "MARKET_CAP"
    METRIC_ADV_20D = "ADV_20D"              # 20-day average daily volume (dollars)
    METRIC_ADV_20D_SHARES = "ADV_20D_SHARES"  # 20-day average daily volume (shares)
    METRIC_OPEN_INTEREST = "OPEN_INTEREST"

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA_SQL)
        self._conn.commit()
        logger.info("FundamentalsStore initialized at %s", self._db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Writes ───────────────────────────────────────────────────────────

    def add_record(self, record: FundamentalRecord) -> None:
        self._conn.execute(
            """INSERT INTO fundamentals_pit
               (canonical_id, metric_name, value, published_at_ns, period_end_ns, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                record.canonical_id,
                record.metric_name,
                record.value,
                record.published_at_ns,
                record.period_end_ns,
                record.source,
            ),
        )
        self._conn.commit()

    def add_records_batch(self, records: list[FundamentalRecord]) -> None:
        self._conn.executemany(
            """INSERT INTO fundamentals_pit
               (canonical_id, metric_name, value, published_at_ns, period_end_ns, source)
               VALUES (?, ?, ?, ?, ?, ?)""",
            [
                (r.canonical_id, r.metric_name, r.value, r.published_at_ns, r.period_end_ns, r.source)
                for r in records
            ],
        )
        self._conn.commit()

    # ── Point-in-time reads ──────────────────────────────────────────────

    def get_as_of(
        self,
        canonical_id: int,
        metric_name: str,
        as_of_ns: int,
    ) -> FundamentalRecord | None:
        row = self._conn.execute(
            """SELECT * FROM fundamentals_pit
               WHERE canonical_id = ?
                 AND metric_name = ?
                 AND published_at_ns <= ?
               ORDER BY published_at_ns DESC
               LIMIT 1""",
            (canonical_id, metric_name, as_of_ns),
        ).fetchone()
        if row is None:
            return None
        return FundamentalRecord(
            canonical_id=row["canonical_id"],
            metric_name=row["metric_name"],
            value=row["value"],
            published_at_ns=row["published_at_ns"],
            period_end_ns=row["period_end_ns"],
            source=row["source"],
        )

    def get_all_as_of(
        self,
        canonical_id: int,
        as_of_ns: int,
    ) -> dict[str, FundamentalRecord]:
        rows = self._conn.execute(
            """SELECT f1.* FROM fundamentals_pit f1
               INNER JOIN (
                   SELECT canonical_id, metric_name, MAX(published_at_ns) as max_pub
                   FROM fundamentals_pit
                   WHERE canonical_id = ? AND published_at_ns <= ?
                   GROUP BY canonical_id, metric_name
               ) f2 ON f1.canonical_id = f2.canonical_id
                    AND f1.metric_name = f2.metric_name
                    AND f1.published_at_ns = f2.max_pub""",
            (canonical_id, as_of_ns),
        ).fetchall()

        return {
            row["metric_name"]: FundamentalRecord(
                canonical_id=row["canonical_id"],
                metric_name=row["metric_name"],
                value=row["value"],
                published_at_ns=row["published_at_ns"],
                period_end_ns=row["period_end_ns"],
                source=row["source"],
            )
            for row in rows
        }

    def get_history(
        self,
        canonical_id: int,
        metric_name: str,
        start_ns: int | None = None,
        end_ns: int | None = None,
    ) -> list[FundamentalRecord]:
        query = """SELECT * FROM fundamentals_pit
                   WHERE canonical_id = ? AND metric_name = ?"""
        params: list = [canonical_id, metric_name]
        if start_ns is not None:
            query += " AND published_at_ns >= ?"
            params.append(start_ns)
        if end_ns is not None:
            query += " AND published_at_ns <= ?"
            params.append(end_ns)
        query += " ORDER BY published_at_ns ASC"

        rows = self._conn.execute(query, params).fetchall()
        return [
            FundamentalRecord(
                canonical_id=r["canonical_id"],
                metric_name=r["metric_name"],
                value=r["value"],
                published_at_ns=r["published_at_ns"],
                period_end_ns=r["period_end_ns"],
                source=r["source"],
            )
            for r in rows
        ]

    def get_earnings_surprise(
        self,
        canonical_id: int,
        as_of_ns: int,
    ) -> float | None:
        actual = self.get_as_of(canonical_id, self.METRIC_EPS, as_of_ns)
        estimate = self.get_as_of(canonical_id, self.METRIC_ANALYST_EST_EPS, as_of_ns)
        if actual is None or estimate is None:
            return None
        if estimate.value == 0:
            return None
        # Simplified surprise — in production this would use historical
        # surprise distribution for Z-scoring
        return (actual.value - estimate.value) / abs(estimate.value)

    def count_records(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM fundamentals_pit").fetchone()
        return row[0]

    def to_dataframe(self, canonical_id: int | None = None) -> pd.DataFrame:
        query = "SELECT * FROM fundamentals_pit"
        params = []
        if canonical_id is not None:
            query += " WHERE canonical_id = ?"
            params.append(canonical_id)
        query += " ORDER BY canonical_id, metric_name, published_at_ns"
        return pd.read_sql(query, self._conn, params=params)
