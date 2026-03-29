
from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.data.ingest.base import AssetClass

logger = logging.getLogger(__name__)

@dataclass(slots=True)
class Instrument:
    canonical_id: int
    exchange: str
    ticker: str
    valid_from_ns: int
    valid_to_ns: int | None       # None means currently active
    asset_class: AssetClass
    sector: str | None
    currency: str
    action_type: str | None       # RENAME, SPLIT, MERGER, DELIST, IPO
    split_ratio: float | None
    dividend_amount: float | None
    underlying_id: int | None     # for derivatives, FK to canonical_id

class SymbolMaster:

    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS symbol_master (
        row_id            INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_id      INTEGER NOT NULL,
        exchange          TEXT NOT NULL,
        ticker            TEXT NOT NULL,
        valid_from_ns     INTEGER NOT NULL,
        valid_to_ns       INTEGER,
        asset_class       TEXT NOT NULL,
        sector            TEXT,
        currency          TEXT NOT NULL,
        action_type       TEXT,
        split_ratio       REAL,
        dividend_amount   REAL,
        underlying_id     INTEGER
    );

    CREATE INDEX IF NOT EXISTS idx_canonical_id
        ON symbol_master(canonical_id);

    CREATE INDEX IF NOT EXISTS idx_ticker_time
        ON symbol_master(ticker, valid_from_ns, valid_to_ns);

    CREATE INDEX IF NOT EXISTS idx_asset_class
        ON symbol_master(asset_class);

    CREATE INDEX IF NOT EXISTS idx_valid_to
        ON symbol_master(valid_to_ns);

    CREATE INDEX IF NOT EXISTS idx_action_type
        ON symbol_master(action_type);
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._conn.executescript(self.SCHEMA_SQL)
        self._conn.commit()
        self._next_id: int | None = None
        logger.info("SymbolMaster initialized at %s", self._db_path)

    def close(self) -> None:
        self._conn.close()

    # ── Writes ───────────────────────────────────────────────────────────

    def _get_next_id(self) -> int:
        if self._next_id is None:
            row = self._conn.execute(
                "SELECT COALESCE(MAX(canonical_id), 0) + 1 FROM symbol_master WHERE action_type IS NULL"
            ).fetchone()
            self._next_id = row[0]
        cid = self._next_id
        self._next_id += 1
        return cid

    def add_instrument(
        self,
        exchange: str,
        ticker: str,
        valid_from_ns: int,
        asset_class: AssetClass,
        currency: str,
        sector: str | None = None,
        valid_to_ns: int | None = None,
        action_type: str | None = None,
        split_ratio: float | None = None,
        dividend_amount: float | None = None,
        underlying_id: int | None = None,
        canonical_id: int | None = None,
    ) -> int:
        if canonical_id is None:
            canonical_id = self._get_next_id()

        self._conn.execute(
            """INSERT INTO symbol_master
               (canonical_id, exchange, ticker, valid_from_ns, valid_to_ns,
                asset_class, sector, currency, action_type, split_ratio,
                dividend_amount, underlying_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                canonical_id,
                exchange,
                ticker,
                valid_from_ns,
                valid_to_ns,
                asset_class.value if isinstance(asset_class, AssetClass) else asset_class,
                sector,
                currency,
                action_type,
                split_ratio,
                dividend_amount,
                underlying_id,
            ),
        )
        self._conn.commit()
        logger.debug("Added instrument: %s/%s id=%d", exchange, ticker, canonical_id)
        return canonical_id

    def record_corporate_action(
        self,
        canonical_id: int,
        action_type: str,
        effective_ns: int,
        split_ratio: float | None = None,
        dividend_amount: float | None = None,
        new_ticker: str | None = None,
    ) -> None:
        instrument = self.get_by_id(canonical_id)
        if instrument is None:
            raise ValueError(f"Unknown canonical_id: {canonical_id}")

        if action_type == "DELIST":
            self._conn.execute(
                "UPDATE symbol_master SET valid_to_ns = ? WHERE canonical_id = ? AND action_type IS NULL AND valid_to_ns IS NULL",
                (effective_ns, canonical_id),
            )
        elif action_type == "SPLIT":
            # Add a split action record (separate row to track the event)
            self.add_instrument(
                exchange=instrument.exchange,
                ticker=instrument.ticker,
                valid_from_ns=effective_ns,
                asset_class=AssetClass(instrument.asset_class.value),
                currency=instrument.currency,
                sector=instrument.sector,
                action_type="SPLIT",
                split_ratio=split_ratio,
                canonical_id=canonical_id,
            )
            return  # Skip the commit below, add_instrument commits
        elif action_type == "DIVIDEND":
            self.add_instrument(
                exchange=instrument.exchange,
                ticker=instrument.ticker,
                valid_from_ns=effective_ns,
                asset_class=AssetClass(instrument.asset_class.value),
                currency=instrument.currency,
                sector=instrument.sector,
                action_type="DIVIDEND",
                dividend_amount=dividend_amount,
                canonical_id=canonical_id,
            )
            return
        elif action_type == "RENAME" and new_ticker:
            # Close old ticker record
            self._conn.execute(
                "UPDATE symbol_master SET valid_to_ns = ? WHERE canonical_id = ? AND valid_to_ns IS NULL AND action_type IS NULL",
                (effective_ns, canonical_id),
            )
            # Create new ticker record with same canonical_id
            self.add_instrument(
                exchange=instrument.exchange,
                ticker=new_ticker,
                valid_from_ns=effective_ns,
                asset_class=AssetClass(instrument.asset_class.value),
                currency=instrument.currency,
                sector=instrument.sector,
                action_type="RENAME",
                canonical_id=canonical_id,
            )
            return
        elif action_type == "MERGER":
            self._conn.execute(
                "UPDATE symbol_master SET valid_to_ns = ? WHERE canonical_id = ? AND action_type IS NULL AND valid_to_ns IS NULL",
                (effective_ns, canonical_id),
            )

        self._conn.commit()

    # ── Reads ────────────────────────────────────────────────────────────

    def _row_to_instrument(self, row: sqlite3.Row) -> Instrument:
        ac = row["asset_class"]
        return Instrument(
            canonical_id=row["canonical_id"],
            exchange=row["exchange"],
            ticker=row["ticker"],
            valid_from_ns=row["valid_from_ns"],
            valid_to_ns=row["valid_to_ns"],
            asset_class=AssetClass(ac) if ac in AssetClass.__members__ else AssetClass(ac),
            sector=row["sector"],
            currency=row["currency"],
            action_type=row["action_type"],
            split_ratio=row["split_ratio"],
            dividend_amount=row["dividend_amount"],
            underlying_id=row["underlying_id"],
        )

    def get_by_id(self, canonical_id: int) -> Instrument | None:
        row = self._conn.execute(
            """SELECT * FROM symbol_master
               WHERE canonical_id = ?
               ORDER BY valid_from_ns DESC LIMIT 1""",
            (canonical_id,),
        ).fetchone()
        return self._row_to_instrument(row) if row else None

    def resolve_ticker(self, ticker: str, as_of_ns: int) -> int | None:
        row = self._conn.execute(
            """SELECT canonical_id FROM symbol_master
               WHERE ticker = ?
                 AND valid_from_ns <= ?
                 AND (valid_to_ns IS NULL OR valid_to_ns > ?)
                 AND (action_type IS NULL OR action_type = 'RENAME')
               ORDER BY valid_from_ns DESC LIMIT 1""",
            (ticker, as_of_ns, as_of_ns),
        ).fetchone()
        return row["canonical_id"] if row else None

    def get_ticker_at(self, canonical_id: int, as_of_ns: int) -> str | None:
        row = self._conn.execute(
            """SELECT ticker FROM symbol_master
               WHERE canonical_id = ?
                 AND valid_from_ns <= ?
                 AND (valid_to_ns IS NULL OR valid_to_ns > ?)
                 AND (action_type IS NULL OR action_type = 'RENAME')
               ORDER BY valid_from_ns DESC LIMIT 1""",
            (canonical_id, as_of_ns, as_of_ns),
        ).fetchone()
        return row["ticker"] if row else None

    def get_splits(self, canonical_id: int) -> list[tuple[int, float]]:
        rows = self._conn.execute(
            """SELECT valid_from_ns, split_ratio FROM symbol_master
               WHERE canonical_id = ? AND action_type = 'SPLIT'
               ORDER BY valid_from_ns ASC""",
            (canonical_id,),
        ).fetchall()
        return [(r["valid_from_ns"], r["split_ratio"]) for r in rows]

    def get_dividends(self, canonical_id: int) -> list[tuple[int, float]]:
        rows = self._conn.execute(
            """SELECT valid_from_ns, dividend_amount FROM symbol_master
               WHERE canonical_id = ? AND action_type = 'DIVIDEND'
               ORDER BY valid_from_ns ASC""",
            (canonical_id,),
        ).fetchall()
        return [(r["valid_from_ns"], r["dividend_amount"]) for r in rows]

    def get_corporate_actions(
        self, canonical_id: int, start_ns: int | None = None, end_ns: int | None = None
    ) -> list[Instrument]:
        query = """SELECT * FROM symbol_master
                   WHERE canonical_id = ? AND action_type IS NOT NULL"""
        params: list = [canonical_id]
        if start_ns is not None:
            query += " AND valid_from_ns >= ?"
            params.append(start_ns)
        if end_ns is not None:
            query += " AND valid_from_ns <= ?"
            params.append(end_ns)
        query += " ORDER BY valid_from_ns ASC"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_instrument(r) for r in rows]

    # ── Universe queries ─────────────────────────────────────────────────

    def get_active_instruments(
        self,
        as_of_ns: int,
        asset_class: AssetClass | None = None,
    ) -> list[Instrument]:
        query = """SELECT * FROM symbol_master
                   WHERE valid_from_ns <= ?
                     AND (valid_to_ns IS NULL OR valid_to_ns > ?)
                     AND (action_type IS NULL OR action_type = 'RENAME')"""
        params: list = [as_of_ns, as_of_ns]
        if asset_class is not None:
            query += " AND asset_class = ?"
            params.append(asset_class.value)
        query += " ORDER BY canonical_id"
        rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_instrument(r) for r in rows]

    def get_delisted_instruments(
        self,
        existed_at_ns: int,
        delisted_before_ns: int,
    ) -> list[Instrument]:
        rows = self._conn.execute(
            """SELECT * FROM symbol_master
               WHERE valid_from_ns <= ?
                 AND valid_to_ns IS NOT NULL
                 AND valid_to_ns < ?
                 AND (action_type IS NULL OR action_type = 'RENAME')
               ORDER BY canonical_id""",
            (existed_at_ns, delisted_before_ns),
        ).fetchall()
        return [self._row_to_instrument(r) for r in rows]

    def count_all(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM symbol_master").fetchone()
        return row[0]

    def count_active(self, as_of_ns: int | None = None) -> int:
        if as_of_ns is None:
            row = self._conn.execute(
                "SELECT COUNT(DISTINCT canonical_id) FROM symbol_master WHERE valid_to_ns IS NULL"
            ).fetchone()
        else:
            row = self._conn.execute(
                """SELECT COUNT(DISTINCT canonical_id) FROM symbol_master
                   WHERE valid_from_ns <= ?
                     AND (valid_to_ns IS NULL OR valid_to_ns > ?)""",
                (as_of_ns, as_of_ns),
            ).fetchone()
        return row[0]

    def to_dataframe(self) -> pd.DataFrame:
        return pd.read_sql("SELECT * FROM symbol_master ORDER BY canonical_id, valid_from_ns",
                           self._conn)
