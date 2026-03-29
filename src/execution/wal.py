
from __future__ import annotations

import json
import os
import sqlite3
import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional

class OrderState(Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"
    ERROR = "ERROR"

@dataclass
class WALEntry:
    sequence_id: int
    timestamp_ns: int
    order_id: str
    state: OrderState
    symbol_id: int
    side: int              # +1 buy, -1 sell
    order_type: str        # MARKET, LIMIT
    size: int
    limit_price: Optional[float] = None
    filled_size: int = 0
    avg_fill_price: float = 0.0
    broker_order_id: Optional[str] = None
    error_msg: str = ""
    metadata: str = ""     # JSON blob for extra context

class WriteAheadLog:

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=FULL")
        self._create_tables()
        self._seq = self._max_sequence() + 1

    def _create_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS wal_entries (
                sequence_id INTEGER PRIMARY KEY,
                timestamp_ns INTEGER NOT NULL,
                order_id TEXT NOT NULL,
                state TEXT NOT NULL,
                symbol_id INTEGER NOT NULL,
                side INTEGER NOT NULL,
                order_type TEXT NOT NULL,
                size INTEGER NOT NULL,
                limit_price REAL,
                filled_size INTEGER DEFAULT 0,
                avg_fill_price REAL DEFAULT 0.0,
                broker_order_id TEXT,
                error_msg TEXT DEFAULT '',
                metadata TEXT DEFAULT ''
            )
        """)
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_wal_order_id ON wal_entries(order_id)
        """)
        self.conn.commit()

    def _max_sequence(self) -> int:
        row = self.conn.execute("SELECT MAX(sequence_id) FROM wal_entries").fetchone()
        return row[0] if row[0] is not None else 0

    def append(self, entry: WALEntry) -> int:
        entry.sequence_id = self._seq
        entry.timestamp_ns = entry.timestamp_ns or time.time_ns()
        self.conn.execute("""
            INSERT INTO wal_entries
            (sequence_id, timestamp_ns, order_id, state, symbol_id, side,
             order_type, size, limit_price, filled_size, avg_fill_price,
             broker_order_id, error_msg, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.sequence_id, entry.timestamp_ns, entry.order_id,
            entry.state.value, entry.symbol_id, entry.side,
            entry.order_type, entry.size, entry.limit_price,
            entry.filled_size, entry.avg_fill_price,
            entry.broker_order_id, entry.error_msg, entry.metadata,
        ))
        self.conn.commit()
        self._seq += 1
        return entry.sequence_id

    def get_order_history(self, order_id: str) -> list[WALEntry]:
        rows = self.conn.execute(
            "SELECT * FROM wal_entries WHERE order_id = ? ORDER BY sequence_id",
            (order_id,),
        ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def get_latest_state(self, order_id: str) -> Optional[WALEntry]:
        row = self.conn.execute(
            "SELECT * FROM wal_entries WHERE order_id = ? ORDER BY sequence_id DESC LIMIT 1",
            (order_id,),
        ).fetchone()
        return self._row_to_entry(row) if row else None

    def get_open_orders(self) -> list[WALEntry]:
        terminal = ("FILLED", "CANCELLED", "REJECTED", "ERROR")
        placeholders = ",".join("?" for _ in terminal)
        rows = self.conn.execute(f"""
            SELECT w.* FROM wal_entries w
            INNER JOIN (
                SELECT order_id, MAX(sequence_id) as max_seq
                FROM wal_entries GROUP BY order_id
            ) latest ON w.order_id = latest.order_id AND w.sequence_id = latest.max_seq
            WHERE w.state NOT IN ({placeholders})
        """, terminal).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def replay(self) -> dict[str, WALEntry]:
        rows = self.conn.execute(
            "SELECT * FROM wal_entries ORDER BY sequence_id"
        ).fetchall()
        state: dict[str, WALEntry] = {}
        for row in rows:
            entry = self._row_to_entry(row)
            state[entry.order_id] = entry
        return state

    def entry_count(self) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM wal_entries").fetchone()
        return row[0]

    def _row_to_entry(self, row) -> WALEntry:
        return WALEntry(
            sequence_id=row[0],
            timestamp_ns=row[1],
            order_id=row[2],
            state=OrderState(row[3]),
            symbol_id=row[4],
            side=row[5],
            order_type=row[6],
            size=row[7],
            limit_price=row[8],
            filled_size=row[9],
            avg_fill_price=row[10],
            broker_order_id=row[11],
            error_msg=row[12],
            metadata=row[13],
        )

    def close(self):
        self.conn.close()
