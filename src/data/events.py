from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class EventRecord:
    canonical_id: int
    ticker: str
    published_at_ns: int
    event_type: str
    source: str
    headline: str
    body: str = ""
    sentiment_score: float = 0.0
    relevance: float = 1.0
    novelty: float = 1.0
    confidence: float = 1.0
    metadata: str | None = None


class EventStore:
    SCHEMA_SQL = """
    CREATE TABLE IF NOT EXISTS event_pit (
        id                INTEGER PRIMARY KEY AUTOINCREMENT,
        canonical_id      INTEGER NOT NULL,
        ticker            TEXT NOT NULL,
        published_at_ns   INTEGER NOT NULL,
        event_type        TEXT NOT NULL,
        source            TEXT NOT NULL,
        headline          TEXT NOT NULL,
        body              TEXT NOT NULL DEFAULT '',
        sentiment_score   REAL NOT NULL DEFAULT 0.0,
        relevance         REAL NOT NULL DEFAULT 1.0,
        novelty           REAL NOT NULL DEFAULT 1.0,
        confidence        REAL NOT NULL DEFAULT 1.0,
        metadata          TEXT
    );

    CREATE INDEX IF NOT EXISTS idx_event_lookup
        ON event_pit(canonical_id, published_at_ns);

    CREATE INDEX IF NOT EXISTS idx_event_ticker
        ON event_pit(ticker, published_at_ns);

    CREATE INDEX IF NOT EXISTS idx_event_type
        ON event_pit(event_type, published_at_ns);
    """

    def __init__(self, db_path: str | Path):
        self._db_path = Path(db_path).expanduser().resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(self.SCHEMA_SQL)
        self._conn.commit()
        logger.info("EventStore initialized at %s", self._db_path)

    def close(self) -> None:
        self._conn.close()

    def add_record(self, record: EventRecord) -> None:
        self._conn.execute(
            """INSERT INTO event_pit
               (canonical_id, ticker, published_at_ns, event_type, source,
                headline, body, sentiment_score, relevance, novelty, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                record.canonical_id,
                record.ticker,
                record.published_at_ns,
                record.event_type,
                record.source,
                record.headline,
                record.body,
                record.sentiment_score,
                record.relevance,
                record.novelty,
                record.confidence,
                record.metadata,
            ),
        )
        self._conn.commit()

    def add_records_batch(self, records: list[EventRecord]) -> None:
        self._conn.executemany(
            """INSERT INTO event_pit
               (canonical_id, ticker, published_at_ns, event_type, source,
                headline, body, sentiment_score, relevance, novelty, confidence, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                (
                    r.canonical_id,
                    r.ticker,
                    r.published_at_ns,
                    r.event_type,
                    r.source,
                    r.headline,
                    r.body,
                    r.sentiment_score,
                    r.relevance,
                    r.novelty,
                    r.confidence,
                    r.metadata,
                )
                for r in records
            ],
        )
        self._conn.commit()

    def get_recent(
        self,
        canonical_id: int,
        as_of_ns: int,
        lookback_ns: int,
    ) -> list[EventRecord]:
        rows = self._conn.execute(
            """SELECT * FROM event_pit
               WHERE canonical_id = ?
                 AND published_at_ns <= ?
                 AND published_at_ns >= ?
               ORDER BY published_at_ns DESC""",
            (canonical_id, as_of_ns, max(0, as_of_ns - lookback_ns)),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_recent_by_ticker(
        self,
        ticker: str,
        as_of_ns: int,
        lookback_ns: int,
    ) -> list[EventRecord]:
        rows = self._conn.execute(
            """SELECT * FROM event_pit
               WHERE ticker = ?
                 AND published_at_ns <= ?
                 AND published_at_ns >= ?
               ORDER BY published_at_ns DESC""",
            (ticker, as_of_ns, max(0, as_of_ns - lookback_ns)),
        ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def to_dataframe(
        self,
        canonical_id: int | None = None,
        ticker: str | None = None,
    ) -> pd.DataFrame:
        query = "SELECT * FROM event_pit"
        params: list[object] = []
        clauses: list[str] = []
        if canonical_id is not None:
            clauses.append("canonical_id = ?")
            params.append(canonical_id)
        if ticker is not None:
            clauses.append("ticker = ?")
            params.append(ticker)
        if clauses:
            query += " WHERE " + " AND ".join(clauses)
        query += " ORDER BY published_at_ns ASC"
        return pd.read_sql(query, self._conn, params=params)

    def count_records(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM event_pit").fetchone()
        return int(row[0]) if row else 0

    def _row_to_record(self, row: sqlite3.Row) -> EventRecord:
        return EventRecord(
            canonical_id=row["canonical_id"],
            ticker=row["ticker"],
            published_at_ns=row["published_at_ns"],
            event_type=row["event_type"],
            source=row["source"],
            headline=row["headline"],
            body=row["body"],
            sentiment_score=row["sentiment_score"],
            relevance=row["relevance"],
            novelty=row["novelty"],
            confidence=row["confidence"],
            metadata=row["metadata"],
        )
