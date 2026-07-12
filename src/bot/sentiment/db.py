"""SentimentDB — persists ConsensusSignals to a sentiment_scores table in candles.db."""

from __future__ import annotations

import contextlib
import json
import logging
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from bot.sentiment.models import ConsensusSignal

logger = logging.getLogger(__name__)

_DEFAULT_DB_PATH = Path("candles.db")

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS sentiment_scores (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    asset       TEXT    NOT NULL,
    scored_at   INTEGER NOT NULL,   -- Unix ms UTC
    sentiment   REAL    NOT NULL,   -- -1.0 to 1.0
    confidence  REAL    NOT NULL,   -- 0.0 to 1.0
    agreement   REAL    NOT NULL,   -- 0.0 to 1.0
    sources     TEXT    NOT NULL,   -- JSON array of agent names
    escalated   INTEGER NOT NULL,   -- 0 or 1
    reasoning   TEXT    NOT NULL
);
"""

_CREATE_INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_sentiment_asset_time
    ON sentiment_scores (asset, scored_at DESC);
"""


class SentimentDB:
    """Opens its own SQLite connection to candles.db for sentiment persistence."""

    def __init__(self, db_path: Path = _DEFAULT_DB_PATH) -> None:
        self._path = db_path
        self._conn: sqlite3.Connection | None = None

    def init_db(self) -> None:
        """Create table and index if they do not exist."""
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_TABLE_SQL)
        self._conn.execute(_CREATE_INDEX_SQL)
        self._conn.commit()
        logger.debug("SentimentDB: initialised at %s", self._path)

    def insert_signal(self, signal: ConsensusSignal) -> None:
        """Persist a ConsensusSignal row."""
        if self._conn is None:
            return
        ts_ms = int(signal.scored_at.timestamp() * 1000)
        try:
            self._conn.execute(
                """
                INSERT INTO sentiment_scores
                    (asset, scored_at, sentiment, confidence, agreement,
                     sources, escalated, reasoning)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.asset,
                    ts_ms,
                    signal.sentiment,
                    signal.confidence,
                    signal.agreement,
                    json.dumps(signal.sources),
                    1 if signal.escalated else 0,
                    signal.reasoning,
                ),
            )
            self._conn.commit()
        except sqlite3.Error:
            logger.exception("SentimentDB: failed to insert signal for %s", signal.asset)

    def get_latest(self, asset: str) -> dict[str, Any] | None:
        """Return the most recent row for an asset, or None."""
        if self._conn is None:
            return None
        try:
            cur = self._conn.execute(
                "SELECT * FROM sentiment_scores WHERE asset=? ORDER BY scored_at DESC LIMIT 1",
                (asset,),
            )
            row = cur.fetchone()
            if row is None:
                return None
            return self._row_to_dict(cur.description, row)
        except sqlite3.Error:
            logger.exception("SentimentDB: get_latest failed for %s", asset)
            return None

    def get_history(self, asset: str, limit: int = 50) -> list[dict[str, Any]]:
        """Return the most recent ``limit`` rows for an asset, newest first."""
        if self._conn is None:
            return []
        try:
            cur = self._conn.execute(
                "SELECT * FROM sentiment_scores WHERE asset=? ORDER BY scored_at DESC LIMIT ?",
                (asset, limit),
            )
            rows = cur.fetchall()
            return [self._row_to_dict(cur.description, r) for r in rows]
        except sqlite3.Error:
            logger.exception("SentimentDB: get_history failed for %s", asset)
            return []

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _row_to_dict(description: Any, row: tuple[Any, ...]) -> dict[str, Any]:
        d = {desc[0]: val for desc, val in zip(description, row, strict=False)}
        # Deserialize sources JSON array
        with contextlib.suppress(json.JSONDecodeError, KeyError):
            d["sources"] = json.loads(d["sources"])
        # Convert scored_at ms back to datetime
        with contextlib.suppress(KeyError, ValueError, OSError):
            d["scored_at"] = datetime.fromtimestamp(d["scored_at"] / 1000, tz=UTC)
        return d
