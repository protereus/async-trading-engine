"""SQLite-backed ``macro_features`` persistence (free FRED overlay).

Extracted from ``CandleDB`` as a dedicated store, sharing its connection.
One row per (series_id, observation_date); derived features (1d/5d delta,
z-score) are computed on read in ``bot.macro.fred`` so we never store stale
transforms.
"""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime

_CREATE_MACRO_FEATURES_TABLE = """
CREATE TABLE IF NOT EXISTS macro_features (
    series_id        TEXT    NOT NULL,
    observation_date INTEGER NOT NULL,
    value            REAL    NOT NULL,
    fetched_at       INTEGER NOT NULL,
    PRIMARY KEY (series_id, observation_date)
);
"""

_CREATE_MACRO_FEATURES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_macro_features_series_time
ON macro_features (series_id, observation_date DESC);
"""


class MacroStore:
    """Persistent SQLite store for FRED macro observations.

    Shares its connection with the owning ``CandleDB`` — one writer (the
    asyncio event loop) means no locking contention.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def init_schema(self) -> None:
        self._conn.execute(_CREATE_MACRO_FEATURES_TABLE)
        self._conn.execute(_CREATE_MACRO_FEATURES_INDEX)

    def insert_macro_observations(
        self, series_id: str, observations: list[tuple[int, float]]
    ) -> int:
        """Upsert FRED observations for ``series_id``.

        ``observations`` is ``[(observation_date_ms, value), ...]``.  Existing
        rows are replaced (FRED occasionally revises recent values).  Returns
        the row count written.
        """
        now_ms = int(datetime.now(UTC).timestamp() * 1000)
        rows = [(series_id, obs_date, value, now_ms) for obs_date, value in observations]
        with self._conn:
            self._conn.executemany(
                "INSERT OR REPLACE INTO macro_features "
                "(series_id, observation_date, value, fetched_at) "
                "VALUES (?, ?, ?, ?)",
                rows,
            )
        return len(rows)

    def get_macro_series(self, series_id: str, limit: int | None = None) -> list[tuple[int, float]]:
        """Return observations for ``series_id`` in ascending date order."""
        if limit is not None:
            rows = self._conn.execute(
                "SELECT observation_date, value FROM macro_features "
                "WHERE series_id = ? ORDER BY observation_date DESC LIMIT ?",
                (series_id, limit),
            ).fetchall()
            return list(reversed(rows))
        rows = self._conn.execute(
            "SELECT observation_date, value FROM macro_features "
            "WHERE series_id = ? ORDER BY observation_date ASC",
            (series_id,),
        ).fetchall()
        return list(rows)

    def get_latest_macro_value(self, series_id: str) -> tuple[int, float] | None:
        """Return the most-recent (observation_date_ms, value) for the series, or None."""
        row = self._conn.execute(
            "SELECT observation_date, value FROM macro_features "
            "WHERE series_id = ? ORDER BY observation_date DESC LIMIT 1",
            (series_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None
