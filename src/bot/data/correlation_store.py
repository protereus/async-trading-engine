"""SQLite-backed ``asset_correlations`` persistence.

Extracted from ``CandleDB`` as a dedicated store, sharing its connection.
Persists the rolling pairwise correlation matrix used by TopK's
correlation-bump filter and restored on startup.
"""

from __future__ import annotations

import logging
import sqlite3

logger = logging.getLogger(__name__)

_CREATE_ASSET_CORRELATIONS_TABLE = """
CREATE TABLE IF NOT EXISTS asset_correlations (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    computed_at INTEGER NOT NULL,
    symbol_a    TEXT    NOT NULL,
    symbol_b    TEXT    NOT NULL,
    correlation REAL    NOT NULL
);
"""

_CREATE_ASSET_CORRELATIONS_INDEX = """
CREATE INDEX IF NOT EXISTS idx_asset_correlations_at
ON asset_correlations (computed_at DESC);
"""


class CorrelationStore:
    """Persistent SQLite store for the asset-correlation matrix.

    Shares its connection with the owning ``CandleDB`` — one writer (the
    asyncio event loop) means no locking contention.
    """

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def init_schema(self) -> None:
        self._conn.execute(_CREATE_ASSET_CORRELATIONS_TABLE)
        self._conn.execute(_CREATE_ASSET_CORRELATIONS_INDEX)

    def write_correlations(
        self,
        computed_at: int,
        matrix: dict[str, dict[str, float]],
    ) -> None:
        """Persist a correlation matrix snapshot.

        Each (symbol_a, symbol_b) pair with symbol_a < symbol_b is stored once.
        The full matrix is recoverable by symmetry.
        """
        rows = [
            (computed_at, a, b, corr)
            for a, row in matrix.items()
            for b, corr in row.items()
            if a < b  # store each pair once
        ]
        self._conn.executemany(
            "INSERT INTO asset_correlations (computed_at, symbol_a, symbol_b, correlation) "
            "VALUES (?, ?, ?, ?)",
            rows,
        )
        self._conn.commit()
        logger.debug("asset_correlations: wrote %d pairs at %d", len(rows), computed_at)

    def read_latest_correlations(self) -> dict[str, dict[str, float]]:
        """Read the most-recently written correlation matrix.

        Returns an empty dict if no rows exist yet.
        The full matrix is reconstructed from the stored (a < b) pairs by symmetry.
        """
        row = self._conn.execute("SELECT MAX(computed_at) FROM asset_correlations").fetchone()
        if row is None or row[0] is None:
            return {}
        latest_at: int = row[0]
        pairs = self._conn.execute(
            "SELECT symbol_a, symbol_b, correlation FROM asset_correlations WHERE computed_at = ?",
            (latest_at,),
        ).fetchall()
        matrix: dict[str, dict[str, float]] = {}
        for a, b, corr in pairs:
            matrix.setdefault(a, {})[b] = corr
            matrix.setdefault(b, {})[a] = corr  # symmetric
        return matrix
