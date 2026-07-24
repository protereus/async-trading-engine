"""SQLite-backed persistent candle storage.

Stores confirmed OHLCV candles so the bot can backfill from its own database
on restart, eliminating the ~43-hour warmup penalty from accumulating candles
via live WebSocket streaming.

``CandleDB`` owns the connection and the candle table itself; the
``signal_history`` / ``asset_correlations`` / ``macro_features`` concerns are
extracted into ``SignalHistoryStore`` / ``CorrelationStore`` / ``MacroStore``
(``bot.data.signal_history_store`` / ``correlation_store`` / ``macro_store``),
constructed against the same connection in ``init_db()``. ``CandleDB``'s own
methods for those concerns are thin delegators, so this stays the single
public entry point every caller already uses.
"""

from __future__ import annotations

import logging
import sqlite3

from bot.core.models import Candle
from bot.data.correlation_store import CorrelationStore
from bot.data.macro_store import MacroStore
from bot.data.signal_history_store import SignalHistoryRecord, SignalHistoryRow, SignalHistoryStore

logger = logging.getLogger(__name__)

_CREATE_CANDLES_TABLE = """
CREATE TABLE IF NOT EXISTS candles (
    symbol    TEXT    NOT NULL,
    timestamp INTEGER NOT NULL,
    open      REAL    NOT NULL,
    high      REAL    NOT NULL,
    low       REAL    NOT NULL,
    close     REAL    NOT NULL,
    volume    REAL    NOT NULL,
    PRIMARY KEY (symbol, timestamp)
);
"""

_CREATE_CANDLES_INDEX = """
CREATE INDEX IF NOT EXISTS idx_candles_symbol_ts
ON candles (symbol, timestamp DESC);
"""


class CandleDB:
    """Persistent SQLite store for confirmed OHLCV candles.

    The connection is kept open for the bot's lifetime; one writer
    (the asyncio event loop) means no locking contention.
    """

    def __init__(self, db_path: str = "candles.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._signal_history: SignalHistoryStore | None = None
        self._correlations: CorrelationStore | None = None
        self._macro: MacroStore | None = None

    def init_db(self) -> None:
        """Open the connection, enable WAL mode, create schema."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_CANDLES_TABLE)
        self._conn.execute(_CREATE_CANDLES_INDEX)

        self._signal_history = SignalHistoryStore(self._conn)
        self._signal_history.init_schema()
        self._correlations = CorrelationStore(self._conn)
        self._correlations.init_schema()
        self._macro = MacroStore(self._conn)
        self._macro.init_schema()

        self._conn.commit()
        logger.info("CandleDB initialised at %s", self._db_path)

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def insert_candle(self, candle: Candle) -> None:
        """Insert a single confirmed candle; silently ignores duplicates."""
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        self._conn.execute(
            "INSERT OR IGNORE INTO candles "
            "(symbol, timestamp, open, high, low, close, volume) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                candle.symbol,
                candle.timestamp,
                candle.open,
                candle.high,
                candle.low,
                candle.close,
                candle.volume,
            ),
        )
        self._conn.commit()

    def insert_candles(self, candles: list[Candle]) -> None:
        """Bulk-insert confirmed candles; silently ignores duplicates."""
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        rows = [(c.symbol, c.timestamp, c.open, c.high, c.low, c.close, c.volume) for c in candles]
        with self._conn:  # transaction
            self._conn.executemany(
                "INSERT OR IGNORE INTO candles "
                "(symbol, timestamp, open, high, low, close, volume) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                rows,
            )
        logger.debug("Bulk-inserted %d candles (duplicates ignored)", len(rows))

    def delete_candles_for_symbol(self, symbol: str) -> int:
        """Delete every row for *symbol*; return the count removed.

        Used by the IG-native candle feed at startup to wipe pre-cutover
        TD-scale rows before backfilling fresh IG-level rows.  Necessary
        because ``insert_candle`` uses ``INSERT OR IGNORE`` — a same-
        timestamp insert from a new source would silently keep the old
        row.  Mixing scales in the store would silently corrupt Kronos.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        with self._conn:
            cur = self._conn.execute("DELETE FROM candles WHERE symbol = ?", (symbol,))
        deleted = cur.rowcount if cur.rowcount is not None else 0
        if deleted > 0:
            logger.info("CandleDB: deleted %d rows for symbol %s", deleted, symbol)
        return deleted

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def get_candles(
        self,
        symbol: str,
        limit: int | None = None,
        since: int | None = None,
    ) -> list[Candle]:
        """Return candles in ascending timestamp order.

        Args:
            symbol: Trading pair, e.g. "EUR/USD".
            limit:  If set, return the N most recent candles.
            since:  If set, return only candles with timestamp >= since.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")

        if limit is not None:
            # Fetch the N most recent rows then flip to ascending order
            if since is not None:
                rows = self._conn.execute(
                    "SELECT symbol, timestamp, open, high, low, close, volume "
                    "FROM candles WHERE symbol = ? AND timestamp >= ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (symbol, since, limit),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT symbol, timestamp, open, high, low, close, volume "
                    "FROM candles WHERE symbol = ? "
                    "ORDER BY timestamp DESC LIMIT ?",
                    (symbol, limit),
                ).fetchall()
            rows = list(reversed(rows))
        else:
            if since is not None:
                rows = self._conn.execute(
                    "SELECT symbol, timestamp, open, high, low, close, volume "
                    "FROM candles WHERE symbol = ? AND timestamp >= ? "
                    "ORDER BY timestamp ASC",
                    (symbol, since),
                ).fetchall()
            else:
                rows = self._conn.execute(
                    "SELECT symbol, timestamp, open, high, low, close, volume "
                    "FROM candles WHERE symbol = ? "
                    "ORDER BY timestamp ASC",
                    (symbol,),
                ).fetchall()

        return [
            Candle(
                symbol=r[0],
                timestamp=r[1],
                open=r[2],
                high=r[3],
                low=r[4],
                close=r[5],
                volume=r[6],
                is_confirmed=True,
            )
            for r in rows
        ]

    def get_latest_timestamp(self, symbol: str) -> int | None:
        """Return the maximum stored timestamp for *symbol*, or None."""
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        row = self._conn.execute(
            "SELECT MAX(timestamp) FROM candles WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_earliest_timestamp(self, symbol: str) -> int | None:
        """Return the minimum stored timestamp for *symbol*, or None."""
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        row = self._conn.execute(
            "SELECT MIN(timestamp) FROM candles WHERE symbol = ?", (symbol,)
        ).fetchone()
        return row[0] if row and row[0] is not None else None

    def get_candle_count(self, symbol: str) -> int:
        """Return the number of stored candles for *symbol*."""
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        row = self._conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ?", (symbol,)
        ).fetchone()
        return int(row[0]) if row else 0

    # ------------------------------------------------------------------
    # Signal history — delegates to SignalHistoryStore
    # ------------------------------------------------------------------

    def write_signal_history(
        self,
        scored_at: int,
        symbol: str,
        horizon_bars: int,
        mean_return: float | None,
        direction_confidence: float | None,
        uncertainty: float | None,
        entry_price: float,
        predicted_mfe_pct: float | None = None,
        predicted_mae_pct: float | None = None,
        predicted_volatility: float | None = None,
        monotonicity: float | None = None,
    ) -> None:
        if self._signal_history is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        self._signal_history.write_signal_history(
            scored_at,
            symbol,
            horizon_bars,
            mean_return,
            direction_confidence,
            uncertainty,
            entry_price,
            predicted_mfe_pct=predicted_mfe_pct,
            predicted_mae_pct=predicted_mae_pct,
            predicted_volatility=predicted_volatility,
            monotonicity=monotonicity,
        )

    def write_signal_history_batch(self, rows: list[SignalHistoryRow]) -> None:
        if self._signal_history is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        self._signal_history.write_signal_history_batch(rows)

    def resolve_signal_history(self, now_ms: int) -> int:
        if self._signal_history is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._signal_history.resolve_signal_history(now_ms)

    def compute_gap_spanned_at_horizon(self, symbol: str, scored_at: int, horizon_bars: int) -> int:
        if self._signal_history is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._signal_history.compute_gap_spanned_at_horizon(symbol, scored_at, horizon_bars)

    def get_signal_history(
        self,
        symbol: str | None = None,
        since_ms: int | None = None,
        resolved_only: bool = False,
        exclude_gap_spanned: bool = False,
        limit: int | None = None,
    ) -> list[SignalHistoryRecord]:
        if self._signal_history is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._signal_history.get_signal_history(
            symbol=symbol,
            since_ms=since_ms,
            resolved_only=resolved_only,
            exclude_gap_spanned=exclude_gap_spanned,
            limit=limit,
        )

    # ------------------------------------------------------------------
    # Asset correlations — delegates to CorrelationStore
    # ------------------------------------------------------------------

    def write_correlations(self, computed_at: int, matrix: dict[str, dict[str, float]]) -> None:
        if self._correlations is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        self._correlations.write_correlations(computed_at, matrix)

    def read_latest_correlations(self) -> dict[str, dict[str, float]]:
        if self._correlations is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._correlations.read_latest_correlations()

    # ------------------------------------------------------------------
    # Macro features — delegates to MacroStore
    # ------------------------------------------------------------------

    def insert_macro_observations(
        self, series_id: str, observations: list[tuple[int, float]]
    ) -> int:
        if self._macro is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._macro.insert_macro_observations(series_id, observations)

    def get_macro_series(self, series_id: str, limit: int | None = None) -> list[tuple[int, float]]:
        if self._macro is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._macro.get_macro_series(series_id, limit=limit)

    def get_latest_macro_value(self, series_id: str) -> tuple[int, float] | None:
        if self._macro is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        return self._macro.get_latest_macro_value(series_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            self._signal_history = None
            self._correlations = None
            self._macro = None
            logger.info("CandleDB closed")
