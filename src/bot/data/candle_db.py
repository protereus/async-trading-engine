"""SQLite-backed persistent candle storage.

Stores confirmed OHLCV candles so the bot can backfill from its own database
on restart, eliminating the ~43-hour warmup penalty from accumulating candles
via live WebSocket streaming.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import UTC, datetime
from typing import NotRequired, TypedDict, cast

from bot.core.models import Candle

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

_CREATE_SIGNAL_HISTORY_TABLE = """
CREATE TABLE IF NOT EXISTS signal_history (
    id                          INTEGER PRIMARY KEY AUTOINCREMENT,
    scored_at                   INTEGER NOT NULL,
    symbol                      TEXT    NOT NULL,
    horizon_bars                INTEGER NOT NULL,
    mean_return                 REAL,
    direction_confidence        REAL,
    uncertainty                 REAL,
    predicted_mfe_pct           REAL,
    predicted_mae_pct           REAL,
    predicted_volatility        REAL,
    monotonicity                REAL,
    entry_price                 REAL    NOT NULL,
    realized_return_at_horizon  REAL,
    realized_max_high_pct       REAL,
    realized_min_low_pct        REAL,
    gap_spanned                 INTEGER DEFAULT 0,
    predicted_close_path        BLOB,
    var_closes_at_horizons      BLOB,
    sentiment_score             REAL,
    sentiment_confidence        REAL,
    sentiment_agent_coverage    INTEGER DEFAULT 0,
    sentiment_slow_decay        REAL,
    sentiment_fast_decay        REAL
);
"""

# /10b — ALTER TABLE migrations for existing live DBs.
# Each tuple is (column_name, ALTER statement). Applied idempotently in init_db.
#
# The sentiment-edge measurement harness adds five columns so the analysis
# can decide whether the sentiment overlay carries actual edge.  Coverage
# defaults to 0 so pre-migration rows are honestly bucketed as "absent" by
# the analysis script.
_SIGNAL_HISTORY_MIGRATIONS: tuple[tuple[str, str], ...] = (
    ("gap_spanned", "ALTER TABLE signal_history ADD COLUMN gap_spanned INTEGER DEFAULT 0"),
    ("predicted_close_path", "ALTER TABLE signal_history ADD COLUMN predicted_close_path BLOB"),
    ("sentiment_score", "ALTER TABLE signal_history ADD COLUMN sentiment_score REAL"),
    (
        "sentiment_confidence",
        "ALTER TABLE signal_history ADD COLUMN sentiment_confidence REAL",
    ),
    (
        "sentiment_agent_coverage",
        "ALTER TABLE signal_history ADD COLUMN sentiment_agent_coverage INTEGER DEFAULT 0",
    ),
    ("sentiment_slow_decay", "ALTER TABLE signal_history ADD COLUMN sentiment_slow_decay REAL"),
    ("sentiment_fast_decay", "ALTER TABLE signal_history ADD COLUMN sentiment_fast_decay REAL"),
    # 2026-06-11 — per-draw Pass-2 closes at every candidate
    # ranking horizon, so per-H direction_confidence is computable offline
    # ahead of the 2026-06-18 threshold recalibration.
    (
        "var_closes_at_horizons",
        "ALTER TABLE signal_history ADD COLUMN var_closes_at_horizons BLOB",
    ),
)

_CREATE_SIGNAL_HISTORY_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signal_history_scored_at
ON signal_history (scored_at);
"""

_CREATE_SIGNAL_HISTORY_SYMBOL_INDEX = """
CREATE INDEX IF NOT EXISTS idx_signal_history_symbol
ON signal_history (symbol, scored_at DESC);
"""

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

# free FRED macro overlay.  One row per (series_id, observation_date);
# observation_date is the UTC-midnight ms of the FRED-published date.  Derived
# features (1d/5d delta, z-score) are computed on read so we never store stale
# transforms.
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


_BAR_INTERVAL_MS = 3_600_000  # Bot operates on 1h bars (Twelve Data, IG quotes)


def _detect_gap_spanned(timestamps: list[int], horizon_end_ms: int) -> int:
    """Return 1 if a market-closure gap was present in the horizon window.

    A gap is detected when:
      - any consecutive-candle interval exceeds 1.5 × the expected bar interval, OR
      - the trailing gap (horizon_end_ms − last candle) exceeds 2 × the expected
        bar interval (the realised window is truncated).

    The expected interval is hard-pegged to 1 h — every candle feed in this
    bot delivers 1 h bars, so a modal-based estimate would mis-fire on rows
    with only one or two candles inside the horizon window.

    With ≤ 1 candle we conservatively flag the row: a single observation
    cannot establish whether the rest of the window was usable.

    H-awareness: the persisted ``signal_history.gap_spanned`` column is
    written once at resolve-time using each row's own ``horizon_bars`` (in
    practice always 120, the live pred_len). Diagnostics that want to ask
    the gap question at a shorter H (the horizon sweep) call this
    function directly with a sliced timestamp list and the H-truncated
    ``horizon_end_ms`` — a 24 h window from a Monday-morning row rarely
    crosses a weekend, even though the underlying 120 h window does.
    """
    if len(timestamps) <= 1:
        return 1
    diffs = [timestamps[i] - timestamps[i - 1] for i in range(1, len(timestamps))]
    if any(d > 1.5 * _BAR_INTERVAL_MS for d in diffs):
        return 1
    if horizon_end_ms - timestamps[-1] > 2 * _BAR_INTERVAL_MS:
        return 1
    return 0


class SignalHistoryRow(TypedDict):
    """Insert shape for one ``signal_history`` row (the write-side contract
    between ``rerank_runner`` and :meth:`CandleDB.write_signal_history_batch`).

    The four required keys are the columns the table cannot NULL; everything
    else is analytics-only and lands as NULL when absent.  The two ``bytes``
    payloads are little-endian float32 BLOBs — see
    :meth:`CandleDB.write_signal_history_batch` for the exact layouts.
    """

    scored_at: int
    symbol: str
    horizon_bars: int
    entry_price: float
    mean_return: NotRequired[float | None]
    direction_confidence: NotRequired[float | None]
    uncertainty: NotRequired[float | None]
    predicted_mfe_pct: NotRequired[float | None]
    predicted_mae_pct: NotRequired[float | None]
    predicted_volatility: NotRequired[float | None]
    monotonicity: NotRequired[float | None]
    predicted_close_path: NotRequired[bytes | None]
    var_closes_at_horizons: NotRequired[bytes | None]
    sentiment_score: NotRequired[float | None]
    sentiment_confidence: NotRequired[float | None]
    sentiment_agent_coverage: NotRequired[int | None]
    sentiment_slow_decay: NotRequired[float | None]
    sentiment_fast_decay: NotRequired[float | None]


class SignalHistoryRecord(TypedDict):
    """Read shape returned by :meth:`CandleDB.get_signal_history` — one dict
    per row, keyed by column name.  ``realized_*`` stay ``None`` until
    :meth:`CandleDB.resolve_signal_history` fills them after the horizon
    elapses."""

    id: int
    scored_at: int
    symbol: str
    horizon_bars: int
    mean_return: float | None
    direction_confidence: float | None
    uncertainty: float | None
    predicted_mfe_pct: float | None
    predicted_mae_pct: float | None
    predicted_volatility: float | None
    monotonicity: float | None
    entry_price: float
    realized_return_at_horizon: float | None
    realized_max_high_pct: float | None
    realized_min_low_pct: float | None
    gap_spanned: int
    predicted_close_path: bytes | None
    var_closes_at_horizons: bytes | None


class CandleDB:
    """Persistent SQLite store for confirmed OHLCV candles.

    The connection is kept open for the bot's lifetime; one writer
    (the asyncio event loop) means no locking contention.
    """

    def __init__(self, db_path: str = "candles.db") -> None:
        self._db_path = db_path
        self._conn: sqlite3.Connection | None = None

    def init_db(self) -> None:
        """Open the connection, enable WAL mode, create schema."""
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute(_CREATE_CANDLES_TABLE)
        self._conn.execute(_CREATE_CANDLES_INDEX)
        self._conn.execute(_CREATE_SIGNAL_HISTORY_TABLE)
        self._conn.execute(_CREATE_SIGNAL_HISTORY_INDEX)
        self._conn.execute(_CREATE_SIGNAL_HISTORY_SYMBOL_INDEX)
        self._conn.execute(_CREATE_ASSET_CORRELATIONS_TABLE)
        self._conn.execute(_CREATE_ASSET_CORRELATIONS_INDEX)
        self._conn.execute(_CREATE_MACRO_FEATURES_TABLE)
        self._conn.execute(_CREATE_MACRO_FEATURES_INDEX)
        self._apply_signal_history_migrations()
        self._conn.commit()
        logger.info("CandleDB initialised at %s", self._db_path)

    def _apply_signal_history_migrations(self) -> None:
        """Idempotently apply ALTER TABLE migrations to signal_history.

        SQLite's ``ALTER TABLE ... ADD COLUMN`` has no ``IF NOT EXISTS`` form
        before 3.35, so we check ``PRAGMA table_info`` first.  Migrations are
        intentionally additive (new columns only); existing rows get the
        column's DEFAULT.
        """
        if self._conn is None:
            return
        existing = {
            row[1] for row in self._conn.execute("PRAGMA table_info(signal_history)").fetchall()
        }
        for column, stmt in _SIGNAL_HISTORY_MIGRATIONS:
            if column in existing:
                continue
            self._conn.execute(stmt)
            logger.info("signal_history: applied migration ADD COLUMN %s", column)

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
    # Signal history
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
        """Insert one signal row per asset per rerank. Realised fields filled later.

        Thin single-row wrapper over :meth:`write_signal_history_batch` so the
        ``signal_history`` INSERT column shape lives in exactly one place (the
        batch writer); the analytics-only columns it doesn't take are written
        NULL.
        """
        self.write_signal_history_batch(
            [
                {
                    "scored_at": scored_at,
                    "symbol": symbol,
                    "horizon_bars": horizon_bars,
                    "mean_return": mean_return,
                    "direction_confidence": direction_confidence,
                    "uncertainty": uncertainty,
                    "predicted_mfe_pct": predicted_mfe_pct,
                    "predicted_mae_pct": predicted_mae_pct,
                    "predicted_volatility": predicted_volatility,
                    "monotonicity": monotonicity,
                    "entry_price": entry_price,
                }
            ]
        )

    def write_signal_history_batch(
        self,
        rows: list[SignalHistoryRow],
    ) -> None:
        """Bulk-insert signal rows.

        The required/optional key split is expressed by
        :class:`SignalHistoryRow`; optional keys may additionally include
        ``predicted_close_path`` as a ``bytes`` payload (— encode as
        little-endian float32 array of the full predicted close path).
        Missing keys → NULL in the column.

        ``var_closes_at_horizons`` is little-endian float32,
        shape ``(sample_count, len(CANDIDATE_HORIZONS))`` row-major: row i is
        Pass-2 draw i, column j the predicted close at
        ``CANDIDATE_HORIZONS[j]`` (NaN where the rollout was shorter than H).
        480 B/row at 20 draws × 6 horizons.  Decode with
        ``np.frombuffer(blob, "<f4").reshape(-1, len(CANDIDATE_HORIZONS))``.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        params = [
            (
                r["scored_at"],
                r["symbol"],
                r["horizon_bars"],
                r.get("mean_return"),
                r.get("direction_confidence"),
                r.get("uncertainty"),
                r.get("predicted_mfe_pct"),
                r.get("predicted_mae_pct"),
                r.get("predicted_volatility"),
                r.get("monotonicity"),
                r["entry_price"],
                r.get("predicted_close_path"),
                r.get("var_closes_at_horizons"),
                r.get("sentiment_score"),
                r.get("sentiment_confidence"),
                # Coverage is an honest integer 0..6 — see the rerank-time
                # capture path in ``bot.strategy.rerank_runner``.  Defaulting
                # to 0 when the row didn't carry sentiment data is correct:
                # the analysis script treats coverage==0 as the ABSENT bucket.
                int(cov) if isinstance(cov := r.get("sentiment_agent_coverage"), int) else 0,
                r.get("sentiment_slow_decay"),
                r.get("sentiment_fast_decay"),
            )
            for r in rows
        ]
        with self._conn:
            self._conn.executemany(
                """
                INSERT INTO signal_history
                    (scored_at, symbol, horizon_bars, mean_return, direction_confidence,
                     uncertainty, predicted_mfe_pct, predicted_mae_pct, predicted_volatility,
                     monotonicity, entry_price, predicted_close_path, var_closes_at_horizons,
                     sentiment_score, sentiment_confidence, sentiment_agent_coverage,
                     sentiment_slow_decay, sentiment_fast_decay)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                params,
            )
        logger.debug("signal_history: bulk-inserted %d rows", len(rows))

    def resolve_signal_history(self, now_ms: int) -> int:
        """Fill realized_* columns for rows whose horizon has elapsed.

        For each unresolved row where scored_at + horizon_bars*3_600_000 < now_ms,
        look up the actual candles from the candles table and compute:
          - realized_return_at_horizon: (close_at_horizon - entry_price) / entry_price
          - realized_max_high_pct: (max_high_over_horizon - entry_price) / entry_price
          - realized_min_low_pct: (entry_price - min_low_over_horizon) / entry_price
          - gap_spanned: 1 when the horizon window contains a market-closure
            gap (weekend, overnight) — defined as any consecutive-candle gap
            exceeding 1.5× the expected 1 h bar interval, or the trailing gap
            between the last in-window candle and ``horizon_end_ms`` exceeding
            2× that interval (see ``_detect_gap_spanned``, which hard-pegs the
            interval to 1 h).  Set so diagnostics can exclude these rows from
            RankIC (the realised return is measured against a truncated
            window and is downward-biased).

        Returns the number of rows resolved.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")

        # Find rows past their horizon with no realized data yet
        pending = self._conn.execute(
            """
            SELECT id, symbol, scored_at, horizon_bars, entry_price
            FROM signal_history
            WHERE realized_return_at_horizon IS NULL
              AND scored_at + horizon_bars * 3600000 < ?
            """,
            (now_ms,),
        ).fetchall()

        if not pending:
            return 0

        resolved = 0
        for row_id, symbol, scored_at, horizon_bars, entry_price in pending:
            horizon_end_ms = scored_at + horizon_bars * 3_600_000
            # Fetch candles covering the prediction window
            candles = self._conn.execute(
                """
                SELECT timestamp, close, high, low FROM candles
                WHERE symbol = ? AND timestamp > ? AND timestamp <= ?
                ORDER BY timestamp ASC
                """,
                (symbol, scored_at, horizon_end_ms),
            ).fetchall()

            if not candles:
                continue

            timestamps = [r[0] for r in candles]
            closes = [r[1] for r in candles]
            highs = [r[2] for r in candles]
            lows = [r[3] for r in candles]

            realized_return = (closes[-1] - entry_price) / entry_price
            realized_max_high_pct = (max(highs) - entry_price) / entry_price
            realized_min_low_pct = (entry_price - min(lows)) / entry_price
            gap_spanned = _detect_gap_spanned(timestamps, horizon_end_ms)

            self._conn.execute(
                """
                UPDATE signal_history
                SET realized_return_at_horizon = ?,
                    realized_max_high_pct = ?,
                    realized_min_low_pct = ?,
                    gap_spanned = ?
                WHERE id = ?
                """,
                (
                    realized_return,
                    realized_max_high_pct,
                    realized_min_low_pct,
                    gap_spanned,
                    row_id,
                ),
            )
            resolved += 1

        self._conn.commit()
        if resolved:
            logger.info("signal_history: resolved %d rows", resolved)
        return resolved

    def compute_gap_spanned_at_horizon(self, symbol: str, scored_at: int, horizon_bars: int) -> int:
        """Recompute gap_spanned for an arbitrary horizon H without writing.

        The persisted ``gap_spanned`` answers "was the window gappy at the
        row's own horizon_bars?" (in practice 120 h). The horizon
        sweep needs to ask the same question at a shorter H — e.g. a Monday
        09:00 row spans the weekend at H=120 (gap=1) but is clean at H=24.

        Returns 1 when no candles cover the window, matching the resolver's
        conservative default.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        horizon_end_ms = scored_at + horizon_bars * _BAR_INTERVAL_MS
        rows = self._conn.execute(
            "SELECT timestamp FROM candles "
            "WHERE symbol = ? AND timestamp > ? AND timestamp <= ? "
            "ORDER BY timestamp ASC",
            (symbol, scored_at, horizon_end_ms),
        ).fetchall()
        timestamps = [int(r[0]) for r in rows]
        return _detect_gap_spanned(timestamps, horizon_end_ms)

    def get_signal_history(
        self,
        symbol: str | None = None,
        since_ms: int | None = None,
        resolved_only: bool = False,
        exclude_gap_spanned: bool = False,
        limit: int | None = None,
    ) -> list[SignalHistoryRecord]:
        """Read signal_history rows for diagnostics. Returns dicts keyed by column name.

        Set ``exclude_gap_spanned=True`` to drop rows whose horizon window was
        truncated by a market-closure gap (weekend / overnight).  These rows
        carry a downward-biased ``realized_return_at_horizon`` and are excluded
        from primary RankIC by the diagnostic.

        ``predicted_close_path`` and ``var_closes_at_horizons`` are returned
        as raw ``bytes`` (or None).  The latter decodes as little-endian
        float32, shape ``(sample_count, len(CANDIDATE_HORIZONS))`` row-major —
        see ``write_signal_history_batch`` for the full layout.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")

        clauses: list[str] = []
        params: list[object] = []
        if symbol is not None:
            clauses.append("symbol = ?")
            params.append(symbol)
        if since_ms is not None:
            clauses.append("scored_at >= ?")
            params.append(since_ms)
        if resolved_only:
            clauses.append("realized_return_at_horizon IS NOT NULL")
        if exclude_gap_spanned:
            clauses.append("COALESCE(gap_spanned, 0) = 0")

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        limit_clause = ""
        if limit:
            limit_clause = "LIMIT ?"
            params.append(int(limit))

        rows = self._conn.execute(
            f"""
            SELECT id, scored_at, symbol, horizon_bars, mean_return, direction_confidence,
                   uncertainty, predicted_mfe_pct, predicted_mae_pct, predicted_volatility,
                   monotonicity, entry_price, realized_return_at_horizon,
                   realized_max_high_pct, realized_min_low_pct,
                   COALESCE(gap_spanned, 0) AS gap_spanned,
                   predicted_close_path, var_closes_at_horizons
            FROM signal_history
            {where}
            ORDER BY scored_at DESC
            {limit_clause}
            """,
            params,
        ).fetchall()

        cols = [
            "id",
            "scored_at",
            "symbol",
            "horizon_bars",
            "mean_return",
            "direction_confidence",
            "uncertainty",
            "predicted_mfe_pct",
            "predicted_mae_pct",
            "predicted_volatility",
            "monotonicity",
            "entry_price",
            "realized_return_at_horizon",
            "realized_max_high_pct",
            "realized_min_low_pct",
            "gap_spanned",
            "predicted_close_path",
            "var_closes_at_horizons",
        ]
        return [cast(SignalHistoryRecord, dict(zip(cols, r, strict=True))) for r in rows]

    # ------------------------------------------------------------------
    # Asset correlations
    # ------------------------------------------------------------------

    def write_correlations(
        self,
        computed_at: int,
        matrix: dict[str, dict[str, float]],
    ) -> None:
        """Persist a correlation matrix snapshot.

        Each (symbol_a, symbol_b) pair with symbol_a < symbol_b is stored once.
        The full matrix is recoverable by symmetry.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
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
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
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

    # ------------------------------------------------------------------
    # Macro features
    # ------------------------------------------------------------------

    def insert_macro_observations(
        self, series_id: str, observations: list[tuple[int, float]]
    ) -> int:
        """Upsert FRED observations for ``series_id``.

        ``observations`` is ``[(observation_date_ms, value), ...]``.  Existing
        rows are replaced (FRED occasionally revises recent values).  Returns
        the row count written.
        """
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
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
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
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
        if self._conn is None:
            raise RuntimeError("CandleDB.init_db() must be called first")
        row = self._conn.execute(
            "SELECT observation_date, value FROM macro_features "
            "WHERE series_id = ? ORDER BY observation_date DESC LIMIT 1",
            (series_id,),
        ).fetchone()
        return (row[0], row[1]) if row else None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the database connection."""
        if self._conn is not None:
            self._conn.close()
            self._conn = None
            logger.info("CandleDB closed")
