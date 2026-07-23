"""Data access layer for the dashboard — strictly read-only.

State sources:
  * ``bot_state.json`` — atomic JSON snapshot written by ``StateManager``.
    Reads use ``json.load`` with a single open(); if a write is in progress
    we may read a partial file, which raises ``JSONDecodeError`` — caller
    falls back to the previous cached payload.
  * ``candles.db`` — SQLite opened with ``mode=ro`` so we cannot mutate.
    WAL mode lets the bot keep writing while we read.

Never opens a writable handle to either file.  Imports from ``bot`` are
allowed only for pure-stdlib helpers (e.g. ``trading_hours``) — never
anything that would transitively load torch or the Kronos predictor.
"""

from __future__ import annotations

import array
import json
import os
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any

from bot.data.eodhd_symbols import EODHD_UNIVERSE
from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS
from bot.execution.ig_quote_scale import ig_display_price as _ig_display_price
from bot.execution.ig_quote_scale import ig_quote_scale as _ig_quote_scale
from bot.risk.risk_config import RiskConfig
from bot.trading_hours import (
    MARKET_CATEGORIES,
    is_market_open,
    is_safe_for_entry,
    market_category,
    seconds_until_open,
)
from webgui.diagnostics import (
    host_loadavg,
    host_memory,
    host_uptime_s,
    journalctl_errors,
    journalctl_events,
    process_memory_mb,
    read_rerank_status_raw,
    rerank_progress,
    rerank_status,
    systemd_status,
)

# Same construction as ``main.py``'s ``risk_config = RiskConfig()`` — no env
# overrides exist for these fields (plain pydantic BaseModel, not
# BaseSettings), so the class defaults *are* the live values.
_RISK_DEFAULTS = RiskConfig()

# Mirrors ``bot.monitoring.health._FEED_STALENESS_MS`` (3h) for the freshness
# panel's stale styling — hourly bars confirm at each :00, so a healthy feed's
# latest candle is normally < 1h old.
_FEED_STALENESS_S = 3 * 3_600


def _risk_utilization(
    open_positions: int, total_risk_gbp: float, equity: float, caps: RiskConfig
) -> dict[str, Any]:
    """Open-position count / total risk-on vs the entry-gating caps, plus
    0-100 bar-width percentages pre-clamped for the template."""
    total_risk_pct = (total_risk_gbp / equity) if equity > 0 else 0.0
    return {
        "open_positions": open_positions,
        "max_open_positions": caps.max_open_positions,
        "open_pct_of_cap": min(100.0, open_positions * 100.0 / caps.max_open_positions)
        if caps.max_open_positions
        else 0.0,
        "total_risk_gbp": total_risk_gbp,
        "total_risk_pct": total_risk_pct,
        "max_total_risk_pct": caps.max_total_risk_pct,
        "risk_pct_of_cap": min(100.0, total_risk_pct * 100.0 / caps.max_total_risk_pct)
        if caps.max_total_risk_pct
        else 0.0,
    }


def _ro_conn(db_path: Path) -> sqlite3.Connection:
    """SQLite connection opened in read-only mode against the live WAL DB."""
    uri = f"file:{db_path}?mode=ro"
    con = sqlite3.connect(uri, uri=True, timeout=5.0)
    con.row_factory = sqlite3.Row
    return con


def _decode_close_path(blob: bytes | None) -> list[float]:
    """Decode a ``predicted_close_path`` BLOB to a float list.

    Mirrors ``scripts/signal_diagnostics._decode_path``: a little-endian
    float32 array of length ``pred_len`` (the full Pass-1 predicted close
    path).  Byte-swaps on the rare big-endian host so the geometry is
    correct regardless of architecture.  Returns ``[]`` on empty/corrupt
    input so callers can skip the chart cleanly.
    """
    if not blob:
        return []
    arr = array.array("f")
    try:
        arr.frombytes(bytes(blob))
    except (ValueError, TypeError):
        return []
    if sys.byteorder == "big":
        arr.byteswap()
    return list(arr)


class DashboardData:
    """One-stop reader for the dashboard endpoints.

    Holds paths and a small in-memory cache so a torn ``bot_state.json`` read
    falls back to the last known-good payload instead of returning an error.
    """

    def __init__(self, state_path: Path, db_path: Path, unit: str = "trading-bot") -> None:
        self._state_path = state_path
        self._db_path = db_path
        self._unit = unit
        self._last_state: dict[str, Any] = {}
        # rerank_status.json lives next to bot_state.json (the bot writes both
        # to its cwd).  See src/bot/state/rerank_status.py.
        self._rerank_status_path = state_path.parent / "rerank_status.json"

    # ------------------------------------------------------------------
    # bot_state.json
    # ------------------------------------------------------------------

    def read_state(self) -> dict[str, Any]:
        """Best-effort read of bot_state.json with last-good fallback."""
        try:
            with self._state_path.open() as f:
                self._last_state = json.load(f)
        except (OSError, json.JSONDecodeError):
            pass
        return self._last_state

    # ------------------------------------------------------------------
    # candles.db
    # ------------------------------------------------------------------

    def latest_candle_close(self, symbol: str) -> tuple[int, float] | None:
        """Return ``(timestamp_ms, close)`` of the most recent candle for *symbol*."""
        with _ro_conn(self._db_path) as con:
            row = con.execute(
                "SELECT timestamp, close FROM candles WHERE symbol = ? "
                "ORDER BY timestamp DESC LIMIT 1",
                (symbol,),
            ).fetchone()
        if row is None:
            return None
        return int(row["timestamp"]), float(row["close"])

    def feed_freshness(self) -> list[dict[str, Any]]:
        """Last-candle age per candle source, for the diagnostics tab.

        Metals (XAU/XAG) are IG-native since 2026-06-19 (``IG_NATIVE_CANDLE_SYMBOLS``,
        mirroring ``HealthMonitor._check_feed_staleness``'s grouping); everything
        else is EODHD-sourced except during a TwelveData failover. ``candles``
        has no per-row source column, so which provider is *live* for the
        non-metal group comes from the ``CANDLE_EXCHANGE`` env var, not the DB.
        """
        now_ms = int(time.time() * 1000)
        with _ro_conn(self._db_path) as con:
            rows = con.execute(
                "SELECT symbol, MAX(timestamp) AS ts FROM candles GROUP BY symbol"
            ).fetchall()
        freshest_by_symbol: dict[str, int] = {
            r["symbol"]: int(r["ts"]) for r in rows if r["ts"] is not None
        }
        metals = {s for s in EODHD_UNIVERSE if s in IG_NATIVE_CANDLE_SYMBOLS}
        others = {s for s in EODHD_UNIVERSE if s not in IG_NATIVE_CANDLE_SYMBOLS}
        primary_label = (
            "TwelveData (failover active)"
            if os.environ.get("CANDLE_EXCHANGE", "eodhd") == "twelvedata"
            else "EODHD (FX + US shares)"
        )

        def _group(label: str, symbols: set[str]) -> dict[str, Any]:
            ts_values = [freshest_by_symbol[s] for s in symbols if s in freshest_by_symbol]
            latest_ms = max(ts_values) if ts_values else None
            age_s = (now_ms - latest_ms) // 1000 if latest_ms is not None else None
            return {
                "label": label,
                "symbol_count": len(symbols),
                "latest_ms": latest_ms,
                "age_s": age_s,
                "stale": age_s is not None and age_s > _FEED_STALENESS_S,
            }

        return [_group(primary_label, others), _group("IG-native metals (XAU/XAG)", metals)]

    def latest_signals(self, limit: int = 28) -> list[dict[str, Any]]:
        """Return the most recent rerank's signal_history rows, ranked by mean_return.

        Picks the highest ``scored_at`` and returns every row at that timestamp.
        ``limit`` is a safety cap.

        Each row gets a ``market_open`` boolean added, derived live from
        ``is_safe_for_entry(symbol)`` — the same strict gate the bot uses to
        decide whether an entry is allowed *right now*.  Lets the template
        distinguish "signal passes Kronos thresholds AND market is open"
        (truly tradeable) from "signal passes thresholds but markets are
        closed for the weekend / maintenance window" (would-be-tradeable).
        """
        with _ro_conn(self._db_path) as con:
            latest_ts = con.execute("SELECT MAX(scored_at) AS ts FROM signal_history").fetchone()[
                "ts"
            ]
            if latest_ts is None:
                return []
            rows = con.execute(
                "SELECT scored_at, symbol, mean_return, direction_confidence, "
                "uncertainty, predicted_mfe_pct, predicted_mae_pct, monotonicity, "
                "entry_price, realized_return_at_horizon, gap_spanned "
                "FROM signal_history WHERE scored_at = ? "
                "ORDER BY mean_return DESC LIMIT ?",
                (latest_ts, limit),
            ).fetchall()
        result: list[dict[str, Any]] = []
        for r in rows:
            d = dict(r)
            d["market_open"] = is_safe_for_entry(d["symbol"])
            result.append(d)
        return result

    def signal_accuracy(self, max_rows: int = 500) -> dict[str, Any]:
        """Summarise calibration over the last *max_rows* resolved signals.

        Returns ``hit_rate`` (sign-of-predicted vs sign-of-realised), sample
        size, and per-quintile mean realised return — a crude RankIC-lite.
        """
        with _ro_conn(self._db_path) as con:
            rows = con.execute(
                "SELECT mean_return, realized_return_at_horizon "
                "FROM signal_history "
                "WHERE realized_return_at_horizon IS NOT NULL AND gap_spanned = 0 "
                "ORDER BY scored_at DESC LIMIT ?",
                (max_rows,),
            ).fetchall()
        if not rows:
            return {"sample_size": 0, "hit_rate": None, "quintiles": []}
        pairs = [(float(r["mean_return"]), float(r["realized_return_at_horizon"])) for r in rows]
        hits = sum(1 for pred, act in pairs if (pred > 0) == (act > 0))
        hit_rate = hits / len(pairs)
        # 5-quantile by predicted return
        sorted_by_pred = sorted(pairs, key=lambda p: p[0])
        n = len(sorted_by_pred)
        quintiles = []
        for q in range(5):
            lo, hi = n * q // 5, n * (q + 1) // 5
            slice_ = sorted_by_pred[lo:hi]
            if slice_:
                mean_actual = sum(a for _, a in slice_) / len(slice_)
                mean_pred = sum(p for p, _ in slice_) / len(slice_)
                quintiles.append(
                    {"q": q + 1, "n": len(slice_), "pred": mean_pred, "actual": mean_actual}
                )
        return {"sample_size": len(pairs), "hit_rate": hit_rate, "quintiles": quintiles}

    # ------------------------------------------------------------------
    # Model-performance visualisations (predicted vs realised)
    #
    # Borrowed from the vendored Kronos web UI's prediction-vs-actual
    # comparison.  All three read only *resolved* signal_history rows and
    # return pre-computed SVG geometry so the template can draw inline
    # charts with no JS / CDN dependency.  Everything is expressed as a
    # percentage return from ``entry_price`` — scale-invariant, so the
    # D3 / metals scale cutovers never enter the picture (the predicted
    # path and ``entry_price`` share the Kronos-context scale; their ratio
    # does not), and no candle re-join is needed.
    # ------------------------------------------------------------------

    def _resolved_cutoff(self, con: sqlite3.Connection) -> int | None:
        """Most-recent ``scored_at`` among resolved rows, or None.

        Touches only columns present in every ``signal_history`` revision,
        so it doubles as a guard: when it returns None the column-heavy
        SELECTs below are skipped and never reference ``predicted_*``
        columns a minimal (test) schema may lack.
        """
        row = con.execute(
            "SELECT MAX(scored_at) AS ts FROM signal_history "
            "WHERE realized_return_at_horizon IS NOT NULL"
        ).fetchone()
        return int(row["ts"]) if row and row["ts"] is not None else None

    def prediction_vs_realized_path(self) -> dict[str, Any] | None:
        """SVG geometry for the most-recent fully-resolved prediction.

        Decodes ``predicted_close_path`` (the float32 path BLOB) into a
        predicted close trajectory as % return from entry, and overlays the
        realised outcome: the actual high/low envelope
        (``realized_max_high_pct`` / ``realized_min_low_pct``) as a band and
        ``realized_return_at_horizon`` as the terminal marker.  This is the
        dashboard analogue of the Kronos web UI's prediction-vs-actual
        candlestick comparison, minus the candle re-join (and so immune to
        the D3 / metals scale cutovers).  Returns None when no resolved row
        carries a path blob.
        """
        with _ro_conn(self._db_path) as con:
            if self._resolved_cutoff(con) is None:
                return None
            row = con.execute(
                "SELECT scored_at, symbol, horizon_bars, entry_price, mean_return, "
                "predicted_mfe_pct, predicted_mae_pct, realized_return_at_horizon, "
                "realized_max_high_pct, realized_min_low_pct, predicted_close_path "
                "FROM signal_history "
                "WHERE realized_return_at_horizon IS NOT NULL "
                "  AND COALESCE(gap_spanned, 0) = 0 "
                "  AND predicted_close_path IS NOT NULL "
                "ORDER BY scored_at DESC LIMIT 1"
            ).fetchone()
        if row is None:
            return None
        entry = float(row["entry_price"] or 0.0)
        closes = _decode_close_path(row["predicted_close_path"])
        if entry <= 0 or len(closes) < 2:
            return None
        pred_pct = [(c - entry) / entry * 100.0 for c in closes]
        realized_ret = float(row["realized_return_at_horizon"]) * 100.0
        hi = float(row["realized_max_high_pct"] or 0.0) * 100.0
        lo = -float(row["realized_min_low_pct"] or 0.0) * 100.0
        # Vertical domain spans the predicted path, the realised envelope and 0.
        lo_dom = min(min(pred_pct), lo, realized_ret, 0.0)
        hi_dom = max(max(pred_pct), hi, realized_ret, 0.0)
        pad_dom = max((hi_dom - lo_dom) * 0.08, 0.05)
        lo_dom -= pad_dom
        hi_dom += pad_dom
        width, height, pad = 320.0, 150.0, 8.0
        n = len(pred_pct)
        span = hi_dom - lo_dom

        def _x(i: int) -> float:
            return pad + (width - 2 * pad) * (i / (n - 1))

        def _y(v: float) -> float:
            return pad + (height - 2 * pad) * (hi_dom - v) / span

        return {
            "symbol": row["symbol"],
            "scored_at_ms": int(row["scored_at"]),
            "horizon_bars": int(row["horizon_bars"]),
            "width": width,
            "height": height,
            "pred_points": " ".join(f"{_x(i):.1f},{_y(v):.1f}" for i, v in enumerate(pred_pct)),
            "pred_end": {"x": _x(n - 1), "y": _y(pred_pct[-1])},
            "pred_return": pred_pct[-1],
            "realized_return": realized_ret,
            "realized_point": {"x": _x(n - 1), "y": _y(realized_ret)},
            "band": {"y_hi": _y(hi), "y_lo": _y(lo), "hi": hi, "lo": lo},
            "zero_y": _y(0.0),
            "error": realized_ret - pred_pct[-1],
        }

    def calibration_scatter(self, max_rows: int = 300) -> dict[str, Any] | None:
        """SVG geometry for a predicted-vs-realised calibration scatter.

        Plots each resolved signal as (predicted ``mean_return``, realised
        return), both in %, against a ``y = x`` perfect-calibration diagonal,
        and overlays the quintile-mean reliability polyline — the visual
        companion to :meth:`signal_accuracy`'s quintile table.  Returns None
        when there are no resolved rows.
        """
        with _ro_conn(self._db_path) as con:
            if self._resolved_cutoff(con) is None:
                return None
            rows = con.execute(
                "SELECT mean_return, realized_return_at_horizon "
                "FROM signal_history "
                "WHERE realized_return_at_horizon IS NOT NULL "
                "  AND COALESCE(gap_spanned, 0) = 0 AND mean_return IS NOT NULL "
                "ORDER BY scored_at DESC LIMIT ?",
                (max_rows,),
            ).fetchall()
        pairs = [
            (float(r["mean_return"]) * 100.0, float(r["realized_return_at_horizon"]) * 100.0)
            for r in rows
        ]
        if not pairs:
            return None
        axis_max = max((max(abs(p), abs(a)) for p, a in pairs), default=1.0)
        axis_max = max(axis_max, 0.1)
        size, pad = 200.0, 10.0

        def _x(v: float) -> float:
            return pad + (size - 2 * pad) * (v + axis_max) / (2 * axis_max)

        def _y(v: float) -> float:
            return pad + (size - 2 * pad) * (axis_max - v) / (2 * axis_max)

        points = [{"x": _x(p), "y": _y(a), "correct": (p > 0) == (a > 0)} for p, a in pairs]
        # Quintile means by predicted return — same buckets as signal_accuracy.
        sorted_pairs = sorted(pairs, key=lambda pa: pa[0])
        nq = len(sorted_pairs)
        quintile = []
        for q in range(5):
            a_i, b_i = nq * q // 5, nq * (q + 1) // 5
            sl = sorted_pairs[a_i:b_i]
            if sl:
                mean_pred = sum(p for p, _ in sl) / len(sl)
                mean_actual = sum(a for _, a in sl) / len(sl)
                quintile.append({"x": _x(mean_pred), "y": _y(mean_actual)})
        return {
            "size": size,
            "pad": pad,
            "axis_max": axis_max,
            "zero_x": _x(0.0),
            "zero_y": _y(0.0),
            "diag": {
                "x1": _x(-axis_max),
                "y1": _y(-axis_max),
                "x2": _x(axis_max),
                "y2": _y(axis_max),
            },
            "points": points,
            "quintile_points": " ".join(f"{q['x']:.1f},{q['y']:.1f}" for q in quintile),
            "sample_size": len(pairs),
        }

    def resolved_signals(self, limit: int = 12) -> list[dict[str, Any]]:
        """Most-recent resolved signals with per-signal predicted-vs-realised error.

        The dashboard analogue of the Kronos web UI's gap analysis: for each
        resolved row, the predicted ``mean_return`` against the realised
        return, the signed error, and whether the direction call was
        correct.  Rows whose horizon window was truncated by a market-closure
        gap are flagged (their realised return is downward-biased).
        """
        with _ro_conn(self._db_path) as con:
            if self._resolved_cutoff(con) is None:
                return []
            rows = con.execute(
                "SELECT scored_at, symbol, mean_return, realized_return_at_horizon, "
                "COALESCE(gap_spanned, 0) AS gap_spanned "
                "FROM signal_history "
                "WHERE realized_return_at_horizon IS NOT NULL AND mean_return IS NOT NULL "
                "ORDER BY scored_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            pred = float(r["mean_return"])
            actual = float(r["realized_return_at_horizon"])
            out.append(
                {
                    "scored_at": int(r["scored_at"]),
                    "symbol": r["symbol"],
                    "pred_return": pred,
                    "realized_return": actual,
                    "error": actual - pred,
                    "correct": (pred > 0) == (actual > 0),
                    "gap_spanned": int(r["gap_spanned"]),
                }
            )
        return out

    def recent_correlation_pairs(self, limit: int = 5) -> list[dict[str, Any]]:
        """Top *limit* pairs by ``|correlation|`` from the most recent matrix snapshot."""
        with _ro_conn(self._db_path) as con:
            latest_ts = con.execute(
                "SELECT MAX(computed_at) AS ts FROM asset_correlations"
            ).fetchone()["ts"]
            if latest_ts is None:
                return []
            rows = con.execute(
                "SELECT symbol_a, symbol_b, correlation FROM asset_correlations "
                "WHERE computed_at = ? "
                "ORDER BY ABS(correlation) DESC LIMIT ?",
                (latest_ts, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def recent_sentiment(self, limit: int = 28) -> list[dict[str, Any]]:
        """Most recent sentiment score per asset (latest row per asset)."""
        with _ro_conn(self._db_path) as con:
            rows = con.execute(
                "SELECT asset, scored_at, sentiment, confidence, agreement, escalated "
                "FROM sentiment_scores s "
                "WHERE scored_at = (SELECT MAX(scored_at) FROM sentiment_scores "
                "                   WHERE asset = s.asset) "
                "ORDER BY sentiment DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(r) for r in rows]

    def market_reference(self, now: Any = None) -> list[dict[str, Any]]:
        """Live universe grouped by market, with hours and a closed-countdown.

        Walks ``EODHD_UNIVERSE`` (the single source of truth for the traded
        symbols) and buckets each into its :func:`market_category`.  Per market
        we report the human schedule, the live open/closed flag, and — when
        closed — seconds until the next open for the countdown.  Categories are
        emitted in the fixed :data:`MARKET_CATEGORIES` order so the table is
        stable across refreshes.
        """
        members: dict[str, list[str]] = {}
        for bot_key in EODHD_UNIVERSE:
            members.setdefault(market_category(bot_key), []).append(bot_key)

        out: list[dict[str, Any]] = []
        for cat, (label, schedule) in MARKET_CATEGORIES.items():
            symbols = sorted(members.get(cat, []))
            if not symbols:
                continue
            probe = symbols[0]  # all members of a category share one schedule
            is_open = is_market_open(probe, now)
            out.append(
                {
                    "category": cat,
                    "label": label,
                    "schedule": schedule,
                    "symbols": symbols,
                    "count": len(symbols),
                    "is_open": is_open,
                    "opens_in_s": None if is_open else seconds_until_open(probe, now),
                }
            )
        return out

    # ------------------------------------------------------------------
    # Aggregate "snapshot" — one dict for the dashboard
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        state = self.read_state()
        positions_raw = state.get("positions", {})
        tp_state = state.get("take_profit_state", {}) or {}
        positions: list[dict[str, Any]] = []
        now_ms = int(time.time() * 1000)
        for sym, pos in positions_raw.items():
            entry_ig = float(pos.get("entry_price", 0.0))
            opened_at = int(pos.get("opened_at", 0))
            qty = float(pos.get("quantity", 0.0))
            # Current price: prefer the heartbeat-refreshed live IG level stored
            # in state (``current_price``, ≤60 s old) so the dashboard P&L tracks
            # IG; fall back to the last *closed* hourly candle (up to ~1 h stale)
            # only when no live price has been written yet.  ``current_price`` is
            # always an IG level; the candle is candle-source units (IG level for
            # D3-native symbols, ETF/face value otherwise) → × scale to IG level.
            current_display: float | None = None
            ig_current: float | None = None
            unreal_pnl_pct: float | None = None
            state_current_ig = float(pos.get("current_price", 0.0))
            if state_current_ig > 0:
                ig_current = state_current_ig
            else:
                latest = self.latest_candle_close(sym)
                if latest is not None:
                    ig_current = latest[1] * _ig_quote_scale(sym)
            if ig_current is not None:
                current_display = _ig_display_price(sym, ig_current)
                if entry_ig > 0:
                    unreal_pnl_pct = (ig_current - entry_ig) / entry_ig * 100
            # Stop level (margin breakers + take-profit ratchets).  Initial stop comes
            # from ``entry_stop_pct`` × entry_price; once breakeven or
            # trailing arms, ``current_trailing_stop`` supersedes it.
            stop_pct: float | None = None
            stop_level_ig: float | None = None
            stop_display: float | None = None
            tp = tp_state.get(sym)
            if isinstance(tp, dict) and entry_ig > 0:
                tp_entry = float(tp.get("entry_price", 0.0))
                base_entry = tp_entry if tp_entry > 0 else entry_ig
                trail = tp.get("current_trailing_stop")
                if trail is not None:
                    stop_level_ig = float(trail)
                    stop_pct = (base_entry - stop_level_ig) / base_entry
                else:
                    raw_pct = tp.get("entry_stop_pct")
                    if isinstance(raw_pct, (int, float)):
                        stop_pct = float(raw_pct)
                        stop_level_ig = base_entry * (1.0 - stop_pct)
                if stop_level_ig is not None:
                    stop_display = _ig_display_price(sym, stop_level_ig)
            positions.append(
                {
                    "symbol": sym,
                    "entry_price": entry_ig,
                    "entry_display": _ig_display_price(sym, entry_ig),
                    "current_display": current_display,
                    "quantity": qty,
                    "opened_at": opened_at,
                    "opened_ago_s": (now_ms - opened_at) // 1000 if opened_at else None,
                    "unrealised_pnl_pct": unreal_pnl_pct,
                    "stop_pct": stop_pct,
                    "stop_level_ig": stop_level_ig,
                    "stop_display": stop_display,
                }
            )
        # Risk-on utilization vs the entry-gating caps (RiskConfig defaults —
        # ``risk.py`` gate 6b).  ``stop_level_ig`` above already folds in any
        # armed trailing stop, so this mirrors ``RiskBudgetLedger.live_risk_gbp``
        # (quantity × remaining distance to the effective stop) without needing
        # the separately-persisted ``risk_budgets`` ledger.
        total_risk_gbp = sum(
            p["quantity"] * max(0.0, p["entry_price"] - p["stop_level_ig"])
            for p in positions
            if p["stop_level_ig"] is not None
        )
        risk = state.get("risk", {})
        last_heartbeat = int(state.get("last_heartbeat", 0))
        heartbeat_age_s = (now_ms - last_heartbeat) // 1000 if last_heartbeat else None
        equity = float(state.get("equity", 0.0))
        peak = float(state.get("peak_equity", 0.0))
        cash = float(state.get("cash", 0.0))
        open_pnl = float(state.get("open_pnl", 0.0))
        drawdown_pct = ((peak - equity) / peak * 100) if peak > 0 else 0.0
        return {
            "now_ms": now_ms,
            "equity": equity,
            "cash": cash,
            "open_pnl": open_pnl,
            "peak_equity": peak,
            "drawdown_pct": drawdown_pct,
            "pnl_24h": float(state.get("pnl_24h", 0.0)),
            "trading_halted": bool(risk.get("trading_halted", False)),
            "consecutive_losses": int(risk.get("consecutive_losses", 0)),
            "last_heartbeat_ms": last_heartbeat,
            "heartbeat_age_s": heartbeat_age_s,
            "bot_started_at_ms": int(state.get("bot_started_at", 0)),
            "positions": positions,
            "risk_utilization": _risk_utilization(
                len(positions), total_risk_gbp, equity, _RISK_DEFAULTS
            ),
            "markets": self.market_reference(),
            "feed_freshness": self.feed_freshness(),
            "signals": self.latest_signals(),
            "accuracy": self.signal_accuracy(),
            "prediction_path": self.prediction_vs_realized_path(),
            "calibration": self.calibration_scatter(),
            "resolved": self.resolved_signals(),
            "correlations": self.recent_correlation_pairs(),
            "sentiment": self.recent_sentiment(),
            "service": self._service_diagnostics(),
            "host": self._host_diagnostics(),
            "events": journalctl_events(self._unit, limit=20),
            "log_errors": journalctl_errors(self._unit, limit=15),
            "rerank_progress": rerank_status(self._rerank_status_path)
            or rerank_progress(self._unit),
        }

    # ------------------------------------------------------------------
    # Host + service diagnostics (systemd, /proc, journalctl)
    # ------------------------------------------------------------------

    def _service_diagnostics(self) -> dict[str, Any]:
        info = systemd_status(self._unit)
        pid = info.get("pid", 0)
        info["bot_mem_mb"] = process_memory_mb(pid) if pid else None
        # Latest rerank timestamp for context — pulled from signal_history.
        with _ro_conn(self._db_path) as con:
            row = con.execute("SELECT MAX(scored_at) AS ts FROM signal_history").fetchone()
        last_scored_at = int(row["ts"]) if row and row["ts"] else 0
        info["last_rerank_ms"] = last_scored_at
        info["last_rerank_age_s"] = (
            int(time.time()) - last_scored_at // 1000 if last_scored_at else None
        )
        # ``next_rerank_at`` (wall-clock seconds) persists in rerank_status.json
        # across reranks — read raw (bypassing rerank_status()'s in_progress
        # gate) so the always-on countdown keeps working between reranks.
        raw_status = read_rerank_status_raw(self._rerank_status_path)
        next_rerank_at = raw_status.get("next_rerank_at") if raw_status else None
        info["next_rerank_at_ms"] = (
            int(next_rerank_at * 1000) if isinstance(next_rerank_at, (int, float)) else None
        )
        return info

    def _host_diagnostics(self) -> dict[str, Any]:
        return {
            "uptime_s": host_uptime_s(),
            "memory": host_memory(),
            "loadavg": host_loadavg(),
        }
