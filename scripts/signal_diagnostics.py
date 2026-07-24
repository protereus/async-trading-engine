"""Signal diagnostics — weekly report on Kronos signal quality.

Reads signal_history rows from candles.db and computes:

  1. RankIC by asset class (Spearman + directional hit-rate).
  2. RankIC per asset (top/bottom 5 in main report; full table in appendix).
  3. Confidence calibration — decile bucket hit-rate.
  4. RankIC × confidence decile cross-tab — does the gate threshold sit at the
     right confidence?
  5. RankIC × predicted-MFE bucket — do low-MFE predictions carry alpha?
  6. Entries-only hit-rate (tradeable subset) vs all rows.
  7. MFE / MAE accuracy + realisation timing.
  8. Predicted-vs-realised volatility ratio.
  9. Optimal ranking horizon by
     asset class, derived from the full predicted-close path that 10b logs to
     ``signal_history.predicted_close_path``.
 10. (10c-prep, with ``--horizon-sweep``) Per-H confidence calibration —
     direction confidence recomputed at each candidate H from the Pass-2
     ``var_closes_at_horizons`` draws, cross-tabbed decile × hit-rate ×
     RankIC per (asset class, H).
 11. (10c-prep, with ``--horizon-sweep``) Entry-filter grid —
     min_confidence × min_predicted_return sweep at each H, live cell
     marked.  Rows without the 10c-prep blob are counted and skipped.

The ``--horizon-sweep`` sections also report **net-of-cost** realised return
(after round-trip spread + per-roll funding): section 8 adds a net-mean-return
grid + cost breakdown, section 10 adds ``mRealNet``/``HitNet%`` columns.
RankIC is a rank stat and barely moves net-of-cost — the *level* (mean return,
hit-rate) is what costs destroy, and funding grows with H, so the net columns
are the P&L-relevant ones for the H + threshold choice.  Spread reuses
``ig_margin.SLIPPAGE_PCT`` (conservative); tune with ``--cost-spread-mult`` /
``--no-spread`` / ``--no-funding``.

Weekend-spanning rows (``gap_spanned=1``) are excluded by default — the
realised return there is measured against a truncated window and biases
forex RankIC downward.  Pass ``--include-gap-rows`` to opt back in.

The persisted ``gap_spanned`` flag answers the question at each row's own
``horizon_bars`` (120 h in production). The ``--horizon-sweep``
section re-evaluates the gap test at every candidate H using the candles
table, so a Monday-morning row that's gap-spanned at H=120 (window crosses
a weekend) can still contribute clean data at H=24.

Output is written to journal/signal_diagnostics_YYYY-MM-DD.txt.
Do not auto-tune off these numbers — give the human the numbers, let the
human decide.

Usage:
    uv run python scripts/signal_diagnostics.py
    uv run python scripts/signal_diagnostics.py --db /path/to/candles.db --days 60
    uv run python scripts/signal_diagnostics.py --horizon-sweep
    uv run python scripts/signal_diagnostics.py --since 2026-06-03 --horizon-sweep
"""

from __future__ import annotations

import argparse
import os
import sqlite3
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

try:
    import numpy as np
except ImportError:
    print("numpy required: uv add numpy", file=sys.stderr)
    sys.exit(1)


def _import_gap_detector() -> Callable[[list[int], int], int]:
    """Import bot.data.signal_history_store._detect_gap_spanned with the src/ path fix.

    Done at first use so ruff can't strip the import as unused (the script
    is normally run via ``uv run`` which sets sys.path correctly, but the
    explicit insert keeps direct ``python scripts/...`` invocations working
    too).
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bot.data.signal_history_store import _detect_gap_spanned

    return _detect_gap_spanned


def _candidate_horizons() -> tuple[int, ...]:
    """Candidate ranking horizons — single source of truth in the strategy.

    The Pass-2 ``var_closes_at_horizons`` blob is written column-per-horizon
    in this exact order, so the sweep and the blob decode must agree.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bot.strategy.topk_strategy import CANDIDATE_HORIZONS

    return CANDIDATE_HORIZONS


def _import_cost_fns() -> tuple[
    Callable[[str], float], Callable[..., object], Callable[..., float]
]:
    """Import the live IG cost primitives with the same src/ path shim.

    Single-sourced so the diagnostic's net-of-cost view and the live bot agree:

      * ``slippage_pct_for(symbol)`` — per-asset-class cost as a fraction of
        price.  It is calibrated as worst-case slippage past a stop (fatter
        than the dealing spread), so the net view is **conservative** — the
        ``--cost-spread-mult`` flag lets the human sensitivity-test.
      * ``classify_symbol(symbol)`` → IG ``AssetClass`` for the funding rate.
      * ``daily_funding_pct(asset_class=, side=, now_utc=)`` — signed daily
        funding as a fraction of notional, weekday ×3 multipliers baked in.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from bot.risk.funding import daily_funding_pct
    from bot.risk.ig_margin import classify_symbol, slippage_pct_for

    return slippage_pct_for, classify_symbol, daily_funding_pct


_ROLL_HOUR_UTC = 22  # IG overnight roll fires at 22:00 UTC (see bot.risk.funding)


def _window_roll_times(scored_at_ms: int, h_end_ms: int) -> list[datetime]:
    """22:00-UTC overnight rolls strictly after entry, up to and incl. window end.

    A LONG opened at ``scored_at`` and held to ``h_end_ms`` pays funding once
    per roll crossed.  Counting the actual crossings (rather than ceil(H/24)
    days) keeps a sub-day hold that never crosses 22:00 at zero funding and
    lets the weekday ×3 land on the right horizons.
    """
    start = datetime.fromtimestamp(scored_at_ms / 1000, tz=UTC)
    end = datetime.fromtimestamp(h_end_ms / 1000, tz=UTC)
    roll = start.replace(hour=_ROLL_HOUR_UTC, minute=0, second=0, microsecond=0)
    if roll <= start:
        roll += timedelta(days=1)
    rolls: list[datetime] = []
    while roll <= end:
        rolls.append(roll)
        roll += timedelta(days=1)
    return rolls


@dataclass
class _CostConfig:
    """How the net-of-cost view charges spread + funding (CLI-controlled)."""

    spread_mult: float = 1.0
    include_spread: bool = True
    include_funding: bool = True

    def describe(self) -> str:
        parts = []
        if self.include_spread:
            parts.append(f"spread=SLIPPAGE_PCT×{self.spread_mult:g}")
        else:
            parts.append("spread=off")
        parts.append("funding=on" if self.include_funding else "funding=off")
        return ", ".join(parts)


# ---------------------------------------------------------------------------
# Asset classification
# ---------------------------------------------------------------------------

# Legacy TwelveData-era sets — kept only as a fallback so signal_history rows
# scored before the EODHD cutover (2026-06-03) still classify.  The live
# universe is classified from ``bot.data.eodhd_symbols.EODHD_UNIVERSE`` below.
_FOREX = {
    "EUR/USD",
    "GBP/USD",
    "USD/JPY",
    "AUD/USD",
    "USD/CAD",
    "USD/CHF",
    "NZD/USD",
    "EUR/GBP",
    "EUR/JPY",
    "GBP/JPY",
    "AUD/JPY",
    "EUR/AUD",
    "EUR/CAD",
    "GBP/CAD",
    "EUR/CHF",
    "GBP/CHF",
    "USD/SEK",
    "USD/NOK",
    "CAD/JPY",
    "GBP/AUD",
    "GBP/NZD",
}
_METALS = {"XAU/USD", "SLV"}
_INDICES = {"SPY", "QQQ", "DIA", "USO", "UNG"}

# Lazily-built bot_key → diagnostics class from the live EODHD universe.
# ``EODHD_UNIVERSE`` is the single source of truth (do not duplicate the
# mapping here); its ``asset_class="equity"`` is relabelled ``us_equity`` to
# distinguish single names from the retired ``index/etf`` legacy cohort.
_EODHD_CLASS_CACHE: dict[str, str] | None = None


def _eodhd_class_map() -> dict[str, str]:
    """Import EODHD_UNIVERSE with the same src/ path fix as the gap detector."""
    global _EODHD_CLASS_CACHE
    if _EODHD_CLASS_CACHE is None:
        sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
        from bot.data.eodhd_symbols import EODHD_UNIVERSE

        _EODHD_CLASS_CACHE = {
            key: ("us_equity" if sym.asset_class == "equity" else sym.asset_class)
            for key, sym in EODHD_UNIVERSE.items()
        }
    return _EODHD_CLASS_CACHE


def _asset_class(symbol: str) -> str:
    """EODHD universe first (forex / metal / us_equity), legacy sets as fallback."""
    eodhd_class = _eodhd_class_map().get(symbol)
    if eodhd_class is not None:
        return eodhd_class
    if symbol in _FOREX:
        return "forex"
    if symbol in _METALS:
        return "metal"
    if symbol in _INDICES:
        return "index/etf"
    return "other"


# ---------------------------------------------------------------------------
# Core statistics helpers
# ---------------------------------------------------------------------------


def _spearman_ic(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation (RankIC)."""
    if len(x) < 3:
        return float("nan")
    arr_x = np.array(x, dtype=float)
    arr_y = np.array(y, dtype=float)
    rx = np.argsort(np.argsort(arr_x)).astype(float)
    ry = np.argsort(np.argsort(arr_y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _hit_rate(pred_returns: list[float], realized_returns: list[float]) -> float:
    """Fraction of rows where predicted direction == realised direction."""
    if not pred_returns:
        return float("nan")
    hits = sum((p >= 0) == (r >= 0) for p, r in zip(pred_returns, realized_returns, strict=True))
    return hits / len(pred_returns)


# ---------------------------------------------------------------------------
# Row loading
# ---------------------------------------------------------------------------


def _load_rows(
    db_path: str,
    since_ms: int,
    include_gap: bool,
) -> list[sqlite3.Row]:
    """Load resolved signal_history rows from the DB.

    Returns sqlite3.Row objects so columns can be accessed by name regardless
    of schema-column order.  Probes which optional /10b columns exist
    so the diagnostic works against pre-migration DBs.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signal_history)").fetchall()}
    gap_select = (
        "COALESCE(gap_spanned, 0) AS gap_spanned"
        if "gap_spanned" in existing_cols
        else "0 AS gap_spanned"
    )
    path_select = (
        "predicted_close_path"
        if "predicted_close_path" in existing_cols
        else "NULL AS predicted_close_path"
    )
    gap_clause = (
        ""
        if include_gap or "gap_spanned" not in existing_cols
        else " AND COALESCE(gap_spanned, 0) = 0"
    )
    rows = conn.execute(
        f"""
        SELECT symbol, mean_return, direction_confidence, uncertainty,
               predicted_mfe_pct, predicted_mae_pct, predicted_volatility,
               monotonicity, entry_price,
               realized_return_at_horizon, realized_max_high_pct, realized_min_low_pct,
               {gap_select},
               horizon_bars, scored_at,
               {path_select}
        FROM signal_history
        WHERE scored_at >= ?
          AND realized_return_at_horizon IS NOT NULL
          {gap_clause}
        ORDER BY scored_at ASC
        """,
        (since_ms,),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Section 1+2: RankIC by asset class and per asset
# ---------------------------------------------------------------------------


def _per_asset_rankic(rows: list[sqlite3.Row]) -> list[tuple[str, int, float, float]]:
    """Per-symbol (symbol, N, RankIC, hit-rate), RankIC-descending, N ≥ 5 only.

    Shared by section 2 (top/bottom 5) and the full-table appendix.
    """
    by_symbol: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        pred = row["mean_return"]
        realized = row["realized_return_at_horizon"]
        if pred is None or realized is None:
            continue
        by_symbol.setdefault(row["symbol"], []).append((pred, realized))

    per_asset: list[tuple[str, int, float, float]] = []
    for sym, pairs in by_symbol.items():
        if len(pairs) < 5:
            continue
        preds = [p for p, _ in pairs]
        reals = [r for _, r in pairs]
        per_asset.append((sym, len(pairs), _spearman_ic(preds, reals), _hit_rate(preds, reals)))
    per_asset.sort(key=lambda t: float("-inf") if np.isnan(t[2]) else t[2], reverse=True)
    return per_asset


def _section_appendix(rows: list[sqlite3.Row], lines: list[str]) -> None:
    """Full per-asset RankIC table (all symbols, N ≥ 5) — promised by the
    module docstring but historically truncated to top/bottom 5."""
    lines.append("\n── Appendix — Per-Asset RankIC (full table, N ≥ 5) ──────────────────")
    per_asset = _per_asset_rankic(rows)
    if not per_asset:
        lines.append("  Not enough rows per symbol yet (need ≥ 5 each).")
        return
    lines.append(f"{'Symbol':<14} {'Class':<12} {'N':>5} {'RankIC':>8} {'Hit%':>7}")
    lines.append("-" * 51)
    for sym, n, ic, hr in per_asset:
        lines.append(f"{sym:<14} {_asset_class(sym):<12} {n:>5} {ic:>8.3f} {hr * 100:>6.1f}%")


def _section_rankic(rows: list[sqlite3.Row], lines: list[str]) -> None:
    by_class: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        pred = row["mean_return"]
        realized = row["realized_return_at_horizon"]
        if pred is None or realized is None:
            continue
        ac = _asset_class(row["symbol"])
        by_class.setdefault(ac, []).append((pred, realized))

    lines.append("\n── 1. RankIC by Asset Class ─────────────────────────────────────────")
    lines.append(f"{'Class':<12} {'N':>5} {'RankIC':>8} {'Hit%':>7}")
    lines.append("-" * 38)
    all_preds, all_realized = [], []
    for ac in sorted(by_class):
        preds = [p for p, _ in by_class[ac]]
        reals = [r for _, r in by_class[ac]]
        ic = _spearman_ic(preds, reals)
        hr = _hit_rate(preds, reals)
        lines.append(f"{ac:<12} {len(preds):>5} {ic:>8.3f} {hr * 100:>6.1f}%")
        all_preds.extend(preds)
        all_realized.extend(reals)
    lines.append("-" * 38)
    lines.append(
        f"{'ALL':<12} {len(all_preds):>5} "
        f"{_spearman_ic(all_preds, all_realized):>8.3f} "
        f"{_hit_rate(all_preds, all_realized) * 100:>6.1f}%"
    )

    # Per-asset table — top/bottom 5 by RankIC (N ≥ 5 only)
    lines.append("\n── 2. Per-Asset RankIC (top / bottom 5, N ≥ 5) ──────────────────────")
    per_asset = _per_asset_rankic(rows)

    if not per_asset:
        lines.append("  Not enough rows per symbol yet (need ≥ 5 each).")
        return

    lines.append(f"{'Symbol':<14} {'N':>5} {'RankIC':>8} {'Hit%':>7}")
    lines.append("-" * 38)
    for sym, n, ic, hr in per_asset[:5]:
        lines.append(f"{sym:<14} {n:>5} {ic:>8.3f} {hr * 100:>6.1f}%")
    if len(per_asset) > 10:
        lines.append("  ...")
    for sym, n, ic, hr in per_asset[-5:]:
        if (sym, n, ic, hr) in per_asset[:5]:
            continue
        lines.append(f"{sym:<14} {n:>5} {ic:>8.3f} {hr * 100:>6.1f}%")


# ---------------------------------------------------------------------------
# Section 3+4: Confidence calibration and RankIC × confidence
# ---------------------------------------------------------------------------


def _section_confidence(rows: list[sqlite3.Row], lines: list[str]) -> None:
    triples = [
        (r["direction_confidence"], r["mean_return"], r["realized_return_at_horizon"])
        for r in rows
        if r["direction_confidence"] is not None
        and r["mean_return"] is not None
        and r["realized_return_at_horizon"] is not None
    ]
    lines.append("\n── 3. Confidence Calibration ────────────────────────────────────────")
    lines.append("  (well-calibrated: the 0.70 decile should hit ~70 % of the time)")
    if not triples:
        lines.append("  No rows with direction_confidence.")
        return
    confs = np.array([t[0] for t in triples])
    edges = np.percentile(confs, np.arange(0, 110, 10))
    lines.append(f"{'Conf bucket':<16} {'N':>5} {'Hit%':>7} {'RankIC':>8}")
    lines.append("-" * 40)
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        bucket = [(p, r) for c, p, r in triples if lo <= c < hi]
        if not bucket:
            continue
        bp = [p for p, _ in bucket]
        br = [r for _, r in bucket]
        lines.append(
            f"[{lo:.2f}, {hi:.2f})  {len(bucket):>5} "
            f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f}"
        )


# ---------------------------------------------------------------------------
# Section 5: RankIC × predicted-MFE bucket
# ---------------------------------------------------------------------------


def _section_mfe_buckets(rows: list[sqlite3.Row], lines: list[str]) -> None:
    triples = [
        (r["predicted_mfe_pct"], r["mean_return"], r["realized_return_at_horizon"])
        for r in rows
        if r["predicted_mfe_pct"] is not None
        and r["mean_return"] is not None
        and r["realized_return_at_horizon"] is not None
    ]
    lines.append("\n── 4. RankIC × Predicted-MFE Bucket ─────────────────────────────────")
    if not triples:
        lines.append("  No path-aware rows yet (predicted_mfe_pct null).")
        return
    mfes = np.array([t[0] for t in triples])
    edges = np.percentile(mfes, [0, 25, 50, 75, 100])
    lines.append(f"{'MFE bucket':<20} {'N':>5} {'Hit%':>7} {'RankIC':>8}")
    lines.append("-" * 44)
    labels = ["Q1 (low MFE)", "Q2", "Q3", "Q4 (high MFE)"]
    for label, lo, hi in zip(labels, edges[:-1], edges[1:], strict=True):
        bucket = [(p, r) for m, p, r in triples if lo <= m <= hi]
        if not bucket:
            continue
        bp = [p for p, _ in bucket]
        br = [r for _, r in bucket]
        lines.append(
            f"{label:<20} {len(bucket):>5} "
            f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f}"
        )


# ---------------------------------------------------------------------------
# Section 6: Entries-only hit-rate (tradeable subset proxy)
# ---------------------------------------------------------------------------


def _section_entries_only(
    rows: list[sqlite3.Row],
    lines: list[str],
    min_confidence: float,
    min_predicted_return: float,
    max_uncertainty: float,
) -> None:
    lines.append("\n── 5. Entry-Subset Hit-Rate (proxy: re-applies live tradeable filter) ")
    lines.append(
        f"  Live thresholds: conf ≥ {min_confidence}, return ≥ {min_predicted_return}, "
        f"uncertainty ≤ {max_uncertainty}"
    )
    all_pairs = []
    entry_pairs = []
    for r in rows:
        p, real = r["mean_return"], r["realized_return_at_horizon"]
        if p is None or real is None:
            continue
        all_pairs.append((p, real))
        c = r["direction_confidence"] or 0.0
        u = r["uncertainty"] or 0.0
        if p >= min_predicted_return and c >= min_confidence and u <= max_uncertainty:
            entry_pairs.append((p, real))
    lines.append(f"{'Subset':<16} {'N':>5} {'Hit%':>7} {'RankIC':>8}")
    lines.append("-" * 40)
    if all_pairs:
        bp = [p for p, _ in all_pairs]
        br = [r for _, r in all_pairs]
        lines.append(
            f"{'all rows':<16} {len(all_pairs):>5} "
            f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f}"
        )
    if entry_pairs:
        bp = [p for p, _ in entry_pairs]
        br = [r for _, r in entry_pairs]
        lines.append(
            f"{'would-enter':<16} {len(entry_pairs):>5} "
            f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f}"
        )
    else:
        lines.append("  No rows clear the live tradeable filter.")


# ---------------------------------------------------------------------------
# Section 7: MFE / MAE accuracy + realisation timing
# ---------------------------------------------------------------------------


def _section_mfe_mae_accuracy(rows: list[sqlite3.Row], lines: list[str]) -> None:
    lines.append("\n── 6. MFE / MAE Accuracy (±20 % tolerance) ──────────────────────────")
    mfe_rows = [
        (r["predicted_mfe_pct"], r["realized_max_high_pct"])
        for r in rows
        if r["predicted_mfe_pct"] is not None and r["realized_max_high_pct"] is not None
    ]
    mae_rows = [
        (r["predicted_mae_pct"], r["realized_min_low_pct"])
        for r in rows
        if r["predicted_mae_pct"] is not None and r["realized_min_low_pct"] is not None
    ]
    if mfe_rows:
        mfe_within = sum(abs(p - r) <= 0.2 * abs(r) for p, r in mfe_rows if r != 0)
        mae = float(np.mean([abs(p - r) for p, r in mfe_rows]))
        within_pct = mfe_within / len(mfe_rows) * 100
        lines.append(f"  MFE: N={len(mfe_rows)}  within±20%={within_pct:.1f}%  MAE={mae:.4f}")
    else:
        lines.append("  MFE: no data (path metrics not yet active)")
    if mae_rows:
        mae_within = sum(abs(p - r) <= 0.2 * abs(r) for p, r in mae_rows if r != 0)
        mae_err = float(np.mean([abs(p - r) for p, r in mae_rows]))
        within_pct = mae_within / len(mae_rows) * 100
        lines.append(f"  MAE: N={len(mae_rows)}  within±20%={within_pct:.1f}%  MAE={mae_err:.4f}")
    else:
        lines.append("  MAE: no data (path metrics not yet active)")

    # Capture fraction summary (proxy for TP calibration in 10f):
    # what fraction of realised MFE was at or above 0.85 × predicted MFE?
    if mfe_rows:
        capture_hits = sum(1 for p, r in mfe_rows if p > 0 and r >= 0.85 * p)
        lines.append(
            f"  Capture rate (realised MFE ≥ 0.85 × predicted): "
            f"{capture_hits}/{len(mfe_rows)} = {capture_hits / len(mfe_rows) * 100:.1f}%"
        )


# ---------------------------------------------------------------------------
# Section 8: predicted-vs-realised volatility
# ---------------------------------------------------------------------------


def _section_vol_accuracy(rows: list[sqlite3.Row], lines: list[str]) -> None:
    lines.append("\n── 7. Predicted vs Realised Volatility ──────────────────────────────")
    triples = [
        (
            r["symbol"],
            r["predicted_volatility"],
            r["realized_max_high_pct"],
            r["realized_min_low_pct"],
        )
        for r in rows
        if r["predicted_volatility"] is not None
        and r["realized_max_high_pct"] is not None
        and r["realized_min_low_pct"] is not None
    ]
    if not triples:
        lines.append("  No data.")
        return
    by_class: dict[str, list[tuple[float, float]]] = {}
    for sym, pv, rmh, rml in triples:
        realised_range = abs(rmh) + abs(rml)  # crude proxy: total MFE + MAE
        by_class.setdefault(_asset_class(sym), []).append((pv, realised_range))
    lines.append(f"{'Class':<12} {'N':>5} {'pred/real':>10}")
    lines.append("-" * 32)
    for ac in sorted(by_class):
        pairs = by_class[ac]
        pred_mean = float(np.mean([p for p, _ in pairs]))
        real_mean = float(np.mean([r for _, r in pairs]))
        ratio = pred_mean / real_mean if real_mean else float("nan")
        lines.append(f"{ac:<12} {len(pairs):>5} {ratio:>10.3f}")
    lines.append("  (ratio ~1.0 = well-calibrated; <1 = under-predicts; >1 = over-predicts)")


# ---------------------------------------------------------------------------
# Section 9 (10b): Horizon-sweep — RankIC × H × asset class
# ---------------------------------------------------------------------------


def _decode_path(blob: bytes | None) -> np.ndarray | None:
    """Decode a predicted_close_path BLOB written by

    Format: little-endian float32 array of length ``pred_len``.  Returns None
    when the column is empty.
    """
    if not blob:
        return None
    try:
        arr = np.frombuffer(blob, dtype="<f4")
    except ValueError:
        return None
    return arr if arr.size > 0 else None


_HOUR_MS = 3_600_000

# The D3 cutover (2026-05-31) replaced TD-ETF historical candles for USO/UNG/SLV
# with IG-native bars of a *different* instrument (WTI / NatGas / silver-spot
# rather than the USO/UNG/SLV ETFs themselves).  Pre-D3 signal_history rows
# carry TD-scale ``entry_price`` (USO ≈ 130, UNG ≈ 12, SLV ≈ 70) and a
# ``predicted_close_path`` generated against TD-scale Kronos context — joining
# either to today's candles table is meaningless.
# ``scripts/recompute_d3_realised.py`` repairs ``realized_*`` columns by
# re-fetching TD ETF history; the sweep recomputes realised from candles each
# pass and so has to exclude the pre-D3 ETF cohort.  Value-based detection
# (rather than a timestamp) handles the mid-day deploy cleanly.
_D3_AFFECTED_SYMBOLS: frozenset[str] = frozenset({"USO", "UNG", "SLV"})
_PRE_D3_TD_VALUE_THRESHOLD: dict[str, float] = {"USO": 500.0, "UNG": 100.0, "SLV": 500.0}


def _is_pre_d3_etf_row(symbol: str, entry_price: float) -> bool:
    """True for USO/UNG/SLV signal_history rows scored before the D3 cutover.

    Detected by TD-scale ``entry_price``: pre-D3 values are well below the
    IG-native scale they flipped to on 2026-05-31, so a value threshold is
    more reliable than a fixed cutover timestamp (the deploy was mid-day).
    """
    if symbol not in _D3_AFFECTED_SYMBOLS:
        return False
    return entry_price < _PRE_D3_TD_VALUE_THRESHOLD[symbol]


# Metals flipped from EODHD ETF-proxy prices to IG-native $/oz on 2026-06-19
# .  The candles
# table was rebuilt IG-native for the *whole* history, but signal_history rows
# scored before the cutover keep their old-scale entry_price — joining the two
# yields a ~10x (XAU) / ~100x (XAG) phantom return that poisons the §8/§9/§10
# metal cells.  Detected by an entry_price value threshold (a mid-day deploy →
# more reliable than a fixed timestamp), exactly like the pre-D3 ETF guard.
_PRE_IGNATIVE_METAL_THRESHOLD: dict[str, float] = {"XAU/USD": 2000.0, "XAG/USD": 1000.0}


def _is_pre_ignative_metal_row(symbol: str, entry_price: float) -> bool:
    """True for XAU/XAG signal_history rows scored before the IG-native cutover.

    Pre-cutover ``entry_price`` sits in the old EODHD ETF-proxy scale
    (XAU ~400, XAG ~62), far below the IG-native $/oz scale the candles were
    rebuilt to (XAU ~4200, XAG ~6450), so a value threshold cleanly separates
    the two cohorts without hardcoding the mid-day deploy timestamp.
    """
    threshold = _PRE_IGNATIVE_METAL_THRESHOLD.get(symbol)
    if threshold is None:
        return False
    return entry_price < threshold


@dataclass
class _HRecord:
    """One usable (row, H) point from the per-H candle join."""

    symbol: str
    asset_class: str
    horizon: int
    scored_at: int  # rerank timestamp (ms) — groups symbols into one cross-section
    pred_ret: float  # from predicted_close_path[H-1] vs entry
    real_ret: float  # last in-window candle close vs entry (gross, on mid for FX)
    confidence: float | None  # per-H direction confidence (None pre-Part-B rows)
    # Net-of-cost (10c-prep §1): both fractions of position value, ≥ 0 for the
    # long-only book.  ``real_ret_net = real_ret − spread_pct − funding_pct``.
    spread_pct: float = 0.0
    funding_pct: float = 0.0

    @property
    def cost_pct(self) -> float:
        return self.spread_pct + self.funding_pct

    @property
    def real_ret_net(self) -> float:
        return self.real_ret - self.spread_pct - self.funding_pct


@dataclass
class _HCollection:
    """Everything the --horizon-sweep sections share: records + drop counters."""

    records: list[_HRecord]
    gap_drops: dict[int, int]
    miss_drops: dict[int, int]
    rows_with_var: int  # rows carrying a var_closes_at_horizons blob
    rows_without_var: int  # resolved+path rows without one (pre-10c-prep)
    error: str | None = None


def _per_h_confidence(
    var_arr: np.ndarray | None, h_idx: int, entry: float, pred_ret: float
) -> float | None:
    """Direction confidence at one horizon from the Pass-2 draws.

    Mirrors the live formula (fraction of draws agreeing with the sign of
    the Pass-1 mean) but evaluated at column ``h_idx`` of the decoded
    ``var_closes_at_horizons`` matrix.  NaN draws (rollout shorter than H)
    are excluded; all-NaN → None.
    """
    if var_arr is None:
        return None
    draws = var_arr[:, h_idx]
    draws = draws[~np.isnan(draws)]
    if draws.size == 0:
        return None
    if pred_ret >= 0:
        return float(np.mean(draws >= entry))
    return float(np.mean(draws < entry))


def _collect_h_records(
    db_path: str, since_ms: int, cost_cfg: _CostConfig | None = None
) -> _HCollection:
    """Per-(row, H) joined data with H-aware gap filtering.

    The persisted ``signal_history.gap_spanned`` reflects each row's own
    ``horizon_bars``, which excludes nearly every weekday forex row from the
    sweep cohort at H=120. Here we re-fetch rows *including* persisted
    gap-spanned ones, then for every (row, H) pair recompute gap_spanned
    against the H-truncated window via ``_detect_gap_spanned``. A
    Monday-morning row clean at H=24 contributes at that H even when its
    120 h window crossed a weekend.

    Respects the report cutoff (``--since`` / ``--days``) so the sweep
    cohort matches the headline sections — essential for the post-cutover
    recalibration, where pre-2026-06-03 rows are a different distribution.
    """
    horizons = _candidate_horizons()
    detect_gap = _import_gap_detector()
    cost_cfg = cost_cfg or _CostConfig()
    slippage_pct_for, classify_symbol, daily_funding_pct = _import_cost_fns()
    empty_counters = dict.fromkeys(horizons, 0)

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(signal_history)").fetchall()}
    if "predicted_close_path" not in existing_cols:
        conn.close()
        return _HCollection(
            [],
            dict(empty_counters),
            dict(empty_counters),
            0,
            0,
            error="predicted_close_path column missing — no 10b data yet.",
        )
    var_select = (
        "var_closes_at_horizons"
        if "var_closes_at_horizons" in existing_cols
        else "NULL AS var_closes_at_horizons"
    )

    # Re-load resolved rows that have a path BLOB, ignoring the persisted
    # gap filter; the per-H gap test below decides inclusion at each H.
    have_path = conn.execute(
        f"""
        SELECT symbol, entry_price, scored_at, predicted_close_path, {var_select}
        FROM signal_history
        WHERE realized_return_at_horizon IS NOT NULL
          AND predicted_close_path IS NOT NULL
          AND scored_at >= ?
        ORDER BY scored_at ASC
        """,
        (since_ms,),
    ).fetchall()
    if not have_path:
        conn.close()
        return _HCollection(
            [],
            dict(empty_counters),
            dict(empty_counters),
            0,
            0,
            error="predicted_close_path empty for all resolved rows in window — no 10b data yet.",
        )

    records: list[_HRecord] = []
    gap_drops: dict[int, int] = dict(empty_counters)
    miss_drops: dict[int, int] = dict(empty_counters)
    rows_with_var = 0
    rows_without_var = 0
    h_max = max(horizons)

    for r in have_path:
        path = _decode_path(r["predicted_close_path"])
        if path is None:
            continue
        scored_at = r["scored_at"]
        entry = r["entry_price"]
        sym = r["symbol"]
        # Pre-D3 USO/UNG/SLV: candles table is now a different instrument from
        # what was traded — no valid join exists.  See comment near
        # ``_PRE_D3_TD_VALUE_THRESHOLD``.
        if _is_pre_d3_etf_row(sym, entry):
            continue
        # Same un-joinable-cohort problem for pre-IG-native metals (2026-06-19):
        # old-scale entry_price joined against rebuilt IG-native candles.
        if _is_pre_ignative_metal_row(sym, entry):
            continue
        ac = _asset_class(sym)

        # Spread is per-asset-class, constant across H (round-trip proxy =
        # SLIPPAGE_PCT × mult, charged once).  Funding is computed per H below
        # since it scales with the holding window.
        spread_pct = (
            slippage_pct_for(sym) * cost_cfg.spread_mult if cost_cfg.include_spread else 0.0
        )
        ig_class = classify_symbol(sym) if cost_cfg.include_funding else None

        # Decode the Pass-2 per-horizon draws once per row.  Layout doc:
        # CandleDB.write_signal_history_batch (LE float32, draws × horizons).
        var_arr: np.ndarray | None = None
        var_blob = r["var_closes_at_horizons"]
        if var_blob:
            flat = np.frombuffer(var_blob, dtype="<f4")
            if flat.size and flat.size % len(horizons) == 0:
                var_arr = flat.reshape(-1, len(horizons))
        if var_arr is not None:
            rows_with_var += 1
        else:
            rows_without_var += 1

        # One window query per row — covers every candidate H by slicing.
        candles = conn.execute(
            """
            SELECT timestamp, close FROM candles
            WHERE symbol = ? AND timestamp > ? AND timestamp <= ?
            ORDER BY timestamp ASC
            """,
            (sym, scored_at, scored_at + h_max * _HOUR_MS),
        ).fetchall()
        if not candles:
            for H in horizons:
                if len(path) >= H:
                    miss_drops[H] += 1
            continue
        timestamps = [int(c[0]) for c in candles]
        closes = [float(c[1]) for c in candles]

        for h_idx, H in enumerate(horizons):
            if len(path) < H:
                continue
            h_end_ms = scored_at + H * _HOUR_MS
            cut = 0
            for cut, ts in enumerate(timestamps, start=1):
                if ts > h_end_ms:
                    cut -= 1
                    break
            slice_ts = timestamps[:cut]
            if not slice_ts:
                miss_drops[H] += 1
                continue
            if detect_gap(slice_ts, h_end_ms):
                gap_drops[H] += 1
                continue
            real_close = closes[cut - 1]
            pred_close = float(path[H - 1])
            pred_ret = (pred_close - entry) / entry
            real_ret = (real_close - entry) / entry
            funding_pct = 0.0
            if ig_class is not None:
                funding_pct = sum(
                    daily_funding_pct(asset_class=ig_class, side="BUY", now_utc=roll)
                    for roll in _window_roll_times(scored_at, h_end_ms)
                )
            records.append(
                _HRecord(
                    symbol=sym,
                    asset_class=ac,
                    horizon=H,
                    scored_at=scored_at,
                    pred_ret=pred_ret,
                    real_ret=real_ret,
                    confidence=_per_h_confidence(var_arr, h_idx, entry, pred_ret),
                    spread_pct=spread_pct,
                    funding_pct=funding_pct,
                )
            )

    conn.close()
    return _HCollection(records, gap_drops, miss_drops, rows_with_var, rows_without_var)


def _section_horizon_sweep(lines: list[str], coll: _HCollection) -> None:
    """RankIC × H × asset class from the shared per-H join."""
    lines.append("\n── 8. Horizon Sweep — RankIC × H (H-aware gap filter) ────")
    horizons = _candidate_horizons()
    if coll.error:
        lines.append(f"  {coll.error}")
        return

    by_class_h: dict[str, dict[int, list[tuple[float, float]]]] = {}
    for rec in coll.records:
        by_class_h.setdefault(rec.asset_class, {}).setdefault(rec.horizon, []).append(
            (rec.pred_ret, rec.real_ret)
        )

    gap_line = " ".join(f"H={H:>3}: {coll.gap_drops[H]:>4}" for H in horizons)
    miss_line = " ".join(f"H={H:>3}: {coll.miss_drops[H]:>4}" for H in horizons)
    if not by_class_h:
        lines.append("  No usable horizon data after per-H gap filtering.")
        lines.append("  Gap-dropped:  " + gap_line)
        lines.append("  Missing-data: " + miss_line)
        return

    lines.append(f"{'Class':<12} " + " ".join(f"H={H:>3}".rjust(10) for H in horizons))
    lines.append("-" * (12 + 11 * len(horizons)))
    for ac in sorted(by_class_h):
        cells = [f"{ac:<12}"]
        for H in horizons:
            pairs = by_class_h[ac].get(H)
            if not pairs:
                cells.append(f"{'-':>10}")
                continue
            ic = _spearman_ic([p for p, _ in pairs], [r for _, r in pairs])
            cells.append(f"{ic:>7.3f}({len(pairs):>3})")
        lines.append(" ".join(cells))
    lines.append("\n  Gap-dropped per H:  " + gap_line)
    lines.append("  Missing-data per H: " + miss_line)

    # Net mean realised return per (class, H) — after spread + funding.  RankIC
    # is a rank stat and barely moves net-of-cost (the per-class cost is ~flat
    # within a cell); the *level* is what funding destroys, and funding grows
    # with H, so this grid — not the RankIC grid above — should drive the
    # per-class H choice.
    net_by_class_h: dict[str, dict[int, list[float]]] = {}
    fund_by_class_h: dict[str, dict[int, list[float]]] = {}
    spread_by_class: dict[str, float] = {}
    for rec in coll.records:
        net_by_class_h.setdefault(rec.asset_class, {}).setdefault(rec.horizon, []).append(
            rec.real_ret_net
        )
        fund_by_class_h.setdefault(rec.asset_class, {}).setdefault(rec.horizon, []).append(
            rec.funding_pct
        )
        spread_by_class[rec.asset_class] = rec.spread_pct  # constant per class

    lines.append("\n  Net mean realised return per (class, H)  [after spread + funding]")
    lines.append(f"{'Class':<12} " + " ".join(f"H={H:>3}".rjust(10) for H in horizons))
    lines.append("-" * (12 + 11 * len(horizons)))
    for ac in sorted(net_by_class_h):
        cells = [f"{ac:<12}"]
        for H in horizons:
            vals = net_by_class_h[ac].get(H)
            cells.append(f"{float(np.mean(vals)):>+9.4f}" if vals else f"{'-':>10}")
        lines.append(" ".join(cells))

    # Cost breakdown (bps) so the spread-vs-funding split is visible: spread is
    # constant per class; funding is the mean over rows (varies by entry weekday
    # and grows with H).
    lines.append("\n  Cost breakdown (bps): spread const per class | funding mean over rows")
    lines.append(
        "  Spread (round-trip): "
        + " | ".join(f"{ac} {spread_by_class[ac] * 1e4:.1f}" for ac in sorted(spread_by_class))
    )
    lines.append(
        f"  Funding by (class, H): {'Class':<8} " + " ".join(f"H={H:>3}".rjust(8) for H in horizons)
    )
    for ac in sorted(fund_by_class_h):
        cells = [f"  {'':<22}{ac:<8}"]
        for H in horizons:
            vals = fund_by_class_h[ac].get(H)
            cells.append(f"{float(np.mean(vals)) * 1e4:>8.1f}" if vals else f"{'-':>8}")
        lines.append(" ".join(cells))

    lines.append(
        "\n  Recommendation: per asset class, pick the H with the highest RankIC AND "
        "positive net mean return."
    )
    lines.append("  Recalibrate min_confidence at the chosen H before adopting it.")


def _section_cross_sectional_rankic(
    lines: list[str], coll: _HCollection, min_symbols: int = 3, min_history: int = 20
) -> None:
    """Cross-sectional RankIC × H — the metric TopK actually lives on.

    At each rerank (``scored_at``) the strategy ranks symbols *against each other*
    and picks the top-k. This measures exactly that: per rerank, Spearman-correlate
    predicted vs realised return *across symbols*, then average over reranks. It is
    distinct from §8, whose per-class RankIC pools symbols **and** time (and so can
    look fine while cross-sectional ranking — picking *which* symbol — is poor; the
    Kronos fine-tune A/B, 2026-06-26, showed exactly that split).

    Two columns: ``raw`` is today's TopK score (``pred_ret``); ``norm`` ranks by a
    per-symbol **causal z-score** (each symbol's pred_ret minus its own past mean,
    over std), which removes the per-symbol calibration bias that wrecks the raw
    cross-sectional ranking. Offline (2026-06-26) this flipped cross-sec RankIC
    from −0.14 to +0.3…+0.5; this section confirms whether that holds on live data.
    """
    lines.append("\n── 8b. Cross-Sectional RankIC × H — TopK ranking (rank symbols per rerank) ──")
    if coll.error:
        lines.append(f"  {coll.error}")
        return

    by_h: dict[int, list[_HRecord]] = {}
    for rec in coll.records:
        by_h.setdefault(rec.horizon, []).append(rec)
    if not by_h:
        lines.append("  No usable cross-sections after per-H gap filtering.")
        return

    # Per-(symbol, H) causal z-score of pred_ret: past-only mean/std, warmup of
    # ``min_history`` reranks. Keyed by id(record).
    norm: dict[int, float] = {}
    for recs in by_h.values():
        by_sym: dict[str, list[_HRecord]] = {}
        for r in recs:
            by_sym.setdefault(r.symbol, []).append(r)
        for rs in by_sym.values():
            rs.sort(key=lambda r: r.scored_at)
            preds = [r.pred_ret for r in rs]
            for i, r in enumerate(rs):
                if i >= min_history:
                    m = float(np.mean(preds[:i]))
                    s = float(np.std(preds[:i]))
                    if s > 0:
                        norm[id(r)] = (r.pred_ret - m) / s

    lines.append(
        f"  (cross-section = symbols at one rerank, ≥{min_symbols} required; "
        f"norm = per-symbol causal z, {min_history}-rerank warmup)"
    )
    lines.append(
        f"{'H':>5} {'raw RankIC':>11} {'norm RankIC':>12} {'n_reranks':>10} {'med_syms':>9}"
    )
    lines.append("-" * 52)
    for H in _candidate_horizons():
        ts_groups: dict[int, list[_HRecord]] = {}
        for r in by_h.get(H, []):
            ts_groups.setdefault(r.scored_at, []).append(r)
        raw_ics: list[float] = []
        norm_ics: list[float] = []
        sizes: list[int] = []
        for rs in ts_groups.values():
            if len(rs) >= min_symbols:
                v = _spearman_ic([r.pred_ret for r in rs], [r.real_ret for r in rs])
                if not np.isnan(v):
                    raw_ics.append(v)
                    sizes.append(len(rs))
            nr = [r for r in rs if id(r) in norm]
            if len(nr) >= min_symbols:
                vn = _spearman_ic([norm[id(r)] for r in nr], [r.real_ret for r in nr])
                if not np.isnan(vn):
                    norm_ics.append(vn)
        raw_s = f"{float(np.mean(raw_ics)):>11.3f}" if raw_ics else f"{'-':>11}"
        norm_s = f"{float(np.mean(norm_ics)):>12.3f}" if norm_ics else f"{'-':>12}"
        med_s = f"{int(np.median(sizes)):>9}" if sizes else f"{'-':>9}"
        lines.append(f"{H:>5} {raw_s} {norm_s} {len(raw_ics):>10} {med_s}")
    lines.append("\n  raw = today's TopK score; norm = per-symbol-z (deployable in select_top_k).")
    lines.append(
        "  If norm ≫ raw and positive, cross-sectional ranking is fixable by normalizing the "
        "score — no fine-tune needed."
    )


def _section_per_h_confidence(lines: list[str], coll: _HCollection) -> None:
    """Confidence decile × hit-rate × RankIC per (asset class, H) (10c-prep).

    Confidence here is recomputed at each H from the Pass-2
    ``var_closes_at_horizons`` draws, answering the 10b caveat directly:
    what ``min_confidence`` keeps gate selectivity constant at the chosen H.
    """
    lines.append("\n── 9. Per-H Confidence Calibration (10c-prep, from Pass-2 draws) ────")
    if coll.error:
        lines.append(f"  {coll.error}")
        return
    lines.append(
        f"  Rows with var_closes_at_horizons: {coll.rows_with_var} used, "
        f"{coll.rows_without_var} skipped (pre-10c-prep, no blob)"
    )
    recs = [r for r in coll.records if r.confidence is not None]
    if not recs:
        lines.append("  No rows carry per-H draws yet — wait for post-deploy reranks.")
        return

    by_class_h: dict[tuple[str, int], list[_HRecord]] = {}
    for rec in recs:
        by_class_h.setdefault((rec.asset_class, rec.horizon), []).append(rec)

    for (ac, H), cell in sorted(by_class_h.items()):
        lines.append(f"\n  {ac} @ H={H}  (N={len(cell)})")
        lines.append(f"  {'Conf bucket':<16} {'N':>5} {'Hit%':>7} {'RankIC':>8}")
        lines.append("  " + "-" * 40)
        confs = np.array([r.confidence for r in cell], dtype=float)
        edges = np.percentile(confs, np.arange(0, 110, 10))
        for i, (lo, hi) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
            last = i == len(edges) - 2
            bucket = [
                r
                for r in cell
                if r.confidence is not None
                and lo <= r.confidence
                and (r.confidence <= hi if last else r.confidence < hi)
            ]
            if not bucket:
                continue
            bp = [r.pred_ret for r in bucket]
            br = [r.real_ret for r in bucket]
            lines.append(
                f"  [{lo:.2f}, {hi:.2f}]  {len(bucket):>5} "
                f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f}"
            )


# Threshold grids swept by section 10.  The live gate values must be members
# so the report can mark the operating point.
_GRID_MIN_CONFIDENCE = (0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_GRID_MIN_PREDICTED_RETURN = (0.001, 0.002, 0.003, 0.005)


def _section_entry_filter_grid(
    lines: list[str],
    coll: _HCollection,
    live_min_confidence: float,
    live_min_predicted_return: float,
) -> None:
    """min_confidence × min_predicted_return sweep at each candidate H.

    Predicted return is computed at H from ``predicted_close_path``;
    confidence at H from the Pass-2 draws.  Print, don't auto-pick — the
    human chooses the cell.
    """
    lines.append("\n── 10. Entry-Filter Grid — conf × return at each H (10c-prep) ───────")
    if coll.error:
        lines.append(f"  {coll.error}")
        return
    lines.append(
        f"  Rows with var_closes_at_horizons: {coll.rows_with_var} used, "
        f"{coll.rows_without_var} skipped (pre-10c-prep, no blob)"
    )
    recs = [r for r in coll.records if r.confidence is not None]
    if not recs:
        lines.append("  No rows carry per-H draws yet — wait for post-deploy reranks.")
        return

    horizons = _candidate_horizons()
    for H in horizons:
        h_recs = [r for r in recs if r.horizon == H]
        if not h_recs:
            continue
        lines.append(f"\n  H={H}  (N={len(h_recs)} row-points; '*' marks the live gate)")
        lines.append(
            f"    {'conf>=':>7} {'ret>=':>7} {'N':>5} {'Hit%':>7} {'RankIC':>8} "
            f"{'mReal':>8} {'mRealNet':>9} {'HitNet%':>8}"
        )
        for conf_min in _GRID_MIN_CONFIDENCE:
            for ret_min in _GRID_MIN_PREDICTED_RETURN:
                live = (
                    abs(conf_min - live_min_confidence) < 1e-9
                    and abs(ret_min - live_min_predicted_return) < 1e-9
                )
                mark = "*" if live else " "
                cell = [
                    r
                    for r in h_recs
                    if r.confidence is not None
                    and r.confidence >= conf_min
                    and r.pred_ret >= ret_min
                ]
                if not cell:
                    lines.append(
                        f"  {mark} {conf_min:>7.2f} {ret_min:>7.3f} {0:>5} "
                        f"{'-':>7} {'-':>8} {'-':>8} {'-':>9} {'-':>8}"
                    )
                    continue
                bp = [r.pred_ret for r in cell]
                br = [r.real_ret for r in cell]
                bn = [r.real_ret_net for r in cell]
                # Net hit-rate: fraction profitable AFTER costs (all entries are
                # LONG with pred_ret ≥ ret_min > 0, so this is "closed up net").
                hit_net = sum(1 for rn in bn if rn >= 0) / len(bn)
                lines.append(
                    f"  {mark} {conf_min:>7.2f} {ret_min:>7.3f} {len(cell):>5} "
                    f"{_hit_rate(bp, br) * 100:>6.1f}% {_spearman_ic(bp, br):>8.3f} "
                    f"{float(np.mean(br)):>8.4f} {float(np.mean(bn)):>+9.4f} {hit_net * 100:>7.1f}%"
                )
    lines.append("\n  mRealNet/HitNet% are after spread + funding — the P&L-relevant columns.")
    lines.append("  Print-only — pick thresholds by hand from the table above.")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

# Two events shifted the signal distribution; rows scored before them are a
# different cohort from today's signals.  Printed in every report header so a
# future reader knows why ``--since 2026-06-03`` is the canonical
# recalibration invocation.
_COHORT_NOTE = (
    "Cohort boundaries: 2026-06-02 eval()-fix | 2026-06-03 EODHD cutover — "
    "recalibration uses --since 2026-06-03"
)


def _parse_since_ms(value: str) -> int:
    """Parse ``--since YYYY-MM-DD`` as UTC midnight; return epoch ms."""
    dt = datetime.strptime(value, "%Y-%m-%d").replace(tzinfo=UTC)
    return int(dt.timestamp() * 1000)


def generate_report(
    db_path: str,
    lookback_days: int,
    include_gap: bool,
    horizon_sweep: bool,
    min_confidence: float,
    min_predicted_return: float,
    max_uncertainty: float,
    since_ms: int | None = None,
    cost_cfg: _CostConfig | None = None,
) -> str:
    now_ms = int(datetime.now(tz=UTC).timestamp() * 1000)
    if since_ms is None:
        since_ms = now_ms - lookback_days * 86_400_000
        window_desc = f"--days {lookback_days}"
    else:
        window_desc = "--since (overrides --days)"
    cutoff_str = datetime.fromtimestamp(since_ms / 1000, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    rows = _load_rows(db_path, since_ms, include_gap=include_gap)
    gap_count = sum(1 for r in rows if r["gap_spanned"])

    lines: list[str] = []
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    lines.append(f"Signal Diagnostics Report — {date_str}")
    lines.append(f"DB: {db_path}  |  Cutoff: {cutoff_str} ({window_desc})")
    lines.append(_COHORT_NOTE)
    if horizon_sweep:
        cost_cfg = cost_cfg or _CostConfig()
        lines.append(
            f"Net-of-cost (sweep sections): {cost_cfg.describe()}.  "
            "Spread reuses ig_margin.SLIPPAGE_PCT (worst-case-slippage proxy → "
            "conservative); FX returns are on mid, equities on last-trade "
            "(slightly over-penalised)."
        )
    lines.append(
        f"Resolved rows: {len(rows)}  "
        f"({'incl.' if include_gap else 'excl.'} gap-spanned; gap rows in set: {gap_count})"
    )
    lines.append("=" * 72)

    if not rows and not horizon_sweep:
        lines.append("\nNo resolved rows yet — wait for horizon_bars hours after first rerank.")
        return "\n".join(lines)

    if rows:
        _section_rankic(rows, lines)
        _section_confidence(rows, lines)
        _section_mfe_buckets(rows, lines)
        _section_entries_only(rows, lines, min_confidence, min_predicted_return, max_uncertainty)
        _section_mfe_mae_accuracy(rows, lines)
        _section_vol_accuracy(rows, lines)
    else:
        # At H=120 a 5-day window crossing any weekend is gap-spanned, so an
        # empty headline cohort is the NORM, not an error (e.g. every
        # post-cutover row the week after a weekend).  The horizon sweep
        # below applies its own per-H gap test and must still run — it is
        # the decision input for the recalibration.
        lines.append(
            "\nNo resolved rows in window after the gap filter — headline sections "
            "skipped (--include-gap-rows opts back in); horizon-sweep sections "
            "below use the per-H gap test and are unaffected."
        )

    if horizon_sweep:
        coll = _collect_h_records(db_path, since_ms, cost_cfg=cost_cfg)
        _section_horizon_sweep(lines, coll)
        _section_cross_sectional_rankic(lines, coll)
        _section_per_h_confidence(lines, coll)
        _section_entry_filter_grid(lines, coll, min_confidence, min_predicted_return)

    if rows:
        _section_appendix(rows, lines)

    lines.append("\n" + "=" * 72)
    lines.append("Note: run `uv run python scripts/signal_diagnostics.py` weekly.")
    lines.append("Do not auto-tune on these numbers — present to human for review.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Kronos signal diagnostics report")
    parser.add_argument(
        "--db",
        default="candles.db",
        help="Path to candles.db (default: candles.db in the working directory)",
    )
    parser.add_argument("--days", type=int, default=30, help="Lookback window in days")
    parser.add_argument(
        "--since",
        default=None,
        metavar="YYYY-MM-DD",
        help="UTC cutoff date; overrides --days when supplied "
        "(--since 2026-06-03 is the canonical post-cutover recalibration window)",
    )
    parser.add_argument(
        "--output-dir", default="journal", help="Directory for output file (default: journal/)"
    )
    parser.add_argument(
        "--include-gap-rows",
        action="store_true",
        help="Include rows where horizon spanned a market closure (default: excluded)",
    )
    parser.add_argument(
        "--horizon-sweep",
        action="store_true",
        help="Add horizon-sweep section: RankIC × H × asset class from logged paths",
    )
    # Defaults mirror the LIVE entry gate (TopKConfig / .env), not the
    # pre-calibration values — section 5's "would-enter" subset is only
    # meaningful if it re-applies the thresholds the bot actually trades with.
    parser.add_argument("--min-confidence", type=float, default=0.80)
    parser.add_argument("--min-predicted-return", type=float, default=0.003)
    parser.add_argument("--max-uncertainty", type=float, default=10.0)
    # Net-of-cost knobs for the --horizon-sweep sections (10c-prep §1).
    parser.add_argument(
        "--cost-spread-mult",
        type=float,
        default=1.0,
        help="Scale the spread term (SLIPPAGE_PCT proxy is conservative; try 0.5 "
        "to approximate the true dealing spread). Default 1.0.",
    )
    parser.add_argument(
        "--no-spread", action="store_true", help="Drop the spread cost from the net-of-cost view"
    )
    parser.add_argument(
        "--no-funding", action="store_true", help="Drop the funding cost from the net-of-cost view"
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: DB not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    report = generate_report(
        args.db,
        args.days,
        include_gap=args.include_gap_rows,
        horizon_sweep=args.horizon_sweep,
        min_confidence=args.min_confidence,
        min_predicted_return=args.min_predicted_return,
        max_uncertainty=args.max_uncertainty,
        since_ms=_parse_since_ms(args.since) if args.since else None,
        cost_cfg=_CostConfig(
            spread_mult=args.cost_spread_mult,
            include_spread=not args.no_spread,
            include_funding=not args.no_funding,
        ),
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out_path = output_dir / f"signal_diagnostics_{date_str}.txt"
    out_path.write_text(report + "\n", encoding="utf-8")

    print(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
