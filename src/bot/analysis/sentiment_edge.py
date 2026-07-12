"""Sentiment-edge measurement harness — analysis layer.

Reads resolved ``signal_history`` rows and partitions them by the sentiment
captured at signal time, then computes hit rate / mean realized return /
Sharpe-proxy per partition.  The goal is to decide — from our own data —
whether enabling the sentiment gate would carry edge before we flip
``sentiment_gate_enabled``.

The harness tests both *readings*:

* **momentum** — gate ADMITS when sentiment agrees with Kronos direction
  (LONG signal + bullish sentiment, SHORT signal + bearish sentiment).
  This is the current design assumption.
* **contrarian** — gate ADMITS when sentiment OPPOSES Kronos direction.
  Forex sentiment tools (CFTC COT, retail-positioning indicators) treat
  positioning as contrarian on liquid pairs; the same may hold for our
  USD-major / metals / index-ETF universe.

The partition is three-way:

* **AGREE** — sentiment direction matches the reading.
* **DISAGREE** — sentiment direction opposes the reading.
* **ABSENT** — ``sentiment_agent_coverage == 0`` or ``sentiment_score IS NULL``;
  always reported separately so the reader can see how much data the
  partial-overlay days produced.

For each (asset class, magnitude bucket, partition, reading) cell we
compute ``Stats`` from ``bot.analysis.short_backtest`` so the output is
directly comparable to the shorting-edge report's numbers.

Cells with ``n < MIN_CELL_N`` are flagged in the text report — small-sample
hit rates are noise, not signal.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from bot.risk.sectors import sector_for

# ---------------------------------------------------------------------------
# Asset-class mapping
#
# The production ``sector_for`` resolver returns a granular sector
# (``fx_usd`` / ``fx_eur_cross`` / …).  The user-facing decision groups
# every FX pair under ``fx_major``; metals and equity_index pass through.
# Anything else (energy, fx_jpy_cross when there's no JPY position…) goes
# to ``other`` so the harness doesn't pretend to have edge in places we
# never trade.
# ---------------------------------------------------------------------------
_FX_SECTORS: frozenset[str] = frozenset({"fx_usd", "fx_eur_cross", "fx_gbp_cross", "fx_jpy_cross"})


def asset_class_for(symbol: str) -> str:
    """Collapse the production sector into the three buckets the
    measurement harness reports: ``fx_major`` / ``metals`` / ``equity_index``
    / ``other``."""
    sector = sector_for(symbol)
    if sector in _FX_SECTORS:
        return "fx_major"
    if sector in ("metals", "equity_index"):
        return sector
    return "other"


_ASSET_CLASSES_REPORTED: tuple[str, ...] = ("fx_major", "metals", "equity_index", "other")

# Sentiment magnitude buckets (absolute value of the sentiment score).
# Boundaries are intentionally aligned to the ±0.3 thresholds the live
# gate currently uses, plus a "strong" bucket so the report can surface
# any separation that only appears at extremes.
_MAG_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("weak", 0.0, 0.15),
    ("moderate", 0.15, 0.30),
    ("strong", 0.30, 0.60),
    ("extreme", 0.60, 1.01),  # 1.01 so |sentiment| == 1.0 falls in extreme
)


# Minimum cell size before we trust hit-rate / mean numbers.  Anything
# below this gets a ⚠ marker in the text report and is excluded from
# directional summaries.
MIN_CELL_N = 30


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SentimentSignal:
    """One resolved ``signal_history`` row with its sentiment context."""

    scored_at_ms: int
    symbol: str
    mean_return: float  # Kronos prediction
    realized_return: float  # actual move at horizon
    sentiment_score: float | None
    sentiment_agent_coverage: int  # 0..6

    @property
    def asset_class(self) -> str:
        return asset_class_for(self.symbol)

    @property
    def kronos_direction(self) -> int:
        """+1 for LONG signal, -1 for SHORT, 0 for flat (treated as
        directionless and excluded from the partition)."""
        if self.mean_return > 0:
            return 1
        if self.mean_return < 0:
            return -1
        return 0

    @property
    def sentiment_direction(self) -> int | None:
        if self.sentiment_score is None or self.sentiment_agent_coverage == 0:
            return None
        if self.sentiment_score > 0:
            return 1
        if self.sentiment_score < 0:
            return -1
        return 0

    def magnitude_bucket(self) -> str | None:
        """Bucket the absolute sentiment magnitude.  Returns None when
        the row has no usable sentiment (routed to ABSENT)."""
        s = self.sentiment_score
        if s is None or self.sentiment_agent_coverage == 0:
            return None
        a = abs(s)
        for name, lo, hi in _MAG_BUCKETS:
            if lo <= a < hi:
                return name
        return _MAG_BUCKETS[-1][0]  # |s| == upper bound


@dataclass(frozen=True)
class Stats:
    n: int
    hit_rate: float
    mean_pnl: float
    median_pnl: float
    stdev: float
    sharpe_proxy: float


def _stats(pnls: list[float]) -> Stats:
    n = len(pnls)
    if n == 0:
        return Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0)
    hit = sum(1 for x in pnls if x > 0) / n
    mean = statistics.mean(pnls)
    median = statistics.median(pnls)
    stdev = statistics.stdev(pnls) if n > 1 else 0.0
    sharpe = mean / stdev if stdev > 0 else 0.0
    return Stats(
        n=n,
        hit_rate=hit,
        mean_pnl=mean,
        median_pnl=median,
        stdev=stdev,
        sharpe_proxy=sharpe,
    )


@dataclass(frozen=True)
class CellResult:
    """Stats for one (asset_class, reading, partition, mag_bucket) cell."""

    asset_class: str
    reading: str  # "momentum" | "contrarian"
    partition: str  # "agree" | "disagree" | "absent"
    mag_bucket: str | None  # None for absent
    stats: Stats


@dataclass(frozen=True)
class HarnessResult:
    """Top-level result.  ``cells`` is dense over the cross of dimensions,
    so the text report can iterate it without re-grouping."""

    n_signals_total: int
    n_signals_with_sentiment: int
    cells: list[CellResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Partition logic
# ---------------------------------------------------------------------------


def _signal_pnl(s: SentimentSignal, reading: str) -> float:
    """The P&L proxy that the *reading* would attribute to *s*.

    Under both readings the trade is Kronos-directional (we go LONG if
    ``mean_return > 0``, SHORT if < 0).  The sentiment partition only
    decides whether that trade was *admitted* by the gate, not the sign
    of the realised return — that comes from the candle itself.

    For LONG: pnl = realized_return.
    For SHORT: pnl = -realized_return.
    """
    if s.kronos_direction == 1:
        return s.realized_return
    if s.kronos_direction == -1:
        return -s.realized_return
    return 0.0


def partition_signal(s: SentimentSignal, reading: str) -> str:
    """Route a signal to ``agree`` / ``disagree`` / ``absent`` under
    *reading*.

    momentum: AGREE when sentiment direction matches Kronos direction.
    contrarian: AGREE when sentiment direction OPPOSES Kronos direction.
    """
    if reading not in ("momentum", "contrarian"):
        raise ValueError(f"unknown reading {reading!r}")
    sent_dir = s.sentiment_direction
    if sent_dir is None or s.kronos_direction == 0:
        return "absent"
    if reading == "momentum":
        return "agree" if sent_dir == s.kronos_direction else "disagree"
    # contrarian
    return "agree" if sent_dir != s.kronos_direction else "disagree"


# ---------------------------------------------------------------------------
# Top-level computation
# ---------------------------------------------------------------------------


def compute_harness(signals: Iterable[SentimentSignal]) -> HarnessResult:
    """Run the harness over *signals*.  Returns a dense ``HarnessResult``
    keyed by (asset_class, reading, partition, magnitude bucket)."""
    sigs = list(signals)
    n_total = len(sigs)
    n_with_sent = sum(
        1 for s in sigs if s.sentiment_agent_coverage > 0 and s.sentiment_score is not None
    )

    cells: list[CellResult] = []
    for ac in _ASSET_CLASSES_REPORTED:
        ac_sigs = [s for s in sigs if s.asset_class == ac]
        for reading in ("momentum", "contrarian"):
            # ABSENT cell (no magnitude bucket — sentiment is missing).
            absent_pnls = [
                _signal_pnl(s, reading) for s in ac_sigs if partition_signal(s, reading) == "absent"
            ]
            cells.append(
                CellResult(
                    asset_class=ac,
                    reading=reading,
                    partition="absent",
                    mag_bucket=None,
                    stats=_stats(absent_pnls),
                )
            )
            # AGREE / DISAGREE cells, bucketed by sentiment magnitude.
            for partition in ("agree", "disagree"):
                for bucket_name, _lo, _hi in _MAG_BUCKETS:
                    pnls = [
                        _signal_pnl(s, reading)
                        for s in ac_sigs
                        if partition_signal(s, reading) == partition
                        and s.magnitude_bucket() == bucket_name
                    ]
                    cells.append(
                        CellResult(
                            asset_class=ac,
                            reading=reading,
                            partition=partition,
                            mag_bucket=bucket_name,
                            stats=_stats(pnls),
                        )
                    )
    return HarnessResult(
        n_signals_total=n_total,
        n_signals_with_sentiment=n_with_sent,
        cells=cells,
    )


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------


def load_signals(db_path: str, since_ms: int | None = None) -> list[SentimentSignal]:
    """Pull every resolved signal_history row, optionally filtered by
    ``scored_at >= since_ms``.  Only rows with ``realized_return_at_horizon``
    set are returned — pending rows aren't useful to the harness."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        sql = (
            "SELECT scored_at, symbol, mean_return, "
            "       realized_return_at_horizon, sentiment_score, "
            "       sentiment_agent_coverage "
            "FROM signal_history "
            "WHERE realized_return_at_horizon IS NOT NULL "
            "  AND mean_return IS NOT NULL"
        )
        params: tuple[Any, ...] = ()
        if since_ms is not None:
            sql += " AND scored_at >= ?"
            params = (since_ms,)
        sql += " ORDER BY scored_at ASC"
        rows = conn.execute(sql, params).fetchall()
    finally:
        conn.close()
    return [
        SentimentSignal(
            scored_at_ms=int(r["scored_at"]),
            symbol=str(r["symbol"]),
            mean_return=float(r["mean_return"]),
            realized_return=float(r["realized_return_at_horizon"]),
            sentiment_score=(
                float(r["sentiment_score"]) if r["sentiment_score"] is not None else None
            ),
            sentiment_agent_coverage=int(r["sentiment_agent_coverage"] or 0),
        )
        for r in rows
    ]


# ---------------------------------------------------------------------------
# Reports
# ---------------------------------------------------------------------------


def _fmt_stats(s: Stats) -> str:
    if s.n == 0:
        return "n=0  (no data)"
    flag = " ⚠" if s.n < MIN_CELL_N else ""
    return (
        f"n={s.n:>4}{flag}  hit={s.hit_rate * 100:5.1f}%  "
        f"mean={s.mean_pnl * 100:+6.2f}%  med={s.median_pnl * 100:+6.2f}%  "
        f"σ={s.stdev * 100:5.2f}%  sharpe≈{s.sharpe_proxy:+.2f}"
    )


def format_report(result: HarnessResult, *, since_label: str | None = None) -> str:
    """Human-readable summary.  Cells with ``n < MIN_CELL_N`` are flagged
    with a ⚠ so the reader doesn't anchor on noise.

    The report is organised so the eye can compare AGREE vs DISAGREE
    within one asset-class / reading / magnitude row.  ABSENT is the
    no-sentiment baseline — its hit / mean / sharpe is what the gate is
    competing against."""
    lines: list[str] = []
    cov_pct = (
        100.0 * result.n_signals_with_sentiment / result.n_signals_total
        if result.n_signals_total
        else 0.0
    )
    lines.append("=" * 84)
    lines.append("Sentiment-edge harness — measurement-only (does NOT change gate state)")
    if since_label is not None:
        lines.append(f"Window: {since_label}")
    lines.append(
        f"Resolved signals: {result.n_signals_total}  "
        f"(with sentiment coverage > 0: {result.n_signals_with_sentiment}, "
        f"{cov_pct:.1f}%)"
    )
    lines.append(f"Small-cell warn threshold: n < {MIN_CELL_N}  (flagged ⚠)")
    lines.append("=" * 84)

    # Group cells for printing.
    by_key: dict[tuple[str, str], list[CellResult]] = {}
    for c in result.cells:
        by_key.setdefault((c.asset_class, c.reading), []).append(c)

    for ac in _ASSET_CLASSES_REPORTED:
        for reading in ("momentum", "contrarian"):
            cells = by_key.get((ac, reading), [])
            absent = next((c for c in cells if c.partition == "absent"), None)
            lines.append("")
            lines.append(f"[{ac}]  reading={reading}")
            if absent is not None:
                lines.append(f"  absent      {_fmt_stats(absent.stats)}")
            for bucket_name, _lo, _hi in _MAG_BUCKETS:
                a = next(
                    (c for c in cells if c.partition == "agree" and c.mag_bucket == bucket_name),
                    None,
                )
                d = next(
                    (c for c in cells if c.partition == "disagree" and c.mag_bucket == bucket_name),
                    None,
                )
                if a is None and d is None:
                    continue
                a_str = _fmt_stats(a.stats) if a is not None else "n=0"
                d_str = _fmt_stats(d.stats) if d is not None else "n=0"
                lines.append(f"  |s| {bucket_name:9s}")
                lines.append(f"    agree     {a_str}")
                lines.append(f"    disagree  {d_str}")

    lines.append("")
    lines.append("=" * 84)
    lines.append("How to read this:")
    lines.append("  * If AGREE outperforms ABSENT and DISAGREE underperforms ABSENT — at")
    lines.append("    sample sizes above the warn threshold and consistently across asset")
    lines.append("    classes — the gate is doing useful work under that reading.")
    lines.append("  * If the separation only appears in the 'strong'/'extreme' buckets,")
    lines.append("    the ±0.30 threshold is approximately right but the boundary may want")
    lines.append("    moving.  If separation is absent at every magnitude, the gate is")
    lines.append("    noise and should NOT be enabled.")
    lines.append("  * Compare the momentum and contrarian readings side-by-side.  Liquid")
    lines.append("    FX often shows contrarian edge on positioning data; spot-check that")
    lines.append("    against our own signals before flipping the sign.")
    lines.append("=" * 84)
    return "\n".join(lines)


def report_to_json(result: HarnessResult, *, since_label: str | None = None) -> dict[str, Any]:
    """Machine-parseable mirror of ``format_report``.  Keys mirror the
    text-report layout so diffs across weeks are straightforward."""
    return {
        "window": since_label,
        "n_signals_total": result.n_signals_total,
        "n_signals_with_sentiment": result.n_signals_with_sentiment,
        "min_cell_n": MIN_CELL_N,
        "cells": [
            {
                "asset_class": c.asset_class,
                "reading": c.reading,
                "partition": c.partition,
                "magnitude_bucket": c.mag_bucket,
                "n": c.stats.n,
                "hit_rate": c.stats.hit_rate,
                "mean_pnl": c.stats.mean_pnl,
                "median_pnl": c.stats.median_pnl,
                "stdev": c.stats.stdev,
                "sharpe_proxy": c.stats.sharpe_proxy,
                "small_sample": c.stats.n < MIN_CELL_N,
            }
            for c in result.cells
        ],
    }
