"""Paper backtest of Kronos-driven shorting strategies.

Reads resolved rows from ``signal_history`` (those past their prediction
horizon, so ``realized_return_at_horizon`` is filled) and computes the
P&L of five hypothetical strategies as if every qualifying signal were
opened and held to horizon close.  No stops, no trails, no transaction
costs — this is a *directional-edge* measurement, not a P&L forecast.

Strategies
----------
A.  ``pure_short``        SHORT when ``mean_return ≤ −min_return``.
B.  ``metals_contrarian`` SHORT metals (XAU/USD, XAG/USD) when Kronos says
                          LONG with conviction.  Driven by the May 2026
                          finding that Kronos has a structural UP bias
                          on precious metals while the market trended
                          DOWN.
C.  ``combined_short``    A ∪ B — pure short on non-metals,
                          contrarian short on metals.
D.  ``long_baseline``     LONG when ``mean_return ≥ +min_return``.
                          Mirrors the live bot's selection rule with
                          metals excluded.  Reported alongside the
                          short variants as a sanity anchor.
E.  ``dual_side``         D ∪ A ∪ B — full ambidextrous strategy.

Why this lives in src/bot/analysis (not scripts/)
-------------------------------------------------
The strategy filters and stats math are pure functions and worth
unit-testing.  The ``scripts/short_strategy_backtest.py`` CLI is a thin
argparse/DB-IO wrapper around this module.
"""

from __future__ import annotations

import sqlite3
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass, field

from bot.risk.sectors import sector_for

# ---------------------------------------------------------------------------
# Strategy identifiers — string-keyed so JSON output stays stable
# ---------------------------------------------------------------------------

STRATEGY_A = "A_pure_short"
STRATEGY_B = "B_metals_contrarian"
STRATEGY_C = "C_combined_short"
STRATEGY_D = "D_long_baseline"
STRATEGY_E = "E_dual_side"

# The two symbols excluded from long selection but used by the metals-
# contrarian short strategy.  Kept here (not imported from BotConfig) so
# the backtest stays runnable against any DB without spinning up the full
# bot config — a notebook-friendliness concession.  Pre-EODHD signal_history
# rows keyed "SLV" predate the 2026-06-03 cutover (and its scale fixes) and
# are out of scope for this backtest.
METALS_UNIVERSE: frozenset[str] = frozenset({"XAU/USD", "XAG/USD"})


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Signal:
    """One resolved row from ``signal_history``."""

    scored_at_ms: int
    symbol: str
    mean_return: float
    uncertainty: float
    realized_return: float

    @property
    def sector(self) -> str:
        return sector_for(self.symbol)


@dataclass(frozen=True)
class BacktestConfig:
    """Thresholds applied to every strategy.

    Mirrors the live bot's tradeable-signal filter so the backtest's
    pass/fail boundary lines up with what the strategy actually sees in
    production.  Override on the CLI to test alternate calibrations.
    """

    min_return: float = 0.003
    max_uncertainty: float = 10.0
    metals: frozenset[str] = METALS_UNIVERSE


@dataclass
class Trade:
    """A single hypothetical fill — symbol + P&L proxy."""

    symbol: str
    pnl: float
    sector: str = field(default="")

    def __post_init__(self) -> None:
        if not self.sector:
            object.__setattr__(self, "sector", sector_for(self.symbol))


@dataclass(frozen=True)
class Stats:
    """Summary statistics for a list of trade P&L proxies."""

    n: int
    hit_rate: float
    mean: float
    median: float
    stdev: float
    sharpe_proxy: float
    cum: float
    best: float
    worst: float


@dataclass(frozen=True)
class StrategyResult:
    """A strategy's full backtest output.

    ``trades`` is the list of qualifying signals turned into P&L proxies;
    ``stats`` is the aggregate; ``by_sector`` is the per-sector breakdown.
    """

    name: str
    trades: list[Trade]
    stats: Stats
    by_sector: dict[str, Stats]


# ---------------------------------------------------------------------------
# Stats math
# ---------------------------------------------------------------------------


def compute_stats(pnls: list[float]) -> Stats:
    """Compute the summary block over a P&L list.

    Returns a zero-filled ``Stats`` for an empty list so callers don't need
    to branch.  ``sharpe_proxy`` is mean/stdev (NOT annualised; we don't
    know per-trade duration well enough to annualise meaningfully).
    """
    n = len(pnls)
    if n == 0:
        return Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    hit = sum(1 for x in pnls if x > 0) / n
    mean = statistics.mean(pnls)
    median = statistics.median(pnls)
    stdev = statistics.stdev(pnls) if n > 1 else 0.0
    sharpe = mean / stdev if stdev > 0 else 0.0
    return Stats(
        n=n,
        hit_rate=hit,
        mean=mean,
        median=median,
        stdev=stdev,
        sharpe_proxy=sharpe,
        cum=sum(pnls),
        best=max(pnls),
        worst=min(pnls),
    )


def _by_sector(trades: list[Trade]) -> dict[str, Stats]:
    grouped: dict[str, list[float]] = {}
    for t in trades:
        grouped.setdefault(t.sector, []).append(t.pnl)
    return {sec: compute_stats(pnls) for sec, pnls in sorted(grouped.items())}


# ---------------------------------------------------------------------------
# Strategies — each returns the list of qualifying trades.  Keep the
# filter logic inline (no helper) so the rule is obvious from one read.
# ---------------------------------------------------------------------------


def _result(name: str, trades: list[Trade]) -> StrategyResult:
    return StrategyResult(
        name=name,
        trades=trades,
        stats=compute_stats([t.pnl for t in trades]),
        by_sector=_by_sector(trades),
    )


def apply_pure_short(signals: Iterable[Signal], cfg: BacktestConfig) -> StrategyResult:
    """A — Pure short: any symbol where Kronos predicts down with conviction."""
    trades = [
        Trade(s.symbol, -s.realized_return)
        for s in signals
        if s.mean_return <= -cfg.min_return and s.uncertainty <= cfg.max_uncertainty
    ]
    return _result(STRATEGY_A, trades)


def apply_metals_contrarian(signals: Iterable[Signal], cfg: BacktestConfig) -> StrategyResult:
    """B — Short metals when Kronos says LONG with conviction.

    The 17-day demo data showed Kronos predicts UP on XAU/SLV 62.8 % of
    the time while the market was UP only 15.4 % — strong contrarian
    edge.  Re-evaluate periodically; if metals enter a clean trend the
    bias should reset and this rule should be retired.
    """
    trades = [
        Trade(s.symbol, -s.realized_return)
        for s in signals
        if s.symbol in cfg.metals
        and s.mean_return >= cfg.min_return
        and s.uncertainty <= cfg.max_uncertainty
    ]
    return _result(STRATEGY_B, trades)


def apply_combined_short(signals: Iterable[Signal], cfg: BacktestConfig) -> StrategyResult:
    """C — A on non-metals ∪ B on metals.  Metals never enter via A."""
    trades: list[Trade] = []
    for s in signals:
        is_metal = s.symbol in cfg.metals
        if is_metal:
            if s.mean_return >= cfg.min_return and s.uncertainty <= cfg.max_uncertainty:
                trades.append(Trade(s.symbol, -s.realized_return))
        else:
            if s.mean_return <= -cfg.min_return and s.uncertainty <= cfg.max_uncertainty:
                trades.append(Trade(s.symbol, -s.realized_return))
    return _result(STRATEGY_C, trades)


def apply_long_baseline(signals: Iterable[Signal], cfg: BacktestConfig) -> StrategyResult:
    """D — Long when Kronos says up with conviction, excluding metals.

    Matches the live bot's selection rule (calibrated thresholds +
    ``topk_exclude_from_selection``).  Reported alongside the short
    strategies so the reader can compare directly without rerunning a
    second tool.
    """
    trades = [
        Trade(s.symbol, s.realized_return)
        for s in signals
        if s.symbol not in cfg.metals
        and s.mean_return >= cfg.min_return
        and s.uncertainty <= cfg.max_uncertainty
    ]
    return _result(STRATEGY_D, trades)


def apply_dual_side(signals: Iterable[Signal], cfg: BacktestConfig) -> StrategyResult:
    """E — D (long non-metals) ∪ A (short non-metals) ∪ B (contrarian metals).

    Symmetric ambidextrous strategy: every signal that clears the
    threshold in either direction becomes a trade.  Highest cumulative
    P&L proxy in the May 2026 data but also widest variance.
    """
    trades: list[Trade] = []
    for s in signals:
        if s.uncertainty > cfg.max_uncertainty:
            continue
        is_metal = s.symbol in cfg.metals
        if is_metal:
            if s.mean_return >= cfg.min_return:
                trades.append(Trade(s.symbol, -s.realized_return))
        else:
            if s.mean_return >= cfg.min_return:
                trades.append(Trade(s.symbol, s.realized_return))
            elif s.mean_return <= -cfg.min_return:
                trades.append(Trade(s.symbol, -s.realized_return))
    return _result(STRATEGY_E, trades)


_STRATEGY_FNS = {
    STRATEGY_A: apply_pure_short,
    STRATEGY_B: apply_metals_contrarian,
    STRATEGY_C: apply_combined_short,
    STRATEGY_D: apply_long_baseline,
    STRATEGY_E: apply_dual_side,
}


def run_all(
    signals: Iterable[Signal], cfg: BacktestConfig | None = None
) -> dict[str, StrategyResult]:
    """Run every registered strategy and return results keyed by name."""
    cfg = cfg or BacktestConfig()
    signals_list = list(signals)
    return {name: fn(signals_list, cfg) for name, fn in _STRATEGY_FNS.items()}


# ---------------------------------------------------------------------------
# DB loader
# ---------------------------------------------------------------------------


def load_signals(db_path: str, since_ms: int | None = None) -> list[Signal]:
    """Load resolved signals from ``signal_history``.

    A ``since_ms`` filter trims to the rolling-window view.  Connection
    is read-only — safe to point at the live DB while the bot is
    writing (SQLite WAL handles concurrent readers).
    """
    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        if since_ms is None:
            cur.execute(
                """
                SELECT scored_at, symbol, mean_return, uncertainty,
                       realized_return_at_horizon
                FROM signal_history
                WHERE realized_return_at_horizon IS NOT NULL
                  AND mean_return IS NOT NULL
                  AND uncertainty IS NOT NULL
                """
            )
        else:
            cur.execute(
                """
                SELECT scored_at, symbol, mean_return, uncertainty,
                       realized_return_at_horizon
                FROM signal_history
                WHERE realized_return_at_horizon IS NOT NULL
                  AND mean_return IS NOT NULL
                  AND uncertainty IS NOT NULL
                  AND scored_at >= ?
                """,
                (since_ms,),
            )
        return [
            Signal(
                scored_at_ms=row[0],
                symbol=row[1],
                mean_return=row[2],
                uncertainty=row[3],
                realized_return=row[4],
            )
            for row in cur.fetchall()
        ]
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _fmt_stats(s: Stats) -> str:
    return (
        f"n={s.n:>5}  hit={s.hit_rate:>5.1%}  "
        f"mean={s.mean:>+.5f}  median={s.median:>+.5f}  "
        f"σ={s.stdev:>.5f}  Sharpe-proxy={s.sharpe_proxy:>+.3f}  "
        f"cum={s.cum:>+.4f}  best={s.best:>+.4f}  worst={s.worst:>+.4f}"
    )


def _top_symbols(trades: list[Trade], k: int = 5) -> str:
    if not trades:
        return ""
    counts = Counter(t.symbol for t in trades)
    return ", ".join(f"{sym}={n}" for sym, n in counts.most_common(k))


def format_report(
    results: dict[str, StrategyResult],
    *,
    n_signals: int,
    window_days: int | None,
    cfg: BacktestConfig,
) -> str:
    """Render a human-readable text report.

    Mirrors the structure of the ad-hoc analysis run during the strategy
    sketch — keeps consecutive runs comparable by eye for drift tracking.
    """
    lines: list[str] = []
    window = f"last {window_days} days" if window_days is not None else "full history"
    lines.append("=== Kronos Short-Strategy Paper Backtest ===")
    lines.append(
        f"Window: {window}   min_return={cfg.min_return}   max_uncertainty={cfg.max_uncertainty}"
    )
    lines.append(f"Resolved signals in scope: {n_signals}")
    lines.append("")

    # Top-line table — one row per strategy
    lines.append(
        f"{'strategy':<22}  {'n':>5}  {'hit':>6}  {'mean':>9}  "
        f"{'median':>9}  {'σ':>8}  {'Sharpe':>8}  {'cum':>9}"
    )
    lines.append("-" * 92)
    for name, res in results.items():
        s = res.stats
        lines.append(
            f"{name:<22}  {s.n:>5}  {s.hit_rate:>5.1%}  "
            f"{s.mean:>+9.5f}  {s.median:>+9.5f}  "
            f"{s.stdev:>8.5f}  {s.sharpe_proxy:>+8.3f}  {s.cum:>+9.4f}"
        )
    lines.append("")

    # Per-strategy detail blocks
    for name, res in results.items():
        lines.append(f"--- {name} ---")
        lines.append(f"  {_fmt_stats(res.stats)}")
        if res.trades:
            lines.append(f"  top symbols: {_top_symbols(res.trades)}")
        if res.by_sector:
            lines.append("  per sector:")
            for sector, stats in res.by_sector.items():
                lines.append(
                    f"    {sector:<16} n={stats.n:>4}  "
                    f"hit={stats.hit_rate:>5.1%}  mean={stats.mean:>+.5f}"
                )
        lines.append("")

    return "\n".join(lines)


def report_to_json(
    results: dict[str, StrategyResult],
    *,
    n_signals: int,
    window_days: int | None,
    cfg: BacktestConfig,
) -> dict[str, object]:
    """Render the same data as a JSON-serialisable dict for diff-over-time.

    Snapshot files dropped under ``journal/`` can be diffed across runs
    to see whether a strategy's edge persists, decays, or reverses.
    """
    return {
        "window_days": window_days,
        "config": {
            "min_return": cfg.min_return,
            "max_uncertainty": cfg.max_uncertainty,
        },
        "n_signals_in_scope": n_signals,
        "strategies": {
            name: {
                "n": res.stats.n,
                "hit_rate": res.stats.hit_rate,
                "mean": res.stats.mean,
                "median": res.stats.median,
                "stdev": res.stats.stdev,
                "sharpe_proxy": res.stats.sharpe_proxy,
                "cum": res.stats.cum,
                "best": res.stats.best,
                "worst": res.stats.worst,
                "by_sector": {
                    sec: {
                        "n": s.n,
                        "hit_rate": s.hit_rate,
                        "mean": s.mean,
                        "stdev": s.stdev,
                    }
                    for sec, s in res.by_sector.items()
                },
            }
            for name, res in results.items()
        },
    }
