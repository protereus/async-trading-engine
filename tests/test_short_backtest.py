"""Tests for bot.analysis.short_backtest.

Covers strategy filters, stats math, per-sector breakdown, and the
DB loader against a temporary SQLite file matching the live
signal_history schema.
"""

from __future__ import annotations

import json
import math
import sqlite3
from pathlib import Path

import pytest

from bot.analysis.short_backtest import (
    STRATEGY_A,
    STRATEGY_B,
    STRATEGY_C,
    STRATEGY_D,
    STRATEGY_E,
    BacktestConfig,
    Signal,
    Stats,
    apply_combined_short,
    apply_dual_side,
    apply_long_baseline,
    apply_metals_contrarian,
    apply_pure_short,
    compute_stats,
    format_report,
    load_signals,
    report_to_json,
    run_all,
)

# ---------------------------------------------------------------------------
# Synthetic-signal helpers
# ---------------------------------------------------------------------------


def _sig(
    symbol: str,
    mr: float,
    rz: float,
    unc: float = 1.0,
    *,
    ts: int = 1_700_000_000_000,
) -> Signal:
    return Signal(
        scored_at_ms=ts,
        symbol=symbol,
        mean_return=mr,
        uncertainty=unc,
        realized_return=rz,
    )


# ---------------------------------------------------------------------------
# Stats math
# ---------------------------------------------------------------------------


class TestComputeStats:
    def test_empty_list_returns_zero_stats(self) -> None:
        s = compute_stats([])
        assert s == Stats(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    def test_single_winner(self) -> None:
        s = compute_stats([0.05])
        assert s.n == 1
        assert s.hit_rate == 1.0
        assert s.mean == 0.05
        assert s.median == 0.05
        assert s.stdev == 0.0  # stdev of a single value
        assert s.sharpe_proxy == 0.0  # stdev=0 → safe default
        assert s.best == 0.05
        assert s.worst == 0.05
        assert s.cum == 0.05

    def test_hit_rate_counts_strict_positives(self) -> None:
        """Zero P&L is NOT a hit — the bot eats spread; we only count net wins."""
        s = compute_stats([0.0, 0.01, -0.01, 0.02])
        assert s.n == 4
        assert s.hit_rate == pytest.approx(0.5)

    def test_sharpe_proxy_is_mean_over_stdev(self) -> None:
        pnls = [0.01, 0.02, -0.01, 0.03, -0.005]
        s = compute_stats(pnls)
        import statistics

        assert s.sharpe_proxy == pytest.approx(statistics.mean(pnls) / statistics.stdev(pnls))


# ---------------------------------------------------------------------------
# Strategy filters
# ---------------------------------------------------------------------------


class TestPureShort:
    """Strategy A — short any symbol where Kronos predicts down with conviction."""

    def test_includes_bearish_under_unc_cap(self) -> None:
        cfg = BacktestConfig(min_return=0.003, max_uncertainty=10.0)
        # mr=-0.005 ≤ -0.003 ✓ ; unc=2.0 ≤ 10.0 ✓
        out = apply_pure_short([_sig("USD/JPY", -0.005, +0.01, unc=2.0)], cfg)
        assert out.stats.n == 1
        # We shorted, price went UP → loss = -realized = -0.01
        assert out.trades[0].pnl == pytest.approx(-0.01)

    def test_excludes_signals_above_unc_cap(self) -> None:
        cfg = BacktestConfig(max_uncertainty=10.0)
        out = apply_pure_short([_sig("EUR/USD", -0.01, -0.005, unc=15.0)], cfg)
        assert out.stats.n == 0

    def test_excludes_bullish_signals(self) -> None:
        cfg = BacktestConfig(min_return=0.003)
        out = apply_pure_short([_sig("EUR/USD", +0.005, -0.002)], cfg)
        assert out.stats.n == 0

    def test_pnl_sign_is_inverted_realized(self) -> None:
        """Going SHORT and price FALLS → we win.  P&L proxy = -realized."""
        cfg = BacktestConfig()
        out = apply_pure_short([_sig("USD/JPY", -0.01, -0.02)], cfg)
        assert out.trades[0].pnl == pytest.approx(0.02)


class TestMetalsContrarian:
    """Strategy B — short metals when Kronos says LONG with conviction."""

    def test_includes_bullish_metal(self) -> None:
        cfg = BacktestConfig()
        out = apply_metals_contrarian([_sig("XAU/USD", +0.01, -0.03)], cfg)
        assert out.stats.n == 1
        assert out.trades[0].pnl == pytest.approx(0.03)

    def test_excludes_non_metals_even_when_bullish(self) -> None:
        cfg = BacktestConfig()
        out = apply_metals_contrarian([_sig("USD/JPY", +0.01, +0.02)], cfg)
        assert out.stats.n == 0

    def test_excludes_bearish_metal(self) -> None:
        """If Kronos already says SHORT metals, B is silent — A handles that case."""
        cfg = BacktestConfig()
        out = apply_metals_contrarian([_sig("XAG/USD", -0.01, +0.02)], cfg)
        assert out.stats.n == 0


class TestCombinedShort:
    """Strategy C — non-metals short via A, metals via contrarian B."""

    def test_metal_never_enters_via_short_branch(self) -> None:
        """Even if Kronos predicts SHORT on metals, C must NOT take it via A —
        only the metals-contrarian path is valid for metals."""
        cfg = BacktestConfig()
        # XAU mr=-0.01 ≤ -0.003 would qualify A, but C excludes metals from A
        out = apply_combined_short([_sig("XAU/USD", -0.01, +0.02)], cfg)
        assert out.stats.n == 0

    def test_non_metal_short_enters(self) -> None:
        cfg = BacktestConfig()
        out = apply_combined_short([_sig("USD/JPY", -0.01, -0.005)], cfg)
        assert out.stats.n == 1
        assert out.trades[0].pnl == pytest.approx(0.005)

    def test_metal_long_signal_becomes_contrarian_short(self) -> None:
        cfg = BacktestConfig()
        out = apply_combined_short([_sig("XAU/USD", +0.01, -0.02)], cfg)
        assert out.stats.n == 1
        assert out.trades[0].pnl == pytest.approx(0.02)


class TestLongBaseline:
    """Strategy D — long non-metals when Kronos predicts up.  Live bot's rule."""

    def test_excludes_metals(self) -> None:
        cfg = BacktestConfig()
        out = apply_long_baseline([_sig("XAU/USD", +0.01, +0.02)], cfg)
        assert out.stats.n == 0

    def test_includes_bullish_non_metal(self) -> None:
        cfg = BacktestConfig()
        out = apply_long_baseline([_sig("EUR/USD", +0.005, +0.002)], cfg)
        assert out.stats.n == 1
        assert out.trades[0].pnl == pytest.approx(0.002)


class TestDualSide:
    """Strategy E — D ∪ A ∪ B."""

    def test_unc_cap_applies_universally(self) -> None:
        cfg = BacktestConfig(max_uncertainty=5.0)
        signals = [
            _sig("EUR/USD", +0.01, +0.005, unc=10.0),  # filtered
            _sig("EUR/USD", -0.01, -0.005, unc=10.0),  # filtered
            _sig("XAU/USD", +0.01, -0.02, unc=10.0),  # filtered
        ]
        out = apply_dual_side(signals, cfg)
        assert out.stats.n == 0

    def test_picks_long_short_and_contrarian_metal(self) -> None:
        cfg = BacktestConfig()
        signals = [
            _sig("EUR/USD", +0.01, +0.005),  # long
            _sig("USD/JPY", -0.01, -0.003),  # short
            _sig("XAU/USD", +0.01, -0.02),  # metals contrarian
            _sig("XAG/USD", -0.01, +0.01),  # metals SHORT — never enters (E does B only)
        ]
        out = apply_dual_side(signals, cfg)
        assert out.stats.n == 3
        pnls = sorted(t.pnl for t in out.trades)
        assert pnls == pytest.approx([0.003, 0.005, 0.02])


# ---------------------------------------------------------------------------
# Per-sector breakdown
# ---------------------------------------------------------------------------


class TestSectorBreakdown:
    def test_groups_by_sector_via_sector_for(self) -> None:
        cfg = BacktestConfig()
        signals = [
            _sig("EUR/USD", -0.01, -0.005),  # fx_usd
            _sig("USD/JPY", -0.01, -0.005),  # fx_usd
            _sig("GBP/JPY", -0.01, -0.005),  # fx_gbp_cross
        ]
        out = apply_pure_short(signals, cfg)
        assert set(out.by_sector.keys()) == {"fx_usd", "fx_gbp_cross"}
        assert out.by_sector["fx_usd"].n == 2
        assert out.by_sector["fx_gbp_cross"].n == 1

    def test_per_sector_aggregates_pnl_correctly(self) -> None:
        cfg = BacktestConfig()
        signals = [
            _sig("EUR/USD", -0.01, +0.01),
            _sig("USD/JPY", -0.01, -0.02),
        ]
        out = apply_pure_short(signals, cfg)
        fx = out.by_sector["fx_usd"]
        # P&L proxies were -0.01 (loss) and 0.02 (win); mean = 0.005, hit = 1/2
        assert fx.n == 2
        assert fx.hit_rate == pytest.approx(0.5)
        assert fx.mean == pytest.approx(0.005)


# ---------------------------------------------------------------------------
# run_all + report rendering
# ---------------------------------------------------------------------------


class TestRunAll:
    def test_emits_every_strategy(self) -> None:
        out = run_all([])
        assert set(out.keys()) == {
            STRATEGY_A,
            STRATEGY_B,
            STRATEGY_C,
            STRATEGY_D,
            STRATEGY_E,
        }


class TestReports:
    def test_text_report_contains_all_strategy_names(self) -> None:
        results = run_all(
            [
                _sig("EUR/USD", +0.005, +0.002),
                _sig("XAU/USD", +0.005, -0.02),
            ]
        )
        text = format_report(results, n_signals=2, window_days=30, cfg=BacktestConfig())
        for name in (STRATEGY_A, STRATEGY_B, STRATEGY_C, STRATEGY_D, STRATEGY_E):
            assert name in text
        assert "Window: last 30 days" in text

    def test_text_report_full_history_label(self) -> None:
        text = format_report({}, n_signals=0, window_days=None, cfg=BacktestConfig())
        assert "Window: full history" in text

    def test_json_payload_is_serialisable_and_typed(self) -> None:
        results = run_all([_sig("EUR/USD", +0.005, +0.002)])
        payload = report_to_json(results, n_signals=1, window_days=30, cfg=BacktestConfig())
        # JSON-roundtrips cleanly (no NaN, no datetimes)
        serialised = json.dumps(payload)
        round_tripped = json.loads(serialised)
        assert round_tripped["window_days"] == 30
        assert round_tripped["config"]["min_return"] == 0.003
        for name in (STRATEGY_A, STRATEGY_D):
            entry = round_tripped["strategies"][name]
            # Sanity: required keys present + numeric
            assert isinstance(entry["n"], int)
            assert isinstance(entry["mean"], (int, float))
            assert not math.isnan(entry["mean"])


# ---------------------------------------------------------------------------
# DB loader — temp SQLite with the live signal_history schema
# ---------------------------------------------------------------------------


@pytest.fixture
def tiny_db(tmp_path: Path) -> str:
    """Build a minimal signal_history table for the loader to read."""
    path = str(tmp_path / "tiny.db")
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE signal_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            scored_at INTEGER NOT NULL,
            symbol TEXT NOT NULL,
            horizon_bars INTEGER NOT NULL,
            mean_return REAL,
            direction_confidence REAL,
            uncertainty REAL,
            predicted_mfe_pct REAL,
            predicted_mae_pct REAL,
            predicted_volatility REAL,
            monotonicity REAL,
            entry_price REAL NOT NULL,
            realized_return_at_horizon REAL,
            realized_max_high_pct REAL,
            realized_min_low_pct REAL
        )
        """
    )
    rows = [
        # ts, symbol, mr, unc, realized
        (1_700_000_000_000, "EUR/USD", 0.005, 1.5, 0.002),
        (1_700_000_001_000, "USD/JPY", -0.004, 2.0, -0.006),
        (1_700_000_002_000, "XAU/USD", 0.008, 1.0, -0.03),
        # Row deliberately NULL on realized — must be skipped by loader
        (1_700_000_003_000, "GBP/USD", 0.005, 1.0, None),
    ]
    for ts, sym, mr, unc, rz in rows:
        conn.execute(
            """
            INSERT INTO signal_history
                (scored_at, symbol, horizon_bars, mean_return, uncertainty,
                 entry_price, realized_return_at_horizon)
            VALUES (?, ?, 120, ?, ?, 1.0, ?)
            """,
            (ts, sym, mr, unc, rz),
        )
    conn.commit()
    conn.close()
    return path


class TestLoader:
    def test_load_signals_filters_unresolved(self, tiny_db: str) -> None:
        signals = load_signals(tiny_db)
        assert len(signals) == 3  # GBP/USD with NULL realized excluded
        symbols = {s.symbol for s in signals}
        assert "GBP/USD" not in symbols

    def test_load_signals_window_filter(self, tiny_db: str) -> None:
        # since_ms after the first two rows → only XAU/USD survives
        signals = load_signals(tiny_db, since_ms=1_700_000_002_000)
        assert len(signals) == 1
        assert signals[0].symbol == "XAU/USD"

    def test_end_to_end_run_against_db(self, tiny_db: str) -> None:
        signals = load_signals(tiny_db)
        results = run_all(signals)
        # XAU was bullish-Kronos, realized DOWN → metals contrarian wins +0.03
        b = results[STRATEGY_B]
        assert b.stats.n == 1
        assert b.trades[0].pnl == pytest.approx(0.03)
        # USD/JPY was bearish-Kronos, realized DOWN → pure short wins +0.006
        a = results[STRATEGY_A]
        assert a.stats.n == 1
        assert a.trades[0].pnl == pytest.approx(0.006)
