"""Tests for scripts/signal_diagnostics.py.

The script is not a package module — load it via importlib so the tests
exercise exactly what ``uv run python scripts/signal_diagnostics.py`` runs.

Covers:
  - EODHD-first asset classification + legacy-set fallback (Part A)
  - ``--since YYYY-MM-DD`` parsing and row-cutoff semantics (Part A)
  - synthetic-DB smoke test: report generates with every section (Part A)
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from bot.core.models import Candle
from bot.data.candle_db import CandleDB

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "signal_diagnostics.py"
_spec = importlib.util.spec_from_file_location("signal_diagnostics", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
sigdiag = importlib.util.module_from_spec(_spec)
# Must be registered before exec_module: the script defines dataclasses under
# ``from __future__ import annotations``, and dataclass processing resolves
# the defining module through sys.modules.
sys.modules["signal_diagnostics"] = sigdiag
_spec.loader.exec_module(sigdiag)

_HOUR_MS = 3_600_000


# ---------------------------------------------------------------------------
# Asset classification (EODHD-first, legacy fallback)
# ---------------------------------------------------------------------------


class TestAssetClass:
    def test_eodhd_forex(self) -> None:
        assert sigdiag._asset_class("EUR/USD") == "forex"
        assert sigdiag._asset_class("AUD/JPY") == "forex"

    def test_eodhd_metals_including_xag(self) -> None:
        # XAG/USD exists only in the EODHD universe — it was unclassified
        # ("other") under the retired TwelveData sets.
        assert sigdiag._asset_class("XAU/USD") == "metal"
        assert sigdiag._asset_class("XAG/USD") == "metal"

    def test_eodhd_us_equities(self) -> None:
        # EODHD_UNIVERSE carries asset_class="equity"; the diagnostics label
        # is "us_equity".  All 14 names must classify.
        from bot.data.eodhd_symbols import EODHD_UNIVERSE

        equities = [k for k, s in EODHD_UNIVERSE.items() if s.asset_class == "equity"]
        assert len(equities) == 14
        for sym in equities:
            assert sigdiag._asset_class(sym) == "us_equity"

    def test_legacy_fallback_for_pre_cutover_rows(self) -> None:
        # Retired TD-era symbols still classify so pre-cutover signal_history
        # rows land in their historical buckets.
        assert sigdiag._asset_class("SPY") == "index/etf"
        assert sigdiag._asset_class("USO") == "index/etf"
        assert sigdiag._asset_class("SLV") == "metal"
        assert sigdiag._asset_class("EUR/CAD") == "forex"  # legacy-only FX cross

    def test_unknown_symbol_is_other(self) -> None:
        assert sigdiag._asset_class("ZZZ") == "other"

    def test_live_universe_class_counts(self) -> None:
        # Expected live split: forex 12, metal 2, us_equity 14.
        from bot.data.eodhd_symbols import EODHD_UNIVERSE

        counts: dict[str, int] = {}
        for sym in EODHD_UNIVERSE:
            counts[sigdiag._asset_class(sym)] = counts.get(sigdiag._asset_class(sym), 0) + 1
        assert counts == {"forex": 12, "metal": 2, "us_equity": 14}


class TestPreIgnativeMetalGuard:
    """The 2026-06-19 IG-native metals cutover left pre-cutover signal_history
    rows with old EODHD ETF-proxy entry_price that no longer joins the rebuilt
    IG-native candles — a phantom ~10x (XAU) / ~100x (XAG) return that poisoned
    the §8/§9/§10 metal cells.  These rows must be dropped, like pre-D3 ETFs."""

    def test_pre_cutover_metals_flagged(self) -> None:
        # Old EODHD ETF-proxy scale: XAU ~400, XAG ~62.
        assert sigdiag._is_pre_ignative_metal_row("XAU/USD", 387.5)
        assert sigdiag._is_pre_ignative_metal_row("XAG/USD", 62.0)

    def test_post_cutover_metals_kept(self) -> None:
        # IG-native $/oz scale: XAU ~4150, XAG ~6450 — both above threshold.
        assert not sigdiag._is_pre_ignative_metal_row("XAU/USD", 4151.1)
        assert not sigdiag._is_pre_ignative_metal_row("XAG/USD", 6454.9)
        # Boundary: entry == threshold is NOT flagged (strict <).
        assert not sigdiag._is_pre_ignative_metal_row("XAU/USD", 2000.0)

    def test_non_metal_symbols_never_flagged(self) -> None:
        # Guard is metals-only; forex/equity entry_prices sit far below the
        # metal thresholds but must pass through untouched.
        assert not sigdiag._is_pre_ignative_metal_row("EUR/USD", 1.08)
        assert not sigdiag._is_pre_ignative_metal_row("AAPL", 195.0)


# ---------------------------------------------------------------------------
# --since parsing
# ---------------------------------------------------------------------------


class TestParseSince:
    def test_parses_utc_midnight(self) -> None:
        expected = int(datetime(2026, 6, 3, tzinfo=UTC).timestamp() * 1000)
        assert sigdiag._parse_since_ms("2026-06-03") == expected

    def test_invalid_format_raises(self) -> None:
        with pytest.raises(ValueError):
            sigdiag._parse_since_ms("03/06/2026")

    def test_since_overrides_days_in_header(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)  # empty schema is enough for the header
        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=30,
            include_gap=False,
            horizon_sweep=False,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
            since_ms=sigdiag._parse_since_ms("2026-06-03"),
        )
        assert "2026-06-03 00:00 UTC" in report
        assert "--since (overrides --days)" in report
        # The fixed cohort note must always appear.
        assert "2026-06-02 eval()-fix" in report
        assert "2026-06-03 EODHD cutover" in report


# ---------------------------------------------------------------------------
# Synthetic-DB smoke test
# ---------------------------------------------------------------------------


def _make_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "candles.db"
    db = CandleDB(str(db_path))
    db.init_db()
    db.close()
    return db_path


def _seed_resolved_rows(db_path: Path, base_ts: int) -> None:
    """Six resolved rows each for EUR/USD (forex) and F (us_equity).

    Hourly candles with no gaps so the resolver marks gap_spanned=0 and the
    rows survive the default exclude-gap filter.
    """
    db = CandleDB(str(db_path))
    db.init_db()
    n_candles = 80
    for sym, px in (("EUR/USD", 1.08), ("F", 16.0)):
        db.insert_candles(
            [
                Candle(
                    symbol=sym,
                    timestamp=base_ts + i * _HOUR_MS,
                    open=px,
                    high=px * (1.001 + 0.0001 * (i % 7)),
                    low=px * 0.999,
                    close=px * (1.0 + 0.0002 * ((i * 13) % 11)),
                    volume=100.0,
                    is_confirmed=True,
                )
                for i in range(n_candles)
            ]
        )
    rows: list[dict[str, object]] = []
    for sym, px in (("EUR/USD", 1.08), ("F", 16.0)):
        # Rising predicted path from the entry price so per-H predicted
        # returns are positive and small (0 → 1 % across 120 bars).
        path_blob = np.asarray(px * (1.0 + np.linspace(0.0, 0.01, 120)), dtype="<f4").tobytes()
        # Only EUR/USD rows carry the 10c-prep Pass-2 blob — F rows exercise
        # the "skipped (no blob)" degradation path.  4 draws × 6 horizons,
        # all above entry → per-H confidence 1.0 on a positive prediction.
        var_blob = np.full((4, 6), px * 1.001, dtype="<f4").tobytes() if sym == "EUR/USD" else None
        for i in range(6):
            rows.append(
                {
                    "scored_at": base_ts + i * _HOUR_MS,
                    "symbol": sym,
                    "horizon_bars": 2,
                    "mean_return": 0.001 * (i + 1) * (1 if sym == "F" else -1),
                    "direction_confidence": 0.70 + 0.04 * i,
                    "uncertainty": 0.5 + 0.1 * i,
                    "entry_price": px,
                    "predicted_mfe_pct": 0.002 + 0.001 * i,
                    "predicted_mae_pct": 0.001,
                    "predicted_volatility": 0.002,
                    "monotonicity": 0.5,
                    "predicted_close_path": path_blob,
                    "var_closes_at_horizons": var_blob,
                }
            )
    db.write_signal_history_batch(rows)
    # Horizons (2 bars) are all in the past relative to "now" → resolve all.
    now_ms = base_ts + (n_candles + 10) * _HOUR_MS
    resolved = db.resolve_signal_history(now_ms)
    assert resolved == 12
    db.close()


class TestCrossSectionalRankIC:
    """Unit tests for §8b — cross-sectional RankIC (rank symbols per rerank)."""

    @staticmethod
    def _coll(recs: list) -> object:
        return sigdiag._HCollection(recs, {}, {}, 0, 0)

    @staticmethod
    def _rec(sym: str, ts: int, h: int, pred: float, real: float) -> object:
        return sigdiag._HRecord(
            symbol=sym,
            asset_class="forex",
            horizon=h,
            scored_at=ts,
            pred_ret=pred,
            real_ret=real,
            confidence=None,
        )

    def test_perfect_cross_sectional_ranking_is_one(self) -> None:
        h = sigdiag._candidate_horizons()[0]
        syms = ("EUR/USD", "GBP/USD", "USD/JPY")
        recs = [
            self._rec(s, ts, h, float(i) / 1000, float(i) / 1000)  # pred order == real order
            for ts in (1_000, 2_000)
            for i, s in enumerate(syms)
        ]
        lines: list[str] = []
        sigdiag._section_cross_sectional_rankic(lines, self._coll(recs))
        text = "\n".join(lines)
        assert "8b. Cross-Sectional RankIC" in text
        row = next(line for line in lines if line.split()[:1] == [str(h)])
        # cols: H | raw RankIC | norm RankIC | n_reranks | med_syms
        assert row.split()[1] == "1.000"  # raw perfect ranking
        assert row.split()[3] == "2"  # two reranks contributed
        assert row.split()[2] == "-"  # norm needs the 20-rerank warmup → none here

    def test_norm_recovers_ranking_that_raw_bias_breaks(self) -> None:
        # 3 symbols with strong opposite biases; the skill (and realised return)
        # rotates each rerank. raw ranking is bias-dominated (wrong); per-symbol
        # causal z-norm removes the bias and recovers the skill order.
        h = sigdiag._candidate_horizons()[0]
        syms = ("EUR/USD", "GBP/USD", "USD/JPY")
        bias = {"EUR/USD": 0.10, "GBP/USD": 0.0, "USD/JPY": -0.10}
        recs = []
        for ts in range(25):  # > 20-rerank warmup so norm engages
            order = [syms[(ts + j) % 3] for j in range(3)]  # rotate the skill order
            skill = {order[0]: 0.03, order[1]: 0.02, order[2]: 0.01}
            for sym in syms:
                recs.append(self._rec(sym, ts, h, bias[sym] + skill[sym], skill[sym]))
        lines: list[str] = []
        sigdiag._section_cross_sectional_rankic(lines, self._coll(recs))
        row = next(line for line in lines if line.split()[:1] == [str(h)])
        raw, norm = float(row.split()[1]), float(row.split()[2])
        assert norm > 0.5  # normalization recovers the skill ranking
        assert norm > raw  # and beats the bias-dominated raw ranking

    def test_thin_cross_sections_below_min_symbols_skipped(self) -> None:
        h = sigdiag._candidate_horizons()[0]
        # Only 2 symbols at the rerank — below min_symbols=3 → no RankIC computed.
        recs = [self._rec(s, 1_000, h, 0.001, 0.001) for s in ("EUR/USD", "GBP/USD")]
        lines: list[str] = []
        sigdiag._section_cross_sectional_rankic(lines, self._coll(recs))
        row = next(line for line in lines if line.split()[:1] == [str(h)])
        assert row.split()[1] == "-"  # no usable cross-section at this H


class TestReportSmoke:
    def test_all_sections_present(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)

        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=36500,  # effectively "everything"
            include_gap=False,
            horizon_sweep=True,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
        )
        for marker in (
            "1. RankIC by Asset Class",
            "2. Per-Asset RankIC",
            "3. Confidence Calibration",
            "4. RankIC × Predicted-MFE Bucket",
            "5. Entry-Subset Hit-Rate",
            "6. MFE / MAE Accuracy",
            "7. Predicted vs Realised Volatility",
            "8. Horizon Sweep",
            "8b. Cross-Sectional RankIC",
            "9. Per-H Confidence Calibration",
            "10. Entry-Filter Grid",
            "Appendix — Per-Asset RankIC (full table",
        ):
            assert marker in report, f"missing section: {marker}"
        # Live classes appear in the class table.
        assert "forex" in report
        assert "us_equity" in report
        # The appendix lists every symbol with N ≥ 5 (both seeded symbols).
        appendix = report.split("Appendix")[1]
        assert "EUR/USD" in appendix
        assert "F " in appendix

    def test_per_h_sections_degrade_and_count_skipped_rows(self, tmp_path: Path) -> None:
        # EUR/USD rows carry the 10c-prep blob, F rows don't — the per-H
        # sections must report used vs skipped and still print data for the
        # rows that have it.
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)

        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=36500,
            include_gap=False,
            horizon_sweep=True,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
        )
        section9 = report.split("9. Per-H Confidence")[1].split("── 10.")[0]
        assert "6 used, 6 skipped" in section9  # 6 EUR/USD with blob, 6 F without
        assert "forex @ H=" in section9  # decile table printed for blob rows
        section10 = report.split("10. Entry-Filter Grid")[1]
        assert "'*' marks the live gate" in section10
        assert "0.80   0.003" in section10  # live cell present in the grid

    def test_sweep_runs_when_headline_cohort_empty(self, tmp_path: Path) -> None:
        """At H=120 every row crossing a weekend is gap-spanned, so an empty
        headline cohort is the norm — the horizon-sweep sections must still
        run (regression: on a real dataset zero rows survived the persisted
        gap filter and the report early-returned)."""
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        db = CandleDB(str(db_path))
        db.init_db()
        # 80 h of candles, but the rows claim a 120 h horizon → trailing gap
        # → resolver marks every row gap_spanned=1.
        db.insert_candles(
            [
                Candle(
                    symbol="EUR/USD",
                    timestamp=base_ts + i * _HOUR_MS,
                    open=1.08,
                    high=1.081,
                    low=1.079,
                    close=1.08 + 0.0001 * (i % 9),
                    volume=0.0,
                    is_confirmed=True,
                )
                for i in range(80)
            ]
        )
        path_blob = np.asarray(1.08 * (1.0 + np.linspace(0.0, 0.01, 120)), dtype="<f4").tobytes()
        db.write_signal_history_batch(
            [
                {
                    "scored_at": base_ts + i * _HOUR_MS,
                    "symbol": "EUR/USD",
                    "horizon_bars": 120,
                    "mean_return": 0.001 * (i + 1),
                    "direction_confidence": 0.8,
                    "uncertainty": 0.5,
                    "entry_price": 1.08,
                    "predicted_close_path": path_blob,
                }
                for i in range(6)
            ]
        )
        assert db.resolve_signal_history(base_ts + 300 * _HOUR_MS) == 6
        # Headline cohort is empty under the default exclude-gap filter.
        assert db.get_signal_history(resolved_only=True, exclude_gap_spanned=True) == []
        db.close()

        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=36500,
            include_gap=False,
            horizon_sweep=True,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
        )
        assert "headline sections skipped" in report
        assert "8. Horizon Sweep" in report
        assert "forex" in report.split("8. Horizon Sweep")[1]  # sweep table has data
        assert "9. Per-H Confidence Calibration" in report
        assert "10. Entry-Filter Grid" in report

    def test_since_cutoff_filters_rows(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)

        # A cutoff after every scored_at leaves zero resolved rows.
        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=30,
            include_gap=False,
            horizon_sweep=False,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
            since_ms=base_ts + 7 * _HOUR_MS,
        )
        assert "Resolved rows: 0" in report

        # A cutoff before them includes all 12.
        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=30,
            include_gap=False,
            horizon_sweep=False,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
            since_ms=base_ts - _HOUR_MS,
        )
        assert "Resolved rows: 12" in report


# ---------------------------------------------------------------------------
# Net-of-cost augmentation
# ---------------------------------------------------------------------------


class TestWindowRollTimes:
    def _ts(self, y: int, mo: int, d: int, h: int) -> int:
        return int(datetime(y, mo, d, h, tzinfo=UTC).timestamp() * 1000)

    def test_subday_hold_crosses_no_roll(self) -> None:
        # Tue 09:00 + 6h = 15:00, never reaches the 22:00 roll.
        start = self._ts(2026, 6, 16, 9)
        assert sigdiag._window_roll_times(start, start + 6 * _HOUR_MS) == []

    def test_five_day_hold_crosses_five_rolls_incl_wed(self) -> None:
        start = self._ts(2026, 6, 16, 9)  # Tuesday
        rolls = sigdiag._window_roll_times(start, start + 120 * _HOUR_MS)
        assert len(rolls) == 5
        assert all(r.hour == 22 for r in rolls)
        assert any(r.weekday() == 2 for r in rolls)  # a Wednesday roll is present

    def test_straddling_the_roll_instant_counts_it(self) -> None:
        # Wed 20:00 + 4h = Thu 00:00 → crosses Wed 22:00 exactly once.
        start = self._ts(2026, 6, 17, 20)
        rolls = sigdiag._window_roll_times(start, start + 4 * _HOUR_MS)
        assert len(rolls) == 1
        assert rolls[0].weekday() == 2 and rolls[0].hour == 22


class TestNetOfCost:
    def test_funding_zero_subday_positive_multiday(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)  # Thu 00:00
        _seed_resolved_rows(db_path, base_ts)
        coll = sigdiag._collect_h_records(str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig())

        fx = [r for r in coll.records if r.asset_class == "forex"]
        # Entries are at 00:00–05:00 UTC; H=6 windows close by 11:00 → no 22:00
        # roll crossed → zero funding.  H=24 crosses the 22:00 roll → positive.
        assert fx, "expected forex records"
        assert all(r.funding_pct == 0.0 for r in fx if r.horizon == 6)
        assert all(r.funding_pct > 0.0 for r in fx if r.horizon == 24)

    def test_net_le_gross_for_long_only_book(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)
        coll = sigdiag._collect_h_records(str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig())
        assert coll.records
        for r in coll.records:
            assert r.cost_pct >= 0.0  # long-only: spread + funding never a credit
            assert r.real_ret_net <= r.real_ret + 1e-12

    def test_spread_reuses_slippage_pct_and_scales(self, tmp_path: Path) -> None:
        from bot.risk.ig_margin import slippage_pct_for

        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)

        full = sigdiag._collect_h_records(str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig())
        fx = next(r for r in full.records if r.symbol == "EUR/USD")
        assert fx.spread_pct == pytest.approx(slippage_pct_for("EUR/USD"))
        # Distinct asset classes draw distinct per-class spreads.
        eq = next(r for r in full.records if r.symbol == "F")
        assert eq.spread_pct == pytest.approx(slippage_pct_for("F"))

        half = sigdiag._collect_h_records(
            str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig(spread_mult=0.5)
        )
        fx_half = next(r for r in half.records if r.symbol == "EUR/USD")
        assert fx_half.spread_pct == pytest.approx(slippage_pct_for("EUR/USD") * 0.5)

    def test_component_isolation_flags(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)

        no_spread = sigdiag._collect_h_records(
            str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig(include_spread=False)
        )
        assert all(r.spread_pct == 0.0 for r in no_spread.records)
        assert any(r.funding_pct > 0.0 for r in no_spread.records)  # funding still on

        no_funding = sigdiag._collect_h_records(
            str(db_path), base_ts - _HOUR_MS, sigdiag._CostConfig(include_funding=False)
        )
        assert all(r.funding_pct == 0.0 for r in no_funding.records)
        assert any(r.spread_pct > 0.0 for r in no_funding.records)  # spread still on

    def test_report_shows_net_columns_and_breakdown(self, tmp_path: Path) -> None:
        db_path = _make_db(tmp_path)
        base_ts = int(datetime(2026, 6, 4, tzinfo=UTC).timestamp() * 1000)
        _seed_resolved_rows(db_path, base_ts)
        report = sigdiag.generate_report(
            str(db_path),
            lookback_days=36500,
            include_gap=False,
            horizon_sweep=True,
            min_confidence=0.80,
            min_predicted_return=0.003,
            max_uncertainty=10.0,
        )
        assert "Net-of-cost (sweep sections):" in report  # header note
        assert "Net mean realised return per (class, H)" in report  # section 8 grid
        assert "Cost breakdown (bps)" in report
        assert "mRealNet" in report and "HitNet%" in report  # section 10 columns
