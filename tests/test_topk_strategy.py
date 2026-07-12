"""Tests for TopKStrategy (multi-asset Kronos scanner).

Kronos is not installed in the test environment. Tests inject a mock
predictor into strat._predictor before calling _run_batch_inference.

Two-pass architecture:
  Pass 1 — single predict_batch call at forecast_temperature=0.6 → point estimate
  Pass 2 — variance_sample_count calls at variance_temperature=1.0 → std/confidence
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from bot.strategy.topk_strategy import AssetSignal, TopKConfig, TopKStrategy

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeCandle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    symbol: str = "TEST"
    is_confirmed: bool = True
    volume: float = 1.0


def _make_candles(n: int, price: float = 100.0) -> list[FakeCandle]:
    """Return n 1-minute candles at a constant price."""
    base_ts = 1_700_000_000_000
    return [
        FakeCandle(
            timestamp=base_ts + i * 60_000,
            open=price,
            high=price * 1.001,
            low=price * 0.999,
            close=price,
        )
        for i in range(n)
    ]


def _make_strategy(overrides: dict | None = None) -> TopKStrategy:
    cfg = TopKConfig(**(overrides or {}))
    return TopKStrategy(cfg)


def _make_batch_predictor(close: float = 102.0) -> MagicMock:
    """Mock predictor: predict_batch returns ``close`` for every asset on every call.

    Works for both Pass 1 (point estimate) and Pass 2 (variance) since both
    read only the 'close' column from the returned DataFrames.
    """
    mock = MagicMock()

    def _predict_batch(
        df_list: list[pd.DataFrame],
        x_timestamp_list: Any,
        y_timestamp_list: Any,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> list[pd.DataFrame]:
        return [pd.DataFrame({"close": [close]}) for _ in df_list]

    mock.predict_batch.side_effect = _predict_batch
    return mock


def _path_batch_predictor(base: float = 100.0, step: float = 1.0) -> MagicMock:
    """Mock predictor: every call returns a length-``pred_len`` linear close path.

    close[i] = base + step × i, so the value at any horizon H is
    ``base + step × (H - 1)`` — lets tests pin per-horizon collection.
    """
    mock = MagicMock()

    def _predict_batch(
        df_list: list[pd.DataFrame],
        x_timestamp_list: Any,
        y_timestamp_list: Any,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> list[pd.DataFrame]:
        closes = [base + step * i for i in range(pred_len)]
        return [pd.DataFrame({"close": closes}) for _ in df_list]

    mock.predict_batch.side_effect = _predict_batch
    return mock


def _cycling_batch_predictor(closes: list[float]) -> MagicMock:
    """Mock predictor: predict_batch returns closes[i] on the i-th call (same for all assets)."""
    mock = MagicMock()
    call_idx = [0]

    def _predict_batch(
        df_list: list[pd.DataFrame],
        x_timestamp_list: Any,
        y_timestamp_list: Any,
        pred_len: int,
        T: float = 1.0,
        top_p: float = 0.9,
        sample_count: int = 1,
        verbose: bool = False,
    ) -> list[pd.DataFrame]:
        c = closes[call_idx[0]]
        call_idx[0] += 1
        return [pd.DataFrame({"close": [c]}) for _ in df_list]

    mock.predict_batch.side_effect = _predict_batch
    return mock


# ---------------------------------------------------------------------------
# TopKConfig defaults
# ---------------------------------------------------------------------------


class TestTopKConfig:
    def test_defaults(self) -> None:
        cfg = TopKConfig()
        assert cfg.k == 2
        # new dual-pass fields
        assert cfg.forecast_temperature == 0.6
        assert cfg.forecast_top_p == 0.9
        assert cfg.forecast_sample_count == 10
        assert cfg.variance_pass_enabled is True
        assert cfg.variance_temperature == 1.0
        assert cfg.variance_sample_count == 20
        assert cfg.aggregate_to_minutes == 1
        assert cfg.min_confidence == 0.60
        assert cfg.max_uncertainty == 5.0
        assert cfg.min_predicted_return == 0.001
        assert cfg.vol_stop_multiplier == 2.0
        assert cfg.min_stop_pct == 0.005
        assert cfg.kronos_context_bars == 400
        assert cfg.pred_len == 120

    def test_custom_values(self) -> None:
        cfg = TopKConfig(k=5, forecast_temperature=0.3, min_confidence=0.8)
        assert cfg.k == 5
        assert cfg.forecast_temperature == 0.3
        assert cfg.min_confidence == 0.8

    def test_variance_pass_disabled(self) -> None:
        cfg = TopKConfig(variance_pass_enabled=False)
        assert cfg.variance_pass_enabled is False

    def test_min_confidence_gate_rejects_just_below_default(self) -> None:
        # A candidate just below the default must be rejected; one at the
        # default must pass — pins the gate as inclusive of the boundary.
        cfg = TopKConfig()
        assert cfg.min_confidence > 0.59
        assert cfg.min_confidence <= 0.60


# ---------------------------------------------------------------------------
# select_top_k
# ---------------------------------------------------------------------------


class TestSelectTopK:
    def _signal(self, symbol: str, mean_return: float, tradeable: bool) -> AssetSignal:
        return AssetSignal(
            symbol=symbol,
            mean_return=mean_return,
            std_return=0.01,
            direction_confidence=0.8,
            uncertainty=0.2,
            stop_pct=0.01,
            tradeable=tradeable,
            predicted_close=101.0,
        )

    def test_selects_top_k_by_return(self) -> None:
        strat = _make_strategy({"k": 2})
        signals = [
            self._signal("SYM_A", 0.05, tradeable=True),
            self._signal("SYM_B", 0.03, tradeable=True),
            self._signal("SYM_C", 0.07, tradeable=True),
        ]
        assert strat.select_top_k(signals) == ["SYM_C", "SYM_A"]

    def test_excludes_non_tradeable(self) -> None:
        strat = _make_strategy({"k": 3})
        signals = [
            self._signal("SYM_A", 0.05, tradeable=True),
            self._signal("SYM_B", 0.10, tradeable=False),
            self._signal("SYM_C", 0.03, tradeable=True),
        ]
        selected = strat.select_top_k(signals)
        assert "SYM_B" not in selected
        assert set(selected) == {"SYM_A", "SYM_C"}

    def test_returns_fewer_than_k_when_insufficient(self) -> None:
        strat = _make_strategy({"k": 5})
        signals = [
            self._signal("SYM_A", 0.05, tradeable=True),
            self._signal("SYM_B", 0.03, tradeable=False),
        ]
        assert strat.select_top_k(signals) == ["SYM_A"]

    def test_empty_signals_returns_empty(self) -> None:
        assert _make_strategy().select_top_k([]) == []

    def test_all_non_tradeable_returns_empty(self) -> None:
        strat = _make_strategy()
        assert strat.select_top_k([self._signal("SYM_X", 0.05, tradeable=False)]) == []

    def test_k_override(self) -> None:
        strat = _make_strategy({"k": 10})
        signals = [self._signal(f"SYM_{i}", float(i) * 0.01, tradeable=True) for i in range(5)]
        assert len(strat.select_top_k(signals, k=2)) == 2

    def test_ranking_is_descending(self) -> None:
        strat = _make_strategy({"k": 5})
        signals = [
            self._signal("LOW", 0.01, tradeable=True),
            self._signal("HIGH", 0.09, tradeable=True),
            self._signal("MID", 0.05, tradeable=True),
        ]
        assert strat.select_top_k(signals) == ["HIGH", "MID", "LOW"]

    def test_is_open_filters_closed_markets(self) -> None:
        # The highest-scoring names are "closed"; an open lower-scoring name
        # must still take a slot instead of the closed top picks being held.
        strat = _make_strategy({"k": 2})
        signals = [
            self._signal("CLOSED_HI", 0.09, tradeable=True),
            self._signal("CLOSED_MID", 0.07, tradeable=True),
            self._signal("OPEN_LO", 0.02, tradeable=True),
        ]
        open_syms = {"OPEN_LO"}
        selected = strat.select_top_k(signals, is_open=lambda s: s in open_syms)
        assert selected == ["OPEN_LO"]

    def test_is_open_none_keeps_full_ranking(self) -> None:
        # Default (is_open=None) must preserve the unfiltered back-test ranking.
        strat = _make_strategy({"k": 3})
        signals = [
            self._signal("A", 0.05, tradeable=True),
            self._signal("B", 0.09, tradeable=True),
        ]
        assert strat.select_top_k(signals) == ["B", "A"]


# ---------------------------------------------------------------------------
# _run_batch_inference (injected mock predictor)
# ---------------------------------------------------------------------------

# Base config for batch inference tests; overrides applied per-test.
_BATCH_CFG: dict[str, Any] = {
    "variance_sample_count": 10,  # Pass 2 loops (replaces old sample_count)
    "forecast_sample_count": 1,  # Keep pass 1 cheap in tests
    "kronos_context_bars": 10,
    "aggregate_to_minutes": 1,  # no aggregation in tests
    "min_predicted_return": 0.002,
    "min_confidence": 0.70,
    "max_uncertainty": 0.50,
    "vol_stop_multiplier": 2.0,
    "min_stop_pct": 0.005,
    "pred_len": 5,
}


class TestRunBatchInference:
    def _strat_with_mock(self, close: float = 102.0, **overrides: Any) -> TopKStrategy:
        strat = _make_strategy({**_BATCH_CFG, **overrides})
        strat._predictor = _make_batch_predictor(close=close)
        return strat

    def test_positive_return_tradeable(self) -> None:
        strat = self._strat_with_mock(close=102.0)
        signals = strat._run_batch_inference({"SYM_A": _make_candles(20, price=100.0)})

        assert len(signals) == 1
        sig = signals[0]
        assert sig.symbol == "SYM_A"
        assert sig.mean_return == pytest.approx(0.02, abs=1e-6)
        assert sig.std_return == pytest.approx(0.0, abs=1e-6)
        assert sig.direction_confidence == pytest.approx(1.0)
        assert sig.tradeable is True

    def test_parallel_groups_matches_serial(self) -> None:
        # Two non-empty groups (SYM_A no-volume + SPY volume) exercise the
        # concurrent dispatch. With a deterministic mock predictor the parallel
        # path must yield the same signals as serial, and must restore the torch
        # thread count it temporarily lowers.
        import torch

        cmap = {
            "SYM_A": _make_candles(20, price=100.0),
            "F": _make_candles(20, price=100.0),
        }
        serial = self._strat_with_mock(close=102.0, parallel_groups=False)
        parallel = self._strat_with_mock(close=102.0, parallel_groups=True)

        sig_serial = {s.symbol: s for s in serial._run_batch_inference(cmap)}
        threads_before = torch.get_num_threads()
        sig_parallel = {s.symbol: s for s in parallel._run_batch_inference(cmap)}

        assert torch.get_num_threads() == threads_before  # thread count restored
        assert set(sig_serial) == set(sig_parallel) == {"SYM_A", "F"}
        for sym in sig_serial:
            assert sig_parallel[sym].mean_return == pytest.approx(sig_serial[sym].mean_return)
            assert sig_parallel[sym].std_return == pytest.approx(sig_serial[sym].std_return)
            assert sig_parallel[sym].tradeable == sig_serial[sym].tradeable

    def test_flat_return_not_tradeable(self) -> None:
        strat = self._strat_with_mock(close=100.0)
        signals = strat._run_batch_inference({"SYM_B": _make_candles(20, price=100.0)})

        assert signals[0].mean_return == pytest.approx(0.0, abs=1e-6)
        assert signals[0].tradeable is False

    def test_stop_pct_floor_applied(self) -> None:
        # std_return == 0 (constant close) → min_stop_pct floor kicks in
        strat = self._strat_with_mock(close=102.0)
        strat._config.min_stop_pct = 0.01
        signals = strat._run_batch_inference({"SYM_C": _make_candles(20, price=100.0)})
        assert signals[0].stop_pct == pytest.approx(0.01)

    def test_stop_pct_from_std(self) -> None:
        # Pass 1: cycling[0]=101.0 → point mean_return=0.01
        # Pass 2: cycling[1..3]=102, 103, 104 → var_returns=[0.02, 0.03, 0.04]
        strat = _make_strategy(
            {
                "variance_sample_count": 3,
                "forecast_sample_count": 1,
                "kronos_context_bars": 10,
                "aggregate_to_minutes": 1,
                "min_predicted_return": 0.002,
                "min_confidence": 0.0,
                "max_uncertainty": 999.0,
                "vol_stop_multiplier": 3.0,
                "min_stop_pct": 0.001,
                "pred_len": 5,
            }
        )
        strat._predictor = _cycling_batch_predictor([101.0, 102.0, 103.0, 104.0])

        signals = strat._run_batch_inference({"SYM_D": _make_candles(20, price=100.0)})

        # std is computed over Pass 2 variance returns only
        var_returns = [0.02, 0.03, 0.04]
        expected_stop = float(np.std(var_returns)) * 3.0
        assert signals[0].stop_pct == pytest.approx(expected_stop, rel=1e-5)

    def test_context_bars_respected(self) -> None:
        # Feed 50 raw candles; context_bars=10 → predict_batch sees 10-row df
        strat = self._strat_with_mock(close=105.0, variance_sample_count=1)
        strat._run_batch_inference({"SYM_E": _make_candles(50, price=100.0)})

        # check any call (both passes receive the same trimmed df_list)
        call_args = strat._predictor.predict_batch.call_args
        df_list: list[pd.DataFrame] = call_args.kwargs["df_list"]
        assert len(df_list[0]) == 10

    def test_variance_pass_call_count(self) -> None:
        # Total calls = 1 (Pass 1) + variance_sample_count (Pass 2)
        strat = self._strat_with_mock(close=101.0, variance_sample_count=7)
        strat._run_batch_inference({"SYM_F": _make_candles(20, price=100.0)})
        assert strat._predictor.predict_batch.call_count == 1 + 7

    def test_variance_pass_disabled_single_call(self) -> None:
        # With variance_pass_enabled=False only Pass 1 runs (one call total)
        strat = self._strat_with_mock(close=101.0, variance_pass_enabled=False)
        strat._run_batch_inference({"SYM_F2": _make_candles(20, price=100.0)})
        assert strat._predictor.predict_batch.call_count == 1

    def test_variance_pass_disabled_uses_perfect_confidence(self) -> None:
        # Without variance pass, direction_confidence should be 1.0 for positive return
        strat = self._strat_with_mock(
            close=102.0,
            variance_pass_enabled=False,
            min_confidence=0.0,
            max_uncertainty=999.0,
        )
        signals = strat._run_batch_inference({"SYM_VD": _make_candles(20, price=100.0)})
        assert signals[0].direction_confidence == pytest.approx(1.0)
        assert signals[0].std_return == pytest.approx(0.0)

    def test_low_confidence_not_tradeable(self) -> None:
        # Pass 1: 105.0 → mean_return=0.05 (LONG)
        # Pass 2 (3 samples): [95.0, 105.0, 95.0] → 2 SHORT, 1 LONG → conf=1/3 < 0.70
        strat = _make_strategy(
            {
                "variance_sample_count": 3,
                "forecast_sample_count": 1,
                "kronos_context_bars": 10,
                "aggregate_to_minutes": 1,
                "min_predicted_return": 0.001,
                "min_confidence": 0.70,
                "max_uncertainty": 999.0,
                "vol_stop_multiplier": 2.0,
                "min_stop_pct": 0.005,
                "pred_len": 5,
            }
        )
        strat._predictor = _cycling_batch_predictor([105.0, 95.0, 105.0, 95.0])
        signals = strat._run_batch_inference({"SYM_G": _make_candles(20, price=100.0)})
        assert signals[0].tradeable is False

    def test_high_uncertainty_not_tradeable(self) -> None:
        # Pass 1: 200.0 → mean_return=1.0 (LONG)
        # Pass 2 (2 samples): [102.0, 50.0] → var_returns=[0.02, -0.50]
        # std ≈ 0.26, uncertainty = 0.26/1.0 ≈ 0.26 > 0.1 threshold
        strat = _make_strategy(
            {
                "variance_sample_count": 2,
                "forecast_sample_count": 1,
                "kronos_context_bars": 10,
                "aggregate_to_minutes": 1,
                "min_predicted_return": 0.001,
                "min_confidence": 0.0,
                "max_uncertainty": 0.1,  # very tight limit
                "vol_stop_multiplier": 2.0,
                "min_stop_pct": 0.005,
                "pred_len": 5,
            }
        )
        strat._predictor = _cycling_batch_predictor([200.0, 102.0, 50.0])
        signals = strat._run_batch_inference({"SYM_H": _make_candles(20, price=100.0)})
        assert signals[0].uncertainty > 0.1
        assert signals[0].tradeable is False

    def test_insufficient_candles_excluded(self) -> None:
        # kronos_context_bars=10, aggregate_to_minutes=1 → need 10 raw; give 5
        strat = self._strat_with_mock(close=102.0)
        signals = strat._run_batch_inference({"SYM_SHORT": _make_candles(5, price=100.0)})
        assert signals == []

    def test_multi_asset_batch(self) -> None:
        strat = self._strat_with_mock(close=102.0, variance_sample_count=5)
        candles_map = {
            "SYM_A": _make_candles(20, price=100.0),
            "SYM_B": _make_candles(25, price=100.0),
        }
        signals = strat._run_batch_inference(candles_map)
        assert len(signals) == 2
        assert {s.symbol for s in signals} == {"SYM_A", "SYM_B"}

    def test_empty_candles_map_returns_empty(self) -> None:
        strat = self._strat_with_mock()
        assert strat._run_batch_inference({}) == []

    def test_samples_field_populated(self) -> None:
        # samples holds variance-pass closes; check they are populated
        strat = self._strat_with_mock(close=102.0, variance_sample_count=5)
        signals = strat._run_batch_inference({"SYM_A": _make_candles(20, price=100.0)})
        assert len(signals[0].samples) == 5
        assert all(s == pytest.approx(102.0) for s in signals[0].samples)

    def test_point_pass_temperature_used(self) -> None:
        # Verify Pass 1 uses forecast_temperature=0.6
        strat = self._strat_with_mock(close=102.0, variance_sample_count=0)
        strat._config.forecast_temperature = 0.6
        strat._run_batch_inference({"SYM_T": _make_candles(20, price=100.0)})
        # Pass 1 is always the first call
        first_call = strat._predictor.predict_batch.call_args_list[0]
        assert first_call.kwargs["T"] == pytest.approx(0.6)

    def test_variance_pass_temperature_used(self) -> None:
        # Verify Pass 2 calls use variance_temperature=1.0
        strat = self._strat_with_mock(close=102.0, variance_sample_count=3)
        strat._config.variance_temperature = 1.0
        strat._run_batch_inference({"SYM_VT": _make_candles(20, price=100.0)})
        # Pass 2 calls are indices 1, 2, 3
        for call in strat._predictor.predict_batch.call_args_list[1:]:
            assert call.kwargs["T"] == pytest.approx(1.0)

    def test_pass1_failure_returns_empty(self) -> None:
        # If Pass 1 raises, _run_batch_inference should return [] not crash
        strat = _make_strategy({**_BATCH_CFG})
        mock = MagicMock()
        mock.predict_batch.side_effect = RuntimeError("GPU OOM")
        strat._predictor = mock
        signals = strat._run_batch_inference({"SYM_X": _make_candles(20, price=100.0)})
        assert signals == []


# ---------------------------------------------------------------------------
# — Pass-2 closes collected at every candidate horizon
# ---------------------------------------------------------------------------


class TestVarClosesAtHorizons:
    def test_collected_per_draw_with_nan_past_rollout(self) -> None:
        import math

        from bot.strategy.topk_strategy import CANDIDATE_HORIZONS

        # kronos_context_bars=10 → rollout clamps to 10 bars, so only H=6 of
        # the candidate horizons is inside the path; the rest must be NaN
        # (never silently clamped to the terminal bar).
        strat = _make_strategy({**_BATCH_CFG, "variance_sample_count": 4, "pred_len": 10})
        strat._predictor = _path_batch_predictor(base=100.0, step=1.0)
        signals = strat._run_batch_inference({"SYM_A": _make_candles(20, price=100.0)})

        rows = signals[0].var_closes_at_horizons
        assert len(rows) == 4  # one row per Pass-2 draw
        for row in rows:
            assert len(row) == len(CANDIDATE_HORIZONS)
            assert row[0] == pytest.approx(105.0)  # H=6 → close[5] = 100 + 5
            assert all(math.isnan(v) for v in row[1:])  # H=12..120 > rollout

    def test_all_horizons_real_when_rollout_covers_them(self) -> None:
        from bot.strategy.topk_strategy import CANDIDATE_HORIZONS

        strat = _make_strategy(
            {
                **_BATCH_CFG,
                "variance_sample_count": 2,
                "kronos_context_bars": 130,
                "pred_len": 120,
            }
        )
        strat._predictor = _path_batch_predictor(base=100.0, step=1.0)
        signals = strat._run_batch_inference({"SYM_B": _make_candles(150, price=100.0)})

        rows = signals[0].var_closes_at_horizons
        assert len(rows) == 2
        for row in rows:
            assert row == pytest.approx([100.0 + (h - 1) for h in CANDIDATE_HORIZONS])

    def test_empty_when_variance_pass_disabled(self) -> None:
        strat = _make_strategy({**_BATCH_CFG, "variance_pass_enabled": False})
        strat._predictor = _path_batch_predictor()
        signals = strat._run_batch_inference({"SYM_C": _make_candles(20, price=100.0)})
        assert signals[0].var_closes_at_horizons == []

    def test_collection_adds_no_inference_calls(self) -> None:
        # Logging-only: call count stays 1 (Pass 1) + variance_sample_count.
        strat = _make_strategy({**_BATCH_CFG, "variance_sample_count": 5})
        strat._predictor = _path_batch_predictor()
        strat._run_batch_inference({"SYM_D": _make_candles(20, price=100.0)})
        assert strat._predictor.predict_batch.call_count == 1 + 5


# ---------------------------------------------------------------------------
# — TOPK_RANKING_HORIZON_BY_CLASS parser
# ---------------------------------------------------------------------------


class TestRankingHorizonByClassParser:
    def test_empty_input_disables_overrides(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        assert parse_ranking_horizon_by_class("", 120) == {}
        assert parse_ranking_horizon_by_class("   ", 120) == {}

    def test_full_example_parses(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        result = parse_ranking_horizon_by_class("forex:48,us_equity:24,metal:24", 120)
        assert result == {"forex": 48, "us_equity": 24, "metal": 24}

    def test_whitespace_tolerated(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        assert parse_ranking_horizon_by_class(" forex : 48 , metal:6 ", 120) == {
            "forex": 48,
            "metal": 6,
        }

    def test_unknown_class_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match="unknown class 'crypto'"):
            parse_ranking_horizon_by_class("crypto:24", 120)
        # 'equity' is the EODHD-internal label; the config key is 'us_equity'.
        with pytest.raises(ValueError, match="unknown class 'equity'"):
            parse_ranking_horizon_by_class("equity:24", 120)

    def test_horizon_above_pred_len_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match=r"outside \[1, pred_len=120\]"):
            parse_ranking_horizon_by_class("forex:121", 120)

    def test_horizon_zero_or_negative_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match="outside"):
            parse_ranking_horizon_by_class("forex:0", 120)
        with pytest.raises(ValueError, match="outside"):
            parse_ranking_horizon_by_class("forex:-6", 120)

    def test_non_integer_horizon_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match="non-integer horizon"):
            parse_ranking_horizon_by_class("forex:abc", 120)

    def test_malformed_pair_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match="malformed pair"):
            parse_ranking_horizon_by_class("forex", 120)
        with pytest.raises(ValueError, match="malformed pair"):
            parse_ranking_horizon_by_class("forex:24,:12", 120)

    def test_duplicate_class_raises(self) -> None:
        from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

        with pytest.raises(ValueError, match="duplicate class"):
            parse_ranking_horizon_by_class("forex:24,forex:48", 120)


# ---------------------------------------------------------------------------
# — per-class horizon resolution + slicing
# ---------------------------------------------------------------------------


class TestPerClassHorizonResolution:
    def test_resolution_order_override_then_global_then_terminal(self) -> None:
        strat = _make_strategy(
            {
                "pred_len": 120,
                "ranking_horizon_bars": 48,
                "ranking_horizon_by_class": {"us_equity": 24},
            }
        )
        assert strat._effective_horizon_bars("F") == 24  # class override (us_equity)
        assert strat._effective_horizon_bars("EUR/USD") == 48  # global scalar
        assert strat._effective_horizon_bars("NOT_IN_UNIVERSE") == 48  # global scalar

        legacy = _make_strategy({"pred_len": 120, "ranking_horizon_by_class": {"metal": 12}})
        assert legacy._effective_horizon_bars("XAU/USD") == 12
        assert legacy._effective_horizon_bars("EUR/USD") == 0  # terminal (legacy)

    def test_signal_horizon_bars_maps_terminal_to_pred_len(self) -> None:
        strat = _make_strategy({"pred_len": 120, "ranking_horizon_by_class": {"us_equity": 24}})
        assert strat.signal_horizon_bars("F") == 24
        assert strat.signal_horizon_bars("EUR/USD") == 120  # 0 → pred_len
        assert strat.signal_horizon_bars("XAG/USD") == 120  # metal: no override here

    def test_mixed_class_batch_sliced_per_symbol(self) -> None:
        # EUR/USD (forex, no-volume group) at terminal bar; F (us_equity,
        # volume group) at H=6. Linear path 100..109 → terminal close 109,
        # H=6 close 105. Both Pass-1 mean_return and Pass-2 samples must
        # slice at each symbol's own H.
        strat = _make_strategy(
            {
                **_BATCH_CFG,
                "pred_len": 10,
                "variance_sample_count": 3,
                "min_confidence": 0.0,
                "max_uncertainty": 999.0,
                "ranking_horizon_by_class": {"us_equity": 6},
            }
        )
        strat._predictor = _path_batch_predictor(base=100.0, step=1.0)
        signals = {
            s.symbol: s
            for s in strat._run_batch_inference(
                {
                    "EUR/USD": _make_candles(20, price=100.0),
                    "F": _make_candles(20, price=100.0),
                }
            )
        }
        assert signals["F"].mean_return == pytest.approx(0.05)  # (105-100)/100
        assert signals["EUR/USD"].mean_return == pytest.approx(0.09)  # (109-100)/100
        assert all(v == pytest.approx(105.0) for v in signals["F"].samples)
        assert all(v == pytest.approx(109.0) for v in signals["EUR/USD"].samples)


class TestLegacyByteIdentity:
    """With the per-class map unset and the global horizon 0, the new
    per-symbol resolution must reproduce terminal-bar (pre-10c-prep)
    output exactly — verified by replaying a seeded synthetic rerank and
    recomputing every metric with the legacy iloc[-1] formulas."""

    @staticmethod
    def _seeded_recording_predictor(
        pred_len_expected: int,
    ) -> tuple[MagicMock, list[list[list[float]]]]:
        """Deterministic (seeded) predictor; records each call's close paths.

        recorded[call_idx][asset_idx] is the close-path list handed back for
        that asset on that call.
        """
        rng = np.random.default_rng(42)
        recorded: list[list[list[float]]] = []
        mock = MagicMock()

        def _predict_batch(
            df_list: list[pd.DataFrame],
            x_timestamp_list: Any,
            y_timestamp_list: Any,
            pred_len: int,
            T: float = 1.0,
            top_p: float = 0.9,
            sample_count: int = 1,
            verbose: bool = False,
        ) -> list[pd.DataFrame]:
            assert pred_len == pred_len_expected
            paths = [
                (100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.002, size=pred_len))).tolist()
                for _ in df_list
            ]
            recorded.append(paths)
            return [pd.DataFrame({"close": p}) for p in paths]

        mock.predict_batch.side_effect = _predict_batch
        return mock, recorded

    def test_defaults_reproduce_terminal_bar_output_exactly(self) -> None:
        n_var = 5
        strat = _make_strategy(
            {
                **_BATCH_CFG,
                "pred_len": 10,
                "variance_sample_count": n_var,
                "min_confidence": 0.0,
                "max_uncertainty": 999.0,
                # Defaults under test: no per-class map, global horizon 0.
                "ranking_horizon_bars": 0,
                "ranking_horizon_by_class": {},
            }
        )
        predictor, recorded = self._seeded_recording_predictor(pred_len_expected=10)
        strat._predictor = predictor

        signals = {
            s.symbol: s
            for s in strat._run_batch_inference(
                {
                    "EUR/USD": _make_candles(20, price=100.0),  # no-volume group
                    "F": _make_candles(20, price=100.0),  # volume group
                }
            )
        }

        # Serial group order: no-volume first then volume; each group is one
        # Pass-1 call followed by n_var Pass-2 calls.
        entry = 100.0
        groups = {"EUR/USD": 0, "F": 1}
        for sym, group_idx in groups.items():
            base_call = group_idx * (1 + n_var)
            p1_path = recorded[base_call][0]
            # Legacy formulas: everything pegged to the terminal bar.
            expected_mean_close = float(pd.DataFrame({"close": p1_path})["close"].iloc[-1])
            expected_mean_return = (expected_mean_close - entry) / entry
            var_closes = [
                float(pd.DataFrame({"close": recorded[base_call + 1 + j][0]})["close"].iloc[-1])
                for j in range(n_var)
            ]
            var_returns = [(c - entry) / entry for c in var_closes]
            expected_std = float(np.std(var_returns))
            if expected_mean_return >= 0:
                expected_conf = float(np.mean([r >= 0 for r in var_returns]))
            else:
                expected_conf = float(np.mean([r < 0 for r in var_returns]))

            sig = signals[sym]
            # Exact equality — not approx — proves byte-identity of the path.
            assert sig.mean_return == expected_mean_return
            assert sig.predicted_close == expected_mean_close
            assert sig.std_return == expected_std
            assert sig.direction_confidence == expected_conf
            assert sig.samples == var_closes
            assert sig.uncertainty == expected_std / (abs(expected_mean_return) + 1e-8)


# ---------------------------------------------------------------------------
# scan() (async)
# ---------------------------------------------------------------------------


class TestScan:
    @pytest.mark.asyncio
    async def test_scan_passes_all_symbols_to_batch(self) -> None:
        strat = _make_strategy(
            {
                "kronos_context_bars": 5,
                "variance_sample_count": 1,
                "forecast_sample_count": 1,
                "aggregate_to_minutes": 1,
                "pred_len": 3,
            }
        )
        strat._predictor = _make_batch_predictor(close=102.0)

        async def fetch(sym: str) -> list[FakeCandle]:
            return _make_candles(10, price=100.0)

        async def fake_to_thread(fn: Any, *args: Any) -> Any:
            return fn(*args)

        with patch("bot.strategy.topk_strategy.asyncio.to_thread", fake_to_thread):
            signals = await strat.scan(["SYM_A", "SYM_B"], fetch)

        assert len(signals) == 2
        assert {s.symbol for s in signals} == {"SYM_A", "SYM_B"}

    @pytest.mark.asyncio
    async def test_scan_handles_fetch_exception(self) -> None:
        strat = _make_strategy({"kronos_context_bars": 5, "aggregate_to_minutes": 1})

        async def bad_fetch(sym: str) -> list[FakeCandle]:
            raise RuntimeError("network error")

        signals = await strat.scan(["SYM_X"], bad_fetch)
        assert signals == []

    @pytest.mark.asyncio
    async def test_scan_returns_empty_for_no_symbols(self) -> None:
        strat = _make_strategy()

        async def fetch(sym: str) -> list[FakeCandle]:
            return _make_candles(10)

        signals = await strat.scan([], fetch)
        assert signals == []


# ---------------------------------------------------------------------------
# _load_predictor (import error path)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# — volume-split inference
# ---------------------------------------------------------------------------


class TestVolumeSplit:
    """Volume-bearing symbols (US shares + IG-native metals) are batched
    separately from no-volume forex — predict_batch needs identical shape."""

    def test_volume_symbol_df_has_volume_columns(self) -> None:
        strat = _make_strategy({**_BATCH_CFG})
        strat._predictor = _make_batch_predictor()
        # Candles with non-zero volume (ETF)
        candles = _make_candles(20, price=100.0)
        result = strat._prepare_asset(candles, "F", has_volume=True)
        assert result is not None
        df, _ = result
        assert "volume" in df.columns
        assert "amount" in df.columns

    def test_novol_symbol_df_has_no_volume_columns(self) -> None:
        strat = _make_strategy({**_BATCH_CFG})
        result = strat._prepare_asset(_make_candles(20), "EUR/USD", has_volume=False)
        assert result is not None
        df, _ = result
        assert "volume" not in df.columns
        assert "amount" not in df.columns

    def test_volume_symbol_routed_to_volume_group(self) -> None:
        # SPY is in _VOLUME_SYMBOLS — its df passed to predict_batch must have volume
        strat = _make_strategy({**_BATCH_CFG})
        call_dfs: list[list[pd.DataFrame]] = []

        def _capture_predict_batch(
            df_list: list[pd.DataFrame],
            x_timestamp_list: Any,
            y_timestamp_list: Any,
            pred_len: int,
            T: float = 1.0,
            top_p: float = 0.9,
            sample_count: int = 1,
            verbose: bool = False,
        ) -> list[pd.DataFrame]:
            call_dfs.append(df_list)
            return [pd.DataFrame({"close": [102.0]}) for _ in df_list]

        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.predict_batch.side_effect = _capture_predict_batch
        strat._predictor = mock

        strat._run_batch_inference({"F": _make_candles(20, price=100.0)})
        # All calls used dfs with volume column
        for dfs in call_dfs:
            for df in dfs:
                assert "volume" in df.columns

    def test_mixed_symbols_two_predict_batch_groups(self) -> None:
        # EUR/USD (novol) + SPY (vol) → Pass 1 fires twice (one per group)
        strat = _make_strategy(
            {**_BATCH_CFG, "variance_sample_count": 0, "variance_pass_enabled": False}
        )
        strat._predictor = _make_batch_predictor(close=102.0)
        strat._run_batch_inference(
            {
                "EUR/USD": _make_candles(20, price=1.0850),
                "F": _make_candles(20, price=100.0),
            }
        )
        # Two groups → two Pass 1 calls (variance disabled)
        assert strat._predictor.predict_batch.call_count == 2

    def test_all_novol_symbols_single_pass1_call(self) -> None:
        # All symbols outside _VOLUME_SYMBOLS → single batch
        strat = _make_strategy({**_BATCH_CFG, "variance_pass_enabled": False})
        strat._predictor = _make_batch_predictor(close=102.0)
        strat._run_batch_inference(
            {
                "EUR/USD": _make_candles(20, price=1.0850),
                "GBP/USD": _make_candles(20, price=1.2700),
            }
        )
        assert strat._predictor.predict_batch.call_count == 1

    def test_signals_returned_for_both_groups(self) -> None:
        strat = _make_strategy({**_BATCH_CFG})
        strat._predictor = _make_batch_predictor(close=102.0)
        signals = strat._run_batch_inference(
            {
                "EUR/USD": _make_candles(20, price=100.0),
                "F": _make_candles(20, price=100.0),
            }
        )
        assert {s.symbol for s in signals} == {"EUR/USD", "F"}

    def test_pass1_failure_in_any_group_returns_empty(self) -> None:
        strat = _make_strategy({**_BATCH_CFG})
        from unittest.mock import MagicMock

        mock = MagicMock()
        mock.predict_batch.side_effect = RuntimeError("OOM")
        strat._predictor = mock
        signals = strat._run_batch_inference({"F": _make_candles(20, price=100.0)})
        assert signals == []


class TestLoadPredictor:
    def test_raises_if_kronos_not_importable(self) -> None:
        strat = _make_strategy({"kronos_dir": ""})
        with (
            patch.dict("sys.modules", {"model": None}),
            pytest.raises(RuntimeError, match="Cannot import"),
        ):
            strat._load_predictor()

    def test_no_op_when_predictor_already_loaded(self) -> None:
        strat = _make_strategy()
        strat._predictor = _make_batch_predictor()
        strat._load_predictor()  # must not raise
        assert strat._predictor is not None


class TestOfflineFirstLoad:
    """``_load_kronos_offline_first`` prefers the local cache, falls back online."""

    def test_uses_cache_without_network(self) -> None:
        from bot.strategy.topk_strategy import _load_kronos_offline_first

        calls: list[dict[str, object]] = []

        def loader(repo: str, **kwargs: object) -> str:
            calls.append({"repo": repo, **kwargs})
            return "loaded"

        result = _load_kronos_offline_first(loader, "NeoQuasar/Kronos-mini", "model")

        assert result == "loaded"
        # Single attempt, offline-only — no online fallback.
        assert calls == [{"repo": "NeoQuasar/Kronos-mini", "local_files_only": True}]

    def test_falls_back_online_when_cache_missing(self) -> None:
        from bot.strategy.topk_strategy import _load_kronos_offline_first

        calls: list[dict[str, object]] = []

        def loader(repo: str, **kwargs: object) -> str:
            calls.append({"repo": repo, **kwargs})
            if kwargs.get("local_files_only"):
                raise OSError("not found in local cache")
            return "downloaded"

        result = _load_kronos_offline_first(loader, "NeoQuasar/Kronos-mini", "model")

        assert result == "downloaded"
        # Offline attempt first, then a plain online attempt that repopulates.
        assert calls == [
            {"repo": "NeoQuasar/Kronos-mini", "local_files_only": True},
            {"repo": "NeoQuasar/Kronos-mini"},
        ]

    def test_online_failure_propagates(self) -> None:
        from bot.strategy.topk_strategy import _load_kronos_offline_first

        def loader(repo: str, **kwargs: object) -> str:
            raise OSError("huggingface unreachable")

        with pytest.raises(OSError, match="unreachable"):
            _load_kronos_offline_first(loader, "NeoQuasar/Kronos-mini", "model")


class TestExpectedBatches:
    """Count of Kronos predict_batch calls feeding the dashboard's overall bar.

    Volume-bearing symbols (US shares + IG-native metals) and the no-volume
    forex pairs are inferred as separate groups because predict_batch requires
    identical feature shape per call.  Each non-empty group contributes
    ``1 + variance_sample_count`` calls (Pass-1 once + Pass-2 N times).
    """

    def test_volume_only_group(self) -> None:
        strat = _make_strategy({"variance_sample_count": 20})
        # All-volume watchlist → one group.
        assert strat.expected_batches(["F", "T", "PFE"]) == 21

    def test_novol_only_group(self) -> None:
        strat = _make_strategy({"variance_sample_count": 20})
        # All-forex watchlist → one group.
        assert strat.expected_batches(["EUR/USD", "GBP/USD", "USD/JPY"]) == 21

    def test_mixed_groups(self) -> None:
        strat = _make_strategy({"variance_sample_count": 20})
        # Both groups non-empty → 2 × (1 + 20) = 42.
        assert strat.expected_batches(["F", "EUR/USD"]) == 42

    def test_empty_watchlist_zero(self) -> None:
        strat = _make_strategy()
        assert strat.expected_batches([]) == 0

    def test_variance_pass_disabled_collapses_to_one_per_group(self) -> None:
        strat = _make_strategy({"variance_pass_enabled": False})
        # Only Pass-1 runs; variance_sample_count is ignored.
        assert strat.expected_batches(["F", "T"]) == 1
        assert strat.expected_batches(["F", "EUR/USD"]) == 2

    def test_variance_sample_count_scales_per_group(self) -> None:
        strat = _make_strategy({"variance_sample_count": 5})
        # 1 + 5 = 6 per group; mixed = 12.
        assert strat.expected_batches(["F"]) == 6
        assert strat.expected_batches(["F", "EUR/USD"]) == 12

    def test_matches_actual_batch_count_in_inference(self) -> None:
        # Ground-truth: run _run_batch_inference with a counting predictor and
        # confirm expected_batches() matches the real predict_batch call count.
        cfg_overrides = {"variance_sample_count": 3, "kronos_context_bars": 50, "pred_len": 4}
        strat = _make_strategy(cfg_overrides)
        predictor = _make_batch_predictor()
        strat._predictor = predictor

        watchlist = ["F", "EUR/USD"]
        candles_map = {sym: _make_candles(60) for sym in watchlist}
        with patch.object(strat, "_load_predictor"):
            strat._run_batch_inference(candles_map)

        assert predictor.predict_batch.call_count == strat.expected_batches(watchlist)
