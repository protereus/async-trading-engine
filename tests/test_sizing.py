"""Direct unit tests for the pure position-sizing math in ``bot.risk.sizing``.

These functions are risk-critical (a pip_value bug once sized FX positions
10,000x too large) and pure, so they earn direct coverage independent of the
RiskManager integration tests that exercise them transitively.
"""

from __future__ import annotations

import pytest

from bot.risk.sizing import compute_ig_size


class TestComputeIgSize:
    def test_non_jpy_fx(self) -> None:
        # £100 risk, 60-point stop (1.20 * 0.5% / 0.0001) -> £1.67/pt.
        stake = compute_ig_size(
            equity_gbp=10_000, risk_pct=0.01, entry_price=1.20, stop_pct=0.005, pip_value=0.0001
        )
        assert stake == pytest.approx(1.67, abs=0.005)

    def test_jpy_pair_pip_value(self) -> None:
        # JPY pip_value 0.01: 150 * 0.5% / 0.01 = 75-point stop -> £1.33/pt.
        stake = compute_ig_size(
            equity_gbp=10_000, risk_pct=0.01, entry_price=150.0, stop_pct=0.005, pip_value=0.01
        )
        assert stake == pytest.approx(1.33, abs=0.005)

    def test_gold_pip_value_one(self) -> None:
        # Gold is IG-native (pip_value 1.0): 4700 * 0.5% = 23.5-point stop.
        stake = compute_ig_size(
            equity_gbp=10_000, risk_pct=0.01, entry_price=4700.0, stop_pct=0.005, pip_value=1.0
        )
        assert stake == pytest.approx(4.26, abs=0.005)

    def test_slippage_buffer_widens_stop_and_shrinks_stake(self) -> None:
        base = compute_ig_size(
            equity_gbp=10_000, risk_pct=0.01, entry_price=1.20, stop_pct=0.005, pip_value=0.0001
        )
        with_buffer = compute_ig_size(
            equity_gbp=10_000,
            risk_pct=0.01,
            entry_price=1.20,
            stop_pct=0.005,
            pip_value=0.0001,
            slippage_buffer_pts=15.0,
        )
        # 60-pt stop -> 75-pt effective stop -> smaller stake.
        assert with_buffer < base
        assert with_buffer == pytest.approx(100 / 75, abs=0.005)

    def test_negative_slippage_buffer_is_clamped_to_zero(self) -> None:
        base = compute_ig_size(
            equity_gbp=10_000, risk_pct=0.01, entry_price=1.20, stop_pct=0.005, pip_value=0.0001
        )
        clamped = compute_ig_size(
            equity_gbp=10_000,
            risk_pct=0.01,
            entry_price=1.20,
            stop_pct=0.005,
            pip_value=0.0001,
            slippage_buffer_pts=-50.0,
        )
        assert clamped == base

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"equity_gbp": 0, "risk_pct": 0.01, "entry_price": 1.2, "stop_pct": 0.005},
            {"equity_gbp": 10_000, "risk_pct": 0.0, "entry_price": 1.2, "stop_pct": 0.005},
            {"equity_gbp": 10_000, "risk_pct": 0.01, "entry_price": 0.0, "stop_pct": 0.005},
            {"equity_gbp": 10_000, "risk_pct": 0.01, "entry_price": 1.2, "stop_pct": 0.0},
            {
                "equity_gbp": 10_000,
                "risk_pct": 0.01,
                "entry_price": 1.2,
                "stop_pct": 0.005,
                "pip_value": 0.0,
            },
        ],
    )
    def test_invalid_inputs_return_zero(self, kwargs: dict[str, float]) -> None:
        assert compute_ig_size(**kwargs) == 0.0
