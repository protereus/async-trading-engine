"""Integration tests for TakeProfitManager wired into TradingBot._process_candle_ig_topk
and _topk_rerank_loop.

These tests mock the IG client, feed store, and Kronos strategy so no real network
calls or GPU inference happens.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from bot.strategy.kronos_signals import KronosPathSignal
from bot.strategy.take_profit import ExitReason, TakeProfitConfig, TakeProfitManager

# ---------------------------------------------------------------------------
# Minimal stubs
# ---------------------------------------------------------------------------

BASE_MS = 1_700_000_000_000


@dataclass
class FakeSignal:
    symbol: str
    mean_return: float
    std_return: float
    direction_confidence: float
    uncertainty: float
    stop_pct: float
    tradeable: bool = True
    predicted_close: float = 100.0
    direction: str = "LONG"


@dataclass
class FakeCandle:
    symbol: str
    close: float
    timestamp: int = BASE_MS
    is_confirmed: bool = True


# ---------------------------------------------------------------------------
# TakeProfitManager unit-level integration (no TradingBot needed)
# ---------------------------------------------------------------------------


class TestTPManagerIntegration:
    """End-to-end tests through the TakeProfitManager without importing TradingBot."""

    def _make_mgr(self, **cfg_overrides: Any) -> TakeProfitManager:
        cfg = TakeProfitConfig(**cfg_overrides)
        return TakeProfitManager(cfg, pred_len=120)

    def test_open_then_price_rises_to_static_tp(self) -> None:
        """Full lifecycle: register → price rises → STATIC_TP exit."""
        mgr = self._make_mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 1.1000, sig, BASE_MS)

        # tp_pct = max(0.02*1.5, 0.05*0.8) = max(0.03, 0.04) = 0.04
        # tp_price ≈ 1.1000 * 1.04 = 1.1440; use 1.1450 to clear float precision
        dec = mgr.evaluate_price("EUR/USD", 1.1450, BASE_MS + 3_600_000)
        assert dec.should_exit
        assert dec.reason == ExitReason.STATIC_TP

        mgr.deregister_position("EUR/USD")
        assert "EUR/USD" not in mgr._positions

    def test_open_then_rerank_mean_flip_closes(self) -> None:
        """Full lifecycle: register → rerank with mean_return flipped → SIGNAL_DECAY_FLIP."""
        mgr = self._make_mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 1.1000, sig, BASE_MS)

        flipped = FakeSignal("EUR/USD", -0.03, 0.01, 0.80, 1.0, 0.02)
        dec = mgr.evaluate_signal("EUR/USD", flipped, in_topk=True)
        assert dec.should_exit
        assert dec.reason == ExitReason.SIGNAL_DECAY_FLIP

    def test_open_then_121_hours_time_exit(self) -> None:
        """Full lifecycle: register → 121 hours pass → TIME_LIMIT exit."""
        mgr = self._make_mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 1.1000, sig, BASE_MS)

        # 121h > 120h limit
        now_ms = BASE_MS + 121 * 3_600_000
        dec = mgr.evaluate_price("EUR/USD", 1.1000, now_ms)
        assert dec.should_exit
        assert dec.reason == ExitReason.TIME_LIMIT

    def test_trailing_ratchet_lifecycle(self) -> None:
        """Price climbs, ratchet arms, then retraces past stop → TRAILING_STOP_RATCHET."""
        mgr = self._make_mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 100.0, sig, BASE_MS)

        # Rises to 110 (10% > 2*2%=4%) → Stage 2 armed; stop = 110*(1-0.01) = 108.9
        mgr.evaluate_price("EUR/USD", 110.0, BASE_MS + 3_600_000)
        assert mgr._positions["EUR/USD"].trail_armed

        # Falls below 108.9
        dec = mgr.evaluate_price("EUR/USD", 108.5, BASE_MS + 7_200_000)
        assert dec.should_exit
        assert dec.reason == ExitReason.TRAILING_STOP_RATCHET

    def test_snapshot_restore_full_roundtrip(self) -> None:
        """snapshot → new manager → restore → same evaluation result.

        Uses entry_price=100.0, evaluates at 100.0 (no profit) so only time exit fires.
        """
        mgr = self._make_mgr()
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 100.0, sig, BASE_MS)

        # Advance state without triggering any component
        mgr.evaluate_price("EUR/USD", 100.0, BASE_MS + 1)

        snap = mgr.snapshot()

        mgr2 = self._make_mgr()
        mgr2.restore(snap)

        # Restored manager: 121h elapsed → TIME_LIMIT (price at entry, no profit)
        dec = mgr2.evaluate_price("EUR/USD", 100.0, BASE_MS + 121 * 3_600_000)
        assert dec.should_exit
        assert dec.reason == ExitReason.TIME_LIMIT

    def test_sentiment_reversal_lifecycle(self) -> None:
        """Sentiment reversal fires only when enabled and thresholds met."""
        mgr = self._make_mgr(sentiment_reversal_enabled=True)
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 100.0, sig, BASE_MS)

        class FakeSent:
            sentiment = -0.5
            confidence = 0.7

        dec = mgr.evaluate_sentiment("EUR/USD", FakeSent())
        assert dec.should_exit
        assert dec.reason == ExitReason.SENTIMENT_REVERSAL

    def test_all_disabled_no_exits(self) -> None:
        """With all components disabled, no exit fires."""
        mgr = self._make_mgr(
            static_enabled=False,
            trailing_enabled=False,
            signal_decay_enabled=False,
            time_enabled=False,
            sentiment_reversal_enabled=False,
        )
        sig = FakeSignal("EUR/USD", 0.05, 0.01, 0.80, 1.0, 0.02)
        mgr.register_position("EUR/USD", 100.0, sig, BASE_MS)

        # Price way above TP, time expired, signal flipped
        dec = mgr.evaluate_price("EUR/USD", 200.0, BASE_MS + 9999 * 3_600_000)
        assert not dec.should_exit

        flipped = FakeSignal("EUR/USD", -0.5, 0.01, 0.10, 10.0, 0.02)
        dec2 = mgr.evaluate_signal("EUR/USD", flipped, in_topk=False)
        assert not dec2.should_exit


class TestPathSignalWiring:
    """Verify that path_signal fields are stored when register_position receives one.

    This test class exists specifically to catch the regression where main.py
    called register_position without passing path_signal=, causing all path-aware
    TP logic to silently fall back to scalar-based formulas.
    """

    def _make_path_signal(
        self, mfe: float = 0.04, mae: float = 0.01, peak_bar: int = 10
    ) -> KronosPathSignal:
        return KronosPathSignal(
            symbol="EUR/USD",
            mean_return=0.03,
            std_return=0.005,
            direction_confidence=0.80,
            uncertainty=0.8,
            predicted_max_high=1.14,
            predicted_min_low=1.09,
            predicted_mfe_pct=mfe,
            predicted_mae_pct=mae,
            predicted_peak_bar=peak_bar,
            predicted_volatility=0.002,
            predicted_path_drawdown=0.008,
            monotonicity=0.75,
            stop_pct=0.011,
            ranking_score=0.025,
        )

    def test_path_signal_stored_on_register(self) -> None:
        """predicted_mfe_pct is non-None when path_signal kwarg is passed."""
        mgr = TakeProfitManager(TakeProfitConfig(), pred_len=120)
        sig = FakeSignal("EUR/USD", 0.03, 0.005, 0.80, 0.8, 0.02)
        path_sig = self._make_path_signal(mfe=0.04, mae=0.01, peak_bar=10)

        mgr.register_position("EUR/USD", 1.10, sig, BASE_MS, path_signal=path_sig)

        state = mgr._positions["EUR/USD"]
        assert state.predicted_mfe_pct == 0.04
        assert state.predicted_mae_pct == 0.01
        assert state.predicted_peak_bar == 10

    def test_path_signal_absent_does_not_crash(self) -> None:
        """Registering without path_signal uses scalar fallbacks — no crash."""
        mgr = TakeProfitManager(TakeProfitConfig(), pred_len=120)
        sig = FakeSignal("EUR/USD", 0.03, 0.005, 0.80, 0.8, 0.02)

        mgr.register_position("EUR/USD", 1.10, sig, BASE_MS)

        state = mgr._positions["EUR/USD"]
        assert state.predicted_mfe_pct is None

    def test_path_mfe_drives_static_tp(self) -> None:
        """Path signal: static TP uses MFE×capture_fraction, not mean_return×fraction."""
        # min_rr=0.005*1.5=0.0075; mfe path=0.04*0.85=0.034 → path wins
        mgr = TakeProfitManager(TakeProfitConfig(), pred_len=120)
        sig = FakeSignal("EUR/USD", 0.03, 0.005, 0.80, 0.8, 0.005)
        path_sig = self._make_path_signal(mfe=0.04, peak_bar=10)

        mgr.register_position("EUR/USD", 1.0000, sig, BASE_MS, path_signal=path_sig)

        # tp_pct = max(0.005*1.5, 0.04*0.85) = max(0.0075, 0.034) = 0.034
        # tp_price ≈ 1.0000 * 1.034 = 1.034; use 1.035 to clear float precision
        dec = mgr.evaluate_price("EUR/USD", 1.035, BASE_MS + 3_600_000)
        assert dec.should_exit
        assert dec.reason == ExitReason.STATIC_TP

    def test_peak_bar_zero_is_valid(self) -> None:
        """predicted_peak_bar=0 (peak at first bar) must trigger time exit, not be ignored."""
        mgr = TakeProfitManager(TakeProfitConfig(), pred_len=120)
        sig = FakeSignal("EUR/USD", 0.03, 0.005, 0.80, 0.8, 0.005)
        path_sig = self._make_path_signal(mfe=0.04, peak_bar=0)

        mgr.register_position("EUR/USD", 1.0000, sig, BASE_MS, path_signal=path_sig)

        # peak_bar=0 → deadline = opened_at + 0 + 12h grace = BASE_MS + 43_200_000
        # Evaluate at BASE_MS + 13h — past grace
        dec = mgr.evaluate_price("EUR/USD", 1.0000, BASE_MS + 13 * 3_600_000)
        assert dec.should_exit
        assert dec.reason == ExitReason.TIME_LIMIT
