"""Unit tests for the direction-aware sentiment gate helper."""

from __future__ import annotations

from bot.execution.ig_convert import apply_sentiment_gate as _apply_sentiment_gate


class TestSentimentGate:
    def test_long_with_bullish_sentiment_passes(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["EUR/USD"],
            signal_returns={"EUR/USD": 0.002},
            sentiment_scores={"EUR/USD": 0.5},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["EUR/USD"]
        assert blocked == []

    def test_long_with_bearish_sentiment_blocks(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["EUR/USD"],
            signal_returns={"EUR/USD": 0.002},
            sentiment_scores={"EUR/USD": -0.5},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == []
        assert blocked == ["EUR/USD"]

    def test_short_with_bearish_sentiment_passes(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["USD/JPY"],
            signal_returns={"USD/JPY": -0.003},
            sentiment_scores={"USD/JPY": -0.4},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["USD/JPY"]
        assert blocked == []

    def test_short_with_bullish_sentiment_blocks(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["USD/JPY"],
            signal_returns={"USD/JPY": -0.003},
            sentiment_scores={"USD/JPY": 0.4},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == []
        assert blocked == ["USD/JPY"]

    def test_missing_sentiment_passes_through(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["XAU/USD"],
            signal_returns={"XAU/USD": 0.005},
            sentiment_scores={},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["XAU/USD"]
        assert blocked == []

    def test_missing_signal_passes_through(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["XAU/USD"],
            signal_returns={},
            sentiment_scores={"XAU/USD": -0.9},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["XAU/USD"]
        assert blocked == []

    def test_neutral_sentiment_blocks_both_directions(self) -> None:
        # Sentiment at 0.0 is below long_threshold=0.3 AND above short_threshold=-0.3
        long_pass, long_block = _apply_sentiment_gate(
            selected=["EUR/USD"],
            signal_returns={"EUR/USD": 0.002},
            sentiment_scores={"EUR/USD": 0.0},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        short_pass, short_block = _apply_sentiment_gate(
            selected=["USD/JPY"],
            signal_returns={"USD/JPY": -0.002},
            sentiment_scores={"USD/JPY": 0.0},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert long_pass == [] and long_block == ["EUR/USD"]
        assert short_pass == [] and short_block == ["USD/JPY"]

    def test_threshold_boundary_inclusive(self) -> None:
        # sentiment == long_threshold should PASS (>= semantics)
        passed, blocked = _apply_sentiment_gate(
            selected=["EUR/USD", "USD/JPY"],
            signal_returns={"EUR/USD": 0.001, "USD/JPY": -0.001},
            sentiment_scores={"EUR/USD": 0.3, "USD/JPY": -0.3},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["EUR/USD", "USD/JPY"]
        assert blocked == []

    def test_zero_return_treated_as_long(self) -> None:
        # mean_return == 0 takes the >= 0 branch (LONG semantics)
        passed, blocked = _apply_sentiment_gate(
            selected=["EUR/USD"],
            signal_returns={"EUR/USD": 0.0},
            sentiment_scores={"EUR/USD": 0.5},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["EUR/USD"]
        assert blocked == []

    def test_mixed_selection(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=["EUR/USD", "USD/JPY", "XAU/USD", "GBP/USD"],
            signal_returns={
                "EUR/USD": 0.002,  # LONG, bullish sentiment → pass
                "USD/JPY": -0.002,  # SHORT, bullish sentiment → block
                "XAU/USD": 0.003,  # LONG, no sentiment → pass through
                "GBP/USD": -0.001,  # SHORT, bearish sentiment → pass
            },
            sentiment_scores={
                "EUR/USD": 0.5,
                "USD/JPY": 0.4,
                "GBP/USD": -0.5,
            },
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == ["EUR/USD", "XAU/USD", "GBP/USD"]
        assert blocked == ["USD/JPY"]

    def test_empty_selection(self) -> None:
        passed, blocked = _apply_sentiment_gate(
            selected=[],
            signal_returns={"EUR/USD": 0.002},
            sentiment_scores={"EUR/USD": 0.5},
            long_threshold=0.3,
            short_threshold=-0.3,
        )
        assert passed == []
        assert blocked == []
