"""Tests for data models and enums."""

from __future__ import annotations

import pytest

from bot.core.models import (
    BotState,
    Candle,
    Direction,
    ErrorType,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    Signal,
)


class TestCandle:
    def test_creation(self) -> None:
        c = Candle(
            timestamp=1_700_000_000_000,
            open=100.0,
            high=110.0,
            low=90.0,
            close=105.0,
            volume=500.0,
            symbol="AVAX/USDT",
            is_confirmed=True,
        )
        assert c.close == 105.0
        assert c.is_confirmed is True

    def test_frozen(self) -> None:
        c = Candle(1, 1.0, 2.0, 0.5, 1.5, 100.0, "AVAX/USDT", True)
        with pytest.raises((AttributeError, TypeError)):
            c.close = 99.0  # type: ignore[misc]


class TestSignal:
    def test_creation(self) -> None:
        s = Signal(
            timestamp=1_700_000_000_000,
            symbol="AVAX/USDT",
            direction=Direction.LONG,
            strength=0.8,
            strategy_name="test_strategy",
            metadata={"rsi": 40.0},
        )
        assert s.direction == Direction.LONG
        assert s.strength == 0.8

    def test_direction_enum(self) -> None:
        assert Direction.LONG.value == "long"
        assert Direction.SHORT.value == "short"
        assert Direction.FLAT.value == "flat"


class TestOrderResult:
    def _make(self) -> OrderResult:
        return OrderResult(
            order_id="exch_order_12345",
            client_order_id="my_order_001",
            symbol="AVAX/USDT",
            side=OrderSide.BUY,
            order_type=OrderType.MARKET,
            status=OrderStatus.FILLED,
            requested_quantity=10.0,
            filled_quantity=10.0,
            average_price=41.5,
            fee=0.01,
            fee_currency="USDT",
            timestamp=1_700_000_000_000,
        )

    def test_creation(self) -> None:
        o = self._make()
        assert o.status == OrderStatus.FILLED

    def test_status_enum_values(self) -> None:
        assert OrderStatus.PENDING.value == "pending"
        assert OrderStatus.FILLED.value == "filled"
        assert OrderStatus.CANCELLED.value == "cancelled"


class TestPosition:
    def test_creation(self) -> None:
        p = Position(
            symbol="AVAX/USDT",
            side=OrderSide.BUY,
            entry_price=40.0,
            quantity=10.0,
            current_price=42.0,
            unrealised_pnl=20.0,
            realised_pnl=0.0,
            opened_at=1_700_000_000_000,
            updated_at=1_700_000_001_000,
        )
        assert p.unrealised_pnl == 20.0


class TestErrorType:
    def test_retryable(self) -> None:
        assert ErrorType.NETWORK_TIMEOUT.is_retryable is True
        assert ErrorType.RATE_LIMIT.is_retryable is True
        assert ErrorType.SERVICE_UNAVAILABLE.is_retryable is True
        assert ErrorType.CONNECTION_ERROR.is_retryable is True

    def test_non_retryable(self) -> None:
        assert ErrorType.AUTHENTICATION_FAILED.is_retryable is False
        assert ErrorType.EXCHANGE_ERROR.is_retryable is False
        assert ErrorType.INVALID_ORDER.is_retryable is False
        assert ErrorType.UNKNOWN_ERROR.is_retryable is False


class TestBotState:
    def test_default_construction(self) -> None:
        state = BotState()
        assert state.equity == 0.0
        assert state.positions == {}

    def test_serialisation_roundtrip(self) -> None:
        state = BotState(
            equity=10_000.0,
            peak_equity=10_500.0,
            pnl_24h=-200.0,
            bot_started_at=1_700_000_000_000,
            last_heartbeat=1_700_000_060_000,
        )
        data = state.to_dict()
        recovered = BotState.from_dict(data)
        assert recovered.equity == 10_000.0
        assert recovered.peak_equity == 10_500.0
        assert recovered.pnl_24h == -200.0
        assert recovered.bot_started_at == 1_700_000_000_000

    def test_roundtrip_with_position(self) -> None:
        state = BotState()
        pos = Position(
            symbol="AVAX/USDT",
            side=OrderSide.BUY,
            entry_price=40.0,
            quantity=5.0,
            current_price=43.0,
            unrealised_pnl=15.0,
            realised_pnl=0.0,
            opened_at=1_700_000_000_000,
            updated_at=1_700_000_001_000,
        )
        state.positions["AVAX/USDT"] = pos
        data = state.to_dict()
        recovered = BotState.from_dict(data)
        assert "AVAX/USDT" in recovered.positions
        assert recovered.positions["AVAX/USDT"].entry_price == 40.0
