"""End-to-end integration tests for ``TradingBot``.

These tests construct a real ``TradingBot`` (broker=ig, candle_exchange=twelvedata,
topk_enabled=True) and drive its hot paths — ``_process_candle_ig_topk``,
``_topk_rerank_loop``, ``_handle_order_filled``, ``_close_position``, and the
startup state-restoration block in ``start()`` — with the IG REST client and the
Kronos predictor stubbed out.  No real network calls, no GPU inference, no
filesystem writes outside ``tmp_path``.

What these tests catch that the unit suite misses:
  * The -pre ETF stop-loss / PnL bug: ``_ig_quote_scale`` returned 1.0 for
    SLV/SPY/QQQ/DIA/USO/UNG, so any ETF position closed instantly with a bogus
    near-100 % loss the moment the first candle arrived.  The unit tests for
    ``_ig_quote_scale`` locked in the bug; only an integration test exercising
    the full per-candle path would have caught it.
  * register_position called without ``path_signal``: the path-aware static-TP
    and time-exit branches stayed dormant in production despite
    shipping the infrastructure.
  * The rerank loop's ``signal_history`` write path silently dropping path
    metrics; the ``asset_correlations`` snapshot never being written.
  * The Telegram ``send_topk_rerank`` ``bumped`` parameter not being threaded
    through from ``_topk_rerank_loop``.
  * Operator-restart resilience: consecutive_losses reset, IG position purge,
    take-profit state restore, correlation matrix restore.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from dataclasses import replace
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.config import BotConfig
from bot.core.event_bus import (
    EVENT_NEW_CANDLE,
    EVENT_POSITION_CLOSED,
)
from bot.core.models import (
    BotState,
    Candle,
    ErrorType,
    ExchangeError,
    OrderResult,
    OrderSide,
    OrderStatus,
    OrderType,
    Position,
    RiskState,
)
from bot.execution.ig_quote_scale import ig_quote_scale as _ig_quote_scale
from bot.main import TradingBot
from bot.strategy.kronos_signals import KronosPathSignal
from bot.strategy.rerank_runner import _snap_size_up_to_grid
from bot.strategy.take_profit import ExitReason
from bot.strategy.topk_strategy import AssetSignal

# ---------------------------------------------------------------------------
# Constants and helpers
# ---------------------------------------------------------------------------

HOUR_MS = 3_600_000
BASE_TS = 1_700_000_000_000  # arbitrary fixed epoch ms

# Kept small so warmup gates fire fast without seeding 400 candles per symbol.
TEST_CONTEXT_BARS = 5


def _candle(symbol: str, ts: int, close: float, *, vol: float = 0.0) -> Candle:
    """Build a flat OHLC candle.  Used by the seed helper and price-walk tests."""
    return Candle(
        symbol=symbol,
        timestamp=ts,
        open=close,
        high=close,
        low=close,
        close=close,
        volume=vol,
        is_confirmed=True,
    )


def _seed_candles(bot: TradingBot, symbol: str, base_close: float, *, count: int = 10) -> None:
    """Populate the in-memory store with `count` flat candles ending at `base_close`."""
    for i in range(count):
        bot.ctx.store.add_candle(_candle(symbol, BASE_TS + i * HOUR_MS, base_close))


def _make_position(
    symbol: str, entry_ig_level: float, *, quantity: float = 1.0, side: OrderSide = OrderSide.BUY
) -> Position:
    """Construct an open Position keyed by IG fill level (matches main.py storage)."""
    return Position(
        symbol=symbol,
        side=side,
        entry_price=entry_ig_level,
        quantity=quantity,
        current_price=entry_ig_level,
        unrealised_pnl=0.0,
        realised_pnl=0.0,
        opened_at=BASE_TS,
        updated_at=BASE_TS,
    )


def _make_signal(
    symbol: str,
    *,
    mean_return: float = 0.005,
    stop_pct: float = 0.01,
    direction_confidence: float = 0.85,
    uncertainty: float = 0.5,
    tradeable: bool = True,
    predicted_close: float | None = None,
) -> AssetSignal:
    return AssetSignal(
        symbol=symbol,
        mean_return=mean_return,
        std_return=stop_pct / 2.0,
        direction_confidence=direction_confidence,
        uncertainty=uncertainty,
        stop_pct=stop_pct,
        tradeable=tradeable,
        predicted_close=predicted_close if predicted_close is not None else 1.0,
        direction="LONG" if mean_return >= 0 else "SHORT",
    )


def _make_path_signal(
    symbol: str,
    *,
    entry_price: float = 1.0,
    mfe: float = 0.02,
    mae: float = 0.005,
    peak_bar: int = 60,
    monotonicity: float = 0.9,
    mean_return: float = 0.005,
    predicted_volatility: float = 0.01,
) -> KronosPathSignal:
    pred_len = 120
    closes = [entry_price * (1.0 + mean_return * (i / pred_len)) for i in range(pred_len)]
    highs = [entry_price * (1.0 + mfe * (i / pred_len)) for i in range(pred_len)]
    lows = [entry_price * (1.0 - mae * (1 - i / pred_len)) for i in range(pred_len)]
    return KronosPathSignal(
        symbol=symbol,
        mean_return=mean_return,
        std_return=0.005,
        direction_confidence=0.85,
        uncertainty=0.5,
        stop_pct=0.01,
        predicted_max_high=entry_price * (1.0 + mfe),
        predicted_min_low=entry_price * (1.0 - mae),
        predicted_mfe_pct=mfe,
        predicted_mae_pct=mae,
        predicted_peak_bar=peak_bar,
        predicted_volatility=predicted_volatility,
        predicted_path_drawdown=mae,
        monotonicity=monotonicity,
        ranking_score=mean_return * abs(monotonicity),
        predicted_closes=closes,
        predicted_highs=highs,
        predicted_lows=lows,
    )


def _ig_level(symbol: str, td_price: float) -> float:
    """Convert a Twelve Data face-value price into the matching IG fill level."""
    return td_price * _ig_quote_scale(symbol)


def _make_order_result(
    *,
    symbol: str,
    side: OrderSide,
    average_price: float,
    filled_quantity: float = 1.0,
    order_id: str = "DEAL_ID_TEST",
    status: OrderStatus = OrderStatus.FILLED,
) -> OrderResult:
    return OrderResult(
        order_id=order_id,
        client_order_id=order_id,
        symbol=symbol,
        side=side,
        order_type=OrderType.MARKET,
        status=status,
        requested_quantity=filled_quantity,
        filled_quantity=filled_quantity,
        average_price=average_price,
        fee=0.0,
        fee_currency="GBP",
        timestamp=BASE_TS,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def bot(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[TradingBot]:
    """Construct a TradingBot wired for IG + Twelve Data + TopK with mocked IG."""
    monkeypatch.chdir(tmp_path)
    config = BotConfig(
        broker="ig",
        bot_env="demo",
        ig_demo_api="test_api_key",
        ig_demo_username="test_user",
        ig_demo_password="test_pass",
        candle_exchange="twelvedata",
        twelve_data_api="test_td_key",
        topk_enabled=True,
        kronos_dir=str(tmp_path / "kronos_src"),
        topk_k=3,
        topk_pred_len=120,
        topk_min_predicted_return=0.001,
        topk_min_confidence=0.70,
        topk_max_uncertainty=2.0,
        topk_min_stop_pct=0.005,
        topk_vol_stop_multiplier=2.0,
        kronos_context_bars=TEST_CONTEXT_BARS,
        topk_watchlist=["EUR/USD", "USD/JPY", "XAU/USD", "GBP/USD"],
    )
    config.validate_config()
    bot = TradingBot(config)

    # Mock IG client (constructed in __init__ but never connected)
    mock_ig = AsyncMock()
    mock_ig.fetch_balance = AsyncMock(
        return_value={"equity": 10_000.0, "margin": 100.0, "balance": 10_000.0, "open_pnl": 0.0}
    )
    # Default `/positions` response — empty list so reconcile is a no-op when
    # no test fixtures set up live IG state.  Tests that need positions on IG
    # (e.g. signal-decay, startup recovery) override this.
    mock_ig.fetch_positions_raw = AsyncMock(return_value=[])
    mock_ig._cst = "test_cst_token"  # for heartbeat connected check
    # require_tradeable returns the IG min deal size (or None); default None =
    # "no minimum known" so the entry path doesn't skip. Tests that exercise the
    # market-closed gate override .side_effect with MarketClosedError.
    mock_ig.require_tradeable = AsyncMock(return_value=None)
    bot.ctx.ig_client = mock_ig

    # The legacy IG-native candle feed (the twelvedata-rollback path this fixture
    # exercises) is spawned by ``bot.start()`` and would do a REAL Lightstreamer
    # connect against the mocked IG client — which throws deep in the LS library
    # ("'<' not supported between instances of 'coroutine' and 'int'") and leaks
    # an unretrieved background-task error into every start()-driven test. No test
    # here needs a live feed (production runs candle_exchange=eodhd, which skips
    # this feed entirely; the start() tests already stub _build_feed_task). Make
    # its run-loop dormant so start() spawns a clean idle task instead.
    async def _dormant_candle_feed(_self: object) -> None:
        await bot.ctx.shutdown_event.wait()

    monkeypatch.setattr("bot.data.ig_candle_feed.IGCandleLSFeed.run", _dormant_candle_feed)

    # Silence Telegram (real alerter is no-op when token+chat are empty, but
    # mocking gives us call-site introspection).
    bot.ctx.alerter = AsyncMock()
    bot.ctx.alerter.send_topk_rerank = AsyncMock(return_value=True)
    bot.ctx.alerter.send_trade_alert = AsyncMock(return_value=True)
    bot.ctx.alerter.alert_take_profit = AsyncMock(return_value=True)
    bot.ctx.alerter.send_startup = AsyncMock(return_value=True)
    bot.ctx.alerter.send_shutdown = AsyncMock(return_value=True)
    bot.ctx.alerter.send_risk_alert = AsyncMock(return_value=True)

    yield bot

    # Cleanup: close DB connection so tmp_path can be cleared.
    if bot.ctx.candle_db._conn is not None:
        bot.ctx.candle_db.close()


@pytest.fixture
def bot_with_topk(bot: TradingBot) -> TradingBot:
    """Bot with two open positions and matching cached topk_signals.

    Used by stop-loss / TP tests: the position is keyed by the candle symbol
    and stored at IG fill level, matching production storage.
    """
    # USD/JPY position at 156.585 FX (IG level 15658.5), stop_pct 1%
    bot.ctx.state.positions["USD/JPY"] = _make_position("USD/JPY", _ig_level("USD/JPY", 156.585))
    bot.ctx.ig_deal_ids["USD/JPY"] = "DEAL_JPY"
    # XAU/USD position, stop_pct 1%.  XAU is IG-native (ig_quote_scale=1.0) so its
    # candles already arrive at IG level and entry/current compare 1:1 — the test
    # still guards that the same scale is applied on both sides of the stop check.
    bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
    bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"

    bot.ctx.topk_signals = [
        _make_signal("USD/JPY", stop_pct=0.01),
        _make_signal("XAU/USD", stop_pct=0.01),
        _make_signal("EUR/USD", stop_pct=0.005),
    ]
    bot.ctx.topk_scanned = True
    bot.ctx.topk_selected = []
    return bot


# ---------------------------------------------------------------------------
# A. Stop-loss across asset classes (covers the original ETF bug)
# ---------------------------------------------------------------------------


class TestStopLossAcrossAssetClasses:
    """``_process_candle_ig_topk`` must compare entry (IG level) and current
    (Twelve Data face value) in matching units.  These tests catch the regression
    that closed every ETF position on the first candle."""

    @pytest.mark.asyncio
    async def test_metal_xau_at_entry_does_not_close(self, bot_with_topk: TradingBot) -> None:
        # XAU entry and current at the same level → no loss.
        pos = bot_with_topk.ctx.state.positions["XAU/USD"]
        await bot_with_topk.ctx.runner.process_candle_ig_topk("XAU/USD", 13.95, pos)
        bot_with_topk.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_metal_xau_small_loss_within_stop_does_not_close(
        self, bot_with_topk: TradingBot
    ) -> None:
        # 0.4% real loss; stop is 1%
        pos = bot_with_topk.ctx.state.positions["XAU/USD"]
        await bot_with_topk.ctx.runner.process_candle_ig_topk("XAU/USD", 13.894, pos)
        bot_with_topk.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_metal_xau_breach_triggers_close(self, bot_with_topk: TradingBot) -> None:
        # 2% real loss → exceeds 1% stop.  Close triggered.
        bot_with_topk.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="XAU/USD", side=OrderSide.SELL, average_price=13.671
            )
        )
        pos = bot_with_topk.ctx.state.positions["XAU/USD"]
        await bot_with_topk.ctx.runner.process_candle_ig_topk("XAU/USD", 13.671, pos)
        bot_with_topk.ctx.ig_client.close_position.assert_awaited_once()
        # Stop-loss path also deregisters from TP manager so a stale state
        # doesn't survive into the next entry.
        assert "XAU/USD" not in bot_with_topk.ctx.tp_manager._positions

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "symbol,entry_td,current_td",
        [
            ("XAU/USD", 73.13, 73.13),  # IG-native metal, scale 1.0
            ("F", 12.20, 12.20),  # US share, scale 100
            ("XOM", 110.28, 110.28),
        ],
    )
    async def test_asset_at_entry_no_close(
        self, bot: TradingBot, symbol: str, entry_td: float, current_td: float
    ) -> None:
        bot.ctx.state.positions[symbol] = _make_position(symbol, _ig_level(symbol, entry_td))
        bot.ctx.ig_deal_ids[symbol] = f"DEAL_{symbol}"
        bot.ctx.topk_signals = [_make_signal(symbol, stop_pct=0.01)]
        bot.ctx.topk_scanned = True
        await bot.ctx.runner.process_candle_ig_topk(
            symbol, current_td, bot.ctx.state.positions[symbol]
        )
        bot.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_jpy_at_entry_does_not_close(self, bot_with_topk: TradingBot) -> None:
        pos = bot_with_topk.ctx.state.positions["USD/JPY"]
        await bot_with_topk.ctx.runner.process_candle_ig_topk("USD/JPY", 156.585, pos)
        bot_with_topk.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_jpy_breach_triggers_close(self, bot_with_topk: TradingBot) -> None:
        bot_with_topk.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="USD/JPY", side=OrderSide.SELL, average_price=15424.2
            )
        )
        # 1.5% real loss exceeds 1% stop
        pos = bot_with_topk.ctx.state.positions["USD/JPY"]
        await bot_with_topk.ctx.runner.process_candle_ig_topk("USD/JPY", 154.242, pos)
        bot_with_topk.ctx.ig_client.close_position.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forex_eurusd_at_entry_does_not_close(self, bot: TradingBot) -> None:
        # EUR/USD pip 0.0001 → IG level = price × 10000.  Storing at face value
        # would compute a 99.99 % loss on the first candle (the production bug
        # this regression test guards against).
        bot.ctx.state.positions["EUR/USD"] = _make_position(
            "EUR/USD", _ig_level("EUR/USD", 1.17225)
        )
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_EU"
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        bot.ctx.topk_scanned = True
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.17225, bot.ctx.state.positions["EUR/USD"]
        )
        bot.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_xau_at_entry_does_not_close(self, bot: TradingBot) -> None:
        # Gold pip_value=1.0, scale collapses to 1× — entry IG level == spot $
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", 4616.52)
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        bot.ctx.topk_signals = [_make_signal("XAU/USD", stop_pct=0.01)]
        bot.ctx.topk_scanned = True
        await bot.ctx.runner.process_candle_ig_topk(
            "XAU/USD", 4616.52, bot.ctx.state.positions["XAU/USD"]
        )
        bot.ctx.ig_client.close_position.assert_not_called()

    @pytest.mark.asyncio
    async def test_stop_pct_falls_back_when_signal_missing(self, bot: TradingBot) -> None:
        """No matching signal → use TOPK_MIN_STOP_PCT."""
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", _ig_level("EUR/USD", 1.20))
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_FB"
        bot.ctx.topk_signals = []  # no cached signal → fall back to min_stop_pct=0.005
        bot.ctx.topk_scanned = True
        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="EUR/USD", side=OrderSide.SELL, average_price=_ig_level("EUR/USD", 1.190)
            )
        )
        # 0.83% loss, exceeds the 0.5% fallback floor
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.190, bot.ctx.state.positions["EUR/USD"]
        )
        bot.ctx.ig_client.close_position.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stop_loss_failure_keeps_tp_state_for_retry(self, bot: TradingBot) -> None:
        """Regression for the production bug found 2026-05-03: when IG rejects
        the stop-loss close (e.g. ``MARKET_CLOSED_WITH_EDITS`` outside trading
        hours), the bot must keep the TP state and ``_ig_deal_ids`` so the next
        candle retries, and surface the failure to the operator via
        ``send_error`` so the silent-stale-position scenario doesn't recur.
        """
        ig_entry = _ig_level("EUR/USD", 1.20)
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", ig_entry)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_RETRY"
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        bot.ctx.topk_scanned = True
        bot.ctx.tp_manager.register_position(
            "EUR/USD", ig_entry, _make_signal("EUR/USD", stop_pct=0.005), BASE_TS
        )
        bot.ctx.ig_client.close_position = AsyncMock(side_effect=RuntimeError("MARKET_CLOSED"))
        # After the close fails, the bot probes /positions to disambiguate
        # a ghost dealId from a flaky endpoint.  IG still reports the position
        # as live → state must be preserved for retry.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.EURUSD.TODAY.IP"},
                    "position": {"dealId": "DEAL_RETRY"},
                }
            ]
        )

        # 0.83% loss > 0.5% stop → triggers the close attempt
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.190, bot.ctx.state.positions["EUR/USD"]
        )

        bot.ctx.ig_client.close_position.assert_awaited_once()
        # Preserved for retry on the next candle
        assert "EUR/USD" in bot.ctx.tp_manager._positions
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_RETRY"
        assert "EUR/USD" in bot.ctx.state.positions
        cast(AsyncMock, bot.ctx.alerter).send_error.assert_awaited()


# ---------------------------------------------------------------------------
# B. Entry path through _process_candle_ig_topk
# ---------------------------------------------------------------------------


class TestEntryPath:
    @pytest.mark.asyncio
    async def test_entry_skipped_when_not_scanned(self, bot: TradingBot) -> None:
        bot.ctx.topk_scanned = False
        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)
        bot.ctx.ig_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_skipped_when_not_in_topk(self, bot: TradingBot) -> None:
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["GBP/USD"]
        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)
        bot.ctx.ig_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_skipped_when_market_closed(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("bot.trading_hours.is_market_open", lambda *a, **kw: False)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)
        bot.ctx.ig_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_placed_when_selected_and_market_open(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        bot.ctx.topk_strategy._path_signals = {
            "EUR/USD": _make_path_signal("EUR/USD", entry_price=1.10)
        }

        bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
                order_id="REF_ENTRY",
            )
        )
        bot.ctx.ig_client.confirm_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=1.10,
                order_id="DEAL_ENTRY",
            )
        )

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        bot.ctx.ig_client.place_order.assert_awaited_once()
        order_arg = bot.ctx.ig_client.place_order.call_args.args[0]
        assert order_arg.epic == "CS.D.EURUSD.TODAY.IP"
        assert order_arg.direction == "BUY"
        assert order_arg.size > 0
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_ENTRY"

    @pytest.mark.asyncio
    async def test_entry_register_position_includes_path_signal(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """regression: the ``path_signal`` kwarg must be wired through
        so that path-aware static TP and time exit can fire."""
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        path_sig = _make_path_signal("EUR/USD", entry_price=1.10, mfe=0.03, mae=0.005, peak_bar=72)
        bot.ctx.topk_strategy._path_signals = {"EUR/USD": path_sig}

        bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
            )
        )
        bot.ctx.ig_client.confirm_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=1.10,
                order_id="DEAL_REG",
            )
        )

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        tp_state = bot.ctx.tp_manager._positions["EUR/USD"]
        assert tp_state.predicted_mfe_pct == pytest.approx(0.03)
        assert tp_state.predicted_mae_pct == pytest.approx(0.005)
        assert tp_state.predicted_peak_bar == 72

    @pytest.mark.asyncio
    async def test_entry_skipped_when_risk_rejects(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RED drawdown tier → risk manager rejects, no order placed.

        ``_process_candle_ig_topk`` calls ``update_equity`` before evaluating
        the order, which recomputes the tier from peak vs current.  Use a
        peak well above current so the tier stays RED through the recompute.
        """
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        # peak=20k, equity=10k → 50 % drawdown → RED (>=15 %)
        bot.ctx.risk_manager._drawdown.set_peak_equity(20_000.0)
        bot.ctx.risk_manager._drawdown._equity = 10_000.0  # direct write bypasses tier callback

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        bot.ctx.ig_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_failure_no_phantom_state(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``confirm_order`` raises after ``place_order`` succeeds → no
        ``_ig_deal_ids`` entry written, no TP-manager registration."""
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]

        bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
            )
        )
        bot.ctx.ig_client.confirm_order = AsyncMock(side_effect=RuntimeError("IG rejected"))

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        assert "EUR/USD" not in bot.ctx.ig_deal_ids
        assert "EUR/USD" not in bot.ctx.tp_manager._positions

    @pytest.mark.asyncio
    async def test_entry_skipped_when_size_zero(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``compute_ig_size`` returns 0 → no order placed.

        We force this by giving the IG client an equity of zero so the sizing
        formula produces 0 stake.
        """
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        bot.ctx.ig_client.fetch_balance = AsyncMock(
            return_value={"equity": 0.0, "margin": 0.0, "balance": 0.0, "open_pnl": 0.0}
        )
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        bot.ctx.ig_client.place_order.assert_not_called()

    @pytest.mark.asyncio
    async def test_entry_retries_on_size_increment_at_snapped_size(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A SIZE_INCREMENT reject snaps the stake UP to the next 0.1 £/pt grid
        and retries once; the second attempt fills and registers state."""
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        # Pin the risk-sized stake to a known off-grid value (0.28 £/pt) so the
        # snap target is deterministic: 0.28 → next 0.1 grid = 0.3.
        monkeypatch.setattr(
            "bot.strategy.rerank_runner.RiskManager.compute_ig_size", lambda *a, **kw: 0.28
        )
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]

        bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
            )
        )
        # First confirm rejects with SIZE_INCREMENT, second fills.
        bot.ctx.ig_client.confirm_order = AsyncMock(
            side_effect=[
                ExchangeError("IG rejected order REF: SIZE_INCREMENT", ErrorType.INVALID_ORDER),
                _make_order_result(
                    symbol="CS.D.EURUSD.TODAY.IP",
                    side=OrderSide.BUY,
                    average_price=1.10,
                    order_id="DEAL_RETRY",
                ),
            ]
        )

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        assert bot.ctx.ig_client.place_order.await_count == 2
        assert bot.ctx.ig_client.place_order.call_args_list[0].args[0].size == pytest.approx(0.28)
        assert bot.ctx.ig_client.place_order.call_args_list[1].args[0].size == pytest.approx(0.3)
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_RETRY"

    @pytest.mark.asyncio
    async def test_entry_size_increment_no_retry_when_already_on_grid(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """SIZE_INCREMENT on a stake already on the 0.1 grid → no pointless
        retry; the slot is left unfilled for the next rerank."""
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        # min deal size forces the stake to land exactly on a grid point (0.3)
        # via the min-deal-size floor check before placement.
        bot.ctx.ig_client.require_tradeable = AsyncMock(return_value=0.3)
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = ["EUR/USD"]
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]

        # Patch sizing so final_size is exactly 0.3 (on the grid).
        monkeypatch.setattr(
            "bot.strategy.rerank_runner.RiskManager.compute_ig_size", lambda *a, **kw: 0.3
        )

        bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
            )
        )
        bot.ctx.ig_client.confirm_order = AsyncMock(
            side_effect=ExchangeError(
                "IG rejected order REF: SIZE_INCREMENT", ErrorType.INVALID_ORDER
            )
        )

        await bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10, None)

        # Placed once, no retry (0.3 is already on the grid), no phantom state.
        assert bot.ctx.ig_client.place_order.await_count == 1
        assert "EUR/USD" not in bot.ctx.ig_deal_ids


@pytest.mark.parametrize(
    ("size", "min_deal", "expected"),
    [
        # Empirically-rejected US-share stakes → next 0.1 grid point.
        (0.28, 0.24, 0.3),
        (0.27, 0.24, 0.3),
        (0.34, 0.24, 0.4),
        (0.53, 0.24, 0.6),
        # Already on the grid → unchanged (caller then skips the retry).
        (0.30, 0.24, 0.3),
        (0.40, 0.24, 0.4),
        # Below the min deal size → clamped up to the first on-grid point >= min.
        (0.05, 0.24, 0.3),
        # No known minimum.
        (0.28, None, 0.3),
    ],
)
def test_snap_size_up_to_grid(size: float, min_deal: float | None, expected: float) -> None:
    assert _snap_size_up_to_grid(size, min_deal) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# C. Take-profit per-candle evaluation through main
# ---------------------------------------------------------------------------


class TestTakeProfitPerCandle:
    """Drives ``_process_candle_ig_topk`` with an open position and verifies
    that the TP manager's verdict fires the unified close path."""

    def _arm(
        self,
        bot: TradingBot,
        symbol: str,
        entry_td: float,
        *,
        path_signal: KronosPathSignal | None = None,
    ) -> Position:
        ig_entry = _ig_level(symbol, entry_td)
        pos = _make_position(symbol, ig_entry)
        bot.ctx.state.positions[symbol] = pos
        bot.ctx.ig_deal_ids[symbol] = f"DEAL_{symbol}"
        sig = _make_signal(symbol, stop_pct=0.01)
        bot.ctx.topk_signals = [sig]
        bot.ctx.topk_scanned = True
        # Open the position at wallclock now so the per-candle path's
        # ``time.time()``-based age calculation doesn't immediately fire the
        # time-limit exit branch.
        opened_now_ms = int(time.time() * 1000)
        bot.ctx.tp_manager.register_position(
            symbol, ig_entry, sig, opened_now_ms, path_signal=path_signal
        )
        return pos

    @pytest.mark.asyncio
    async def test_static_tp_fires_at_path_aware_target(self, bot: TradingBot) -> None:
        """MFE 3 % → target = 3 % × 0.85 = 2.55 % above entry."""
        path_sig = _make_path_signal("EUR/USD", entry_price=1.10, mfe=0.03, mae=0.005, peak_bar=60)
        self._arm(bot, "EUR/USD", 1.10, path_signal=path_sig)
        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP", side=OrderSide.SELL, average_price=1.130
            )
        )

        # Walk price up to 2.7 % (above 2.55 % static-TP target)
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.130, bot.ctx.state.positions["EUR/USD"]
        )
        bot.ctx.ig_client.close_position.assert_awaited_once()
        # alert_take_profit should fire with reason=static_take_profit
        cast(AsyncMock, bot.ctx.alerter).alert_take_profit.assert_awaited()
        args = cast(AsyncMock, bot.ctx.alerter).alert_take_profit.call_args
        assert args.args[1] == ExitReason.STATIC_TP.value

    @pytest.mark.asyncio
    async def test_trailing_breakeven_arms_then_exits(self, bot: TradingBot) -> None:
        """Stage-1 trailing: profit >= 1× stop arms breakeven; subsequent
        retreat below entry+buffer triggers the trailing-breakeven exit.

        Price chosen at +1.2 % profit so it clears the breakeven activation
        (1× stop = 1 %) without tripping the trail activation (2× stop = 2 %)
        or the fallback static-TP target (1.5×stop = 1.5 %).
        """
        self._arm(bot, "EUR/USD", 1.10)  # entry IG 1.10, stop 1 %, no path → fallback TP
        # Step 1: walk to 1.2 % gain → arms breakeven, below static-TP target
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.1132, bot.ctx.state.positions["EUR/USD"]
        )
        st = bot.ctx.tp_manager._positions["EUR/USD"]
        assert st.breakeven_armed is True
        assert st.trail_armed is False
        # Breakeven trail is entry × (1 + breakeven_buffer=0.001), expressed in IG
        # level units to match the registered ig_entry.
        assert st.current_trailing_stop == pytest.approx(
            _ig_level("EUR/USD", 1.10 * 1.001), rel=1e-6
        )

        # Step 2: price retreats below entry+buffer → exit
        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP", side=OrderSide.SELL, average_price=1.099
            )
        )
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.099, bot.ctx.state.positions["EUR/USD"]
        )
        bot.ctx.ig_client.close_position.assert_awaited_once()
        # Reason = TRAILING_STOP_BREAKEVEN (Stage 2 was never armed)
        assert (
            cast(AsyncMock, bot.ctx.alerter).alert_take_profit.call_args.args[1]
            == ExitReason.TRAILING_STOP_BREAKEVEN.value
        )

    @pytest.mark.asyncio
    async def test_trailing_ratchet_arms_above_2x_stop(self, bot: TradingBot) -> None:
        """Stage-2 trailing: profit >= 2× stop arms ratchet; the trailing stop
        floats up with peak_price and exits when price retreats to it.

        Static TP is disabled for this test so the 2.5 %-profit walk-up arms
        the ratchet without first firing static_tp at the 1.5 % fallback
        target.  This isolates the ratchet behaviour.
        """
        # Disable static TP so it doesn't fire at 1.5 % before ratchet at 2 %
        bot.ctx.tp_manager._config.static_enabled = False
        self._arm(bot, "EUR/USD", 1.10)
        # Step 1: walk to +2.5 % → arms ratchet (peak 1.1275)
        peak = 1.1275
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", peak, bot.ctx.state.positions["EUR/USD"]
        )
        st = bot.ctx.tp_manager._positions["EUR/USD"]
        assert st.trail_armed is True
        # Trailing stop = peak × (1 - stop × multiplier=0.5), in IG level units.
        expected_trailing = _ig_level("EUR/USD", peak * (1 - 0.01 * 0.5))
        assert st.current_trailing_stop == pytest.approx(expected_trailing, rel=1e-6)

        # Step 2: price drops below trailing → exit
        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP", side=OrderSide.SELL, average_price=1.118
            )
        )
        await bot.ctx.runner.process_candle_ig_topk(
            "EUR/USD", 1.118, bot.ctx.state.positions["EUR/USD"]
        )
        bot.ctx.ig_client.close_position.assert_awaited_once()
        # Reason = TRAILING_STOP_RATCHET
        assert (
            cast(AsyncMock, bot.ctx.alerter).alert_take_profit.call_args.args[1]
            == ExitReason.TRAILING_STOP_RATCHET.value
        )


# ---------------------------------------------------------------------------
# D. TopK rerank loop end-to-end
# ---------------------------------------------------------------------------


class TestRerankLoop:
    """Runs ``_topk_rerank_loop`` for one full iteration with shutdown asserted
    after the Telegram alert.  Verifies signal_history, correlation snapshot,
    bumped passthrough, equity refresh, and signal-decay closes."""

    def _prepare(
        self,
        bot: TradingBot,
        signals: list[AssetSignal],
        path_signals: dict[str, KronosPathSignal] | None = None,
        returns_map: dict[str, list[float]] | None = None,
    ) -> None:
        """Seed candles + stub the Kronos predictor's scan() with canned signals.

        The strategy's ``select_top_k``, correlation tracker, and path-signal
        plumbing all run unmodified — only the heavyweight inference is bypassed.
        """
        for sym in {s.symbol for s in signals} | set(bot.ctx.candle_symbols):
            _seed_candles(bot, sym, base_close=1.0, count=TEST_CONTEXT_BARS)

        bot.ctx.topk_strategy._path_signals = dict(path_signals or {})
        if returns_map is not None:
            bot.ctx.topk_strategy._correlation_tracker.update(returns_map)

        async def fake_scan(symbols: list[str], fetcher: Any) -> list[AssetSignal]:
            # Don't replace _path_signals here; tests set it via the kw arg above.
            return signals

        bot.ctx.topk_strategy.scan = fake_scan

        # Make send_topk_rerank trip shutdown so the loop body runs once.
        async def stop_after_alert(*args: Any, **kwargs: Any) -> bool:
            bot.ctx.shutdown_event.set()
            return True

        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.side_effect = stop_after_alert

    @pytest.mark.asyncio
    async def test_signal_history_rows_carry_path_metrics(self, bot: TradingBot) -> None:
        sig = _make_signal("EUR/USD", mean_return=0.005)
        path = _make_path_signal("EUR/USD", entry_price=1.0, mfe=0.025, mae=0.004)
        self._prepare(bot, [sig], path_signals={"EUR/USD": path})

        await bot.ctx.runner.topk_rerank_loop()

        rows = bot.ctx.candle_db.get_signal_history(symbol="EUR/USD")
        assert len(rows) == 1
        row = rows[0]
        assert row["mean_return"] == pytest.approx(0.005)
        assert row["predicted_mfe_pct"] == pytest.approx(0.025)
        assert row["predicted_mae_pct"] == pytest.approx(0.004)
        assert row["predicted_volatility"] is not None
        assert row["monotonicity"] is not None

    @pytest.mark.asyncio
    async def test_signal_history_path_metrics_null_when_path_missing(
        self, bot: TradingBot
    ) -> None:
        """If a symbol has no path signal (e.g. degenerate prediction), the
        path columns are NULL and the row still gets written."""
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig], path_signals={})  # no path entry

        await bot.ctx.runner.topk_rerank_loop()

        rows = bot.ctx.candle_db.get_signal_history(symbol="EUR/USD")
        assert len(rows) == 1
        assert rows[0]["predicted_mfe_pct"] is None
        assert rows[0]["predicted_mae_pct"] is None
        assert rows[0]["mean_return"] == pytest.approx(0.005)

    @pytest.mark.asyncio
    async def test_correlation_snapshot_persisted(self, bot: TradingBot) -> None:
        sig_a = _make_signal("EUR/USD", mean_return=0.005)
        sig_b = _make_signal("USD/JPY", mean_return=0.004)
        returns = {
            "EUR/USD": [0.001, -0.002, 0.003, -0.001, 0.002, 0.001, -0.003] * 10,
            "USD/JPY": [-0.001, 0.002, -0.003, 0.001, -0.002, -0.001, 0.003] * 10,
        }
        self._prepare(bot, [sig_a, sig_b], returns_map=returns)

        await bot.ctx.runner.topk_rerank_loop()

        # Wait briefly for the to_thread write to flush
        await asyncio.sleep(0.05)
        matrix = bot.ctx.candle_db.read_latest_correlations()
        assert "EUR/USD" in matrix
        assert "USD/JPY" in matrix["EUR/USD"]

    @pytest.mark.asyncio
    async def test_bumped_threaded_to_telegram(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two strongly-correlated symbols → second one is bumped by the
        correlation filter; the bumped tuple must reach the Telegram payload."""
        # The rerank now filters selection by is_market_open; pin it True so the
        # selection logic is exercised regardless of when the suite runs.
        monkeypatch.setattr("bot.strategy.rerank_runner.is_market_open", lambda *a, **kw: True)
        sig_a = _make_signal("EUR/USD", mean_return=0.010)
        sig_b = _make_signal("USD/JPY", mean_return=0.005)
        # Perfect positive correlation so JPY is bumped vs the higher-ranked EUR/USD
        returns = {
            "EUR/USD": [0.001, 0.002, 0.003, 0.004, 0.005] * 5,
            "USD/JPY": [0.001, 0.002, 0.003, 0.004, 0.005] * 5,
        }
        self._prepare(bot, [sig_a, sig_b], returns_map=returns)
        # Tighten correlation threshold so the bump triggers
        bot.ctx.topk_strategy._correlation_tracker._config.max_correlation = 0.6

        await bot.ctx.runner.topk_rerank_loop()

        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.assert_awaited()
        kwargs = cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.call_args.kwargs
        assert kwargs["bumped"] is not None
        bumped_syms = [t[0] for t in kwargs["bumped"]]
        assert "USD/JPY" in bumped_syms

    @pytest.mark.asyncio
    async def test_equity_refreshed_at_rerank(self, bot: TradingBot) -> None:
        sig = _make_signal("EUR/USD")
        self._prepare(bot, [sig])
        bot.ctx.ig_client.fetch_balance.return_value = {
            "equity": 12_345.0,
            "margin": 200.0,
            "balance": 12_300.0,
            "open_pnl": 45.0,
        }

        await bot.ctx.runner.topk_rerank_loop()

        assert bot.ctx.risk_manager._drawdown.equity == pytest.approx(12_345.0)
        assert bot.ctx.state.cash == pytest.approx(12_300.0)
        assert bot.ctx.state.open_pnl == pytest.approx(45.0)

    @pytest.mark.asyncio
    async def test_equity_fetch_failure_does_not_break_rerank(self, bot: TradingBot) -> None:
        """Network blip on /accounts must not crash the loop — alert still
        sends, just with equity=None."""
        sig = _make_signal("EUR/USD")
        self._prepare(bot, [sig])
        bot.ctx.ig_client.fetch_balance.side_effect = RuntimeError("network blip")

        await bot.ctx.runner.topk_rerank_loop()

        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.assert_awaited()
        kwargs = cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.call_args.kwargs
        assert kwargs["equity"] is None

    @pytest.mark.asyncio
    async def test_alert_uses_live_position_price_not_candle(self, bot: TradingBot) -> None:
        """The per-position price in the alert must come from IG's live
        /positions (so it matches the live aggregate Open P&L), not the last
        closed candle (seeded at 1.0 → IG level 10000)."""
        sig = _make_signal("EUR/USD")
        self._prepare(bot, [sig])
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", _ig_level("EUR/USD", 1.10))
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_EUR"
        # Keep the position through the per-rerank reconcile (it purges anything
        # not on IG before the alert builds).
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {"market": {"epic": "CS.D.EURUSD.TODAY.IP"}, "position": {"dealId": "DEAL_EUR"}}
            ]
        )
        # Live IG position (keyed by EPIC) with a current price distinct from the
        # candle so the source is unambiguous.
        bot.ctx.ig_client.fetch_positions = AsyncMock(
            return_value=[_make_position("CS.D.EURUSD.TODAY.IP", 11050.0)]
        )

        await bot.ctx.runner.topk_rerank_loop()

        prices = cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.call_args.kwargs[
            "current_prices"
        ]
        # Display = IG level / scale: live bid 11050 → 1.1050 (not candle 1.0).
        assert prices["EUR/USD"] == pytest.approx(1.1050)

    @pytest.mark.asyncio
    async def test_alert_falls_back_to_candle_when_positions_fetch_fails(
        self, bot: TradingBot
    ) -> None:
        sig = _make_signal("EUR/USD")
        self._prepare(bot, [sig])
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", _ig_level("EUR/USD", 1.10))
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_EUR"
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {"market": {"epic": "CS.D.EURUSD.TODAY.IP"}, "position": {"dealId": "DEAL_EUR"}}
            ]
        )
        bot.ctx.ig_client.fetch_positions = AsyncMock(side_effect=RuntimeError("blip"))

        await bot.ctx.runner.topk_rerank_loop()

        prices = cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.call_args.kwargs[
            "current_prices"
        ]
        # Fallback to the candle: close 1.0 × scale 10000 → display 1.0.
        assert prices["EUR/USD"] == pytest.approx(1.0)

    @pytest.mark.asyncio
    async def test_signal_decay_flip_closes_open_position(self, bot: TradingBot) -> None:
        """Open position whose mean_return flips sign → SIGNAL_DECAY_FLIP exit
        fires inside the rerank loop's per-position evaluation."""
        # Open an XAU/USD position (IG-native, scale 1.0 → entry == candle level)
        ig_entry = _ig_level("XAU/USD", 13.95)
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", ig_entry)
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        # Tell the per-rerank reconcile that XAU/USD is still open on IG so it's
        # not purged before signal_decay evaluation runs.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.USCGC.TODAY.IP"},
                    "position": {"dealId": "DEAL_XAU"},
                }
            ]
        )
        # Register at the TP manager with positive entry mean_return
        entry_sig = _make_signal("XAU/USD", mean_return=0.02, stop_pct=0.01)
        bot.ctx.tp_manager.register_position("XAU/USD", ig_entry, entry_sig, BASE_TS)

        # New rerank signal flips sign — should close the position
        flipped = _make_signal("XAU/USD", mean_return=-0.01, stop_pct=0.01)
        # Also seed candles for XAU/USD so it shows up in the store's latest-candle lookup
        _seed_candles(bot, "XAU/USD", base_close=13.95, count=TEST_CONTEXT_BARS)
        self._prepare(bot, [flipped])

        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.USCGC.TODAY.IP", side=OrderSide.SELL, average_price=2790.0
            )
        )

        await bot.ctx.runner.topk_rerank_loop()

        bot.ctx.ig_client.close_position.assert_awaited_once()
        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.alert_take_profit.assert_awaited()
        # The exit reason should be the signal-decay flip
        assert alerter.alert_take_profit.call_args.args[1] == ExitReason.SIGNAL_DECAY_FLIP.value

    @pytest.mark.asyncio
    async def test_topk_state_updated_after_rerank(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("bot.strategy.rerank_runner.is_market_open", lambda *a, **kw: True)
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        assert bot.ctx.topk_scanned is True
        assert bot.ctx.topk_selected == ["EUR/USD"]
        assert bot.ctx.topk_signals == [sig]

    @pytest.mark.asyncio
    async def test_periodic_reconcile_purges_externally_closed_position(
        self, bot: TradingBot
    ) -> None:
        """sync regression: an open position absent from IG ``/positions``
        (e.g. IG server-side stop fired between rerank cycles, operator closed
        manually via the web UI) must be purged from ``_state.positions``,
        ``_ig_deal_ids``, and the TP manager during the per-rerank reconcile —
        without waiting for the next bot restart.
        """
        # Seed an XAU/USD position locally as if it were open.  _risk_manager._open_positions
        # is keyed by IG EPIC (see IGClient.OrderResult.symbol=order.epic), not by
        # the candle symbol — seed both so the reconcile purge is exercised against
        # the real key.
        xau_epic = bot.ctx.candle_epic_map["XAU/USD"]
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_GONE"
        bot.ctx.risk_manager._open_positions[xau_epic] = bot.ctx.state.positions["XAU/USD"]
        entry_sig = _make_signal("XAU/USD", stop_pct=0.01)
        bot.ctx.tp_manager.register_position(
            "XAU/USD", _ig_level("XAU/USD", 13.95), entry_sig, BASE_TS
        )
        # IG says the position no longer exists
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])

        # Set up rerank with no XAU/USD signal — purge happens before signal_decay,
        # so signal_decay must not also try to close it.
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        assert "XAU/USD" not in bot.ctx.state.positions
        assert "XAU/USD" not in bot.ctx.ig_deal_ids
        assert "XAU/USD" not in bot.ctx.tp_manager._positions
        assert xau_epic not in bot.ctx.risk_manager._open_positions
        # No close attempt — IG already closed it
        assert (
            not hasattr(bot.ctx.ig_client.close_position, "assert_awaited")
            or bot.ctx.ig_client.close_position.await_count == 0
        )

    @pytest.mark.asyncio
    async def test_periodic_reconcile_keeps_position_still_on_ig(self, bot: TradingBot) -> None:
        """Position present on IG must survive the per-rerank reconcile
        unchanged, with its deal_id refreshed and the risk-manager view
        re-seeded (the post-restart scenario: ``_open_positions`` starts
        empty because it isn't persisted, and on_fill never fires for
        positions opened pre-restart).
        """
        # Simulate post-restart: state.positions populated from disk, but
        # _risk_manager._open_positions is empty (it's never restored).
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_OLD"
        bot.ctx.risk_manager._open_positions.clear()
        bot.ctx.tp_manager.register_position(
            "XAU/USD", _ig_level("XAU/USD", 13.95), _make_signal("XAU/USD", stop_pct=0.01), BASE_TS
        )
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.USCGC.TODAY.IP"},
                    "position": {"dealId": "DEAL_REFRESHED"},
                }
            ]
        )
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        assert "XAU/USD" in bot.ctx.state.positions
        assert bot.ctx.ig_deal_ids["XAU/USD"] == "DEAL_REFRESHED"  # deal_id refreshed
        assert "XAU/USD" in bot.ctx.tp_manager._positions
        # Risk-manager view re-seeded — keyed by EPIC, aliased to the local
        # Position object so a subsequent SELL fill resolves correctly.
        assert "CS.D.USCGC.TODAY.IP" in bot.ctx.risk_manager._open_positions
        assert (
            bot.ctx.risk_manager._open_positions["CS.D.USCGC.TODAY.IP"]
            is bot.ctx.state.positions["XAU/USD"]
        )

    @pytest.mark.asyncio
    async def test_periodic_reconcile_alerts_on_orphan_ig_position(self, bot: TradingBot) -> None:
        """Bidirectional reconcile: an IG position absent from ``_state.positions``
        is logged + Telegram-alerted as an orphan and recorded in
        ``_alerted_orphan_deals`` so it doesn't re-alert on the next rerank.
        Orphans are NOT auto-closed and NOT auto-adopted.
        """
        # No local positions; IG has one orphan (e.g. created by a prior
        # confirm_order 404 race where the order silently filled).
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.USCGC.TODAY.IP"},
                    "position": {
                        "dealId": "ORPHAN_DEAL_1",
                        "direction": "BUY",
                        "size": 0.5,
                        "level": 2800.0,
                    },
                }
            ]
        )
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        # Alerted once, recorded for dedup, NOT adopted into local state
        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.send_error.assert_awaited()
        assert "ORPHAN_DEAL_1" in alerter.send_error.call_args.args[0]
        assert "ORPHAN_DEAL_1" in bot.ctx.alerted_orphan_deals
        assert "XAU/USD" not in bot.ctx.state.positions
        assert "XAU/USD" not in bot.ctx.ig_deal_ids

        # Second rerank with the same orphan still on IG: no duplicate alert
        alerter.send_error.reset_mock()
        await bot.ctx.runner.topk_rerank_loop()
        alerter.send_error.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_periodic_reconcile_flags_duplicate_epic_as_orphan(self, bot: TradingBot) -> None:
        """When IG has TWO positions for the same EPIC (the May-14 USD/NOK
        duplicate-confirm-race pattern), the first claim wires the local
        position's deal_id and the SECOND is flagged as an orphan — instead
        of silently overwriting ``_ig_deal_ids`` with the second one.
        """
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_FROM_LAST_SCAN"
        bot.ctx.tp_manager.register_position(
            "XAU/USD", _ig_level("XAU/USD", 13.95), _make_signal("XAU/USD", stop_pct=0.01), BASE_TS
        )
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.USCGC.TODAY.IP"},
                    "position": {
                        "dealId": "DEAL_MATCH",
                        "direction": "BUY",
                        "size": 1.0,
                        "level": 2790.0,
                    },
                },
                {
                    "market": {"epic": "CS.D.USCGC.TODAY.IP"},
                    "position": {
                        "dealId": "DEAL_ORPHAN",
                        "direction": "BUY",
                        "size": 1.0,
                        "level": 2792.0,
                    },
                },
            ]
        )
        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        # Local XAU/USD was wired to the first IG match; second match is an orphan
        assert bot.ctx.ig_deal_ids["XAU/USD"] == "DEAL_MATCH"
        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.send_error.assert_awaited()
        assert "DEAL_ORPHAN" in alerter.send_error.call_args.args[0]
        assert "DEAL_ORPHAN" in bot.ctx.alerted_orphan_deals
        assert "DEAL_MATCH" not in bot.ctx.alerted_orphan_deals

    @pytest.mark.asyncio
    async def test_periodic_reconcile_resilient_to_ig_failure(self, bot: TradingBot) -> None:
        """A failed ``/positions`` fetch (network blip, 5xx, malformed body)
        must not crash the rerank — local state stays untouched and the rest
        of the loop runs normally."""
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_KEEP"
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(side_effect=RuntimeError("network blip"))

        sig = _make_signal("EUR/USD", mean_return=0.005)
        self._prepare(bot, [sig])

        await bot.ctx.runner.topk_rerank_loop()

        # State preserved — best-effort sync, not a hard failure
        assert "XAU/USD" in bot.ctx.state.positions
        assert bot.ctx.ig_deal_ids["XAU/USD"] == "DEAL_KEEP"
        # And the rerank still completed (signals stored, alert sent)
        assert bot.ctx.topk_scanned is True
        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.assert_awaited()


# ---------------------------------------------------------------------------
# E. Order-filled handler
# ---------------------------------------------------------------------------


class TestOrderFilledHandler:
    @pytest.mark.asyncio
    async def test_buy_fill_adds_position_keyed_by_candle_symbol(self, bot: TradingBot) -> None:
        """Order results carry IG EPICs as ``symbol``; ``_state.positions`` must
        be keyed by the candle symbol (e.g. ``EUR/USD``), not the EPIC."""
        result = _make_order_result(
            symbol="CS.D.EURUSD.TODAY.IP",
            side=OrderSide.BUY,
            average_price=1.10,
            order_id="DEAL_BUY",
        )
        # risk manager needs to know the position too — call on_fill first
        bot.ctx.risk_manager.on_fill(result)
        await bot.ctx.events.handle_order_filled(result)

        assert "EUR/USD" in bot.ctx.state.positions
        assert bot.ctx.state.positions["EUR/USD"].entry_price == pytest.approx(1.10)
        assert "CS.D.EURUSD.TODAY.IP" not in bot.ctx.state.positions

    @pytest.mark.asyncio
    async def test_sell_fill_removes_position_and_emits_event(self, bot: TradingBot) -> None:
        # Pre-populate a position both in state and risk manager
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        # Risk manager keys by EPIC for IG fills
        bot.ctx.risk_manager._open_positions["CS.D.EURUSD.TODAY.IP"] = bot.ctx.state.positions[
            "EUR/USD"
        ]
        emitted_events: list[Any] = []

        async def capture(data: Any) -> None:
            emitted_events.append(data)

        bot.ctx.event_bus.subscribe(EVENT_POSITION_CLOSED, capture)

        sell_result = _make_order_result(
            symbol="CS.D.EURUSD.TODAY.IP",
            side=OrderSide.SELL,
            average_price=1.115,
            order_id="DEAL_SELL",
        )
        await bot.ctx.events.handle_order_filled(sell_result)

        assert "EUR/USD" not in bot.ctx.state.positions
        # Allow the dispatch hop
        await asyncio.sleep(0)
        assert any(e.get("symbol") == "CS.D.EURUSD.TODAY.IP" for e in emitted_events)

    @pytest.mark.asyncio
    async def test_buy_fill_jpy_alert_uses_fx_rate_display(self, bot: TradingBot) -> None:
        result = _make_order_result(
            symbol="CS.D.USDJPY.TODAY.IP",
            side=OrderSide.BUY,
            average_price=15658.5,  # IG level
            order_id="DEAL_JPY_BUY",
        )
        bot.ctx.risk_manager.on_fill(result)
        await bot.ctx.events.handle_order_filled(result)

        cast(AsyncMock, bot.ctx.alerter).send_trade_alert.assert_awaited_once()
        kwargs = cast(AsyncMock, bot.ctx.alerter).send_trade_alert.call_args.kwargs
        # display_price = IG level / 100 = ~156.585 FX
        assert kwargs["display_price"] == pytest.approx(156.585, abs=0.01)
        assert kwargs["display_symbol"] == "USD/JPY"

    @pytest.mark.asyncio
    async def test_buy_fill_metal_alert_uses_ig_level(self, bot: TradingBot) -> None:
        # XAU/USD is on the IG-native candle feed (scale 1.0), so the alert
        # display price is the IG fill level itself (2790) — no ETF-style
        # display conversion.
        result = _make_order_result(
            symbol="CS.D.USCGC.TODAY.IP",
            side=OrderSide.BUY,
            average_price=2790.0,
            order_id="DEAL_XAU_BUY",
        )
        bot.ctx.risk_manager.on_fill(result)
        await bot.ctx.events.handle_order_filled(result)

        kwargs = cast(AsyncMock, bot.ctx.alerter).send_trade_alert.call_args.kwargs
        assert kwargs["display_price"] == pytest.approx(2790.0, abs=0.01)
        assert kwargs["display_symbol"] == "XAU/USD"


# ---------------------------------------------------------------------------
# F. Unified close path
# ---------------------------------------------------------------------------


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_metal_alert_pnl_uses_underlying_return(self, bot: TradingBot) -> None:
        # XAU/USD candle store is in IG-level units (scale 1.0).
        # Entry IG 2790, current IG 2840 → ~+1.79 %.
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", 2790.0)
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        bot.ctx.store.add_candle(_candle("XAU/USD", BASE_TS, 2840.0))

        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.USCGC.TODAY.IP", side=OrderSide.SELL, average_price=2840.0
            )
        )

        await bot.ctx.closer.request_close("XAU/USD", "static_take_profit", "test")

        bot.ctx.ig_client.close_position.assert_awaited_once()
        kwargs_pnl = cast(AsyncMock, bot.ctx.alerter).alert_take_profit.call_args
        # Args (positional): symbol, reason, entry_display, exit_display, pnl_pct, reasoning
        entry_display = kwargs_pnl.args[2]
        exit_display = kwargs_pnl.args[3]
        pnl_pct = kwargs_pnl.args[4]
        # Display is the IG level itself post-D3 (scale 1.0 collapses the conversion).
        assert entry_display == pytest.approx(2790.0, abs=0.01)
        assert exit_display == pytest.approx(2840.0, abs=0.01)
        assert pnl_pct == pytest.approx(1.79, abs=0.05)

    @pytest.mark.asyncio
    async def test_close_jpy_alert_pnl_uses_fx_rate(self, bot: TradingBot) -> None:
        bot.ctx.state.positions["USD/JPY"] = _make_position("USD/JPY", 15658.5)
        bot.ctx.ig_deal_ids["USD/JPY"] = "DEAL_JPY"
        bot.ctx.store.add_candle(_candle("USD/JPY", BASE_TS, 157.052))

        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.USDJPY.TODAY.IP",
                side=OrderSide.SELL,
                average_price=15705.2,
            )
        )

        await bot.ctx.closer.request_close("USD/JPY", "trailing_ratchet", "test")

        kwargs_pnl = cast(AsyncMock, bot.ctx.alerter).alert_take_profit.call_args
        entry_display = kwargs_pnl.args[2]
        exit_display = kwargs_pnl.args[3]
        pnl_pct = kwargs_pnl.args[4]
        assert entry_display == pytest.approx(156.585, abs=0.01)
        assert exit_display == pytest.approx(157.052, abs=0.01)
        # +0.298 %
        assert pnl_pct == pytest.approx(0.298, abs=0.05)

    @pytest.mark.asyncio
    async def test_close_no_position_is_noop(self, bot: TradingBot) -> None:
        """No position for symbol → close_position returns silently."""
        await bot.ctx.closer.request_close("EUR/USD", "static_take_profit", "")
        bot.ctx.ig_client.close_position.assert_not_called()
        cast(AsyncMock, bot.ctx.alerter).alert_take_profit.assert_not_called()

    @pytest.mark.asyncio
    async def test_close_deregisters_from_tp_manager(self, bot: TradingBot) -> None:
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_DR"
        bot.ctx.tp_manager.register_position(
            "EUR/USD", 1.10, _make_signal("EUR/USD", stop_pct=0.005), BASE_TS
        )
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        bot.ctx.ig_client.close_position = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.EURUSD.TODAY.IP",
                side=OrderSide.SELL,
                average_price=1.105,
            )
        )

        assert "EUR/USD" in bot.ctx.tp_manager._positions
        await bot.ctx.closer.request_close("EUR/USD", "static_take_profit", "")
        assert "EUR/USD" not in bot.ctx.tp_manager._positions

    @pytest.mark.asyncio
    async def test_close_position_keeps_tp_state_when_ig_rejects(self, bot: TradingBot) -> None:
        """Regression for the production bug found 2026-05-03:
        when IG rejects the close (e.g. ``MARKET_CLOSED_WITH_EDITS``), the TP
        manager state and ``_ig_deal_ids`` must be preserved so the next
        rerank/candle can retry, and the false-positive ``alert_take_profit``
        Telegram must NOT fire.  An error alert is sent in its place.
        """
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_FAIL"
        bot.ctx.tp_manager.register_position(
            "EUR/USD", 1.10, _make_signal("EUR/USD", stop_pct=0.005), BASE_TS
        )
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        bot.ctx.ig_client.close_position = AsyncMock(side_effect=RuntimeError("MARKET_CLOSED"))
        # After the close fails, the bot probes /positions to disambiguate
        # a ghost dealId from a flaky endpoint.  IG still reports the position
        # as live → state must be preserved for retry.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.EURUSD.TODAY.IP"},
                    "position": {"dealId": "DEAL_FAIL"},
                }
            ]
        )

        await bot.ctx.closer.request_close("EUR/USD", "static_take_profit", "")

        # close_position attempted but IG rejected — preserve everything for retry
        assert "EUR/USD" in bot.ctx.tp_manager._positions
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_FAIL"
        assert "EUR/USD" in bot.ctx.state.positions
        # No false take-profit alert
        cast(AsyncMock, bot.ctx.alerter).alert_take_profit.assert_not_called()
        # Operator gets an error alert instead
        cast(AsyncMock, bot.ctx.alerter).send_error.assert_awaited()

    @pytest.mark.asyncio
    async def test_close_failure_with_ghost_dealid_purges_and_alerts(self, bot: TradingBot) -> None:
        """Regression for the production bug observed 2026-05-11: when a
        ``.TODAY.IP`` contract rolls and the local ``dealId`` becomes a ghost,
        the close call hangs/fails for hours.  The probe must reconcile the
        position out immediately and fire an external-close Telegram alert —
        no spurious ``send_error`` for a position that no longer exists.
        """
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 11000.0)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_GHOST"
        bot.ctx.tp_manager.register_position(
            "EUR/USD", 11000.0, _make_signal("EUR/USD", stop_pct=0.005), BASE_TS
        )
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        bot.ctx.ig_client.close_position = AsyncMock(side_effect=TimeoutError())
        # /positions returns empty → reconciler treats the position as gone.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])

        await bot.ctx.closer.request_close("EUR/USD", "static_take_profit", "")

        assert "EUR/USD" not in bot.ctx.state.positions
        assert "EUR/USD" not in bot.ctx.ig_deal_ids
        assert "EUR/USD" not in bot.ctx.tp_manager._positions
        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.alert_take_profit.assert_awaited()
        assert alerter.alert_take_profit.call_args.args[1] == "reconciled_external_close"
        alerter.send_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_market_closed_defers_close_without_alert_or_reconcile(
        self, bot: TradingBot
    ) -> None:
        """Phase-Hotfix-2026-05-13: when IG raises ``MARKET_CLOSED_WITH_EDITS``
        (the 22:00 UTC funding-tick reject), the close path must NOT reconcile
        (would mis-purge a still-live position) and must NOT send any alert.
        All local state — dealId, TP entry, position — stays intact for the
        next-candle retry once the window ends.
        """
        from bot.core.models import ErrorType, MarketClosedError  # noqa: PLC0415

        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 11000.0)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_LIVE"
        bot.ctx.tp_manager.register_position(
            "EUR/USD", 11000.0, _make_signal("EUR/USD", stop_pct=0.005), BASE_TS
        )
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        bot.ctx.ig_client.close_position = AsyncMock(
            side_effect=MarketClosedError("market closed", ErrorType.MARKET_CLOSED)
        )
        # /positions should NOT be probed; if it is, set an empty list so the
        # assertion below would fail loudly.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])

        await bot.ctx.closer.request_close("EUR/USD", "static_take_profit", "")

        # State preserved
        assert "EUR/USD" in bot.ctx.state.positions
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_LIVE"
        assert "EUR/USD" in bot.ctx.tp_manager._positions
        # No reconcile probe + no alert of any kind
        bot.ctx.ig_client.fetch_positions_raw.assert_not_awaited()
        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.alert_take_profit.assert_not_called()
        alerter.send_error.assert_not_called()

    @pytest.mark.asyncio
    async def test_reconcile_alert_uses_real_ig_fill_when_available(self, bot: TradingBot) -> None:
        """Phase-Hotfix-2026-05-13 (A): when an external close is detected, the
        Telegram alert reports the *actual* IG close level + GBP P&L from
        ``/history/transactions``, not the local Twelve Data candle estimate.
        """
        # Position opened with IG entry level 11000 (== EUR/USD ~1.10).
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 11000.0)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_GONE"
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        # /positions returns empty → reconciler triggers purge path.
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        # /history/transactions returns the matching close at IG level 10800
        # (== EUR/USD 1.0800), with a -3.20 GBP realised loss.
        bot.ctx.ig_client.fetch_closed_transaction = AsyncMock(
            return_value={
                "instrumentName": "EUR/USD",
                "openDateUtc": "...",
                "closeLevel": "10800",
                "openLevel": "11000",
                "profitAndLoss": "E-3.20",
                "size": "+1.00",
                "period": "DFB",
            }
        )

        await bot.ctx.closer.reconcile_positions_with_ig()

        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.alert_take_profit.assert_awaited()
        args = alerter.alert_take_profit.call_args.args
        # alert_take_profit(symbol, reason, entry_display, close_display, pnl_pct, reasoning)
        assert args[0] == "EUR/USD"
        assert args[1] == "reconciled_external_close"
        # close_display = closeLevel / scale = 10800 / 10000 = 1.0800
        assert args[3] == pytest.approx(1.0800, abs=1e-6)
        # pnl_pct = (10800 - 11000) / 11000 * 100 ≈ -1.818%
        assert args[4] == pytest.approx(-1.8181818, abs=1e-4)
        # reasoning mentions the real fill price and real GBP P&L
        assert "10800" in args[5] or "1.0800" in args[5]
        assert "-3.20" in args[5]
        assert "EUR/USD" not in bot.ctx.state.positions

    @pytest.mark.asyncio
    async def test_reconcile_alert_falls_back_to_candle_when_history_empty(
        self, bot: TradingBot
    ) -> None:
        """When ``/history/transactions`` returns no match, the alert falls back
        to the local candle close — but the reasoning is explicitly tagged as
        an *estimate* so the operator can tell."""
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 11000.0)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_GONE"
        bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS, 1.105))

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        bot.ctx.ig_client.fetch_closed_transaction = AsyncMock(return_value=None)

        await bot.ctx.closer.reconcile_positions_with_ig()

        alerter = cast(AsyncMock, bot.ctx.alerter)
        alerter.alert_take_profit.assert_awaited()
        reasoning = alerter.alert_take_profit.call_args.args[5]
        assert "estimated" in reasoning.lower()


# ---------------------------------------------------------------------------
# G. Startup state recovery
# ---------------------------------------------------------------------------


class TestStartupRecovery:
    """Drives ``bot.start()`` end-to-end with the long-running tasks neutered.

    The shutdown event is set before ``start()`` runs so the outer ``await
    self._shutdown_event.wait()`` returns immediately.  Each long-running task
    is replaced with a coroutine that simply waits on the same event so the
    cleanup path in ``shutdown()`` stays well-typed.
    """

    @staticmethod
    def _stub_long_running(bot: TradingBot) -> None:
        async def _idle() -> None:
            await bot.ctx.shutdown_event.wait()

        bot.ctx.runner.subscribe_candle_handler = _idle  # type: ignore[method-assign]
        bot.ctx.health.health_check = _idle  # type: ignore[method-assign]
        bot.ctx.runner.topk_rerank_loop = _idle  # type: ignore[method-assign]
        bot.ctx.runner.signal_resolver_loop = _idle  # type: ignore[method-assign]

        def _build_idle_feed() -> asyncio.Task[None]:
            return asyncio.create_task(_idle())

        bot.ctx.lifecycle.build_feed_task = _build_idle_feed  # type: ignore[method-assign]

    @pytest.mark.asyncio
    async def test_consecutive_losses_reset_at_pause_threshold(self, bot: TradingBot) -> None:
        # Pre-populate state with consecutive_losses == pause threshold (4)
        state = BotState()
        state.risk = RiskState(
            consecutive_losses=bot.ctx.risk_config.consecutive_loss_pause,
            peak_equity=10_000.0,
        )
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert bot.ctx.state.risk.consecutive_losses == 0

    @pytest.mark.asyncio
    async def test_position_purged_when_not_on_ig(self, bot: TradingBot) -> None:
        """A position that no longer exists on IG (closed externally while the
        bot was offline) must be purged from state on restart."""
        state = BotState()
        state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])  # IG: no positions
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert "EUR/USD" not in bot.ctx.state.positions

    @pytest.mark.asyncio
    async def test_position_kept_and_deal_id_reconciled_when_on_ig(self, bot: TradingBot) -> None:
        state = BotState()
        state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.EURUSD.TODAY.IP"},
                    "position": {"dealId": "DEAL_RECON"},
                }
            ]
        )
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert "EUR/USD" in bot.ctx.state.positions
        assert bot.ctx.ig_deal_ids["EUR/USD"] == "DEAL_RECON"

    @pytest.mark.asyncio
    async def test_topk_state_restored_when_fresh(self, bot: TradingBot) -> None:
        now_ms = int(time.time() * 1000)
        state = BotState()
        state.topk_state = {
            "selected": ["EUR/USD"],
            "signals": [
                {
                    "symbol": "EUR/USD",
                    "mean_return": 0.005,
                    "std_return": 0.003,
                    "direction_confidence": 0.8,
                    "uncertainty": 0.4,
                    "stop_pct": 0.005,
                    "tradeable": True,
                    "predicted_close": 1.10,
                    "direction": "LONG",
                }
            ],
            "scanned_at": now_ms - 60_000,  # 1 min ago
        }
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert bot.ctx.topk_scanned is True
        assert bot.ctx.topk_selected == ["EUR/USD"]
        assert len(bot.ctx.topk_signals) == 1
        assert bot.ctx.topk_signals[0].symbol == "EUR/USD"

    @pytest.mark.asyncio
    async def test_topk_state_skipped_when_stale(self, bot: TradingBot) -> None:
        # Older than 4 h → must be skipped
        state = BotState()
        state.topk_state = {
            "selected": ["EUR/USD"],
            "signals": [],
            "scanned_at": int(time.time() * 1000) - 5 * 60 * 60 * 1000,
        }
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert bot.ctx.topk_scanned is False
        assert bot.ctx.topk_selected == []

    @pytest.mark.asyncio
    async def test_correlation_matrix_restored_from_db(self, bot: TradingBot) -> None:
        # Pre-write a correlation snapshot to the candle DB
        bot.ctx.candle_db.write_correlations(
            int(time.time() * 1000),
            {
                "EUR/USD": {"USD/JPY": 0.42},
                "USD/JPY": {"EUR/USD": 0.42},
            },
        )

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        # Tracker should now expose that pair
        corr = bot.ctx.topk_strategy._correlation_tracker.correlation("EUR/USD", "USD/JPY")
        assert corr == pytest.approx(0.42)

    @pytest.mark.asyncio
    async def test_take_profit_state_restored(self, bot: TradingBot) -> None:
        state = BotState()
        # Position keyed by candle symbol; tp_state keyed identically
        state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        state.take_profit_state = {
            "EUR/USD": {
                "symbol": "EUR/USD",
                "entry_price": 1.10,
                "entry_mean_return": 0.005,
                "entry_stop_pct": 0.005,
                "peak_price": 1.115,
                "opened_at_ms": BASE_TS,
                "signal_decay_strikes": 1,
                "breakeven_armed": True,
                "trail_armed": False,
                "current_trailing_stop": 1.105,
                "predicted_mfe_pct": 0.025,
                "predicted_mae_pct": 0.005,
                "predicted_peak_bar": 60,
                "bar_interval_ms": 3_600_000,
            }
        }
        bot.ctx.state_manager.save(state)

        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(
            return_value=[
                {
                    "market": {"epic": "CS.D.EURUSD.TODAY.IP"},
                    "position": {"dealId": "DEAL_TP_RESTORE"},
                }
            ]
        )
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        tp_state = bot.ctx.tp_manager._positions["EUR/USD"]
        assert tp_state.peak_price == pytest.approx(1.115)
        assert tp_state.signal_decay_strikes == 1
        assert tp_state.breakeven_armed is True
        assert tp_state.current_trailing_stop == pytest.approx(1.105)
        assert tp_state.predicted_mfe_pct == pytest.approx(0.025)
        assert tp_state.predicted_peak_bar == 60

    @pytest.mark.asyncio
    async def test_startup_fetches_initial_equity(self, bot: TradingBot) -> None:
        bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        bot.ctx.ig_client.fetch_balance = AsyncMock(
            return_value={"equity": 12_500.0, "margin": 50.0, "balance": 12_500.0, "open_pnl": 0.0}
        )
        self._stub_long_running(bot)
        bot.ctx.shutdown_event.set()
        await bot.start()

        assert bot.ctx.risk_manager._drawdown.equity == pytest.approx(12_500.0)


# ---------------------------------------------------------------------------
# H. Misc: per-candle dispatch and event subscription
# ---------------------------------------------------------------------------


class TestCandleDispatch:
    @pytest.mark.asyncio
    async def test_candle_with_position_runs_topk_path(self, bot: TradingBot) -> None:
        """``_process_candle`` must dispatch confirmed candles to
        ``_process_candle_ig_topk`` when the IG TopK path is active."""
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", 1.10)
        bot.ctx.ig_deal_ids["EUR/USD"] = "DEAL_DISPATCH"
        bot.ctx.topk_signals = [_make_signal("EUR/USD", stop_pct=0.005)]
        bot.ctx.topk_scanned = True
        # Two candles required (the dispatch path checks len(candles) >= 2)
        for i in range(3):
            bot.ctx.store.add_candle(_candle("EUR/USD", BASE_TS + i * HOUR_MS, 1.10))

        called = MagicMock()

        async def fake_topk(epic: str, price: float, pos: Any) -> None:
            called(epic, price, pos)

        bot.ctx.runner.process_candle_ig_topk = fake_topk  # type: ignore[method-assign,assignment]

        await bot.ctx.runner.process_candle(_candle("EUR/USD", BASE_TS + 5 * HOUR_MS, 1.105))

        called.assert_called_once()
        assert called.call_args.args[0] == "EUR/USD"
        assert called.call_args.args[1] == pytest.approx(1.105)

    @pytest.mark.asyncio
    async def test_shutdown_bounded_by_timeouts_when_feed_close_hangs(
        self, bot: TradingBot
    ) -> None:
        """A hung feed.close() must not block shutdown past ~10 s.

        Sets up a fake twelve_data_feed whose close() never returns.  Verifies
        that:
          - shutdown() completes within a wall-time well under the 75 s
            systemd TimeoutStopSec budget,
          - the IG client + Telegram closes still execute (the hung feed
            must NOT short-circuit subsequent steps).
        """
        import time as _time

        async def hang_forever() -> None:
            await asyncio.sleep(120)  # well past any per-step budget

        td_feed = AsyncMock()
        td_feed.close = MagicMock(side_effect=lambda: hang_forever())
        bot.ctx.twelve_data_feed = td_feed

        bot.ctx.ig_client.close = AsyncMock(return_value=None)  # type: ignore[method-assign]
        bot.ctx.alerter.send_shutdown = AsyncMock(return_value=True)

        # No background tasks running for this test
        bot.ctx.tasks = []

        start = _time.monotonic()
        await bot.shutdown()
        elapsed = _time.monotonic() - start

        # Five seconds for the feed-close timeout + a small budget for the
        # subsequent steps.  Real budget is 5 + 10 + 5 = 20 s worst case;
        # 30 s leaves a margin so the test doesn't flake on slow CI.
        assert elapsed < 30.0, f"shutdown took {elapsed:.1f}s — bounded close not working"
        # Despite the hung feed, the IG client + Telegram closes still ran.
        bot.ctx.ig_client.close.assert_awaited_once()
        bot.ctx.alerter.send_shutdown.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_shutdown_continues_when_telegram_hangs(self, bot: TradingBot) -> None:
        """send_shutdown() hanging at the end must not block bot exit."""
        import time as _time

        async def hang_forever() -> bool:
            await asyncio.sleep(120)
            return True

        bot.ctx.alerter.send_shutdown = MagicMock(side_effect=lambda *_a, **_kw: hang_forever())  # type: ignore[method-assign]
        bot.ctx.ig_client.close = AsyncMock(return_value=None)  # type: ignore[method-assign]
        bot.ctx.tasks = []

        start = _time.monotonic()
        await bot.shutdown()
        elapsed = _time.monotonic() - start

        assert elapsed < 15.0, f"shutdown took {elapsed:.1f}s — Telegram hang bypass not working"

    @pytest.mark.asyncio
    async def test_unconfirmed_candle_skipped(self, bot: TradingBot) -> None:
        """``_subscribe_candle_handler`` filters out unconfirmed candles."""

        bot.ctx.topk_scanned = True
        called = MagicMock()

        async def fake_process(c: Any) -> None:
            called(c)

        bot.ctx.runner.process_candle = fake_process  # type: ignore[method-assign,assignment]

        # Run subscriber in background, then emit one unconfirmed candle
        sub_task = asyncio.create_task(bot.ctx.runner.subscribe_candle_handler())
        await asyncio.sleep(0)  # let subscribe register
        await bot.ctx.event_bus.emit(
            EVENT_NEW_CANDLE,
            replace(_candle("EUR/USD", BASE_TS, 1.0), is_confirmed=False),
        )
        await asyncio.sleep(0)
        bot.ctx.shutdown_event.set()
        await sub_task

        called.assert_not_called()


# ---------------------------------------------------------------------------
# Preflight #4 — margin walk-down end-to-end.
# IG_LIVE_RISK_REFERENCE.md §4.3.  TestMarginCircuitBreakers in
# test_risk_manager.py proves the *classifier* trips at the right
# thresholds; this block proves the wiring `AccountUpdate stream →
# RiskManager.update_margin_state → EVENT_MARGIN_BREAKER →
# TradingBot._handle_margin_breaker → _close_position` actually fires the
# de-risking action.  A green classifier with broken wiring would still
# liquidate the account.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
class TestMarginBreakerE2E:
    @staticmethod
    def _account(equity: float, margin: float) -> Any:
        from bot.core.models import AccountUpdate

        return AccountUpdate(
            timestamp=BASE_TS,
            equity=equity,
            margin_required=margin,
            available_to_deal=max(0.0, equity - margin),
            unrealised_pnl=0.0,
        )

    @staticmethod
    def _seed_two_positions(bot: TradingBot) -> None:
        """Seed two open IG positions + their candle prices so
        `_pick_worst_performer` and `_close_position` have everything they
        need.  Also wires the event bus the way ``start()`` would — the
        fixture never calls ``start()``, so EVENT_MARGIN_BREAKER would
        otherwise have no subscriber.  Wiring it here is the legitimate
        E2E setup, not a workaround."""
        # USD/JPY at IG level 15658.5
        bot.ctx.state.positions["USD/JPY"] = _make_position(
            "USD/JPY", _ig_level("USD/JPY", 156.585)
        )
        bot.ctx.ig_deal_ids["USD/JPY"] = "DEAL_JPY"
        _seed_candles(bot, "USD/JPY", 156.585, count=2)
        # XAU/USD at IG level 13.95 × _ig_quote_scale("XAU/USD").  Latest candle
        # below entry so XAU/USD is the deterministic worst performer.
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        bot.ctx.store.add_candle(_candle("XAU/USD", BASE_TS, 13.95 * 0.98))
        bot.ctx.events.wire()

    @pytest.mark.asyncio
    async def test_defensive_close_invokes_close_position_for_worst_performer(
        self, bot: TradingBot
    ) -> None:
        """Drive a transition from NORMAL → HALT_ENTRIES → DEFENSIVE_CLOSE.
        The defensive-close action must close exactly the worst-performing
        position (here XAU/USD) via `_close_position`."""
        self._seed_two_positions(bot)

        close_calls: list[tuple[str, str]] = []

        async def fake_close(symbol: str, reason: str, reasoning: str = "") -> None:
            close_calls.append((symbol, reason))

        bot.ctx.closer.request_close = fake_close  # type: ignore[method-assign,assignment]

        # Pre-prime the breaker into HALT first (eq/margin=0.80) to avoid the
        # "skip first transition" race; then push to DEFENSIVE (0.65).
        bot.ctx.risk_manager.update_margin_state(self._account(equity=10_000, margin=12_500))
        bot.ctx.risk_manager.update_margin_state(self._account(equity=10_000, margin=15_385))

        # Give the event-bus tasks one tick to drain.
        await asyncio.sleep(0.05)

        assert len(close_calls) == 1, (
            f"Expected exactly one position closed on DEFENSIVE transition, got {close_calls}"
        )
        # XAU/USD should be the worst-performer pick
        assert close_calls[0][0] == "XAU/USD"
        assert "margin_breaker_defensive" in close_calls[0][1]

    @pytest.mark.asyncio
    async def test_flatten_invokes_close_position_for_all(self, bot: TradingBot) -> None:
        """EMERGENCY_FLATTEN must close every open position via
        `_close_position`."""
        self._seed_two_positions(bot)

        close_calls: list[str] = []

        async def fake_close(symbol: str, reason: str, reasoning: str = "") -> None:
            close_calls.append(symbol)

        bot.ctx.closer.request_close = fake_close  # type: ignore[method-assign,assignment]

        # Walk straight through: HALT → EMERGENCY
        bot.ctx.risk_manager.update_margin_state(self._account(equity=10_000, margin=12_500))
        bot.ctx.risk_manager.update_margin_state(self._account(equity=10_000, margin=18_182))

        await asyncio.sleep(0.05)

        assert set(close_calls) == {"USD/JPY", "XAU/USD"}, (
            f"EMERGENCY_FLATTEN should close every open position; got {close_calls}"
        )

    @pytest.mark.asyncio
    async def test_halt_entries_does_not_close(self, bot: TradingBot) -> None:
        """The HALT_ENTRIES transition must not close anything — the
        pre-trade gate handles it.  Closing on HALT would be a bug."""
        self._seed_two_positions(bot)

        close_calls: list[str] = []

        async def fake_close(symbol: str, reason: str, reasoning: str = "") -> None:
            close_calls.append(symbol)

        bot.ctx.closer.request_close = fake_close  # type: ignore[method-assign,assignment]

        bot.ctx.risk_manager.update_margin_state(self._account(equity=10_000, margin=12_500))

        await asyncio.sleep(0.05)

        assert close_calls == []


# ---------------------------------------------------------------------------
# Preflight #7 — MARKET_CLOSED_WITH_EDITS handling.
# IG_LIVE_RISK_REFERENCE.md §1.2.  REST-layer rejection is covered by
# tests/test_ig_client.py::test_market_closed_raises_market_closed_error
# (also marked preflight).  This block pins what main.py does with the
# exception on the entry path and the close path — both are *intentional*
# (defer + retry, don't mis-purge) and we want to catch any regression
# that silently swallows the error or eagerly reconciles.
# ---------------------------------------------------------------------------


@pytest.mark.preflight
class TestMarketClosedReconciliation:
    @pytest.mark.asyncio
    async def test_close_path_deferred_when_market_closed(self, bot: TradingBot) -> None:
        """`_close_ig_position` returning None (MarketClosedError caught) must
        leave the position and dealId intact and NOT call reconcile — the
        position is still alive on IG; reconciling would mis-purge it."""
        from bot.execution.ig_client import MarketClosedError

        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        _seed_candles(bot, "XAU/USD", 13.95, count=2)

        # IG close raises MarketClosedError, which _close_ig_position translates
        # to a deferred (None) result.
        bot.ctx.ig_client.close_position = AsyncMock(
            side_effect=MarketClosedError("MARKET_CLOSED_WITH_EDITS", error_type=None)  # type: ignore[arg-type]
        )
        bot.ctx.closer.reconcile_positions_with_ig = AsyncMock()  # type: ignore[method-assign]

        await bot.ctx.closer.request_close("XAU/USD", reason="test_market_closed")

        # Position + dealId preserved for the next retry
        assert "XAU/USD" in bot.ctx.state.positions
        assert bot.ctx.ig_deal_ids.get("XAU/USD") == "DEAL_XAU"
        # No reconcile fired — the deferred branch must NOT call it
        bot.ctx.closer.reconcile_positions_with_ig.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_close_path_failure_triggers_reconcile(self, bot: TradingBot) -> None:
        """For any *non* MARKET_CLOSED failure, `_close_position` must
        reconcile so a ghost-dealId scenario gets purged correctly.  This
        is the inverse of the previous test — the two together ensure the
        deferred vs failed branches don't get conflated."""
        bot.ctx.state.positions["XAU/USD"] = _make_position("XAU/USD", _ig_level("XAU/USD", 13.95))
        bot.ctx.ig_deal_ids["XAU/USD"] = "DEAL_XAU"
        _seed_candles(bot, "XAU/USD", 13.95, count=2)

        bot.ctx.ig_client.close_position = AsyncMock(side_effect=RuntimeError("network bork"))
        bot.ctx.closer.reconcile_positions_with_ig = AsyncMock()  # type: ignore[method-assign]

        await bot.ctx.closer.request_close("XAU/USD", reason="test_failed_close")

        bot.ctx.closer.reconcile_positions_with_ig.assert_awaited_once()


# ---------------------------------------------------------------------------
# topk_exclude_from_selection — symbols still scanned + signal_history-logged
# but never picked for entry.  See BotConfig.topk_exclude_from_selection.
# ---------------------------------------------------------------------------


class TestExcludeFromSelection:
    @pytest.mark.asyncio
    async def test_excluded_symbol_appears_in_signal_history(self, bot: TradingBot) -> None:
        """Kronos still scans XAU/USD; its signal still lands in signal_history."""
        bot.ctx.config.topk_exclude_from_selection = ["XAU/USD"]

        sig_eur = _make_signal("EUR/USD", mean_return=0.005)
        sig_xau = _make_signal("XAU/USD", mean_return=0.010)  # strong signal
        for sym in {"EUR/USD", "XAU/USD"} | set(bot.ctx.candle_symbols):
            _seed_candles(bot, sym, base_close=1.0, count=TEST_CONTEXT_BARS)

        async def fake_scan(symbols: list[str], fetcher: Any) -> list[AssetSignal]:
            return [sig_eur, sig_xau]

        bot.ctx.topk_strategy.scan = fake_scan
        bot.ctx.topk_strategy._path_signals = {}

        async def stop_after_alert(*args: Any, **kwargs: Any) -> bool:
            bot.ctx.shutdown_event.set()
            return True

        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.side_effect = stop_after_alert

        await bot.ctx.runner.topk_rerank_loop()

        # Both rows recorded — collection is independent of selection
        eur_rows = bot.ctx.candle_db.get_signal_history(symbol="EUR/USD")
        xau_rows = bot.ctx.candle_db.get_signal_history(symbol="XAU/USD")
        assert len(eur_rows) == 1
        assert len(xau_rows) == 1
        assert xau_rows[0]["mean_return"] == pytest.approx(0.010)

    @pytest.mark.asyncio
    async def test_excluded_symbol_never_selected_even_when_best(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """XAU/USD has the highest mean_return but is excluded — top-K must
        skip it and select the next-best non-excluded symbol."""
        monkeypatch.setattr("bot.strategy.rerank_runner.is_market_open", lambda *a, **kw: True)
        bot.ctx.config.topk_exclude_from_selection = ["XAU/USD"]

        sig_xau = _make_signal("XAU/USD", mean_return=0.020)  # highest
        sig_eur = _make_signal("EUR/USD", mean_return=0.010)  # 2nd
        for sym in {"EUR/USD", "XAU/USD"} | set(bot.ctx.candle_symbols):
            _seed_candles(bot, sym, base_close=1.0, count=TEST_CONTEXT_BARS)

        async def fake_scan(symbols: list[str], fetcher: Any) -> list[AssetSignal]:
            return [sig_xau, sig_eur]

        bot.ctx.topk_strategy.scan = fake_scan
        bot.ctx.topk_strategy._path_signals = {}

        async def stop_after_alert(*args: Any, **kwargs: Any) -> bool:
            bot.ctx.shutdown_event.set()
            return True

        cast(AsyncMock, bot.ctx.alerter).send_topk_rerank.side_effect = stop_after_alert

        await bot.ctx.runner.topk_rerank_loop()

        # XAU was the best but excluded — EUR should be selected instead
        assert "XAU/USD" not in bot.ctx.topk_selected
        assert "EUR/USD" in bot.ctx.topk_selected


# ---------------------------------------------------------------------------
# EODHD-path integration (production config: candle_exchange="eodhd")
#
# The fixtures above drive the twelvedata rollback path. Production runs the
# 28-symbol EODHD universe with EODHD-first quote scales (gold/silver use a
# calibrated cross-instrument scale; US single-name shares use 100). These
# tests exercise the same hot paths on that path so the production scale
# round-trip and EPIC wiring are covered, not just the rollback path.
# ---------------------------------------------------------------------------


@pytest.fixture
def eodhd_bot(monkeypatch: pytest.MonkeyPatch, tmp_path: Any) -> Iterator[TradingBot]:
    """TradingBot wired for the production EODHD path with a mocked IG client.

    Mirrors the ``bot`` fixture but with ``candle_exchange="eodhd"`` — so the
    EODHD 28-symbol EPIC map and EODHD-first quote scales are in force. Since
    2026-06-19 the lifecycle DOES spawn ``IGCandleLSFeed`` under EODHD (for the
    IG-native metals), so its run-loop is stubbed dormant here too.
    """
    monkeypatch.chdir(tmp_path)
    config = BotConfig(
        broker="ig",
        bot_env="demo",
        ig_demo_api="test_api_key",
        ig_demo_username="test_user",
        ig_demo_password="test_pass",
        candle_exchange="eodhd",
        eodhd_api="test_eodhd_key",
        topk_enabled=True,
        kronos_dir=str(tmp_path / "kronos_src"),
        topk_k=3,
        topk_pred_len=120,
        topk_min_predicted_return=0.001,
        topk_min_confidence=0.70,
        topk_max_uncertainty=2.0,
        topk_min_stop_pct=0.005,
        topk_vol_stop_multiplier=2.0,
        kronos_context_bars=TEST_CONTEXT_BARS,
        topk_watchlist=["EUR/USD", "USD/JPY", "XAU/USD", "MO"],
    )
    config.validate_config()
    bot = TradingBot(config)

    mock_ig = AsyncMock()
    mock_ig.fetch_balance = AsyncMock(
        return_value={"equity": 10_000.0, "margin": 100.0, "balance": 10_000.0, "open_pnl": 0.0}
    )
    mock_ig.fetch_positions_raw = AsyncMock(return_value=[])
    mock_ig._cst = "test_cst_token"
    mock_ig.require_tradeable = AsyncMock(return_value=None)
    bot.ctx.ig_client = mock_ig

    # Metals are IG-native under EODHD now, so start() spawns IGCandleLSFeed;
    # keep its run-loop dormant so no real Lightstreamer connect is attempted.
    async def _dormant_candle_feed(_self: object) -> None:
        await bot.ctx.shutdown_event.wait()

    monkeypatch.setattr("bot.data.ig_candle_feed.IGCandleLSFeed.run", _dormant_candle_feed)

    bot.ctx.alerter = AsyncMock()
    for _m in (
        "send_topk_rerank",
        "send_trade_alert",
        "alert_take_profit",
        "send_startup",
        "send_shutdown",
        "send_risk_alert",
    ):
        setattr(bot.ctx.alerter, _m, AsyncMock(return_value=True))

    yield bot

    if bot.ctx.candle_db._conn is not None:
        bot.ctx.candle_db.close()


class TestEodhdWiring:
    def test_epic_map_is_the_eodhd_universe(self, eodhd_bot: TradingBot) -> None:
        # candle symbol → IG EPIC must come from eodhd_symbols, not the TD map.
        assert eodhd_bot.ctx.candle_epic_map["XAU/USD"] == "CS.D.USCGC.TODAY.IP"
        assert eodhd_bot.ctx.candle_epic_map["MO"] == "SE.D.MO.DAILY.IP"
        assert eodhd_bot.ctx.candle_epic_map["EUR/USD"] == "CS.D.EURUSD.TODAY.IP"
        # Dropped legacy ETF proxies are not in the eodhd universe.
        assert "SPY" not in eodhd_bot.ctx.candle_epic_map

    @pytest.mark.asyncio
    async def test_ig_native_metals_feed_spawned_under_eodhd(self, eodhd_bot: TradingBot) -> None:
        """Regression: under EODHD the lifecycle must still spawn IGCandleLSFeed
        for the IG-native metals (XAU/XAG). It was previously gated out by
        ``candle_exchange != 'eodhd'``, which left metals with no candle source
        once they were excluded from the EODHD feed."""
        from bot.data.ig_candle_feed import IGCandleLSFeed

        eodhd_bot.ctx.ig_client.fetch_positions_raw = AsyncMock(return_value=[])
        TestStartupRecovery._stub_long_running(eodhd_bot)
        eodhd_bot.ctx.shutdown_event.set()
        await eodhd_bot.start()

        assert isinstance(eodhd_bot.ctx.ig_candle_feed, IGCandleLSFeed)
        assert eodhd_bot.ctx.ig_candle_feed._epic_to_symbol == {
            "CS.D.USCGC.TODAY.IP": "XAU/USD",
            "CS.D.USCSI.TODAY.IP": "XAG/USD",
        }


class TestEodhdStopLossScales:
    """Stop-loss on the production EODHD path: entry is stored at IG fill level,
    candles arrive at EODHD face value, and ``ig_quote_scale`` must reconcile
    them. Guards the gold cross-instrument scale (~10.84) and the US-share scale
    (100) the twelvedata tests never exercised — the same class of bug that once
    closed every ETF position on the first candle."""

    @staticmethod
    def _arm(bot: TradingBot, symbol: str, face_price: float, stop_pct: float = 0.01) -> Position:
        pos = _make_position(symbol, _ig_level(symbol, face_price))
        bot.ctx.state.positions[symbol] = pos
        bot.ctx.ig_deal_ids[symbol] = f"DEAL_{symbol}"
        bot.ctx.topk_signals = [_make_signal(symbol, stop_pct=stop_pct)]
        bot.ctx.topk_scanned = True
        bot.ctx.topk_selected = []
        # Isolate the stop *decision* from the close machinery (covered elsewhere).
        bot.ctx.closer.close_ig_position = AsyncMock(return_value=True)  # type: ignore[method-assign]
        return pos

    @pytest.mark.asyncio
    async def test_gold_xau_at_entry_does_not_close(self, eodhd_bot: TradingBot) -> None:
        # GLD candle 412.0 → IG spot ~4467 (scale ~10.84). Same price → no loss.
        pos = self._arm(eodhd_bot, "XAU/USD", 412.0)
        await eodhd_bot.ctx.runner.process_candle_ig_topk("XAU/USD", 412.0, pos)
        eodhd_bot.ctx.closer.close_ig_position.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_gold_xau_breach_triggers_close(self, eodhd_bot: TradingBot) -> None:
        # If the gold scale were wrong (e.g. 1.0) the entry/current mismatch would
        # close instantly; with the calibrated 10.84 scale a real 2% GLD drop is a
        # real 2% IG-level loss → breaches the 1% stop.
        pos = self._arm(eodhd_bot, "XAU/USD", 412.0, stop_pct=0.01)
        await eodhd_bot.ctx.runner.process_candle_ig_topk("XAU/USD", 412.0 * 0.98, pos)
        eodhd_bot.ctx.closer.close_ig_position.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_us_share_mo_at_entry_does_not_close(self, eodhd_bot: TradingBot) -> None:
        # MO $45 → IG level 4500 (cents, scale 100). Same price → no loss.
        pos = self._arm(eodhd_bot, "MO", 45.0)
        await eodhd_bot.ctx.runner.process_candle_ig_topk("MO", 45.0, pos)
        eodhd_bot.ctx.closer.close_ig_position.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_us_share_mo_breach_triggers_close(self, eodhd_bot: TradingBot) -> None:
        pos = self._arm(eodhd_bot, "MO", 45.0, stop_pct=0.01)
        await eodhd_bot.ctx.runner.process_candle_ig_topk("MO", 45.0 * 0.97, pos)
        eodhd_bot.ctx.closer.close_ig_position.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_forex_eurusd_breach_triggers_close(self, eodhd_bot: TradingBot) -> None:
        pos = self._arm(eodhd_bot, "EUR/USD", 1.10, stop_pct=0.005)
        await eodhd_bot.ctx.runner.process_candle_ig_topk("EUR/USD", 1.10 * 0.99, pos)
        eodhd_bot.ctx.closer.close_ig_position.assert_awaited_once()


class TestEodhdEntryPath:
    @pytest.mark.asyncio
    async def test_entry_places_order_with_eodhd_epic(
        self, eodhd_bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # A selected gold signal must place an order against the EODHD EPIC
        # (CS.D.USCGC.TODAY.IP), proving the eodhd EPIC map is wired into entry.
        monkeypatch.setattr("bot.strategy.rerank_runner.is_safe_for_entry", lambda *a, **kw: True)
        eodhd_bot.ctx.topk_scanned = True
        eodhd_bot.ctx.topk_selected = ["XAU/USD"]
        eodhd_bot.ctx.topk_signals = [_make_signal("XAU/USD", stop_pct=0.01)]
        eodhd_bot.ctx.topk_strategy._path_signals = {
            "XAU/USD": _make_path_signal("XAU/USD", entry_price=412.0)
        }
        eodhd_bot.ctx.ig_client.place_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.USCGC.TODAY.IP",
                side=OrderSide.BUY,
                average_price=0.0,
                status=OrderStatus.PENDING,
                order_id="REF_XAU",
            )
        )
        eodhd_bot.ctx.ig_client.confirm_order = AsyncMock(
            return_value=_make_order_result(
                symbol="CS.D.USCGC.TODAY.IP",
                side=OrderSide.BUY,
                average_price=4467.0,
                order_id="DEAL_XAU",
            )
        )

        await eodhd_bot.ctx.runner.process_candle_ig_topk("XAU/USD", 412.0, None)

        eodhd_bot.ctx.ig_client.place_order.assert_awaited_once()
        order_arg = eodhd_bot.ctx.ig_client.place_order.call_args.args[0]
        assert order_arg.epic == "CS.D.USCGC.TODAY.IP"
        assert order_arg.direction == "BUY"
        assert order_arg.size > 0
        assert eodhd_bot.ctx.ig_deal_ids["XAU/USD"] == "DEAL_XAU"


class TestHeartbeatPositionRefresh:
    """The heartbeat refreshes each open position's live IG price into state so
    the dashboard shows a fresh per-position P&L (≤60s) instead of the last
    closed hourly candle."""

    @pytest.mark.asyncio
    async def test_heartbeat_writes_live_position_price_to_state(
        self, bot: TradingBot, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        bot.ctx.state.positions["EUR/USD"] = _make_position("EUR/USD", _ig_level("EUR/USD", 1.10))
        bot.ctx.ig_client.fetch_positions = AsyncMock(
            return_value=[_make_position("CS.D.EURUSD.TODAY.IP", 11050.0)]
        )

        # Drive exactly one heartbeat body, then exit the loop. shield is reduced
        # to identity so we can close the inner wait() coroutine cleanly.
        monkeypatch.setattr("bot.monitoring.health.asyncio.shield", lambda c: c)

        async def _one_shot(awaitable: Any, **_: Any) -> None:
            if asyncio.iscoroutine(awaitable):
                awaitable.close()
            bot.ctx.shutdown_event.set()  # makes `while not is_set()` exit after the body
            raise TimeoutError

        monkeypatch.setattr("bot.monitoring.health.asyncio.wait_for", _one_shot)

        await bot.ctx.health.health_check()

        assert bot.ctx.state.positions["EUR/USD"].current_price == pytest.approx(11050.0)
