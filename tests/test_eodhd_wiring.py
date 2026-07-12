"""EODHD feed-selection wiring (migration step 7).

Verifies that ``CANDLE_EXCHANGE=eodhd`` routes the universe + feed correctly:
the bot keys come from the EODHD symbol map and ``build_feed_task`` selects
``EODHDFeed``. The feed task is cancelled before it runs (no network I/O).
"""

from __future__ import annotations

import asyncio

import pytest

from bot.config import BotConfig
from bot.data.eodhd_feed import EODHDFeed
from bot.data.eodhd_symbols import SYMBOL_EPIC_MAP as EODHD_MAP
from bot.main import TradingBot


def _eodhd_bot(tmp_path: object) -> TradingBot:
    config = BotConfig(
        broker="ig",
        bot_env="demo",
        ig_demo_api="k",
        ig_demo_username="u",
        ig_demo_password="p",
        candle_exchange="eodhd",
        eodhd_api="test_eodhd_key",
        topk_enabled=True,
        kronos_dir=str(tmp_path) + "/kronos_src",
    )
    config.validate_config()
    return TradingBot(config)


def test_eodhd_symbol_routing(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    bot = _eodhd_bot(tmp_path)
    assert set(bot.ctx.candle_symbols) == set(EODHD_MAP)
    assert bot.ctx.candle_epic_map == EODHD_MAP
    assert {"F", "XOM", "XAU/USD", "XAG/USD"} <= set(bot.ctx.candle_symbols)


def test_eodhd_feed_excludes_ig_native_metals(tmp_path: object) -> None:
    """Metals are IG-native (IGCandleLSFeed) since 2026-06-19, so the EODHD feed
    must not own them: excluded from its backfill symbol list AND its 'us' WS
    subscription, while the equities on that endpoint stay."""
    from bot.core.event_bus import EventBus
    from bot.data.eodhd_feed import EODHDFeed
    from bot.data.ig_candle_aggregator import IG_NATIVE_CANDLE_SYMBOLS
    from bot.data.store import DataStore

    config = BotConfig(candle_exchange="eodhd", eodhd_api="k", candle_buffer_size=10)
    feed = EODHDFeed(DataStore(buffer_size=10), EventBus(), config)

    owned = {s.bot_key for s in feed._symbols}
    assert IG_NATIVE_CANDLE_SYMBOLS.isdisjoint(owned)  # no XAU/USD, XAG/USD
    assert {"F", "XOM", "EUR/USD"} <= owned  # equities + FX still owned

    from bot.data.eodhd_symbols import EODHD_UNIVERSE

    metal_ws = {EODHD_UNIVERSE[m].ws_symbol for m in IG_NATIVE_CANDLE_SYMBOLS}
    us_subs = set(feed._ws_symbols("us"))
    assert metal_ws.isdisjoint(us_subs)  # no metal WS codes subscribed
    assert any(s.ws_endpoint == "us" for s in feed._symbols)  # equities still subscribed


@pytest.mark.asyncio
async def test_eodhd_feed_selected(tmp_path: object, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    bot = _eodhd_bot(tmp_path)
    task = bot.ctx.lifecycle.build_feed_task()
    task.cancel()  # cancel before the loop runs run() → no backfill / WS I/O
    with pytest.raises(asyncio.CancelledError):
        await task
    assert isinstance(bot.ctx.eodhd_feed, EODHDFeed)
    assert task.get_name() == "eodhd_feed"
