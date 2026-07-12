"""Feed-staleness detection in the heartbeat (HealthMonitor._check_feed_staleness).

Regression guard for a real outage class: the EODHD WebSocket can go
"connected but silent", freezing every FX/share candle for hours while the
heartbeat's ``connected`` flag (IG REST session) stayed True, so nothing
alerted. The check canaries each feed by its freshest currently-open symbol.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from bot.monitoring import health
from bot.monitoring.health import _FEED_STALENESS_MS, HealthMonitor

NOW_MS = 1_782_240_000_000  # 2026-06-23T18:40:00Z, an arbitrary "now"
FRESH = NOW_MS - 5 * 60_000  # 5 min old
STALE = NOW_MS - 20 * 3_600_000  # 20h old (well past the 3h threshold)

_EODHD = ["EUR/USD", "GBP/USD", "F", "PFE"]
_METALS = ["XAU/USD", "XAG/USD"]


def _make_monitor(latest: dict[str, int | None]) -> tuple[HealthMonitor, AsyncMock]:
    """Build a HealthMonitor over a fake context. ``latest`` maps symbol → candle
    timestamp ms (or None for "no candle")."""

    def get_latest_candle(sym: str) -> object | None:
        ts = latest.get(sym)
        return None if ts is None else SimpleNamespace(timestamp=ts)

    alerter = SimpleNamespace(send_error=AsyncMock())
    ctx = SimpleNamespace(
        candle_symbols=_EODHD + _METALS,
        store=SimpleNamespace(get_latest_candle=get_latest_candle),
        alerter=alerter,
    )
    return HealthMonitor(ctx), alerter.send_error  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _all_markets_open(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "is_market_open", lambda _sym: True)


@pytest.mark.asyncio
async def test_alerts_once_when_eodhd_feed_stale() -> None:
    latest = {s: STALE for s in _EODHD} | {s: FRESH for s in _METALS}
    mon, send_error = _make_monitor(latest)

    await mon._check_feed_staleness(NOW_MS)
    assert send_error.await_count == 1
    msg = send_error.await_args.args[0]
    assert "EODHD" in msg and "stale" in msg.lower()
    assert "EODHD (FX + US shares)" in mon._stale_feeds
    assert "IG-native metals (XAU/XAG)" not in mon._stale_feeds

    # Still stale next heartbeat → no duplicate alert.
    await mon._check_feed_staleness(NOW_MS)
    assert send_error.await_count == 1


@pytest.mark.asyncio
async def test_no_alert_when_all_fresh() -> None:
    mon, send_error = _make_monitor({s: FRESH for s in _EODHD + _METALS})
    await mon._check_feed_staleness(NOW_MS)
    send_error.assert_not_awaited()
    assert not mon._stale_feeds


@pytest.mark.asyncio
async def test_no_alert_when_market_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(health, "is_market_open", lambda _sym: False)
    mon, send_error = _make_monitor({s: STALE for s in _EODHD + _METALS})
    await mon._check_feed_staleness(NOW_MS)
    send_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_fresh_fx_canary_masks_cold_share_at_session_open() -> None:
    """At the US share session open, shares are still cold (last candle from the
    prior session) but FX is live. The freshest-open canary must keep EODHD
    healthy, not false-positive on the cold shares."""
    latest = {"EUR/USD": FRESH, "GBP/USD": FRESH, "F": STALE, "PFE": STALE}
    latest |= {s: FRESH for s in _METALS}
    mon, send_error = _make_monitor(latest)
    await mon._check_feed_staleness(NOW_MS)
    send_error.assert_not_awaited()


@pytest.mark.asyncio
async def test_recovery_emits_alert_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    mon, send_error = _make_monitor({s: STALE for s in _EODHD} | {s: FRESH for s in _METALS})
    await mon._check_feed_staleness(NOW_MS)
    assert send_error.await_count == 1

    # Feed recovers — fresh candles for everyone.
    mon._ctx.store.get_latest_candle = lambda _s: SimpleNamespace(timestamp=FRESH)  # type: ignore[attr-defined]
    with caplog.at_level("INFO", logger="bot.monitoring.health"):
        await mon._check_feed_staleness(NOW_MS)
    assert send_error.await_count == 2
    assert "recovered" in send_error.await_args.args[0].lower()
    assert not mon._stale_feeds
    # The recovery transition must be logged (not Telegram-only), so a
    # postmortem can see when the staleness window closed.
    assert any("FEED RECOVERED" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_threshold_boundary_not_flagged() -> None:
    """A candle just under the threshold is healthy."""
    just_ok = NOW_MS - (_FEED_STALENESS_MS - 60_000)
    mon, send_error = _make_monitor({s: just_ok for s in _EODHD + _METALS})
    await mon._check_feed_staleness(NOW_MS)
    send_error.assert_not_awaited()
