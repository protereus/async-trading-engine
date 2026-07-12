"""EODHD WS reconnect-backoff policy.

Guards against the 2026-06-03 reconnect-storm regression: the loop used to
reset the backoff on mere *connect*, so a flapping endpoint (connect → drop a
few seconds later) reconnected at MIN forever and never escalated toward MAX.
The reset is now gated on connection *uptime*.
"""

from __future__ import annotations

from bot.data.eodhd_feed import (
    _WS_RECONNECT_MAX_S,
    _WS_RECONNECT_MIN_S,
    _WS_STABLE_RESET_S,
    _next_reconnect_backoff,
)


def test_stable_connection_resets_to_min() -> None:
    # A long-lived connection (uptime past the stability threshold) recovers fast.
    assert _next_reconnect_backoff(32.0, uptime_s=_WS_STABLE_RESET_S) == _WS_RECONNECT_MIN_S
    assert _next_reconnect_backoff(60.0, uptime_s=3600.0) == _WS_RECONNECT_MIN_S


def test_flapping_connection_escalates() -> None:
    # Short-lived connections escalate geometrically instead of storming at MIN.
    b = _WS_RECONNECT_MIN_S
    seen = [b]
    for _ in range(10):
        b = _next_reconnect_backoff(b, uptime_s=4.0)  # always drops quickly
        seen.append(b)
    # Strictly increasing until the ceiling, then pinned at MAX.
    assert seen[1] == _WS_RECONNECT_MIN_S * 2
    assert max(seen) == _WS_RECONNECT_MAX_S
    assert seen[-1] == _WS_RECONNECT_MAX_S


def test_failed_connect_escalates() -> None:
    # A connection that never came up (uptime 0) is treated as a failure → escalate.
    assert _next_reconnect_backoff(2.0, uptime_s=0.0) == 4.0


def test_escalation_capped_at_max() -> None:
    assert _next_reconnect_backoff(_WS_RECONNECT_MAX_S, uptime_s=1.0) == _WS_RECONNECT_MAX_S
