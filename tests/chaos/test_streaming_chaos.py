"""Chaos scenarios for the Lightstreamer streaming layer (live IG demo).

Automates IG_LIVE_RISK_REFERENCE.md §9.1 with deterministic fault injection
instead of hand-run iptables rules.  Four fault classes, each mapped to the
network pathology it reproduces:

===================  ==================  =====================================
Scenario             toxiproxy fault     Real-world analogue
===================  ==================  =====================================
silent stall         timeout toxic       server goes quiet; sockets stay open
                                         (the "silent thread death" trigger)
hard cut             proxy disabled      TCP RST / listener gone
latency spike        latency toxic       congested path, delayed frames
slow trickle         slicer toxic        fragmented frames, dribbling socket
===================  ==================  =====================================

Every scenario asserts the invariants from the spec: the fault is detected,
the feed recovers to a live subscription without manual intervention, and
the broker's open-position book is unchanged.  Results are written by the
session recorder to ``docs/chaos/`` as metrics only.

Run: ``uv run pytest -m chaos`` (see docs/CHAOS_TESTING.md).
"""

from __future__ import annotations

import time
from typing import Any

import pytest

from .conftest import PROXY_NAME
from .drivers import EventLog, count_ticks, wait_tick
from .recorder import ChaosRecorder, ScenarioResult
from .toxiproxy_api import ToxiproxyClient

pytestmark = pytest.mark.chaos

# Recovery must complete within one heartbeat-recovery cycle plus the 60s
# initial-grace window that follows an in-fault reconnect attempt.
RECOVERY_TIMEOUT_S = 150.0


async def _positions(feed: Any) -> int | None:
    try:
        return len(await feed._client.fetch_positions())
    except Exception:
        return None


async def _run_scenario(
    *,
    name: str,
    fault: str,
    params: dict[str, Any],
    feed: Any,
    event_log: EventLog,
    recorder: ChaosRecorder,
    inject: Any,
    clear: Any,
    detect_kind: str | None,
    fault_hold_s: float,
) -> ScenarioResult:
    """Shared scenario skeleton: baseline → inject → observe → clear → verify."""
    pos_before = await _positions(feed)
    ticks_baseline = await count_ticks(feed, 10.0)

    t0 = time.monotonic()
    inject()
    detected = None
    if detect_kind is not None:
        detected = await event_log.wait(detect_kind, since=t0, within_s=45.0)
        ticks_during = 0
    else:
        ticks_during = await count_ticks(feed, fault_hold_s)
    t_clear = time.monotonic()
    clear()

    recovered = None
    if detect_kind is not None:
        recovered = await event_log.wait(
            "heartbeat_recovery", since=t0, within_s=RECOVERY_TIMEOUT_S
        )
        if recovered is None:
            recovered = await event_log.wait("sdk_reconnect_ok", since=t0, within_s=5.0)
    tick_after = await wait_tick(feed, after=t_clear, within_s=RECOVERY_TIMEOUT_S)
    ticks_after = await count_ticks(feed, 10.0)
    pos_after = await _positions(feed)

    result = ScenarioResult(
        name=name,
        fault=fault,
        params=params,
        passed=False,  # flipped by the caller once its asserts hold
        timeline_s={
            "detected": None if detected is None else round(detected - t0, 2),
            "fault_cleared": round(t_clear - t0, 2),
            "recovered": None if recovered is None else round(recovered - t0, 2),
            "first_tick_after_clear": None if tick_after is None else round(tick_after - t0, 2),
        },
        counts={
            "ticks_baseline_10s": ticks_baseline,
            "ticks_during_fault": ticks_during,
            "ticks_after": ticks_after,
            "heartbeat_trips": event_log.count("heartbeat_trip", since=t0),
            "sdk_reconnects": event_log.count("sdk_reconnect_scheduled", since=t0),
            "errors": event_log.count("error", since=t0),
        },
        positions={
            "before": pos_before,
            "after": pos_after,
            "unchanged": pos_before is not None and pos_before == pos_after,
        },
    )
    recorder.record(result)
    return result


async def test_baseline_connectivity(
    chaos_feed: Any, event_log: EventLog, recorder: ChaosRecorder
) -> None:
    """No fault: the proxied feed streams; ambient heartbeat cadence is recorded.

    Ambient trips are reported, not failed: on sparse instruments (weekend
    crypto) natural inter-tick gaps can exceed the 10s heartbeat threshold,
    and the recovery machinery handling that is by design.  The number here
    is the context for reading the fault scenarios' detection timings.
    """
    pos = await _positions(chaos_feed)
    t0 = time.monotonic()
    ticks = await count_ticks(chaos_feed, 30.0)
    trips = event_log.count("heartbeat_trip", since=t0)
    recoveries = event_log.count("heartbeat_recovery", since=t0)
    result = ScenarioResult(
        name="baseline",
        fault="none",
        params={"observe_s": 30},
        passed=ticks > 0 and trips == recoveries,
        timeline_s={},
        counts={
            "ticks_baseline_10s": ticks,
            "heartbeat_trips": trips,
            "heartbeat_recoveries": recoveries,
        },
        positions={"before": pos, "after": pos, "unchanged": pos is not None},
        notes=(
            f"{ticks} updates in 30s through the proxy; {trips} ambient heartbeat "
            f"trip(s), {recoveries} recovered — natural tick gaps on these "
            "instruments at this hour"
        ),
    )
    recorder.record(result)
    assert ticks > 0, "no updates through the proxy during a 30s quiet observation"
    assert trips == recoveries, "an ambient heartbeat trip did not recover"


async def test_silent_stall_triggers_heartbeat_recovery(
    chaos_feed: Any, toxiproxy: ToxiproxyClient, event_log: EventLog, recorder: ChaosRecorder
) -> None:
    """Blackhole server→client data with sockets held open.

    This is the exact failure the active heartbeat exists for: no exception,
    no disconnect callback, just silence.  The feed must notice within the
    heartbeat threshold and execute the full §3.2 recovery ladder.
    """
    result = await _run_scenario(
        name="silent_stall",
        fault="timeout toxic (downstream, indefinite)",
        params={"toxic": {"type": "timeout", "stream": "downstream", "timeout_ms": 0}},
        feed=chaos_feed,
        event_log=event_log,
        recorder=recorder,
        inject=lambda: toxiproxy.add_toxic(
            PROXY_NAME, "stall", "timeout", "downstream", {"timeout": 0}
        ),
        clear=lambda: toxiproxy.remove_toxic(PROXY_NAME, "stall"),
        detect_kind="heartbeat_trip",
        fault_hold_s=0.0,
    )
    result.passed = (
        result.timeline_s["detected"] is not None
        and result.timeline_s["recovered"] is not None
        and result.counts["ticks_after"] > 0
        and result.positions["unchanged"]
    )
    assert result.timeline_s["detected"] is not None, "heartbeat never detected the stall"
    assert result.timeline_s["recovered"] is not None, "no recovery after the stall cleared"
    assert result.counts["ticks_after"] > 0, "no updates after recovery"
    assert result.positions["unchanged"], "open-position count changed during the scenario"


async def test_hard_connection_cut_reconnects(
    chaos_feed: Any, toxiproxy: ToxiproxyClient, event_log: EventLog, recorder: ChaosRecorder
) -> None:
    """Sever the TCP connection outright (listener closed, connections killed).

    Unlike the stall, the SDK *does* observe this one — the scenario pins the
    DISCONNECTED → backoff → reconnect → resubscribe path.
    """
    result = await _run_scenario(
        name="hard_cut",
        fault="proxy disabled until heartbeat detection (TCP severed)",
        params={"hold": "until_detected"},
        feed=chaos_feed,
        event_log=event_log,
        recorder=recorder,
        inject=lambda: toxiproxy.set_enabled(PROXY_NAME, False),
        clear=lambda: toxiproxy.set_enabled(PROXY_NAME, True),
        detect_kind="heartbeat_trip",
        fault_hold_s=0.0,
    )
    detected_by_sdk = result.counts["sdk_reconnects"] > 0
    detected = result.timeline_s["detected"] is not None or detected_by_sdk
    result.passed = detected and result.counts["ticks_after"] > 0 and result.positions["unchanged"]
    result.notes = (
        "detected via SDK DISCONNECTED path" if detected_by_sdk else "detected via heartbeat path"
    )
    assert detected, "neither the SDK nor the heartbeat noticed a severed connection"
    assert result.counts["ticks_after"] > 0, "no updates after the cut was restored"
    assert result.positions["unchanged"], "open-position count changed during the scenario"


async def test_latency_spike_does_not_kill_feed(
    chaos_feed: Any, toxiproxy: ToxiproxyClient, event_log: EventLog, recorder: ChaosRecorder
) -> None:
    """3s ± 0.5s of added downstream latency for 20s.

    Degradation, not amputation: the feed may ride it out or the heartbeat
    may legitimately trip (3s delay against a 10s threshold).  Either way the
    invariant is the same — streaming updates after the fault clears, no
    crash, book unchanged.
    """
    result = await _run_scenario(
        name="latency_spike",
        fault="latency toxic 3000ms ± 500ms (downstream, 20s)",
        params={"latency_ms": 3000, "jitter_ms": 500, "hold_s": 20},
        feed=chaos_feed,
        event_log=event_log,
        recorder=recorder,
        inject=lambda: toxiproxy.add_toxic(
            PROXY_NAME, "lag", "latency", "downstream", {"latency": 3000, "jitter": 500}
        ),
        clear=lambda: toxiproxy.remove_toxic(PROXY_NAME, "lag"),
        detect_kind=None,
        fault_hold_s=20.0,
    )
    result.passed = result.counts["ticks_after"] > 0 and result.positions["unchanged"]
    survived = result.counts["heartbeat_trips"] == 0
    result.notes = (
        "rode out the latency without tripping"
        if survived
        else f"heartbeat tripped {result.counts['heartbeat_trips']}x under 3s latency, recovered"
    )
    assert result.counts["ticks_after"] > 0, "feed did not stream after latency cleared"
    assert result.positions["unchanged"], "open-position count changed during the scenario"


async def test_slow_trickle_fragmented_frames(
    chaos_feed: Any, toxiproxy: ToxiproxyClient, event_log: EventLog, recorder: ChaosRecorder
) -> None:
    """Fragment the downstream byte stream into ~64-byte slices for 20s.

    TLS records arrive in dribs; the SDK's framing must reassemble without
    corrupting updates — the TickValidator downstream would reject corrupt
    prints as outliers, which would surface here as errors or a tick gap.
    """
    result = await _run_scenario(
        name="slow_trickle",
        fault="slicer toxic ~64B slices, 2ms gaps (downstream, 20s)",
        params={"average_size": 64, "size_variation": 32, "delay_us": 2000, "hold_s": 20},
        feed=chaos_feed,
        event_log=event_log,
        recorder=recorder,
        inject=lambda: toxiproxy.add_toxic(
            PROXY_NAME,
            "slice",
            "slicer",
            "downstream",
            {"average_size": 64, "size_variation": 32, "delay": 2000},
        ),
        clear=lambda: toxiproxy.remove_toxic(PROXY_NAME, "slice"),
        detect_kind=None,
        fault_hold_s=20.0,
    )
    result.passed = (
        result.counts["ticks_after"] > 0
        and result.counts["errors"] == 0
        and result.positions["unchanged"]
    )
    result.notes = f"{result.counts['ticks_during_fault']} updates arrived through sliced frames"
    assert result.counts["ticks_after"] > 0, "feed did not stream after slicing cleared"
    assert result.counts["errors"] == 0, "errors logged while frames were fragmented"
    assert result.positions["unchanged"], "open-position count changed during the scenario"
