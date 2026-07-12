"""Tests for the IGCandleAggregator (D1 of ).

Pure unit tests against the aggregator's tick → hourly-Candle pipeline.
No LS, no IGClient, no asyncio — just feed ticks at constructed UTMs and
inspect the emit callback.
"""

from __future__ import annotations

from bot.core.models import Candle
from bot.data.ig_candle_aggregator import IGCandleAggregator

# Sample hour-start: 2026-05-29 14:00:00 UTC = 1780041600000 ms
_HOUR_START = 1_780_041_600_000
_NEXT_HOUR_START = _HOUR_START + 3_600_000


def _emit_into(sink: list[Candle]) -> IGCandleAggregator:
    return IGCandleAggregator(emit_callback=sink.append)


class TestWithinHourAccumulation:
    def test_first_tick_sets_open_high_low_close(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START, mid=88.30, market_open=True)
        bucket = agg.current_bucket("USO")
        assert bucket is not None
        assert bucket.open == 88.30
        assert bucket.high == 88.30
        assert bucket.low == 88.30
        assert bucket.close == 88.30
        assert bucket.was_subscribed_at_open is True
        assert sink == []  # nothing emitted yet; hour hasn't rolled over

    def test_subsequent_ticks_update_high_low_close_not_open(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START + 60_000, mid=88.30, market_open=True)
        agg.ingest_tick("USO", _HOUR_START + 120_000, mid=88.50, market_open=True)
        agg.ingest_tick("USO", _HOUR_START + 180_000, mid=88.20, market_open=True)
        agg.ingest_tick("USO", _HOUR_START + 240_000, mid=88.40, market_open=True)
        bucket = agg.current_bucket("USO")
        assert bucket is not None
        # First tick was at minute 1, so was_subscribed_at_open is False
        assert bucket.was_subscribed_at_open is False
        assert bucket.open == 88.30  # first observed mid
        assert bucket.high == 88.50
        assert bucket.low == 88.20
        assert bucket.close == 88.40

    def test_ltv_accumulates_across_ticks(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START, mid=88.30, market_open=True, ltv=10.0)
        agg.ingest_tick("USO", _HOUR_START + 60_000, mid=88.30, market_open=True, ltv=20.0)
        # Zero LTV is treated as missing, not a real zero-volume minute
        agg.ingest_tick("USO", _HOUR_START + 120_000, mid=88.30, market_open=True, ltv=0.0)
        bucket = agg.current_bucket("USO")
        assert bucket is not None
        assert bucket.volume == 30.0


class TestHourRollover:
    def test_rollover_emits_completed_candle(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        # Whole hour worth of ticks starting at minute 0
        for offset_min, mid in [(0, 88.0), (30, 89.0), (45, 87.5), (59, 88.5)]:
            agg.ingest_tick(
                "USO",
                _HOUR_START + offset_min * 60_000,
                mid=mid,
                market_open=True,
                ltv=10.0,
            )
        # First tick of the next hour triggers rollover + emit
        agg.ingest_tick("USO", _NEXT_HOUR_START, mid=88.7, market_open=True, ltv=10.0)

        assert len(sink) == 1
        c = sink[0]
        assert c.symbol == "USO"
        assert c.timestamp == _HOUR_START
        assert c.open == 88.0
        assert c.high == 89.0
        assert c.low == 87.5
        assert c.close == 88.5
        assert c.volume == 40.0
        assert c.is_confirmed is True

    def test_partial_first_bucket_dropped_on_rollover(self) -> None:
        """First-ever tick lands at minute 5 → bucket is held but never
        emitted.  The next hour's bucket starts fresh and IS emitted."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        # First tick well after hour 0 — was_subscribed_at_open=False
        agg.ingest_tick("USO", _HOUR_START + 5 * 60_000, mid=88.0, market_open=True)
        # Roll over: partial bucket should be DROPPED
        agg.ingest_tick("USO", _NEXT_HOUR_START, mid=89.0, market_open=True)
        assert sink == []
        # The new bucket starts at the new hour, eligible for emission
        next_bucket = agg.current_bucket("USO")
        assert next_bucket is not None
        assert next_bucket.was_subscribed_at_open is True

    def test_full_first_bucket_then_partial_handover(self) -> None:
        """First bucket subscribed in minute 0 → eligible.  After it
        rolls over, the second bucket (which inherits because it starts
        at minute 0 of the next hour) is also eligible."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START + 30_000, mid=88.0, market_open=True)
        agg.ingest_tick("USO", _HOUR_START + 30 * 60_000, mid=89.0, market_open=True)
        agg.ingest_tick("USO", _NEXT_HOUR_START, mid=90.0, market_open=True)
        assert len(sink) == 1
        assert sink[0].timestamp == _HOUR_START
        assert sink[0].open == 88.0
        assert sink[0].close == 89.0

    def test_per_symbol_buckets_isolated(self) -> None:
        """A USO rollover doesn't affect a UNG in-progress bucket."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START + 30_000, mid=88.0, market_open=True)
        agg.ingest_tick("UNG", _HOUR_START + 30_000, mid=11.0, market_open=True)
        # Roll USO forward; UNG bucket should remain in-progress
        agg.ingest_tick("USO", _NEXT_HOUR_START, mid=89.0, market_open=True)
        assert len(sink) == 1
        assert sink[0].symbol == "USO"
        ung_bucket = agg.current_bucket("UNG")
        assert ung_bucket is not None
        assert ung_bucket.hour_start_ms == _HOUR_START


class TestMarketClosedGate:
    def test_closed_ticks_are_dropped(self) -> None:
        """LS emits ticks 24/7 even for closed markets (probe finding).
        Aggregator must gate on market_open=False and discard them."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("FTSE", _HOUR_START, mid=10400.0, market_open=False)
        agg.ingest_tick("FTSE", _HOUR_START + 60_000, mid=10410.0, market_open=False)
        assert agg.current_bucket("FTSE") is None
        assert sink == []

    def test_open_then_closed_then_open_keeps_open_data(self) -> None:
        """A few ticks in an open window get a bucket; ticks during the
        closed gap don't pollute it; ticks after reopen are dropped if
        they're in a different hour OR continued if same hour."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START + 30_000, mid=88.0, market_open=True)
        # Simulated close window — discarded
        agg.ingest_tick("USO", _HOUR_START + 90_000, mid=99.0, market_open=False)
        # Reopen — same hour, accumulates into existing bucket
        agg.ingest_tick("USO", _HOUR_START + 120_000, mid=89.0, market_open=True)
        bucket = agg.current_bucket("USO")
        assert bucket is not None
        # The 99.0 tick must NOT have polluted high
        assert bucket.high == 89.0
        assert bucket.low == 88.0


class TestDefensiveInputs:
    def test_non_positive_utm_dropped(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", utm_ms=0, mid=88.0, market_open=True)
        agg.ingest_tick("USO", utm_ms=-1, mid=88.0, market_open=True)
        assert agg.current_bucket("USO") is None

    def test_non_positive_mid_dropped(self) -> None:
        """Missing / zero price is a feed problem; never enter the
        bucket — would otherwise contaminate OHLC with 0."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START, mid=0.0, market_open=True)
        agg.ingest_tick("USO", _HOUR_START + 60_000, mid=-1.5, market_open=True)
        assert agg.current_bucket("USO") is None

    def test_out_of_order_tick_within_active_bucket_dropped(self) -> None:
        """An earlier-hour tick arriving after a later one (LS reconnect
        snapshot quirk) must not retroactively pull the bucket
        back into a prior hour."""
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START + 60_000, mid=88.0, market_open=True)
        # Engineered out-of-order: tick with utm_ms BEFORE the bucket's
        # hour_start — should be dropped, bucket stays at HOUR_START
        agg.ingest_tick("USO", _HOUR_START - 60_000, mid=99.0, market_open=True)
        bucket = agg.current_bucket("USO")
        assert bucket is not None
        assert bucket.hour_start_ms == _HOUR_START
        assert bucket.high == 88.0  # 99.0 didn't get in


class TestFlush:
    def test_flush_clears_state(self) -> None:
        sink: list[Candle] = []
        agg = _emit_into(sink)
        agg.ingest_tick("USO", _HOUR_START, mid=88.0, market_open=True)
        agg.flush()
        assert agg.current_bucket("USO") is None
        # Flush does NOT emit partial buckets — confirmed-only semantics
        assert sink == []
