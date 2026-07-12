"""Tests for the per-epic spread-widening monitor (IG_LIVE_RISK_REFERENCE.md §6.1).

Reference: IG_LIVE_RISK_REFERENCE.md §6.1
"""

from __future__ import annotations

import pytest

from bot.risk.spread_monitor import SpreadMonitor


class TestRecord:
    def test_records_positive_spread(self) -> None:
        mon = SpreadMonitor(min_prime=2)
        mon.record("EUR/USD", 1.5)
        assert mon.latest_spread("EUR/USD") == 1.5
        assert mon.sample_count("EUR/USD") == 1

    def test_zero_and_negative_ignored(self) -> None:
        """A bid > offer print is a corrupted tick; the tick validator should
        already reject it, but be defensive here."""
        mon = SpreadMonitor()
        mon.record("EUR/USD", 0.0)
        mon.record("EUR/USD", -0.5)
        assert mon.latest_spread("EUR/USD") is None
        assert mon.sample_count("EUR/USD") == 0

    def test_window_bounded(self) -> None:
        mon = SpreadMonitor(window=10, min_prime=2)
        for i in range(25):
            mon.record("EUR/USD", 1.0 + i * 0.01)
        assert mon.sample_count("EUR/USD") == 10
        # Window kept the LAST 10 — oldest survivor is sample 15 (1.15)
        stats = mon.stats("EUR/USD")
        assert stats is not None
        mean, _ = stats
        # mean of 1.15, 1.16, ..., 1.24 = 1.195
        assert mean == pytest.approx(1.195, abs=1e-6)


class TestStats:
    def test_returns_none_until_min_prime(self) -> None:
        mon = SpreadMonitor(min_prime=10)
        for i in range(9):
            mon.record("EUR/USD", 1.0 + i * 0.01)
        assert mon.stats("EUR/USD") is None
        mon.record("EUR/USD", 1.10)
        assert mon.stats("EUR/USD") is not None

    def test_no_samples_returns_none(self) -> None:
        mon = SpreadMonitor()
        assert mon.stats("EUR/USD") is None

    def test_mean_and_stdev(self) -> None:
        mon = SpreadMonitor(min_prime=2)
        for v in (1.0, 2.0, 3.0, 4.0, 5.0):
            mon.record("EUR/USD", v)
        stats = mon.stats("EUR/USD")
        assert stats is not None
        mean, stdev = stats
        assert mean == pytest.approx(3.0)
        assert stdev == pytest.approx(1.5811, abs=1e-4)  # sample stdev


class TestIsAnomalous:
    def test_under_prime_returns_false(self) -> None:
        """Without enough history we don't gate — otherwise post-restart the
        bot would self-paralyse on every trade."""
        mon = SpreadMonitor(min_prime=100)
        mon.record("EUR/USD", 1.0)
        mon.record("EUR/USD", 100.0)  # huge spread but no baseline yet
        assert mon.is_anomalous("EUR/USD") is False

    def test_normal_spread_not_anomalous(self) -> None:
        """After priming on a tight range, a similar spread isn't flagged."""
        import random

        random.seed(0)
        mon = SpreadMonitor(min_prime=50, n_sigma=2.0)
        for _ in range(100):
            mon.record("EUR/USD", 1.0 + random.gauss(0, 0.05))
        # A new spread close to the mean shouldn't trip
        assert mon.is_anomalous("EUR/USD", 1.05) is False

    def test_spread_above_n_sigma_anomalous(self) -> None:
        import random

        random.seed(0)
        mon = SpreadMonitor(min_prime=50, n_sigma=2.0)
        for _ in range(100):
            mon.record("EUR/USD", 1.0 + random.gauss(0, 0.05))
        mean, stdev = mon.stats("EUR/USD")  # type: ignore[misc]
        # 3σ above mean is definitely anomalous at n_sigma=2
        assert mon.is_anomalous("EUR/USD", mean + 3 * stdev) is True

    def test_uses_latest_when_current_not_passed(self) -> None:
        import random

        random.seed(1)
        mon = SpreadMonitor(min_prime=50, n_sigma=2.0)
        for _ in range(100):
            mon.record("EUR/USD", 1.0 + random.gauss(0, 0.05))
        # Most recent observation is in-range — no anomaly
        assert mon.is_anomalous("EUR/USD") is False

        # Now record a wide print — that becomes the "latest"
        mon.record("EUR/USD", 10.0)
        assert mon.is_anomalous("EUR/USD") is True

    def test_unknown_epic_returns_false(self) -> None:
        mon = SpreadMonitor(min_prime=2)
        assert mon.is_anomalous("UNKNOWN") is False

    def test_zero_or_negative_current_returns_false(self) -> None:
        mon = SpreadMonitor(min_prime=2)
        for _ in range(10):
            mon.record("EUR/USD", 1.0)
        assert mon.is_anomalous("EUR/USD", 0.0) is False
        assert mon.is_anomalous("EUR/USD", -1.0) is False


class TestReset:
    def test_reset_one_epic(self) -> None:
        mon = SpreadMonitor()
        mon.record("EUR/USD", 1.0)
        mon.record("XAU/USD", 5.0)
        mon.reset("EUR/USD")
        assert mon.sample_count("EUR/USD") == 0
        assert mon.latest_spread("EUR/USD") is None
        assert mon.sample_count("XAU/USD") == 1  # unaffected

    def test_reset_all(self) -> None:
        mon = SpreadMonitor()
        mon.record("EUR/USD", 1.0)
        mon.record("XAU/USD", 5.0)
        mon.reset()
        assert mon.sample_count("EUR/USD") == 0
        assert mon.sample_count("XAU/USD") == 0


class TestEpicsIndependent:
    def test_per_epic_isolation(self) -> None:
        import random

        random.seed(0)
        mon = SpreadMonitor(min_prime=50, n_sigma=2.0)
        # Prime EUR/USD tight, XAU/USD wide
        for _ in range(100):
            mon.record("EUR/USD", 1.0 + random.gauss(0, 0.05))
            mon.record("XAU/USD", 50.0 + random.gauss(0, 5.0))

        # A spread of 10 is wild for EUR/USD but normal-ish for XAU/USD
        assert mon.is_anomalous("EUR/USD", 10.0) is True
        assert mon.is_anomalous("XAU/USD", 55.0) is False
