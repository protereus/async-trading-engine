"""Per-epic spread-widening monitor (IG_LIVE_RISK_REFERENCE.md §6.1).

Reference: ``IG_LIVE_RISK_REFERENCE.md §6.1`` — spreads on IG are integrated
into the bid-ask quote, not a line-item fee.  Every position opens at an
immediate loss equal to the spread.  Spreads widen with volatility and
illiquidity; UK non-FTSE 350 small-caps were recently widened from 0.35 %
to 0.50 % on DFBs.  Entering a trade during a transient spread blow-out
turns a marginal positive-EV signal into a guaranteed loser.

This module maintains a per-epic rolling window of observed spreads (in IG
points) and exposes:
  - ``record(epic, spread_pts)`` — feed an observed spread sample
  - ``stats(epic)`` → ``(mean, stdev) | None`` (None until min-prime samples)
  - ``is_anomalous(epic, current_spread=None)`` — True if spread > mean + Nσ
  - ``latest_spread(epic)`` — most recent observed value

Scope decisions:
  - In-memory only.  30-day persistence (the spec target) is a follow-up;
    on restart the window rebuilds from incoming ticks.  At 1-min sampling
    over 28 epics that's < 1 MB and primes in ~100 minutes of clock-time
    market hours.
  - Default Nσ = 2.0 per §6.1 spec; tunable per instance.
  - Sampling cadence is the caller's responsibility (typically once per
    confirmed candle close to keep memory in check).
"""

from __future__ import annotations

import statistics
from collections import deque

# 30 days × 24h × 60 min = 43,200 1-min samples per epic, the spec target.
_DEFAULT_WINDOW = 43_200
# Need at least this many samples before gating any trades — without enough
# history we'd false-positive on the first wide print after startup.
_DEFAULT_MIN_PRIME = 200
# §6.1 — halt above mean + 2σ.
_DEFAULT_N_SIGMA = 2.0


class SpreadMonitor:
    """Per-epic rolling spread-anomaly detector.

    Single-event-loop only — no locks.  Caller is expected to invoke
    ``record`` from one async path (typically ``_handle_chart_update``) and
    ``is_anomalous`` from another (pre-trade gate).  Float reads/writes are
    atomic under CPython's GIL so cross-task access is safe.
    """

    def __init__(
        self,
        *,
        window: int = _DEFAULT_WINDOW,
        min_prime: int = _DEFAULT_MIN_PRIME,
        n_sigma: float = _DEFAULT_N_SIGMA,
    ) -> None:
        self._window = window
        self._min_prime = min_prime
        self._n_sigma = n_sigma
        self._samples: dict[str, deque[float]] = {}
        self._latest: dict[str, float] = {}

    def record(self, epic: str, spread_pts: float) -> None:
        """Feed an observed spread.  Negative/zero values are ignored (a
        bid > offer print is a corrupted tick that the validator already
        rejects; this is a defensive belt-and-braces)."""
        if spread_pts <= 0:
            return
        if epic not in self._samples:
            self._samples[epic] = deque(maxlen=self._window)
        self._samples[epic].append(spread_pts)
        self._latest[epic] = spread_pts

    def latest_spread(self, epic: str) -> float | None:
        return self._latest.get(epic)

    def sample_count(self, epic: str) -> int:
        samples = self._samples.get(epic)
        return len(samples) if samples is not None else 0

    def stats(self, epic: str) -> tuple[float, float] | None:
        """Return (mean, stdev) over the rolling window for *epic*, or None
        if we don't yet have ``min_prime`` samples."""
        samples = self._samples.get(epic)
        if samples is None or len(samples) < self._min_prime:
            return None
        mean = statistics.fmean(samples)
        stdev = statistics.stdev(samples) if len(samples) > 1 else 0.0
        return mean, stdev

    def is_anomalous(self, epic: str, current_spread: float | None = None) -> bool:
        """True if *current_spread* (or the latest observed) > mean + Nσ.

        Until the rolling window has ``min_prime`` samples this returns
        False — we don't gate trades on undersized history (the alternative
        is a self-fulfilling "skip everything after restart" loop)."""
        if current_spread is None:
            current_spread = self._latest.get(epic)
        if current_spread is None or current_spread <= 0:
            return False
        s = self.stats(epic)
        if s is None:
            return False
        mean, stdev = s
        return current_spread > mean + self._n_sigma * stdev

    def reset(self, epic: str | None = None) -> None:
        """Drop rolling state for *epic* (or all epics if None)."""
        if epic is None:
            self._samples.clear()
            self._latest.clear()
            return
        self._samples.pop(epic, None)
        self._latest.pop(epic, None)
