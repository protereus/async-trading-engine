"""Correlation-aware position selection.

CorrelationTracker maintains a rolling Pearson correlation matrix of 1h returns
across the asset universe. Recomputed at each rerank from the last
``lookback_bars`` candles prepared by TopKStrategy.

select_uncorrelated filters a pre-ranked candidate list so that no two selected
assets have |Pearson correlation| > max_correlation (default 0.65).  The filter
preserves rank order — the highest-scoring uncorrelated symbol is always preferred
over a lower-scoring one.

When correlation_max=1.0 no candidate can ever be bumped (|corr| ≤ 1 always),
so the filter is a no-op and existing behaviour is reproduced exactly.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class CorrelationConfig:
    enabled: bool = True
    lookback_bars: int = 200  # bars of return history used; ~8 days of 1h bars
    max_correlation: float = 0.65  # corr threshold; set 1.0 to disable filtering
    # Long-only books: two assets that are strongly *negatively* correlated and
    # both held LONG are offsetting (variance-reducing), i.e. diversifying — not
    # redundant.  When ``long_only`` is set the filter bumps only on high
    # *positive* correlation (corr > max_correlation); the legacy default keys on
    # |corr|, which wrongly bumps the negatively-correlated (hedging) candidate.
    long_only: bool = False


class CorrelationTracker:
    """Rolling Pearson correlation matrix across the asset universe.

    Usage::

        tracker = CorrelationTracker()
        tracker.update({"EUR/USD": [...returns...], "GBP/USD": [...returns...]})
        selected, bumped = tracker.select_uncorrelated(["EUR/USD", "GBP/USD", "XAU/USD"], k=2)
    """

    def __init__(self, config: CorrelationConfig | None = None) -> None:
        self._config = config or CorrelationConfig()
        # _matrix[a][b] = Pearson correlation between a and b
        self._matrix: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Matrix computation
    # ------------------------------------------------------------------

    def update(self, returns_map: dict[str, list[float]]) -> None:
        """Recompute the Pearson correlation matrix from return series.

        Parameters
        ----------
        returns_map:
            ``{symbol: [float]}`` — percentage-change (or log-return) series per
            asset, typically ``df["close"].pct_change().dropna().tolist()``.
            Each series is trimmed to the last ``lookback_bars`` observations before
            computing the matrix.  Symbols with fewer than 2 observations are ignored.
        """
        lookback = self._config.lookback_bars
        trimmed: dict[str, list[float]] = {}
        for sym, rets in returns_map.items():
            tail = rets[-lookback:]
            if len(tail) >= 2:
                trimmed[sym] = tail

        symbols = list(trimmed)
        if len(symbols) < 2:
            logger.debug("CorrelationTracker: fewer than 2 symbols — matrix not updated")
            return

        df = pd.DataFrame(trimmed)
        corr = df.corr(method="pearson")

        # pandas-stubs declares `DataFrame.loc[a, b]` as a wide union of
        # pandas scalars (incl. date/timedelta/bytes).  At runtime a Pearson
        # correlation matrix only ever holds numeric cells, so the `float`
        # cast is always valid — silence mypy's arg-type warning here.
        self._matrix = {
            s: {t: float(corr.loc[s, t]) for t in symbols if t != s}  # type: ignore[arg-type]
            for s in symbols
        }
        logger.debug("CorrelationTracker: updated %d×%d matrix", len(symbols), len(symbols))

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def correlation(self, sym_a: str, sym_b: str) -> float | None:
        """Return the Pearson correlation between two symbols, or None if unknown."""
        return self._matrix.get(sym_a, {}).get(sym_b)

    # ------------------------------------------------------------------
    # Correlation-filtered selection
    # ------------------------------------------------------------------

    def select_uncorrelated(
        self,
        ranked_candidates: list[str],
        k: int,
    ) -> tuple[list[str], list[tuple[str, str, float]]]:
        """Apply the correlation filter to a pre-ranked candidate list.

        Iterates ``ranked_candidates`` in order (best first).  A candidate is
        accepted only if |correlation| with every already-accepted symbol is
        ≤ ``max_correlation``.  Stops as soon as ``k`` symbols are accepted or
        the list is exhausted.

        When the internal matrix is empty (no ``update`` call yet) or
        ``enabled=False``, returns ``ranked_candidates[:k]`` with no bumps.

        Parameters
        ----------
        ranked_candidates:
            Symbols ordered from best to worst signal.
        k:
            Maximum number of symbols to select.

        Returns
        -------
        ``(selected, bumped)`` where ``bumped`` is a list of
        ``(bumped_symbol, blocking_symbol, corr_value)`` triples.
        """
        cfg = self._config
        if not cfg.enabled or not self._matrix:
            return ranked_candidates[:k], []

        selected: list[str] = []
        bumped: list[tuple[str, str, float]] = []

        for symbol in ranked_candidates:
            blocker_sym = ""
            blocker_corr = 0.0
            for already in selected:
                corr = self._matrix.get(symbol, {}).get(already)
                if corr is None:
                    continue
                # Long-only book bumps on positive corr only; legacy keys on |corr|.
                effective = corr if cfg.long_only else abs(corr)
                if effective > cfg.max_correlation:
                    blocker_sym = already
                    blocker_corr = corr
                    break

            if blocker_sym:
                bumped.append((symbol, blocker_sym, blocker_corr))
                logger.info(
                    "CorrelationTracker: %s bumped (|corr|=%.2f with %s)",
                    symbol,
                    abs(blocker_corr),
                    blocker_sym,
                )
            else:
                selected.append(symbol)

            if len(selected) >= k:
                break

        return selected, bumped

    # ------------------------------------------------------------------
    # Persistence helpers (called by candle_db integration)
    # ------------------------------------------------------------------

    def snapshot(self) -> dict[str, dict[str, float]]:
        """Return a serialisable copy of the current matrix."""
        return {s: dict(row) for s, row in self._matrix.items()}

    def restore(self, data: dict[str, dict[str, float]]) -> None:
        """Restore from a previously snapshotted matrix (e.g. from SQLite)."""
        self._matrix = {s: dict(row) for s, row in data.items()}
        logger.info("CorrelationTracker: restored %d-asset matrix", len(self._matrix))
