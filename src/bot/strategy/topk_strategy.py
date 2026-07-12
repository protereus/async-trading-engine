"""Top-K multi-asset probabilistic strategy — Kronos predict_batch sole signal source.

Architecture
------------
Every ``rerank_interval_minutes``:

1. Fetch 1h candles for every symbol in the watchlist from the in-memory store
   (Twelve Data delivers pre-built 1h bars; ``aggregate_to_minutes=1`` is the
   default — set higher only if a sub-hour feed is wired in).
2. Slice the last ``kronos_context_bars`` bars (≈400 → ~17 days of 1h data).
3. Dual-pass MC inference:
   - Pass 1: one ``predict_batch`` at ``forecast_temperature=0.6, sample_count=10``
     for a sharp averaged OHLC path (point estimate).
   - Pass 2: ``variance_sample_count`` calls at ``variance_temperature=1.0,
     sample_count=1`` for the empirical return spread (variance estimate).
   Total per group: ``1 + variance_sample_count`` calls. Volume-bearing assets and
   no-volume forex are batched as separate groups because predict_batch
   requires identical feature shape.
4. Per asset: compute mean_return, std_return, directional_confidence, uncertainty,
   and a KronosPathSignal (predicted MFE/MAE/peak-bar/monotonicity).
5. Filter: direction==LONG AND confidence>=min_confidence AND
           uncertainty<=max_uncertainty AND mean_return>=min_predicted_return
           AND predicted_volatility<=predicted_vol_max.
6. Rank tradeable signals by ranking_score (mean_return × |monotonicity| ×
   MFE-confirmation) when path signal present, else mean_return; select top-K.
7. Correlation filter bumps highly-correlated candidates.
8. stop_pct = max(std_return × vol_stop_multiplier, predicted_mae_pct × 1.10,
                 min_stop_pct).

predict_batch constraint: all series in a single batch call must have identical
context length. Each (volume / no-volume) group is aligned to its own min_len
before the call. Assets failing the minimum-bar threshold are excluded entirely.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from math import nan
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable  # noqa: F401

    from bot.config import BotConfig
    from bot.core.models import Candle  # noqa: F401

import numpy as np
import pandas as pd

from bot.core.models import PersistedAssetSignal
from bot.data.eodhd_symbols import EODHD_UNIVERSE
from bot.data.eodhd_symbols import VOLUME_SYMBOLS as _EODHD_VOLUME_SYMBOLS
from bot.strategy import _kronos_progress
from bot.strategy.correlation import CorrelationConfig, CorrelationTracker
from bot.strategy.kronos_signals import KronosPathSignal, extract_path_signal

logger = logging.getLogger(__name__)

# Symbols whose feed carries volume → batched separately so Kronos receives a
# consistent feature shape (volume-aware tokeniser path). The EODHD universe's
# volume-bearing symbols: the 14 US shares plus the IG-native metals XAU/XAG
# (LS LTV carries volume); FX carries none.
_VOLUME_SYMBOLS: frozenset[str] = _EODHD_VOLUME_SYMBOLS

# Candidate ranking horizons (1h bars) evaluated by the offline calibration
# tooling.  The Pass-2 variance loop records each draw's
# predicted close at every one of these bars so per-H direction_confidence is
# computable offline from ``signal_history.var_closes_at_horizons`` — the
# 10b caveat requires confidence recalibration *at the chosen H*, and Pass-2
# endpoints used to exist only at the terminal bar.  Shared with
# ``scripts/signal_diagnostics.py`` (imported there; do not duplicate).
CANDIDATE_HORIZONS: tuple[int, ...] = (6, 12, 24, 48, 72, 120)

# (point_preds, var_closes, var_closes_at_horizons) for one homogeneous
# inference group — see TopKStrategy._run_inference_group.
_GroupResult = tuple[
    dict[str, pd.DataFrame],
    dict[str, list[float]],
    dict[str, list[list[float]]],
]

# One batched, shape-homogeneous inference group:
# (symbols, model frames, x-timestamps, y-timestamps, pred_len).
_GroupJob = tuple[list[str], list[pd.DataFrame], list[pd.Series], list[pd.Series], int]


def ranking_class_for(symbol: str) -> str | None:
    """Asset class used for per-class ranking-horizon resolution.

    Sourced from ``EODHD_UNIVERSE`` (the live universe's single source of
    truth); its ``asset_class="equity"`` is relabelled ``us_equity`` to match
    the diagnostics report.  Symbols outside the universe (legacy rollback
    feeds, tests) return None and fall back to the global horizon.
    """
    entry = EODHD_UNIVERSE.get(symbol)
    if entry is None:
        return None
    return "us_equity" if entry.asset_class == "equity" else entry.asset_class


# Class names accepted by TOPK_RANKING_HORIZON_BY_CLASS — exactly the classes
# ``ranking_class_for`` can resolve for the live universe.
RANKING_HORIZON_CLASSES: frozenset[str] = frozenset(
    {"us_equity" if s.asset_class == "equity" else s.asset_class for s in EODHD_UNIVERSE.values()}
)


def parse_ranking_horizon_by_class(raw: str, pred_len: int) -> dict[str, int]:
    """Parse ``TOPK_RANKING_HORIZON_BY_CLASS`` ("forex:48,us_equity:24,metal:24").

    Raises ValueError on malformed pairs, unknown class names, or H outside
    [1, pred_len] — callers (``BotConfig.validate_config``) surface this at
    startup rather than silently mis-slicing.  Empty/blank input → ``{}``
    (per-class overrides disabled).
    """
    result: dict[str, int] = {}
    if not raw.strip():
        return result
    for pair in raw.split(","):
        cls, sep, h_str = pair.strip().partition(":")
        cls = cls.strip()
        if not sep or not cls or not h_str.strip():
            raise ValueError(
                f"TOPK_RANKING_HORIZON_BY_CLASS: malformed pair {pair!r} "
                "(expected 'class:bars', e.g. 'us_equity:24')"
            )
        if cls not in RANKING_HORIZON_CLASSES:
            raise ValueError(
                f"TOPK_RANKING_HORIZON_BY_CLASS: unknown class {cls!r} "
                f"(valid: {', '.join(sorted(RANKING_HORIZON_CLASSES))})"
            )
        try:
            horizon = int(h_str.strip())
        except ValueError as exc:
            raise ValueError(
                f"TOPK_RANKING_HORIZON_BY_CLASS: non-integer horizon {h_str.strip()!r} "
                f"for class {cls!r}"
            ) from exc
        if not 1 <= horizon <= pred_len:
            raise ValueError(
                f"TOPK_RANKING_HORIZON_BY_CLASS: horizon {horizon} for class {cls!r} "
                f"outside [1, pred_len={pred_len}]"
            )
        if cls in result:
            raise ValueError(f"TOPK_RANKING_HORIZON_BY_CLASS: duplicate class {cls!r}")
        result[cls] = horizon
    return result


def _load_kronos_offline_first(loader: Callable[..., Any], repo: str, label: str) -> Any:
    """Load a pinned Kronos checkpoint, preferring the local cache.

    Tries ``local_files_only=True`` first so a healthy, pre-cached checkpoint
    never touches the network — a transient HuggingFace 503 otherwise stalls
    every restart ~30s in retry backoff before falling back to the same local
    cache anyway.

    If the checkpoint is missing from the cache for any reason (fresh host, cache
    eviction, partial copy), fall back to a normal online load that downloads and
    repopulates it. The online attempt is allowed to raise — a genuinely
    unreachable HuggingFace with no cache is a hard startup failure either way.
    """
    try:
        return loader(repo, local_files_only=True)
    except Exception as exc:
        logger.warning(
            "Kronos %s '%s' not available from local cache (%s); "
            "downloading from HuggingFace to repopulate",
            label,
            repo,
            exc,
        )
        return loader(repo)


def _slice_index(ranking_horizon_bars: int, pred_len: int) -> int:
    """Index into a length-``pred_len`` predicted series for the ranking signal.

    ``ranking_horizon_bars == 0`` → terminal bar (legacy: -1).
    ``>0`` → clamped to the rollout length so a short rollout never IndexErrors.
    """
    if ranking_horizon_bars <= 0:
        return -1
    return min(ranking_horizon_bars, pred_len) - 1


def _bump_variance_progress(done: int, total: int, t0: float, grp_label: str) -> None:
    """Advance the dashboard batch counter through the silent variance pass.

    Pass-2 calls run verbose=False (no trange), so without this the
    dashboard's inner bar freezes for 40 of the ~42 predict_batch calls
    (~16–22 min).  ``done`` counts samples completed *before* the upcoming
    call; the ``s/call`` unit deliberately avoids the ``s/it`` shape
    ``webgui/diagnostics._TQDM_RE`` parses, so the Pass-1 bar is unaffected.
    """
    now = time.monotonic()
    rate = (now - t0) / done if done else 0.0
    _kronos_progress.bump_batch(
        {
            "current": done,
            "total": total,
            "elapsed_s": int(now - t0),
            "eta_s": int(rate * (total - done)),
            "rate_s_per_it": float(rate),
            "label": f"variance ({grp_label})",
            "unit": "s/call",
        }
    )


def _maybe_log_variance_progress(
    grp_label: str, done: int, total: int, t0: float, last_emit: float
) -> float:
    """Throttled journal line (~every 15 s, plus the final sample) so the
    otherwise-silent variance pass is visible without 20 log lines.  Returns
    the new ``last_emit`` timestamp."""
    now = time.monotonic()
    if now - last_emit < 15.0 and done != total:
        return last_emit
    elapsed = now - t0
    logger.info(
        "Kronos Pass 2 (variance): group=%s %d/%d [%.0fs elapsed, %.1fs/call]",
        grp_label,
        done,
        total,
        elapsed,
        elapsed / done,
    )
    return now


@dataclass
class TopKConfig:
    kronos_dir: str = ""  # Path prepended to sys.path so ``from model import`` works
    kronos_model: str = "NeoQuasar/Kronos-mini"
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-2k"
    kronos_context_bars: int = 400  # Aggregated bars fed to Kronos (2k tokeniser ≤ 2048)

    k: int = 2  # Max simultaneous open positions
    rerank_interval_minutes: int = 60  # Full scan frequency

    pred_len: int = 120  # Prediction horizon in aggregated bars
    aggregate_to_minutes: int = 1  # Resample raw candles before Kronos; 1 = no resampling

    # --- Inference hyperparameters (paper Table 6 — Price/Return/Investment tasks) ---
    # Pass 1 (point estimate): low temperature for sharp directional signal
    forecast_temperature: float = 0.6  # Paper Table 6 — was 1.0 (example script value)
    forecast_top_p: float = 0.90  # Nucleus sampling threshold (unchanged)
    forecast_sample_count: int = 10  # Paper Table 6 — single averaged pass

    # Pass 2 (variance estimate): higher temperature preserves MC spread for uncertainty
    variance_pass_enabled: bool = True  # Second inference pass for std/confidence
    variance_temperature: float = 1.0  # High T to maintain MC variance
    variance_sample_count: int = 20  # Loops for variance; total calls = 1 + 20 = 21

    # Run the two homogeneous groups concurrently to reclaim idle cores on
    # multi-core hosts; output is unchanged. Off by default.
    parallel_groups: bool = False

    # Entry filters — deliberately wide, uncalibrated defaults (see BotConfig).
    min_confidence: float = 0.60  # Min fraction of MC paths agreeing on direction
    max_uncertainty: float = 5.0  # Cap on std_return / |mean_return|
    min_predicted_return: float = 0.001  # 0.1% over the prediction horizon

    vol_stop_multiplier: float = 2.0  # stop_pct = std_return × this
    min_stop_pct: float = 0.005  # Floor stop (0.5% of current price)
    predicted_vol_max: float = 0.05  # Volatility regime filter

    # Correlation filter
    correlation_enabled: bool = True
    correlation_max: float = 0.75  # corr threshold; 1.0 disables filtering
    correlation_lookback_bars: int = 200  # ~8 days of 1h bars
    correlation_long_only: bool = False  # bump on positive corr only (long-only book)

    # horizon-matched ranking slice
    # 0 → use terminal bar (legacy; ranking pegged to pred_len).
    # >0 → slice both Pass 1 mean_close and Pass 2 var_closes at iloc[H-1]
    # so ranking matches the rerank cadence. Path metrics (MFE/MAE/peak/
    # monotonicity) still derive from the full predicted path — they are
    # correctly horizon-agnostic and inform TP/stop, not ranking.
    ranking_horizon_bars: int = 0

    # per-asset-class ranking horizon, e.g. {"us_equity": 24}.
    # At H=120 on 1h bars every US-equity row is gap_spanned by construction
    # (overnight gaps), so equities need a short H while FX may keep a longer
    # one. Resolution per symbol: class override → global ranking_horizon_bars
    # → 0 (terminal bar, legacy). Empty default = global applies to all —
    # byte-identical to pre-10c-prep behaviour. Keys validated by
    # parse_ranking_horizon_by_class (env TOPK_RANKING_HORIZON_BY_CLASS).
    ranking_horizon_by_class: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_bot_config(cls, cfg: BotConfig) -> TopKConfig:
        """Build from the app-wide ``BotConfig``.

        Owns the ``BotConfig.topk_*``/``kronos_*`` → ``TopKConfig`` field
        mapping so the wiring site (``Lifecycle.init_ig``) doesn't carry it —
        same pattern as ``SentimentConfig.from_bot_config``.
        """
        return cls(
            kronos_dir=cfg.kronos_dir,
            kronos_model=cfg.kronos_model,
            kronos_tokenizer=cfg.kronos_tokenizer,
            kronos_context_bars=cfg.kronos_context_bars,
            k=cfg.topk_k,
            rerank_interval_minutes=cfg.topk_rerank_interval_minutes,
            pred_len=cfg.topk_pred_len,
            forecast_temperature=cfg.topk_forecast_temperature,
            forecast_top_p=cfg.topk_forecast_top_p,
            forecast_sample_count=cfg.topk_forecast_sample_count,
            variance_pass_enabled=cfg.topk_variance_pass_enabled,
            variance_temperature=cfg.topk_variance_temperature,
            variance_sample_count=cfg.topk_variance_sample_count,
            parallel_groups=cfg.topk_parallel_groups,
            min_confidence=cfg.topk_min_confidence,
            max_uncertainty=cfg.topk_max_uncertainty,
            min_predicted_return=cfg.topk_min_predicted_return,
            aggregate_to_minutes=cfg.topk_aggregate_to_minutes,
            vol_stop_multiplier=cfg.topk_vol_stop_multiplier,
            min_stop_pct=cfg.topk_min_stop_pct,
            correlation_enabled=cfg.topk_correlation_enabled,
            correlation_max=cfg.topk_correlation_max,
            correlation_lookback_bars=cfg.topk_correlation_lookback_bars,
            correlation_long_only=cfg.topk_correlation_long_only,
            ranking_horizon_bars=cfg.topk_ranking_horizon_bars,
            ranking_horizon_by_class=parse_ranking_horizon_by_class(
                cfg.topk_ranking_horizon_by_class,
                cfg.topk_pred_len,
            ),
        )


@dataclass
class AssetSignal:
    symbol: str  # Candle/DB key (e.g. "BTC/USDT")
    mean_return: float  # Mean expected return across MC samples
    std_return: float  # Std of returns across MC samples
    direction_confidence: float  # Fraction of samples agreeing with mean direction
    uncertainty: float  # CV = std / (|mean| + ε) — higher is noisier
    stop_pct: float  # Recommended stop as fraction of current price
    tradeable: bool  # Passes all entry filters
    predicted_close: float  # Mean predicted closing price at horizon
    direction: Literal["LONG", "SHORT"] = "LONG"
    samples: list[float] = field(default_factory=list)  # All MC predicted-close values
    # Per Pass-2 draw, the predicted close at each CANDIDATE_HORIZONS bar
    # (NaN where the rollout was shorter than H).  Logging-only: persisted to
    # ``signal_history.var_closes_at_horizons`` so per-H confidence can be
    # recalibrated offline.  No live filter reads this.
    var_closes_at_horizons: list[list[float]] = field(default_factory=list)

    def to_persist(self) -> PersistedAssetSignal:
        """The nine scalar fields that survive a restart in
        ``BotState.topk_state``.  The per-draw diagnostic lists (``samples``,
        ``var_closes_at_horizons``) are rerank-scoped and intentionally
        dropped — a restored signal only needs to drive entry/exit gating
        until the next rerank replaces it."""
        return PersistedAssetSignal(
            symbol=self.symbol,
            mean_return=self.mean_return,
            std_return=self.std_return,
            direction_confidence=self.direction_confidence,
            uncertainty=self.uncertainty,
            stop_pct=self.stop_pct,
            tradeable=self.tradeable,
            predicted_close=self.predicted_close,
            direction=self.direction,
        )

    @classmethod
    def from_persist(cls, data: PersistedAssetSignal) -> AssetSignal:
        """Inverse of :meth:`to_persist` — rebuilds a signal saved by a
        previous process."""
        return cls(
            symbol=data["symbol"],
            mean_return=data["mean_return"],
            std_return=data["std_return"],
            direction_confidence=data["direction_confidence"],
            uncertainty=data["uncertainty"],
            stop_pct=data["stop_pct"],
            tradeable=data["tradeable"],
            predicted_close=data["predicted_close"],
            direction=data["direction"],
        )


class TopKStrategy:
    """Multi-asset scanner: ranks symbols by Kronos signal, selects top-K.

    Kronos predictor is lazy-loaded on the first ``scan()`` call.
    """

    def __init__(self, config: TopKConfig) -> None:
        self._config = config
        self._predictor: Any = None
        self._path_signals: dict[str, KronosPathSignal] = {}
        self._last_bumped: list[tuple[str, str, float]] = []
        # Subset of _last_bumped whose bumped symbol would otherwise have been in
        # the top-k by score — i.e. correlation actually changed the held
        # portfolio. The alert reports only these; trailing also-rans are noise.
        self._last_material_bumped: list[tuple[str, str, float]] = []
        # (dropped, blocker) set last surfaced in the rerank alert. The alert
        # dedups against this so a steady-state bump (e.g. GBP/USD ↔ EUR/USD
        # every rerank) is shown once and only re-surfaces when it changes.
        self._last_alerted_bump_key: frozenset[tuple[str, str]] = frozenset()
        self._correlation_tracker = CorrelationTracker(
            CorrelationConfig(
                enabled=config.correlation_enabled,
                max_correlation=config.correlation_max,
                lookback_bars=config.correlation_lookback_bars,
                long_only=config.correlation_long_only,
            )
        )

    # ------------------------------------------------------------------
    # Ranking-horizon resolution
    # ------------------------------------------------------------------

    def _effective_horizon_bars(self, symbol: str) -> int:
        """Ranking horizon for *symbol*: class override → global → 0 (terminal)."""
        cfg = self._config
        if cfg.ranking_horizon_by_class:
            cls = ranking_class_for(symbol)
            if cls is not None and cls in cfg.ranking_horizon_by_class:
                return cfg.ranking_horizon_by_class[cls]
        return cfg.ranking_horizon_bars

    def signal_horizon_bars(self, symbol: str) -> int:
        """Horizon (bars) at which *symbol*'s ranking metrics were sliced.

        Written to ``signal_history.horizon_bars`` so the resolver realises
        returns over the same window the prediction was scored at —
        realised-at-H must match predicted-at-H or calibration numbers are
        garbage.  The legacy 0 (terminal bar) maps to ``pred_len``.
        """
        horizon = self._effective_horizon_bars(symbol)
        return horizon if horizon > 0 else self._config.pred_len

    # -- Accessors for collaborators (keep _correlation_tracker / _path_signals
    #    private to the strategy rather than reached into across seams) --------

    def path_signal_for(self, symbol: str) -> KronosPathSignal | None:
        """Latest extracted path signal for *symbol* (``None`` if none)."""
        return self._path_signals.get(symbol)

    def snapshot_correlation(self) -> dict[str, dict[str, float]]:
        """Serialisable snapshot of the rolling correlation matrix."""
        return self._correlation_tracker.snapshot()

    def restore_correlation(self, data: dict[str, dict[str, float]]) -> None:
        """Restore the rolling correlation matrix from a persisted snapshot."""
        self._correlation_tracker.restore(data)

    # ------------------------------------------------------------------
    # Kronos loading (once, then cached)
    # ------------------------------------------------------------------

    def _load_predictor(self) -> None:
        """Import and instantiate KronosPredictor (blocking — call via to_thread)."""
        if self._predictor is not None:
            return
        cfg = self._config

        # Patch ``tqdm.trange`` before Kronos's ``model.kronos`` resolves it at
        # import time. Without this, the autoregressive progress bar reaches
        # journald as ``\r``-separated control bytes and corrupts neighbouring
        # log entries (see ``_kronos_progress`` docstring).
        from bot.strategy._kronos_progress import install as _install_kronos_progress

        _install_kronos_progress()

        try:
            from model import Kronos, KronosPredictor, KronosTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Cannot import 'model' (Kronos). Set KRONOS_DIR to a "
                "Kronos checkout containing the model/ package. "
                f"Original error: {exc}"
            ) from exc

        model = _load_kronos_offline_first(Kronos.from_pretrained, cfg.kronos_model, "model")
        tokenizer = _load_kronos_offline_first(
            KronosTokenizer.from_pretrained, cfg.kronos_tokenizer, "tokenizer"
        )
        # fast_decode: exact last-position decode (skips proj_s1/proj_s2 over the
        # discarded non-final window positions). Bit-identical output, ~11.6% faster
        # scan-wall. Validated bit-exact against the stock decode path.
        self._predictor = KronosPredictor(
            model,
            tokenizer,
            max_context=cfg.kronos_context_bars,
            fast_decode=True,
            compile_model=True,
        )
        logger.info(
            "Kronos predictor loaded: model=%s tokenizer=%s",
            cfg.kronos_model,
            cfg.kronos_tokenizer,
        )

    # ------------------------------------------------------------------
    # Data preparation
    # ------------------------------------------------------------------

    def _prepare_asset(
        self,
        candles: list[Any],
        symbol: str,
        has_volume: bool = False,
    ) -> tuple[pd.DataFrame, float] | None:
        """Aggregate and slice candles for one asset.

        Returns ``(df, current_price)`` with df indexed by UTC DatetimeIndex,
        or ``None`` if there are insufficient bars.

        The returned df has exactly ``min(len_after_agg, kronos_context_bars)``
        rows — the caller aligns all assets to the global minimum before batching.
        When ``has_volume=True``, the DataFrame includes ``volume`` and ``amount``
        columns so Kronos can use its volume-aware tokeniser path.
        """
        cfg = self._config
        min_raw = cfg.kronos_context_bars * cfg.aggregate_to_minutes
        if len(candles) < min_raw:
            logger.warning(
                "TopK: %s has only %d raw candles (need %d for %d×%dm bars) — skipping",
                symbol,
                len(candles),
                min_raw,
                cfg.kronos_context_bars,
                cfg.aggregate_to_minutes,
            )
            return None

        raw: dict[str, list[float]] = {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
        }
        if has_volume:
            raw["volume"] = [c.volume for c in candles]
            raw["amount"] = [c.volume * c.close for c in candles]

        df_raw = pd.DataFrame(
            raw,
            index=pd.to_datetime([c.timestamp for c in candles], unit="ms", utc=True),
        )

        if cfg.aggregate_to_minutes > 1:
            rule = f"{cfg.aggregate_to_minutes}min"
            agg: dict[str, str] = {"open": "first", "high": "max", "low": "min", "close": "last"}
            if has_volume:
                agg["volume"] = "sum"
                agg["amount"] = "sum"
            # `dict[str, str]` (column → reduction-fn name) is the canonical
            # pandas agg signature, but pandas-stubs narrows it to
            # `Mapping[Hashable, Callable | str]`.  Same runtime behaviour.
            df_raw = df_raw.resample(rule).agg(agg).dropna()  # type: ignore[arg-type]

        df = df_raw.iloc[-cfg.kronos_context_bars :]
        if len(df) < 2:
            return None

        return df, float(df["close"].iloc[-1])

    def _build_timestamps(self, df: pd.DataFrame, pred_len: int) -> tuple[pd.Series, pd.Series]:
        """Build x_timestamp and y_timestamp Series for predict_batch.

        Uses the modal bar interval across all consecutive diffs to avoid
        inflated gaps at weekends / market closures corrupting y_timestamp.
        """
        x_timestamp = pd.Series(df.index, name="timestamp")
        diffs = pd.Series(df.index[1:]) - pd.Series(df.index[:-1])
        interval = diffs.mode()[0] if len(diffs) > 0 else (df.index[-1] - df.index[-2])
        y_timestamp = pd.Series(
            pd.date_range(
                start=df.index[-1] + interval,
                periods=pred_len,
                freq=interval,
                tz="UTC",
            ),
            name="timestamp",
        )
        return x_timestamp, y_timestamp

    # ------------------------------------------------------------------
    # Batch MC inference (synchronous — wrap in asyncio.to_thread)
    # ------------------------------------------------------------------

    def _run_inference_group(
        self,
        symbols: list[str],
        dfs: list[pd.DataFrame],
        x_timestamps: list[pd.Series],
        y_timestamps: list[pd.Series],
        pred_len: int,
    ) -> _GroupResult | None:
        """Two-pass predict_batch for one homogeneous group (volume or no-volume).

        Pass 1 (point estimate): one call at forecast_temperature for a sharp OHLC path.
        Pass 2 (variance, optional): variance_sample_count calls at variance_temperature.

        Returns ``(point_preds, var_closes, var_closes_at_horizons)`` or
        ``None`` if Pass 1 raises.
        """
        # Group label for log lines (Pass 1 already draws a tqdm bar; the
        # variance pass below is otherwise silent — see the timing note there).
        grp_label = "volume" if (symbols and symbols[0] in _VOLUME_SYMBOLS) else "no-volume"

        point_preds = self._run_pass1_point_estimate(
            symbols, dfs, x_timestamps, y_timestamps, pred_len, grp_label
        )
        if point_preds is None:
            return None
        var_closes, var_closes_h = self._run_pass2_variance(
            symbols, dfs, x_timestamps, y_timestamps, pred_len, grp_label
        )
        return point_preds, var_closes, var_closes_h

    def _run_pass1_point_estimate(
        self,
        symbols: list[str],
        dfs: list[pd.DataFrame],
        x_timestamps: list[pd.Series],
        y_timestamps: list[pd.Series],
        pred_len: int,
        grp_label: str,
    ) -> dict[str, pd.DataFrame] | None:
        """Pass 1: one predict_batch at forecast_temperature for a sharp OHLC path.

        Returns the per-symbol point-estimate frames, or ``None`` if the call
        raises (a Pass-1 failure is fatal for the whole rerank).
        """
        cfg = self._config
        point_preds: dict[str, pd.DataFrame] = {}
        _p1_t0 = time.monotonic()
        logger.info(
            "Kronos Pass 1 (point estimate): group=%s syms=%d sample_count=%d pred_len=%d",
            grp_label,
            len(symbols),
            cfg.forecast_sample_count,
            pred_len,
        )
        try:
            pred_list: list[pd.DataFrame] = self._predictor.predict_batch(
                df_list=dfs,
                x_timestamp_list=x_timestamps,
                y_timestamp_list=y_timestamps,
                pred_len=pred_len,
                T=cfg.forecast_temperature,
                top_p=cfg.forecast_top_p,
                sample_count=cfg.forecast_sample_count,
                verbose=True,
            )
            for i, sym in enumerate(symbols):
                point_preds[sym] = pred_list[i]
        except Exception:
            logger.exception("predict_batch Pass 1 failed — aborting rerank")
            return None
        logger.info(
            "Kronos Pass 1 done: group=%s syms=%d in %.1fs",
            grp_label,
            len(symbols),
            time.monotonic() - _p1_t0,
        )
        return point_preds

    def _run_pass2_variance(
        self,
        symbols: list[str],
        dfs: list[pd.DataFrame],
        x_timestamps: list[pd.Series],
        y_timestamps: list[pd.Series],
        pred_len: int,
        grp_label: str,
    ) -> tuple[dict[str, list[float]], dict[str, list[list[float]]]]:
        """Pass 2 (optional): variance_sample_count single draws at variance_temperature.

        Slices the same bar index as Pass 1 — per symbol (class override →
        global ranking_horizon_bars → 0 = terminal), so std_return /
        direction_confidence / uncertainty are consistent with mean_return at
        that symbol's effective H.  Returns ``(var_closes, var_closes_h)``;
        both empty per symbol when the variance pass is disabled.

        These calls run verbose=False, so Kronos draws no tqdm bar and they are
        otherwise invisible in the journal — historically ~16–22 min of silent
        wall time per rerank (40 of the ~42 predict_batch calls).  We emit a
        start line, a throttled progress line (~every 15 s, plus the last), and
        a completion line so the variance pass is legible in the logs.  The
        format deliberately avoids the ``N/N [MM:SS<MM:SS, Xs/it]`` shape that
        ``webgui/diagnostics._TQDM_RE`` parses (``s/call`` not ``s/it``), so the
        dashboard's Pass-1 progress bar is unaffected.
        """
        cfg = self._config
        var_closes: dict[str, list[float]] = {s: [] for s in symbols}
        # per draw, the close at every CANDIDATE_HORIZONS bar
        # (logging only — live metrics below still come from var_closes at the
        # configured ranking slice).
        var_closes_h: dict[str, list[list[float]]] = {s: [] for s in symbols}
        if not cfg.variance_pass_enabled:
            return var_closes, var_closes_h

        slice_idx_by_sym = {
            s: _slice_index(self._effective_horizon_bars(s), pred_len) for s in symbols
        }
        n_var = cfg.variance_sample_count
        var_t0 = time.monotonic()
        var_last_emit = var_t0
        logger.info(
            "Kronos Pass 2 (variance): group=%s syms=%d samples=%d (silent inference)",
            grp_label,
            len(symbols),
            n_var,
        )
        for sample_idx in range(n_var):
            _bump_variance_progress(sample_idx, n_var, var_t0, grp_label)
            try:
                var_list: list[pd.DataFrame] = self._predictor.predict_batch(
                    df_list=dfs,
                    x_timestamp_list=x_timestamps,
                    y_timestamp_list=y_timestamps,
                    pred_len=pred_len,
                    T=cfg.variance_temperature,
                    top_p=cfg.forecast_top_p,
                    sample_count=1,
                    verbose=False,
                )
                for i, sym in enumerate(symbols):
                    draw_closes = var_list[i]["close"]
                    var_closes[sym].append(float(draw_closes.iloc[slice_idx_by_sym[sym]]))
                    # NaN where the (clamped) rollout is shorter than H so
                    # offline readers can't mistake a terminal-bar value
                    # for a genuine bar-H close.
                    var_closes_h[sym].append(
                        [
                            float(draw_closes.iloc[h - 1]) if h <= len(draw_closes) else nan
                            for h in CANDIDATE_HORIZONS
                        ]
                    )
            except Exception:
                logger.exception("predict_batch Pass 2 sample %d failed — skipping", sample_idx)
            var_last_emit = _maybe_log_variance_progress(
                grp_label, sample_idx + 1, n_var, var_t0, var_last_emit
            )
        logger.info(
            "Kronos Pass 2 done: group=%s syms=%d samples=%d in %.1fs",
            grp_label,
            len(symbols),
            n_var,
            time.monotonic() - var_t0,
        )
        return var_closes, var_closes_h

    def _run_groups_parallel(
        self,
        jobs: list[_GroupJob],
    ) -> list[_GroupResult | None]:
        """Run ``_run_inference_group`` for every group concurrently.

        Thread-safe: ``auto_regressive_inference`` runs under ``torch.no_grad()``
        with all mutable buffers local to the call and the model weights read
        only, so multiple threads can drive one predictor at once.  We split the
        box's cores evenly across the groups (``set_num_threads`` is process-
        global, so this is symmetric — the benchmark found no gain from
        oversubscription) and restore the original thread count afterwards.

        Trade-off: the dashboard's overall batch counter and the Pass-1/Pass-2
        log lines interleave across the two groups while this runs; the lines
        stay readable because each is tagged with its group.
        """
        import os
        from concurrent.futures import ThreadPoolExecutor

        import torch

        n = len(jobs)
        per_group = max(1, (os.cpu_count() or n) // n)
        prev_threads = torch.get_num_threads()
        torch.set_num_threads(per_group)
        logger.info(
            "TopK parallel groups: %d groups × %d torch threads (was %d)",
            n,
            per_group,
            prev_threads,
        )
        try:
            with ThreadPoolExecutor(max_workers=n) as pool:
                futures = [pool.submit(self._run_inference_group, *job) for job in jobs]
                return [f.result() for f in futures]
        finally:
            torch.set_num_threads(prev_threads)

    def _prepare_assets(
        self, candles_map: dict[str, list[Candle]]
    ) -> dict[str, tuple[pd.DataFrame, float]]:
        """Build each asset's model frame + current price and refresh correlation.

        Returns ``{symbol: (df, current_price)}`` for assets with sufficient
        data (empty when none qualify).  Also updates the rolling correlation
        matrix from the prepared assets' close-return series — but
        only when at least one asset qualifies, matching the prior inline path.
        """
        cfg = self._config
        prepared: dict[str, tuple[pd.DataFrame, float]] = {}
        for symbol, candles in candles_map.items():
            result = self._prepare_asset(candles, symbol, has_volume=(symbol in _VOLUME_SYMBOLS))
            if result is not None:
                prepared[symbol] = result

        if not prepared:
            return prepared

        # Update rolling correlation matrix from close-price returns
        returns_map: dict[str, list[float]] = {
            s: df["close"].pct_change().dropna().tail(cfg.correlation_lookback_bars).tolist()
            for s, (df, _) in prepared.items()
        }
        self._correlation_tracker.update(returns_map)
        return prepared

    def _build_group_jobs(self, prepared: dict[str, tuple[pd.DataFrame, float]]) -> list[_GroupJob]:
        """Split prepared assets into the no-volume and volume groups and build
        each group's batched, length-aligned inference inputs.

        The two groups are independent (different feature shape), so they can
        run serially or concurrently — the computation, temperatures and sample
        counts are identical either way, so the signals are statistically
        unchanged.  Empty groups are skipped; no-volume precedes volume.
        """
        cfg = self._config
        vol_syms = [s for s in prepared if s in _VOLUME_SYMBOLS]
        novol_syms = [s for s in prepared if s not in _VOLUME_SYMBOLS]

        group_jobs: list[_GroupJob] = []
        for group_syms in (novol_syms, vol_syms):
            if not group_syms:
                continue
            grp_dfs = [prepared[s][0] for s in group_syms]
            min_len = min(len(df) for df in grp_dfs)
            grp_dfs = [df.iloc[-min_len:] for df in grp_dfs]
            grp_pred_len = min(cfg.pred_len, min_len)
            x_ts = [self._build_timestamps(df, grp_pred_len)[0] for df in grp_dfs]
            y_ts = [self._build_timestamps(df, grp_pred_len)[1] for df in grp_dfs]
            group_jobs.append((group_syms, grp_dfs, x_ts, y_ts, grp_pred_len))
        return group_jobs

    def _compute_asset_signal(
        self,
        symbol: str,
        point_df: pd.DataFrame,
        current_price: float,
        var_closes: list[float],
        var_closes_at_horizons: list[list[float]],
    ) -> tuple[AssetSignal, KronosPathSignal | None]:
        """Turn one asset's two-pass predictions into an ``AssetSignal`` (+ path signal).

        Slices the point estimate at the symbol's effective ranking horizon for
        ``mean_return``, derives ``std_return`` / ``direction_confidence`` /
        ``uncertainty`` from the variance draws, applies the path-signal stop
        widening, and the vol-regime + LONG-only tradeability gate.  Returns the
        signal and the extracted ``KronosPathSignal`` (``None`` when extraction
        failed); the caller registers the path signal.
        """
        cfg = self._config
        # /10c-prep: slice predicted close at the symbol's
        # effective ranking horizon (class override → global → 0 =
        # terminal, identical to pre-10b). Path metrics below still use
        # the full predicted path via extract_path_signal.
        effective_horizon = self._effective_horizon_bars(symbol)
        slice_idx = _slice_index(effective_horizon, len(point_df))
        mean_close = float(point_df["close"].iloc[slice_idx])
        mean_return = (mean_close - current_price) / current_price

        if var_closes and cfg.variance_pass_enabled:
            var_returns = [(c - current_price) / current_price for c in var_closes]
            std_return = float(np.std(var_returns))
            if mean_return >= 0:
                direction_confidence = float(np.mean([r >= 0 for r in var_returns]))
            else:
                direction_confidence = float(np.mean([r < 0 for r in var_returns]))
        else:
            std_return = 0.0
            direction_confidence = 1.0 if mean_return >= 0 else 0.0
            var_closes = []

        direction: Literal["LONG", "SHORT"] = "LONG" if mean_return >= 0 else "SHORT"
        uncertainty = std_return / (abs(mean_return) + 1e-8)
        stop_pct = max(std_return * cfg.vol_stop_multiplier, cfg.min_stop_pct)

        # Attempt path signal extraction from Pass 1 OHLC output
        path_sig = extract_path_signal(
            symbol=symbol,
            pred_df=point_df,
            entry_price=current_price,
            std_return=std_return,
            direction_confidence=direction_confidence,
            vol_stop_multiplier=cfg.vol_stop_multiplier,
            min_stop_pct=cfg.min_stop_pct,
            predicted_vol_max=cfg.predicted_vol_max,
            ranking_horizon_bars=effective_horizon,
        )
        if path_sig is not None:
            # Use path-derived stop: widen if model's own predicted MAE exceeds variance stop
            path_stop = max(stop_pct, path_sig.predicted_mae_pct * 1.10)
            stop_pct = max(path_stop, cfg.min_stop_pct)
            path_sig.stop_pct = stop_pct

        # Vol regime filter: path signal must not predict extreme volatility
        vol_ok = path_sig is None or path_sig.predicted_volatility <= cfg.predicted_vol_max

        # Only LONG signals are traded (IG spread bet; no short selling in this bot)
        tradeable = (
            direction == "LONG"
            and mean_return >= cfg.min_predicted_return
            and direction_confidence >= cfg.min_confidence
            and uncertainty <= cfg.max_uncertainty
            and vol_ok
        )

        logger.info(
            "TopK | %s | dir=%s | return=%.4f | conf=%.2f | uncert=%.2f "
            "| stop=%.3f%% | tradeable=%s | var_samples=%d | path=%s | vol=%s",
            symbol,
            direction,
            mean_return,
            direction_confidence,
            uncertainty,
            stop_pct * 100,
            tradeable,
            len(var_closes),
            "ok" if path_sig is not None else "missing",
            "yes" if symbol in _VOLUME_SYMBOLS else "no",
        )

        signal = AssetSignal(
            symbol=symbol,
            mean_return=mean_return,
            std_return=std_return,
            direction_confidence=direction_confidence,
            uncertainty=uncertainty,
            stop_pct=stop_pct,
            tradeable=tradeable,
            predicted_close=mean_close,
            direction=direction,
            samples=var_closes,
            var_closes_at_horizons=var_closes_at_horizons,
        )
        return signal, path_sig

    def _execute_inference_jobs(
        self, group_jobs: list[_GroupJob]
    ) -> (
        tuple[dict[str, pd.DataFrame], dict[str, list[float]], dict[str, list[list[float]]]] | None
    ):
        """Run the group jobs (serial, or parallel under TOPK_PARALLEL_GROUPS)
        and merge each group's point-prediction / variance dicts.

        Returns ``None`` if any group's Pass 1 failed — fatal for the whole rerank.
        """
        cfg = self._config
        all_point_preds: dict[str, pd.DataFrame] = {}
        all_var_closes: dict[str, list[float]] = {}
        all_var_closes_h: dict[str, list[list[float]]] = {}

        # TOPK_PARALLEL_GROUPS — run the two groups concurrently to reclaim
        # otherwise-idle cores (see scripts/bench_parallel_groups.py to measure
        # the gain on your hardware).  Off by default; the serial path is the
        # long-standing behaviour.
        if cfg.parallel_groups and len(group_jobs) > 1:
            group_results = self._run_groups_parallel(group_jobs)
        else:
            group_results = [self._run_inference_group(*job) for job in group_jobs]

        for group_result in group_results:
            if group_result is None:
                return None  # Pass 1 failure is fatal for the whole rerank
            grp_preds, grp_var_closes, grp_var_closes_h = group_result
            all_point_preds.update(grp_preds)
            all_var_closes.update(grp_var_closes)
            all_var_closes_h.update(grp_var_closes_h)

        return all_point_preds, all_var_closes, all_var_closes_h

    def _compute_all_signals(
        self,
        group_jobs: list[_GroupJob],
        all_point_preds: dict[str, pd.DataFrame],
        current_prices: dict[str, float],
        all_var_closes: dict[str, list[float]],
        all_var_closes_h: dict[str, list[list[float]]],
    ) -> tuple[list[AssetSignal], dict[str, KronosPathSignal]]:
        """Compute each asset's signal metrics and collect its path signal.

        Returns ``(signals, path_signals)``; the caller owns assigning the
        path-signal map to ``self._path_signals`` so this stays side-effect free.
        """
        # The job list preserves the original no-volume-then-volume group order, so
        # flattening it reproduces the prior ``novol_syms + vol_syms`` ordering.
        all_symbols = [s for job in group_jobs for s in job[0]]
        signals: list[AssetSignal] = []
        path_signals: dict[str, KronosPathSignal] = {}
        for symbol in all_symbols:
            signal, path_sig = self._compute_asset_signal(
                symbol,
                all_point_preds[symbol],
                current_prices[symbol],
                all_var_closes[symbol],
                all_var_closes_h.get(symbol, []),
            )
            if path_sig is not None:
                path_signals[symbol] = path_sig
            signals.append(signal)

        return signals, path_signals

    def _run_batch_inference(self, candles_map: dict[str, list[Candle]]) -> list[AssetSignal]:
        """Two-pass batch inference across all eligible assets.

        Assets are split into two groups before batching (predict_batch requires
        identical feature shape): the volume-bearing symbols (the 14 US shares +
        the IG-native metals XAU/XAG) get ``open, high, low, close, volume,
        amount`` columns; the no-volume forex pairs get ``open, high, low,
        close`` only.

        Pass 1 (point estimate): one predict_batch call with forecast_temperature=0.6.
        Pass 2 (variance estimate, optional): variance_sample_count loops at T=1.0.
        Total calls: 2 (Pass 1 vol+novol) + 2×variance_sample_count when both groups
        are non-empty; reduces to 1 + variance_sample_count for a single group.
        """
        self._load_predictor()
        cfg = self._config

        # Per-class overrides slice ranking at a shorter H, where std_return
        # is structurally smaller — the std_return × vol_stop_multiplier term
        # tightens. The mae_pct × 1.10 and min_stop_pct floors still bound
        # stops (MAE stays full-path), but vol_stop_multiplier may need
        # revisiting post-flip;
        if cfg.ranking_horizon_by_class:
            logger.warning(
                "TopK per-class ranking horizons active: %s — std_return at short H "
                "tightens the vol-stop term (floors still apply)",
                cfg.ranking_horizon_by_class,
            )

        # Prepare every asset (volume columns for ETF symbols) and refresh the
        # rolling correlation matrix from their close returns.
        prepared = self._prepare_assets(candles_map)
        if not prepared:
            logger.warning("TopK: no assets with sufficient data for batch inference")
            return []

        current_prices = {s: prepared[s][1] for s in prepared}

        # Split into volume-bearing / no-volume groups and build each group's
        # batched inputs (predict_batch requires identical feature shape).
        group_jobs = self._build_group_jobs(prepared)

        # Execute all predict_batch calls
        job_results = self._execute_inference_jobs(group_jobs)
        if job_results is None:
            return []  # Pass 1 failure is fatal for the whole rerank

        all_point_preds, all_var_closes, all_var_closes_h = job_results

        # Compute signal metrics per asset and extract path signals
        signals, self._path_signals = self._compute_all_signals(
            group_jobs,
            all_point_preds,
            current_prices,
            all_var_closes,
            all_var_closes_h,
        )
        return signals

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def expected_batches(self, symbols: list[str]) -> int:
        """Count Kronos ``predict_batch`` calls a ``scan(symbols)`` would issue.

        Groups *symbols* into volume-bearing vs no-volume (same split as
        ``_run_batch_inference``) and returns
        ``n_non_empty_groups * (1 + variance_sample_count)`` when the variance
        pass is enabled, else ``n_non_empty_groups``.  Used by the dashboard
        to draw an overall-rerank progress bar.
        """
        cfg = self._config
        has_vol = any(s in _VOLUME_SYMBOLS for s in symbols)
        has_novol = any(s not in _VOLUME_SYMBOLS for s in symbols)
        groups = int(has_vol) + int(has_novol)
        per_group = 1 + (cfg.variance_sample_count if cfg.variance_pass_enabled else 0)
        return groups * per_group

    async def scan(
        self,
        symbols: list[str],
        candle_fetcher: Callable[[str], Awaitable[list[Candle]]],
    ) -> list[AssetSignal]:
        """Fetch candles for all symbols and run batch MC inference.

        Parameters
        ----------
        symbols:
            Candle/DB keys to scan (e.g. ``["BTC/USDT", "ETH/USDT"]``).
        candle_fetcher:
            Async callable ``(symbol: str) -> list[Candle]``.

        Returns
        -------
        List of AssetSignal (unordered).  Call ``select_top_k`` to rank.
        """
        candles_map: dict[str, list[Candle]] = {}
        for symbol in symbols:
            try:
                candles = await candle_fetcher(symbol)
                candles_map[symbol] = candles
            except Exception:
                logger.exception("TopK candle fetch failed for %s", symbol)

        if not candles_map:
            return []

        return await asyncio.to_thread(self._run_batch_inference, candles_map)

    def select_top_k(
        self,
        signals: list[AssetSignal],
        k: int | None = None,
        is_open: Callable[[str], bool] | None = None,
    ) -> list[str]:
        """Return up to k symbols with the highest tradeable LONG returns.

        When path signals are available, rank by ranking_score
        (mean_return × |monotonicity| × MFE_confirmation) rather than raw
        mean_return, rewarding directional cleanliness and MFE confirmation.
        Falls back to mean_return when path signals are absent.

        ``is_open`` — optional market-open predicate.  When supplied, assets
        whose market is currently closed are dropped from the candidate pool so
        the k slots aren't consumed by names that can't be entered right now
        (e.g. US shares overnight / on NYSE holidays while FX stays open).  The
        exit path compensates by not penalising a *held* position for missing
        the top-K while its own market is shut — see ``topk_rerank_loop``.  Pass
        None (the default) for the unfiltered ranking used by back-tests.
        """
        if k is None:
            k = self._config.k
        tradeable = [s for s in signals if s.tradeable and (is_open is None or is_open(s.symbol))]

        def _score(sig: AssetSignal) -> float:
            path = self._path_signals.get(sig.symbol)
            return path.ranking_score if path is not None else sig.mean_return

        ranked = sorted(tradeable, key=_score, reverse=True)
        ranked_symbols = [s.symbol for s in ranked]

        # correlation filter — bumps highly-correlated candidates
        selected, bumped = self._correlation_tracker.select_uncorrelated(ranked_symbols, k)
        self._last_bumped = bumped
        # A bump is "material" only if the bumped symbol sits in the top-k by
        # score (ranked_symbols[:k]) — without the filter it would have been
        # held, so correlation genuinely changed the portfolio. Bumps of
        # lower-ranked also-rans never affected what we hold; the alert skips
        # them to stay signal, not noise.
        naive_top_k = set(ranked_symbols[:k])
        self._last_material_bumped = [b for b in bumped if b[0] in naive_top_k]
        for sym, blocker, corr in bumped:
            logger.info(
                "TopK: %s bumped (|corr|=%.2f with %s); next candidate selected instead",
                sym,
                abs(corr),
                blocker,
            )

        logger.info(
            "TopK selection: %d tradeable / %d scanned → top-%d: %s (bumped: %s)",
            len(tradeable),
            len(signals),
            k,
            selected,
            [b[0] for b in bumped] or "none",
        )
        return selected
