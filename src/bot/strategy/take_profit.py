"""Take-profit manager — five independent exit components for open positions.

Priority order (stop-loss, priority 1, is handled externally in main.py):
  2. Trailing stop — Stage 2 ratchet  (TRAILING_STOP_RATCHET)
  3. Trailing stop — Stage 1 breakeven (TRAILING_STOP_BREAKEVEN)
  4. Signal decay  — mean_return flip  (SIGNAL_DECAY_FLIP, via evaluate_signal)
  5. Static take-profit                (STATIC_TP)
  6. Time-based exit                   (TIME_LIMIT)
  7. Signal decay  — strikes exhausted (SIGNAL_DECAY_STRIKES, via evaluate_signal)
  8. Sentiment reversal                (SENTIMENT_REVERSAL, via evaluate_sentiment)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from bot.config import BotConfig

logger = logging.getLogger(__name__)


class ExitReason(StrEnum):
    HOLD = "hold"
    STATIC_TP = "static_take_profit"
    TRAILING_STOP_BREAKEVEN = "trailing_breakeven"
    TRAILING_STOP_RATCHET = "trailing_ratchet"
    SIGNAL_DECAY_FLIP = "signal_decay_mean_flip"
    SIGNAL_DECAY_STRIKES = "signal_decay_strikes_exhausted"
    TIME_LIMIT = "time_limit"
    SENTIMENT_REVERSAL = "sentiment_reversal"


@dataclass
class ExitDecision:
    should_exit: bool
    reason: ExitReason
    reasoning: str = ""
    target_price: float | None = None  # for logging only


@dataclass
class TakeProfitConfig:
    static_enabled: bool = True
    trailing_enabled: bool = True
    signal_decay_enabled: bool = True
    time_enabled: bool = True
    sentiment_reversal_enabled: bool = False

    min_rr_multiplier: float = 1.5
    kronos_target_fraction: float = 0.80
    # path-aware TP
    kronos_mfe_capture_fraction: float = 0.85  # fraction of predicted MFE to target
    kronos_mae_buffer: float = 1.10  # stop floor = predicted_mae_pct × this
    time_post_peak_grace_hours: float = 12.0  # hours past predicted_peak_bar before time exit

    breakeven_activation_mult: float = 1.0
    breakeven_buffer: float = 0.001
    trail_activation_mult: float = 2.0
    trail_multiplier: float = 0.5

    signal_decay_min_confidence: float = 0.55
    signal_decay_max_uncertainty: float = 3.0
    signal_decay_max_strikes: int = 2
    # Top-K eviction is tracked on its own counter so a position whose Kronos
    # signal (confidence + uncertainty) is still healthy is not force-closed
    # the moment another asset edges it out of the rerank top-K.  Counted in
    # reranks; defaults to 6 (~6 h at the live 60-min cadence).
    signal_decay_max_topk_misses: int = 6

    time_horizon_multiplier: float = 1.0

    sentiment_reversal_threshold: float = -0.3
    sentiment_reversal_min_confidence: float = 0.6

    @classmethod
    def from_bot_config(cls, cfg: BotConfig) -> TakeProfitConfig:
        """Build from the app-wide ``BotConfig``.

        Owns the ``BotConfig.tp_*`` → ``TakeProfitConfig`` field mapping so
        the wiring site (``Lifecycle.init_ig``) doesn't carry it — same
        pattern as ``SentimentConfig.from_bot_config``.
        """
        return cls(
            static_enabled=cfg.tp_static_enabled,
            trailing_enabled=cfg.tp_trailing_enabled,
            signal_decay_enabled=cfg.tp_signal_decay_enabled,
            time_enabled=cfg.tp_time_enabled,
            sentiment_reversal_enabled=cfg.tp_sentiment_reversal_enabled,
            min_rr_multiplier=cfg.tp_min_rr_multiplier,
            kronos_target_fraction=cfg.tp_kronos_target_fraction,
            breakeven_activation_mult=cfg.tp_breakeven_activation_mult,
            breakeven_buffer=cfg.tp_breakeven_buffer,
            trail_activation_mult=cfg.tp_trail_activation_mult,
            trail_multiplier=cfg.tp_trail_multiplier,
            signal_decay_min_confidence=cfg.tp_signal_decay_min_confidence,
            signal_decay_max_uncertainty=cfg.tp_signal_decay_max_uncertainty,
            signal_decay_max_strikes=cfg.tp_signal_decay_max_strikes,
            signal_decay_max_topk_misses=cfg.tp_signal_decay_max_topk_misses,
            time_horizon_multiplier=cfg.tp_time_horizon_multiplier,
            sentiment_reversal_threshold=cfg.tp_sentiment_reversal_threshold,
            sentiment_reversal_min_confidence=cfg.tp_sentiment_reversal_min_confidence,
        )


@dataclass
class PositionTPState:
    symbol: str
    entry_price: float
    entry_mean_return: float
    entry_stop_pct: float
    peak_price: float
    opened_at_ms: int
    signal_decay_strikes: int = 0
    topk_miss_strikes: int = 0
    breakeven_armed: bool = False
    trail_armed: bool = False
    current_trailing_stop: float | None = None  # absolute price
    # path-aware fields (None when path signal was unavailable at entry)
    predicted_mfe_pct: float | None = None
    predicted_mae_pct: float | None = None
    predicted_peak_bar: int | None = None
    bar_interval_ms: int = 3_600_000  # 1h default; used for peak_bar timing


class TakeProfitManager:
    """Per-position take-profit evaluation across five independent components."""

    def __init__(self, config: TakeProfitConfig, pred_len: int) -> None:
        self._config = config
        self._pred_len = pred_len
        self._positions: dict[str, PositionTPState] = {}

    def register_position(
        self,
        symbol: str,
        entry_price: float,
        entry_signal: object,
        opened_at_ms: int,
        path_signal: object | None = None,
        bar_interval_ms: int = 3_600_000,
    ) -> None:
        """Register a new open position for TP tracking.

        path_signal: optional KronosPathSignal for path-aware TP.
        bar_interval_ms: milliseconds per candle bar (default 1h = 3_600_000).
        """
        mean_return = float(getattr(entry_signal, "mean_return", 0.0))
        stop_pct = float(getattr(entry_signal, "stop_pct", 0.0))
        predicted_mfe_pct: float | None = None
        predicted_mae_pct: float | None = None
        predicted_peak_bar: int | None = None
        if path_signal is not None:
            _raw_mfe = getattr(path_signal, "predicted_mfe_pct", None)
            _raw_mae = getattr(path_signal, "predicted_mae_pct", None)
            _raw_peak = getattr(path_signal, "predicted_peak_bar", None)
            predicted_mfe_pct = float(_raw_mfe) if _raw_mfe is not None else None
            predicted_mae_pct = float(_raw_mae) if _raw_mae is not None else None
            predicted_peak_bar = int(_raw_peak) if _raw_peak is not None else None
        self._positions[symbol] = PositionTPState(
            symbol=symbol,
            entry_price=entry_price,
            entry_mean_return=mean_return,
            entry_stop_pct=stop_pct,
            peak_price=entry_price,
            opened_at_ms=opened_at_ms,
            predicted_mfe_pct=predicted_mfe_pct,
            predicted_mae_pct=predicted_mae_pct,
            predicted_peak_bar=predicted_peak_bar,
            bar_interval_ms=bar_interval_ms,
        )
        logger.info(
            "TP registered: %s entry=%.4f mean_return=%.4f stop_pct=%.4f"
            " mfe_pct=%s mae_pct=%s peak_bar=%s",
            symbol,
            entry_price,
            mean_return,
            stop_pct,
            f"{predicted_mfe_pct:.4f}" if predicted_mfe_pct is not None else "n/a",
            f"{predicted_mae_pct:.4f}" if predicted_mae_pct is not None else "n/a",
            str(predicted_peak_bar) if predicted_peak_bar is not None else "n/a",
        )

    def deregister_position(self, symbol: str) -> None:
        """Remove a position from TP tracking (call on any close)."""
        self._positions.pop(symbol, None)

    def get_trailing_stop(self, symbol: str) -> float | None:
        """Return the current trailing-stop level for *symbol* (None if unarmed).

        Same units as the ``current_price`` argument to ``register_position`` /
        ``evaluate_price`` — IG levels on the IG path.  Read by the risk
        manager's total-risk gate so live risk-on shrinks as the trail moves up.
        """
        state = self._positions.get(symbol)
        return state.current_trailing_stop if state is not None else None

    def _check_trailing_stop(
        self,
        symbol: str,
        state: PositionTPState,
        current_price: float,
        profit_pct: float,
        stop_pct: float,
    ) -> ExitDecision | None:
        """Priorities 2 & 3: arm/ratchet the trailing stop, exit if crossed.

        Mutates *state* (trail_armed / breakeven_armed / current_trailing_stop)
        even when no exit fires — arming persists for future evaluations.
        """
        if not (self._config.trailing_enabled and stop_pct > 0):
            return None

        # Stage 2: ratchet trail — takes priority over Stage 1
        if profit_pct >= stop_pct * self._config.trail_activation_mult:
            state.trail_armed = True
            state.breakeven_armed = True  # Stage 2 implies Stage 1 passed
            trail_pct = stop_pct * self._config.trail_multiplier
            new_trail = state.peak_price * (1.0 - trail_pct)
            if state.current_trailing_stop is None or new_trail > state.current_trailing_stop:
                state.current_trailing_stop = new_trail
                logger.debug(
                    "TP trail ratchet: %s peak=%.4f stop=%.4f",
                    symbol,
                    state.peak_price,
                    new_trail,
                )

        # Stage 1: breakeven — only if Stage 2 not yet armed
        elif (
            not state.breakeven_armed
            and profit_pct >= stop_pct * self._config.breakeven_activation_mult
        ):
            state.breakeven_armed = True
            breakeven_stop = state.entry_price * (1.0 + self._config.breakeven_buffer)
            if state.current_trailing_stop is None or breakeven_stop > state.current_trailing_stop:
                state.current_trailing_stop = breakeven_stop
            logger.debug("TP breakeven armed: %s stop=%.4f", symbol, state.current_trailing_stop)

        # Exit if price crossed any trailing stop level
        if state.current_trailing_stop is not None and current_price <= state.current_trailing_stop:
            reason = (
                ExitReason.TRAILING_STOP_RATCHET
                if state.trail_armed
                else ExitReason.TRAILING_STOP_BREAKEVEN
            )
            return ExitDecision(
                True,
                reason,
                f"price={current_price:.4f} <= trailing_stop={state.current_trailing_stop:.4f}",
                target_price=state.current_trailing_stop,
            )
        return None

    def _check_static_take_profit(
        self, state: PositionTPState, current_price: float, stop_pct: float
    ) -> ExitDecision | None:
        """Priority 5: fixed take-profit target derived from the path/mean-return signal."""
        if not (self._config.static_enabled and stop_pct > 0):
            return None

        if state.predicted_mfe_pct is not None:
            # Phase 9b: path-derived target; fallback is mean_return × fraction
            tp_pct = max(
                stop_pct * self._config.min_rr_multiplier,
                state.predicted_mfe_pct * self._config.kronos_mfe_capture_fraction,
            )
            tp_source = f"target derived from predicted_mfe_pct={state.predicted_mfe_pct:.4f}"
        else:
            tp_pct = max(
                stop_pct * self._config.min_rr_multiplier,
                state.entry_mean_return * self._config.kronos_target_fraction,
            )
            tp_source = (
                f"target from mean_return={state.entry_mean_return:.4f}"
                f" × {self._config.kronos_target_fraction}"
            )
        tp_price = state.entry_price * (1.0 + tp_pct)
        if current_price >= tp_price:
            return ExitDecision(
                True,
                ExitReason.STATIC_TP,
                f"price={current_price:.4f} >= tp_price={tp_price:.4f} ({tp_source})",
                target_price=tp_price,
            )
        return None

    def _check_time_exit(self, state: PositionTPState, now_ms: int) -> ExitDecision | None:
        """Priority 6: time-based exit — grace period past the predicted peak, or max age."""
        if not self._config.time_enabled:
            return None

        if state.predicted_peak_bar is not None:
            # Phase 9b: exit grace_hours after the predicted price peak
            peak_ms = state.opened_at_ms + state.predicted_peak_bar * state.bar_interval_ms
            grace_ms = int(self._config.time_post_peak_grace_hours * 3_600_000)
            deadline_ms = peak_ms + grace_ms
            if now_ms > deadline_ms:
                return ExitDecision(
                    True,
                    ExitReason.TIME_LIMIT,
                    (
                        f"past predicted peak bar {state.predicted_peak_bar} "
                        f"+ {self._config.time_post_peak_grace_hours:.0f}h grace"
                    ),
                )
        else:
            max_age_ms = int(self._pred_len * self._config.time_horizon_multiplier * 3_600_000)
            age_ms = now_ms - state.opened_at_ms
            if age_ms >= max_age_ms:
                return ExitDecision(
                    True,
                    ExitReason.TIME_LIMIT,
                    f"age={age_ms / 3_600_000:.1f}h >= max={max_age_ms / 3_600_000:.1f}h",
                )
        return None

    def evaluate_price(self, symbol: str, current_price: float, now_ms: int) -> ExitDecision:
        """Evaluate trailing stop, static TP, and time exit on each confirmed candle.

        Called on every 1h Twelve Data candle and every 1m Lightstreamer candle.
        Returns the highest-priority exit decision, or HOLD if none fire.
        """
        state = self._positions.get(symbol)
        if state is None:
            return ExitDecision(False, ExitReason.HOLD)

        # Update peak price
        if current_price > state.peak_price:
            state.peak_price = current_price

        profit_pct = (current_price - state.entry_price) / state.entry_price
        stop_pct = state.entry_stop_pct

        trailing_stop_exit = self._check_trailing_stop(
            symbol, state, current_price, profit_pct, stop_pct
        )
        if trailing_stop_exit is not None:
            return trailing_stop_exit

        static_tp_exit = self._check_static_take_profit(state, current_price, stop_pct)
        if static_tp_exit is not None:
            return static_tp_exit

        time_exit = self._check_time_exit(state, now_ms)
        if time_exit is not None:
            return time_exit

        return ExitDecision(False, ExitReason.HOLD)

    def evaluate_signal(
        self, symbol: str, latest_signal: object | None, in_topk: bool
    ) -> ExitDecision:
        """Evaluate signal decay on each TopK rerank.

        Checks (in priority order within this method):
          - mean_return sign flip → immediate SIGNAL_DECAY_FLIP
          - confidence / uncertainty / topk membership → strike counting → SIGNAL_DECAY_STRIKES
        """
        state = self._positions.get(symbol)
        if state is None or not self._config.signal_decay_enabled:
            return ExitDecision(False, ExitReason.HOLD)

        if latest_signal is None:
            state.signal_decay_strikes += 1
            if state.signal_decay_strikes >= self._config.signal_decay_max_strikes:
                return ExitDecision(
                    True,
                    ExitReason.SIGNAL_DECAY_STRIKES,
                    f"no signal; strikes={state.signal_decay_strikes}",
                )
            return ExitDecision(False, ExitReason.HOLD)

        mean_return = float(getattr(latest_signal, "mean_return", 0.0))
        confidence = float(getattr(latest_signal, "direction_confidence", 1.0))
        uncertainty = float(getattr(latest_signal, "uncertainty", 0.0))

        # Immediate exit: mean_return sign flipped relative to entry
        if mean_return * state.entry_mean_return < 0:
            return ExitDecision(
                True,
                ExitReason.SIGNAL_DECAY_FLIP,
                f"mean_return={mean_return:.4f} flipped from entry={state.entry_mean_return:.4f}",
            )

        # Signal-quality strikes (confidence / uncertainty).  These reflect the
        # Kronos signal itself deteriorating, so any unhealthy rerank increments
        # immediately and a healthy rerank resets to 0.
        signal_unhealthy = (
            confidence < self._config.signal_decay_min_confidence
            or uncertainty > self._config.signal_decay_max_uncertainty
        )
        if signal_unhealthy:
            state.signal_decay_strikes += 1
            logger.debug(
                "TP signal strike: %s conf=%.2f unc=%.2f strikes=%d/%d",
                symbol,
                confidence,
                uncertainty,
                state.signal_decay_strikes,
                self._config.signal_decay_max_strikes,
            )
        else:
            state.signal_decay_strikes = 0

        # Top-K eviction strikes (separate counter).  A single eviction does
        # not by itself imply the original thesis is wrong, so we tolerate
        # several consecutive misses before closing.
        if not in_topk:
            state.topk_miss_strikes += 1
            logger.debug(
                "TP topk miss: %s misses=%d/%d (conf=%.2f unc=%.2f healthy)",
                symbol,
                state.topk_miss_strikes,
                self._config.signal_decay_max_topk_misses,
                confidence,
                uncertainty,
            )
        else:
            state.topk_miss_strikes = 0

        if state.signal_decay_strikes >= self._config.signal_decay_max_strikes:
            return ExitDecision(
                True,
                ExitReason.SIGNAL_DECAY_STRIKES,
                (
                    f"strikes={state.signal_decay_strikes}/{self._config.signal_decay_max_strikes} "
                    f"conf={confidence:.2f} unc={uncertainty:.2f}"
                ),
            )

        if state.topk_miss_strikes >= self._config.signal_decay_max_topk_misses:
            return ExitDecision(
                True,
                ExitReason.SIGNAL_DECAY_STRIKES,
                (
                    f"topk_misses={state.topk_miss_strikes}/"
                    f"{self._config.signal_decay_max_topk_misses} "
                    f"conf={confidence:.2f} unc={uncertainty:.2f}"
                ),
            )

        return ExitDecision(False, ExitReason.HOLD)

    def evaluate_sentiment(self, symbol: str, sentiment: object) -> ExitDecision:
        """Evaluate sentiment reversal after each sentiment scan.

        Only active when sentiment_reversal_enabled=True. Closes on confident bearish
        consensus only — does not close on neutral or noisy signals.
        """
        state = self._positions.get(symbol)
        if state is None or not self._config.sentiment_reversal_enabled:
            return ExitDecision(False, ExitReason.HOLD)

        sent_value = float(getattr(sentiment, "sentiment", 0.0))
        confidence = float(getattr(sentiment, "confidence", 0.0))

        if (
            sent_value <= self._config.sentiment_reversal_threshold
            and confidence >= self._config.sentiment_reversal_min_confidence
        ):
            return ExitDecision(
                True,
                ExitReason.SENTIMENT_REVERSAL,
                (
                    f"sentiment={sent_value:.2f} "
                    f"<= threshold={self._config.sentiment_reversal_threshold} "
                    f"conf={confidence:.2f} "
                    f">= {self._config.sentiment_reversal_min_confidence}"
                ),
            )

        return ExitDecision(False, ExitReason.HOLD)

    def snapshot(self) -> dict[str, object]:
        """Serialise all position TP state for JSON persistence."""
        result: dict[str, object] = {}
        for symbol, state in self._positions.items():
            result[symbol] = {
                "symbol": state.symbol,
                "entry_price": state.entry_price,
                "entry_mean_return": state.entry_mean_return,
                "entry_stop_pct": state.entry_stop_pct,
                "peak_price": state.peak_price,
                "opened_at_ms": state.opened_at_ms,
                "signal_decay_strikes": state.signal_decay_strikes,
                "topk_miss_strikes": state.topk_miss_strikes,
                "breakeven_armed": state.breakeven_armed,
                "trail_armed": state.trail_armed,
                "current_trailing_stop": state.current_trailing_stop,
                # path fields
                "predicted_mfe_pct": state.predicted_mfe_pct,
                "predicted_mae_pct": state.predicted_mae_pct,
                "predicted_peak_bar": state.predicted_peak_bar,
                "bar_interval_ms": state.bar_interval_ms,
            }
        return result

    def restore(self, data: dict[str, object]) -> None:
        """Restore position TP state from a snapshot (crash recovery)."""
        self._positions = {}
        for symbol, raw in data.items():
            d = raw if isinstance(raw, dict) else {}
            _mfe = d.get("predicted_mfe_pct")
            _mae = d.get("predicted_mae_pct")
            _peak = d.get("predicted_peak_bar")
            self._positions[symbol] = PositionTPState(
                symbol=str(d.get("symbol", symbol)),
                entry_price=float(d.get("entry_price", 0.0)),
                entry_mean_return=float(d.get("entry_mean_return", 0.0)),
                entry_stop_pct=float(d.get("entry_stop_pct", 0.0)),
                peak_price=float(d.get("peak_price", 0.0)),
                opened_at_ms=int(d.get("opened_at_ms", 0)),
                signal_decay_strikes=int(d.get("signal_decay_strikes", 0)),
                topk_miss_strikes=int(d.get("topk_miss_strikes", 0)),
                breakeven_armed=bool(d.get("breakeven_armed", False)),
                trail_armed=bool(d.get("trail_armed", False)),
                predicted_mfe_pct=float(_mfe) if _mfe is not None else None,
                predicted_mae_pct=float(_mae) if _mae is not None else None,
                predicted_peak_bar=int(_peak) if _peak is not None else None,
                bar_interval_ms=int(d.get("bar_interval_ms", 3_600_000)),
                current_trailing_stop=(
                    float(d["current_trailing_stop"])
                    if d.get("current_trailing_stop") is not None
                    else None
                ),
            )
        logger.info("restored take_profit_state for %d positions", len(self._positions))
