"""Path-aware Kronos signal extraction.

KronosPathSignal extends the scalar AssetSignal metrics with full OHLC path
statistics derived from the Pass 1 (point estimate) predict_batch output.

All path metrics are computed from the averaged OHLC path returned by a single
predict_batch call at forecast_temperature=0.6.  They are available immediately
after Pass 1 — no additional inference calls required.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class KronosPathSignal:
    """Full path-aware signal from one Kronos point-estimate pass.

    Scalar metrics (mean_return, std_return, direction_confidence, uncertainty)
    match the existing AssetSignal contract.  Path metrics are computed from the
    complete predicted OHLC sequence across the pred_len horizon.
    """

    symbol: str

    # --- Existing scalar metrics (preserved from AssetSignal) ---
    mean_return: float  # (predicted_close[-1] - entry) / entry
    std_return: float  # from variance pass
    direction_confidence: float
    uncertainty: float
    stop_pct: float

    # --- Path-aware metrics ---
    predicted_max_high: float  # peak of predicted highs across horizon
    predicted_min_low: float  # trough of predicted lows across horizon
    predicted_mfe_pct: float  # (max_high - entry) / entry  — positive
    predicted_mae_pct: float  # (entry - min_low) / entry   — positive
    predicted_peak_bar: int  # bar index where predicted close peaks
    predicted_volatility: float  # mean of (high - low) / close per bar
    predicted_path_drawdown: float  # max peak-to-trough in predicted close path
    monotonicity: float  # Pearson corr(bar_index, predicted_close) ∈ [-1, 1]

    # --- Ranking score (replaces raw mean_return for select_top_k) ---
    ranking_score: float = 0.0  # computed by extract_path_signal

    # Raw path arrays (kept for persistence / debugging; excluded from equality)
    predicted_closes: list[float] = field(default_factory=list, compare=False)
    predicted_highs: list[float] = field(default_factory=list, compare=False)
    predicted_lows: list[float] = field(default_factory=list, compare=False)


def extract_path_signal(
    symbol: str,
    pred_df: pd.DataFrame,
    entry_price: float,
    std_return: float,
    direction_confidence: float,
    vol_stop_multiplier: float = 2.0,
    min_stop_pct: float = 0.005,
    predicted_vol_max: float = 0.05,
    ranking_horizon_bars: int = 0,
) -> KronosPathSignal | None:
    """Compute KronosPathSignal from a single predict_batch output DataFrame.

    Parameters
    ----------
    pred_df:
        DataFrame returned by predict_batch for one asset.  Must contain
        ``open``, ``high``, ``low``, ``close`` columns.
    entry_price:
        Current (last observed) close price used as the reference for returns.
    std_return:
        Std of returns across the variance pass (Pass 2), used for stop_pct.
    direction_confidence:
        From variance pass.
    predicted_vol_max:
        Assets with predicted_volatility > this are flagged (not filtered here;
        filtering happens in select_top_k_path).
    ranking_horizon_bars:
        — bar index used for ``mean_return`` (1-indexed).  ``0``
        keeps legacy behaviour (final bar).  MFE / MAE / peak / monotonicity
        / path_drawdown remain computed across the **full** predicted path;
        only the ranking-input scalar (mean_return) is sliced.

    Returns
    -------
    KronosPathSignal, or None if the DataFrame is missing required columns.
    """
    required = {"open", "high", "low", "close"}
    if not required.issubset(pred_df.columns):
        return None
    if len(pred_df) == 0:
        return None

    closes = pred_df["close"].to_numpy(dtype=float)
    highs = pred_df["high"].to_numpy(dtype=float)
    lows = pred_df["low"].to_numpy(dtype=float)

    n = len(closes)
    bar_idx = np.arange(n, dtype=float)

    # — slice mean_return at the configured ranking horizon.
    # Default (0) = terminal close, identical to pre-10b behaviour.
    if ranking_horizon_bars <= 0:
        ranking_close = closes[-1]
    else:
        ranking_close = closes[min(ranking_horizon_bars, n) - 1]
    mean_return = (ranking_close - entry_price) / entry_price
    uncertainty = std_return / (abs(mean_return) + 1e-8)
    stop_pct = max(std_return * vol_stop_multiplier, min_stop_pct)

    # Path metrics
    predicted_max_high = float(np.max(highs))
    predicted_min_low = float(np.min(lows))
    predicted_mfe_pct = (predicted_max_high - entry_price) / entry_price
    predicted_mae_pct = (entry_price - predicted_min_low) / entry_price

    peak_bar = int(np.argmax(closes))

    per_bar_range = (highs - lows) / np.where(closes != 0, closes, 1.0)
    predicted_volatility = float(np.mean(per_bar_range))

    # Path drawdown: max peak-to-trough along the predicted close series
    running_max = np.maximum.accumulate(closes)
    drawdowns = (running_max - closes) / np.where(running_max != 0, running_max, 1.0)
    predicted_path_drawdown = float(np.max(drawdowns))

    # Monotonicity: Pearson correlation of bar index vs close
    monotonicity = float(np.corrcoef(bar_idx, closes)[0, 1]) if n >= 2 and closes.std() > 0 else 0.0

    # Ranking score: rewards directional cleanliness and MFE confirmation
    abs_mean = abs(mean_return)
    mfe_confirmation = min(1.0, predicted_mfe_pct / abs_mean) if abs_mean > 1e-9 else 0.0
    ranking_score = mean_return * max(0.0, monotonicity) * mfe_confirmation

    return KronosPathSignal(
        symbol=symbol,
        mean_return=mean_return,
        std_return=std_return,
        direction_confidence=direction_confidence,
        uncertainty=uncertainty,
        stop_pct=stop_pct,
        predicted_max_high=predicted_max_high,
        predicted_min_low=predicted_min_low,
        predicted_mfe_pct=predicted_mfe_pct,
        predicted_mae_pct=predicted_mae_pct,
        predicted_peak_bar=peak_bar,
        predicted_volatility=predicted_volatility,
        predicted_path_drawdown=predicted_path_drawdown,
        monotonicity=monotonicity,
        ranking_score=ranking_score,
        predicted_closes=closes.tolist(),
        predicted_highs=highs.tolist(),
        predicted_lows=lows.tolist(),
    )


def select_top_k_path(
    signals: list[KronosPathSignal],
    k: int,
    min_confidence: float = 0.70,
    max_uncertainty: float = 2.0,
    min_predicted_return: float = 0.001,
    predicted_vol_max: float = 0.05,
) -> list[str]:
    """Select top-k symbols from path signals using ranking_score.

    Tradeable filter (same as AssetSignal):
      - direction == LONG (mean_return >= 0)
      - mean_return >= min_predicted_return
      - direction_confidence >= min_confidence
      - uncertainty <= max_uncertainty
      - predicted_volatility <= predicted_vol_max  (new: regime filter)

    Ranking: by ranking_score descending (rewards monotonicity + MFE confirmation).
    """
    tradeable = [
        s
        for s in signals
        if s.mean_return >= min_predicted_return
        and s.direction_confidence >= min_confidence
        and s.uncertainty <= max_uncertainty
        and s.predicted_volatility <= predicted_vol_max
    ]
    ranked = sorted(tradeable, key=lambda s: s.ranking_score, reverse=True)
    return [s.symbol for s in ranked[:k]]
