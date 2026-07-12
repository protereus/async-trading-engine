"""Pure (CPU) evaluation metrics for the Kronos fine-tune A/B.

Consumes a predictions DataFrame (one row per scored test origin; see
``predict.py`` for the schema) and produces IC / RankIC / hit-rate and a
post-cost signal backtest. No torch — unit-tested locally.

Predictions schema (columns used here):
  symbol, asset_class, origin_ts, current_price, mean_return,
  direction_confidence, uncertainty, tradeable, realized_return.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
import pandas as pd
from costs import round_trip_cost


def _pearson(a: Sequence[float], b: Sequence[float]) -> float:
    """Pearson correlation; NaN when <2 points or a degenerate (zero-variance) input."""
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def ic(pred: Sequence[float], realized: Sequence[float]) -> float:
    """Information coefficient: Pearson corr of predicted vs realised return."""
    return _pearson(pred, realized)


def rank_ic(pred: Sequence[float], realized: Sequence[float]) -> float:
    """Rank IC: Pearson on ranks (== Spearman), no scipy dependency."""
    p = pd.Series(pred, dtype=float)
    r = pd.Series(realized, dtype=float)
    mask = p.notna() & r.notna()
    if mask.sum() < 2:
        return float("nan")
    return _pearson(p[mask].rank().to_numpy(), r[mask].rank().to_numpy())


def hit_rate(pred: Sequence[float], realized: Sequence[float]) -> float:
    """Fraction of rows where predicted and realised return share a sign.

    Rows with a zero predicted or realised return are excluded (no directional bet).
    """
    p = np.asarray(pred, dtype=float)
    r = np.asarray(realized, dtype=float)
    mask = np.isfinite(p) & np.isfinite(r) & (p != 0) & (r != 0)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean(np.sign(p[mask]) == np.sign(r[mask])))


def cross_sectional_rank_ic(df: pd.DataFrame, min_symbols: int = 3) -> float:
    """Mean per-timestamp RankIC across symbols (the cross-sectional TopK view).

    Groups rows by ``origin_ts``; for each timestamp with ≥ ``min_symbols``
    symbols, computes RankIC of ``mean_return`` vs ``realized_return``, then
    averages over timestamps. NaN when no timestamp qualifies.
    """
    vals: list[float] = []
    for _, grp in df.groupby("origin_ts"):
        if len(grp) < min_symbols:
            continue
        v = rank_ic(grp["mean_return"].to_numpy(), grp["realized_return"].to_numpy())
        if math.isfinite(v):
            vals.append(v)
    return float(np.mean(vals)) if vals else float("nan")


def _net_return_stats(net: np.ndarray) -> dict[str, float]:
    """Summary stats for a vector of per-trade net returns."""
    net = net[np.isfinite(net)]
    n = len(net)
    if n == 0:
        return {
            "n_trades": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "std": float("nan"),
            "total": float("nan"),
            "hit_rate": float("nan"),
            "t_stat": float("nan"),
        }
    mean = float(np.mean(net))
    std = float(np.std(net, ddof=1)) if n > 1 else float("nan")
    t_stat = float(mean / (std / math.sqrt(n))) if (std and std > 0) else float("nan")
    return {
        "n_trades": n,
        "mean": mean,
        "median": float(np.median(net)),
        "std": std,
        "total": float(np.sum(net)),
        "hit_rate": float(np.mean(net > 0)),
        "t_stat": t_stat,
    }


def signal_backtest(df: pd.DataFrame, costs: dict[str, float]) -> dict[str, object]:
    """Post-cost edge of the tradeable signals.

    Each ``tradeable`` row is one long trade entered at the origin and held to the
    horizon: ``net = realized_return − round_trip_cost(asset_class)``. Reports the
    net-return distribution pooled and per asset class. This is the primary
    decision metric — a fine-tune only wins if its tradeable signals are
    profitable *after* costs.
    """
    trades = df[df["tradeable"].astype(bool)].copy()
    if trades.empty:
        return {"pooled": _net_return_stats(np.array([])), "by_class": {}}
    trades["cost"] = trades["asset_class"].map(lambda c: round_trip_cost(str(c), costs))
    trades["net_return"] = trades["realized_return"].astype(float) - trades["cost"]

    by_class = {
        str(cls): _net_return_stats(grp["net_return"].to_numpy())
        for cls, grp in trades.groupby("asset_class")
    }
    return {"pooled": _net_return_stats(trades["net_return"].to_numpy()), "by_class": by_class}


def _grouped(df: pd.DataFrame, fn) -> dict[str, float]:
    """Apply ``fn(group)`` per asset_class → {class: value}."""
    return {str(cls): fn(grp) for cls, grp in df.groupby("asset_class")}


def _agg_ics(ics: list[float], rics: list[float]) -> dict[str, float]:
    fi = [v for v in ics if math.isfinite(v)]
    fr = [v for v in rics if math.isfinite(v)]
    return {
        "n_symbols": len(ics),
        "mean_ic": float(np.mean(fi)) if fi else float("nan"),
        "median_ic": float(np.median(fi)) if fi else float("nan"),
        "mean_rank_ic": float(np.mean(fr)) if fr else float("nan"),
        "median_rank_ic": float(np.median(fr)) if fr else float("nan"),
        "n_pos_ic": int(sum(1 for v in fi if v > 0)),
    }


def per_symbol_ic(df: pd.DataFrame) -> dict[str, object]:
    """Within-symbol IC/RankIC aggregated across symbols (overall + per class).

    This is the **correct** headline for a multi-symbol panel: pooled IC mixes a
    spurious between-symbol level effect (Simpson's paradox — Run 1 had pooled IC
    −0.07 while per-symbol IC was +0.45 / FX 12-of-12 positive). ``n_pos_ic`` (how
    many symbols beat 0) is the robust read; magnitudes are inflated by
    overlapping-window autocorrelation, so trust the sign over the value.
    """
    recs: list[tuple[str, float, float]] = []
    for _, g in df.groupby("symbol"):
        p, r = g["mean_return"].to_numpy(), g["realized_return"].to_numpy()
        recs.append((str(g["asset_class"].iloc[0]), ic(p, r), rank_ic(p, r)))
    overall = _agg_ics([r[1] for r in recs], [r[2] for r in recs])
    by_class = {
        cls: _agg_ics([r[1] for r in recs if r[0] == cls], [r[2] for r in recs if r[0] == cls])
        for cls in sorted({r[0] for r in recs})
    }
    return {**overall, "by_class": by_class}


def summarize(df: pd.DataFrame, costs: dict[str, float]) -> dict[str, object]:
    """Full metric bundle for one model's predictions.

    Headline metrics are **per-symbol IC** and **cross-sectional RankIC** (the two
    that matter for a TopK panel). ``ic``/``rank_ic`` ``pooled`` are retained but
    DEPRECATED — pooled IC is Simpson-distorted on a multi-symbol panel (see
    ``per_symbol_ic``); don't use it as a headline.
    """
    pred = df["mean_return"].to_numpy()
    real = df["realized_return"].to_numpy()

    def _ic(g: pd.DataFrame) -> float:
        return ic(g["mean_return"].to_numpy(), g["realized_return"].to_numpy())

    def _ric(g: pd.DataFrame) -> float:
        return rank_ic(g["mean_return"].to_numpy(), g["realized_return"].to_numpy())

    def _hr(g: pd.DataFrame) -> float:
        return hit_rate(g["mean_return"].to_numpy(), g["realized_return"].to_numpy())

    return {
        "n_origins": int(len(df)),
        "n_symbols": int(df["symbol"].nunique()),
        "n_tradeable": int(df["tradeable"].astype(bool).sum()),
        # --- headline (panel-correct) ---
        "per_symbol_ic": per_symbol_ic(df),
        "cross_sectional_rank_ic": cross_sectional_rank_ic(df),
        "signal_backtest": signal_backtest(df, costs),
        # --- deprecated: pooled IC is Simpson-distorted, kept for continuity ---
        "ic": {"pooled": ic(pred, real), "by_class": _grouped(df, _ic)},
        "rank_ic": {"pooled": rank_ic(pred, real), "by_class": _grouped(df, _ric)},
        "hit_rate": {"pooled": hit_rate(pred, real), "by_class": _grouped(df, _hr)},
    }
