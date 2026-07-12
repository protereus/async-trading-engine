"""Prototype A — does cross-sectional normalization of the predicted return lift
cross-sectional RankIC? (offline, no GPU, no live-strategy change.)

TopK ranks symbols by predicted return at each rerank, but a "+0.3%" means very
different things across symbols/classes, and some symbols are systematically more
bullish — so the raw cross-sectional ranking is mis-calibrated (Run 1: cross-sec
RankIC ≈ −0.14 despite positive per-symbol IC). This script re-scores a
predictions CSV under several normalizations and reports cross-sectional RankIC
for each, so we can see if normalization alone fixes the ranking before touching
``select_top_k``.

Schemes (✓ = causal/deployable as-is, ⚠ = uses full-sample stats → optimistic
upper bound, only a screen):
  raw              ✓  mean_return as-is (today's behaviour)
  vol_norm         ✓  mean_return / (std_return + eps)   — predicted return per unit risk
  psym_z_causal    ✓  per-symbol z-score from an EXPANDING (past-only) mean/std
  psym_demean_la   ⚠  mean_return − per-symbol full-sample mean
  psym_z_la        ⚠  per-symbol full-sample z-score (bias + scale removed)

Usage::

    uv run python scripts/kronos_eval/xsec_normalize.py kronos_ab/results/predictions_baseline.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from costs import load_costs, round_trip_cost  # type: ignore[import-not-found]  # noqa: E402
from metrics import rank_ic  # type: ignore[import-not-found]  # noqa: E402

_EPS = 1e-9
_MIN_EXPANDING = 20  # rows before a per-symbol expanding stat is trusted


def _xsec_rankic(df: pd.DataFrame, score_col: str, min_symbols: int = 3) -> tuple[float, int, int]:
    """Mean per-rerank RankIC of ``score_col`` vs realized, across symbols."""
    vals: list[float] = []
    sizes: list[int] = []
    for _, g in df.groupby("origin_ts"):
        sub = g[[score_col, "realized_return"]].dropna()
        if len(sub) < min_symbols:
            continue
        v = rank_ic(sub[score_col].to_numpy(), sub["realized_return"].to_numpy())
        if np.isfinite(v):
            vals.append(v)
            sizes.append(len(sub))
    if not vals:
        return float("nan"), 0, 0
    return float(np.mean(vals)), len(vals), int(np.median(sizes))


def _topk_backtest(
    df: pd.DataFrame, score_col: str, k: int, costs: dict[str, float]
) -> dict[str, float]:
    """At each rerank pick the top-k symbols by ``score_col``, go long, hold to the
    horizon, net of round-trip cost. Isolates whether better *ranking* picks better
    symbols. NOTE: overlapping windows → trades are autocorrelated; compare the
    *relative* net across schemes, don't trust the absolute t-stat.
    """
    nets: list[float] = []
    for _, g in df.groupby("origin_ts"):
        sub = g.dropna(subset=[score_col])
        if sub.empty:
            continue
        for _, row in sub.nlargest(k, score_col).iterrows():
            cost = round_trip_cost(str(row["asset_class"]), costs)
            nets.append(float(row["realized_return"]) - cost)
    arr = np.array(nets, dtype=float)
    if not arr.size:
        return {"n_picks": 0, "mean": float("nan"), "hit": float("nan")}
    return {"n_picks": int(arr.size), "mean": float(arr.mean()), "hit": float((arr > 0).mean())}


def _add_scores(df: pd.DataFrame) -> list[str]:
    """Add score columns for each scheme; return their names in display order."""
    df["raw"] = df["mean_return"]
    df["vol_norm"] = df["mean_return"] / (df["std_return"].abs() + _EPS)

    # Per-symbol causal z-score: expanding mean/std over the symbol's own past,
    # shifted by 1 so the current row is excluded (no lookahead).
    df.sort_values(["symbol", "origin_ts"], inplace=True)
    grp = df.groupby("symbol")["mean_return"]
    exp_mean = grp.transform(lambda s: s.shift(1).expanding(_MIN_EXPANDING).mean())
    exp_std = grp.transform(lambda s: s.shift(1).expanding(_MIN_EXPANDING).std())
    df["psym_z_causal"] = (df["mean_return"] - exp_mean) / (exp_std.abs() + _EPS)

    # Full-sample per-symbol stats (lookahead — upper bound only).
    sym_mean = df.groupby("symbol")["mean_return"].transform("mean")
    sym_std = df.groupby("symbol")["mean_return"].transform("std")
    df["psym_demean_la"] = df["mean_return"] - sym_mean
    df["psym_z_la"] = (df["mean_return"] - sym_mean) / (sym_std.abs() + _EPS)

    return ["raw", "vol_norm", "psym_z_causal", "psym_demean_la", "psym_z_la"]


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("predictions")
    p.add_argument("--min-symbols", type=int, default=3)
    p.add_argument("--topk", type=int, default=3, help="k for the post-cost top-k backtest.")
    p.add_argument("--costs", default=None, help="JSON overriding default round-trip costs.")
    p.add_argument(
        "--class",
        dest="asset_class",
        default=None,
        help="Restrict to one asset_class (e.g. forex).",
    )
    args = p.parse_args()

    df = pd.read_csv(args.predictions)
    df["origin_ts"] = pd.to_datetime(df["origin_ts"], utc=True)
    if args.asset_class:
        df = df[df["asset_class"] == args.asset_class].copy()

    schemes = _add_scores(df)
    costs = load_costs(args.costs)

    causal = {"raw", "vol_norm", "psym_z_causal"}
    print(f"\n=== cross-sectional ranking + post-cost top-{args.topk}: {args.predictions} ===")
    scope = args.asset_class or "all classes"
    print(f"scope: {scope}  ·  min {args.min_symbols} symbols/rerank\n")
    print(
        f"{'scheme':<16} {'causal?':>8} {'x-sec RankIC':>13} "
        f"{'topk net%':>10} {'topk hit':>9} {'n_picks':>8}"
    )
    print("-" * 70)
    raw_ic = raw_net = None
    for s in schemes:
        ic, _n_ts, _med = _xsec_rankic(df, s, args.min_symbols)
        bt = _topk_backtest(df, s, args.topk, costs)
        if s == "raw":
            raw_ic, raw_net = ic, bt["mean"]
        flag = "✓" if s in causal else "⚠ LA"
        has_ic_delta = raw_ic is not None and s != "raw" and np.isfinite(ic)
        d_ic = f" Δ{ic - raw_ic:+.3f}" if has_ic_delta else ""
        net_pct = bt["mean"] * 100
        d_net = (
            f" Δ{(bt['mean'] - raw_net) * 100:+.3f}"
            if (raw_net is not None and s != "raw" and np.isfinite(bt["mean"]))
            else ""
        )
        print(
            f"{s:<16} {flag:>8} {ic:>9.3f}{d_ic:>4} "
            f"{net_pct:>7.3f}{d_net:>3} {bt['hit']:>9.3f} {bt['n_picks']:>8}"
        )
    print("\n✓ = deployable · ⚠ LA = full-sample stats (lookahead → optimistic upper bound)")
    print(
        "x-sec RankIC: ranks symbols correctly at the rerank.  topk net%: mean post-cost return "
        f"per top-{args.topk} pick (overlapping → compare schemes, ignore absolute t-stat)."
    )


if __name__ == "__main__":
    main()
