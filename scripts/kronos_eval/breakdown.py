"""Per-symbol breakdown of a Kronos eval predictions CSV (pure CPU, no torch).

Diagnostic for distinguishing "uniform noise" from "systematic / harness issue":
prints per-symbol IC / RankIC / hit-rate / post-cost net, plus a summary (how many
symbols have IC>0, FX-vs-equity, horizon-span sanity). Optionally overlays a second
predictions file (e.g. fine-tuned) for a per-symbol IC comparison.

Usage::

    uv run python scripts/kronos_eval/breakdown.py kronos_ab/results/predictions_baseline.csv \
        --compare kronos_ab/results/predictions_finetuned.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from costs import load_costs, round_trip_cost  # type: ignore[import-not-found]  # noqa: E402
from metrics import hit_rate, ic, rank_ic  # type: ignore[import-not-found]  # noqa: E402


def _load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["tradeable"] = df["tradeable"].astype(bool)
    df["origin_ts"] = pd.to_datetime(df["origin_ts"], utc=True)
    df["horizon_ts"] = pd.to_datetime(df["horizon_ts"], utc=True)
    return df


def _per_symbol(df: pd.DataFrame, costs: dict[str, float]) -> pd.DataFrame:
    rows = []
    for sym, g in df.groupby("symbol"):
        pred = g["mean_return"].to_numpy()
        real = g["realized_return"].to_numpy()
        tr = g[g["tradeable"]]
        if len(tr):
            cost = round_trip_cost(str(g["asset_class"].iloc[0]), costs)
            net = (tr["realized_return"] - cost).to_numpy()
            net_mean = float(np.mean(net))
            net_hit = float(np.mean(net > 0))
        else:
            net_mean = float("nan")
            net_hit = float("nan")
        span_d = float((g["horizon_ts"] - g["origin_ts"]).dt.total_seconds().median() / 86400)
        rows.append(
            {
                "symbol": sym,
                "class": g["asset_class"].iloc[0],
                "n": len(g),
                "n_tr": len(tr),
                "IC": ic(pred, real),
                "RankIC": rank_ic(pred, real),
                "hit": hit_rate(pred, real),
                "pred%": float(np.mean(pred)) * 100,
                "real%": float(np.mean(real)) * 100,
                "net%": net_mean * 100,
                "net_hit": net_hit,
                "span_d": span_d,
            }
        )
    return pd.DataFrame(rows).sort_values("IC")


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("predictions")
    p.add_argument("--compare", default=None, help="Second predictions CSV (e.g. fine-tuned).")
    p.add_argument("--costs", default=None)
    args = p.parse_args()
    costs = load_costs(args.costs)

    df = _load(args.predictions)
    tab = _per_symbol(df, costs)

    cmp_ic = None
    if args.compare:
        cdf = _load(args.compare)
        cmp_ic = {
            sym: ic(g["mean_return"].to_numpy(), g["realized_return"].to_numpy())
            for sym, g in cdf.groupby("symbol")
        }

    pd.set_option("display.width", 160)
    pd.set_option("display.float_format", lambda x: f"{x:7.3f}")
    print(f"\n=== per-symbol breakdown: {args.predictions} ===")
    show = tab.copy()
    if cmp_ic is not None:
        show["IC_cmp"] = show["symbol"].map(cmp_ic)
        show["dIC"] = show["IC_cmp"] - show["IC"]
    print(show.to_string(index=False))

    # --- summary ---
    fx = tab[tab["class"] == "forex"]
    eq = tab[tab["class"] == "equity"]
    print("\n=== summary ===")
    for name, sub in (("ALL", tab), ("forex", fx), ("equity", eq)):
        pos = int((sub["IC"] > 0).sum())
        print(
            f"{name:7s} n_sym={len(sub):2d}  IC>0: {pos}/{len(sub)}  "
            f"mean_IC={sub['IC'].mean():+.3f}  median_IC={sub['IC'].median():+.3f}  "
            f"mean_RankIC={sub['RankIC'].mean():+.3f}  median_span={sub['span_d'].median():.1f}d"
        )
    # systematic-sign check: is per-symbol IC distribution centered well below 0?
    allmean = tab["IC"].mean()
    verdict = (
        "SYSTEMATIC (IC consistently negative across symbols — check sign/horizon alignment)"
        if (tab["IC"] > 0).mean() < 0.25 and allmean < -0.05
        else "NOISE-LIKE (IC scattered around ~0; no single toxic symbol or sign bug)"
    )
    print(f"\nverdict: {verdict}")
    if cmp_ic is not None:
        improved = int((show["dIC"] > 0).sum())
        print(
            f"fine-tuned improved per-symbol IC on {improved}/{len(show)} symbols "
            f"(mean dIC={show['dIC'].mean():+.3f})"
        )


if __name__ == "__main__":
    main()
