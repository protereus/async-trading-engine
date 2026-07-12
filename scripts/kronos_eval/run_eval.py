"""Orchestrate the Kronos eval: predictions CSV → metrics, and baseline-vs-finetuned compare.

Two CPU-only subcommands (the GPU prediction step is ``predict.py``):

  summarize  --predictions <csv> --label <name> [--costs <json>] [--out-dir results]
             → results/<label>/metrics.json + summary.md

  compare    --baseline <metrics.json> --finetuned <metrics.json>
             → side-by-side delta table (post-cost decision view)

The A/B flow: run ``predict.py`` for the zero-shot model and the fine-tuned model
on the SAME manifest, ``summarize`` each, then ``compare``. The decision metric is
the post-cost signal backtest, not IC.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from costs import load_costs  # type: ignore[import-not-found]  # noqa: E402
from metrics import summarize  # type: ignore[import-not-found]  # noqa: E402


def _fmt(x: object) -> str:
    return f"{x:.4f}" if isinstance(x, float) else str(x)


def _summary_md(label: str, m: dict) -> str:
    bt = m["signal_backtest"]["pooled"]
    psi = m["per_symbol_ic"]
    lines = [
        f"# Kronos eval — {label}",
        "",
        f"- Origins scored: {m['n_origins']} across {m['n_symbols']} symbols "
        f"({m['n_tradeable']} tradeable)",
        "",
        "## Headline (panel-correct)",
        f"- **Per-symbol IC**: mean {_fmt(psi['mean_ic'])}, median {_fmt(psi['median_ic'])}, "
        f"**positive {psi['n_pos_ic']}/{psi['n_symbols']}** "
        f"(per-symbol RankIC mean {_fmt(psi['mean_rank_ic'])})",
        f"- **Cross-sectional RankIC**: {_fmt(m['cross_sectional_rank_ic'])}  "
        f"_(what TopK ranking needs)_",
        f"- _(deprecated)_ pooled IC: {_fmt(m['ic']['pooled'])} — Simpson-distorted on a panel; "
        f"don't headline it",
        "",
        "## Post-cost signal backtest (pooled)",
        f"- Trades: {bt['n_trades']}  ·  mean net: {_fmt(bt['mean'])}  ·  "
        f"total: {_fmt(bt['total'])}  ·  hit-rate: {_fmt(bt['hit_rate'])}  ·  "
        f"t-stat: {_fmt(bt['t_stat'])}",
        "",
        "## Per asset class",
        "",
        "| class | per-sym mean IC | IC>0 | x-sec RankIC* | net trades | mean net | net hit |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    by_bt = m["signal_backtest"]["by_class"]
    for c in sorted(psi["by_class"]):
        cc = psi["by_class"][c]
        b = by_bt.get(c, {})
        lines.append(
            f"| {c} | {_fmt(cc['mean_ic'])} | {cc['n_pos_ic']}/{cc['n_symbols']} | "
            f"{_fmt(m['rank_ic']['by_class'].get(c, float('nan')))} | {b.get('n_trades', 0)} | "
            f"{_fmt(b.get('mean', float('nan')))} | {_fmt(b.get('hit_rate', float('nan')))} |"
        )
    lines += ["", "*per-class RankIC is pooled-within-class (less distorted than cross-asset).", ""]
    return "\n".join(lines)


def _cmd_summarize(args: argparse.Namespace) -> None:
    df = pd.read_csv(args.predictions)
    df["tradeable"] = df["tradeable"].astype(bool)
    metrics = summarize(df, load_costs(args.costs))
    out_dir = Path(args.out_dir) / args.label
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    (out_dir / "summary.md").write_text(_summary_md(args.label, metrics))
    bt = metrics["signal_backtest"]["pooled"]
    psi = metrics["per_symbol_ic"]
    print(
        f"[{args.label}] per-sym IC mean={_fmt(psi['mean_ic'])} "
        f"(pos {psi['n_pos_ic']}/{psi['n_symbols']}) "
        f"x-sec RankIC={_fmt(metrics['cross_sectional_rank_ic'])} "
        f"net_mean={_fmt(bt['mean'])} net_hit={_fmt(bt['hit_rate'])} "
        f"trades={bt['n_trades']} → {out_dir}/"
    )


def _cmd_compare(args: argparse.Namespace) -> None:
    base = json.loads(Path(args.baseline).read_text())
    fine = json.loads(Path(args.finetuned).read_text())

    def row(name: str, b: object, f: object) -> str:
        d = (f - b) if isinstance(b, float) and isinstance(f, float) else ""
        d_str = f"{d:+.4f}" if isinstance(d, float) else ""
        return f"| {name} | {_fmt(b)} | {_fmt(f)} | {d_str} |"

    bbt, fbt = base["signal_backtest"]["pooled"], fine["signal_backtest"]["pooled"]
    bpsi, fpsi = base["per_symbol_ic"], fine["per_symbol_ic"]
    lines = [
        "# Kronos A/B — baseline vs fine-tuned",
        "",
        "| metric | baseline | fine-tuned | Δ |",
        "|---|---:|---:|---:|",
        row("per-symbol IC (mean)", bpsi["mean_ic"], fpsi["mean_ic"]),
        row("per-symbol IC (median)", bpsi["median_ic"], fpsi["median_ic"]),
        f"| per-symbol IC>0 | {bpsi['n_pos_ic']}/{bpsi['n_symbols']} "
        f"| {fpsi['n_pos_ic']}/{fpsi['n_symbols']} | |",
        row(
            "cross-sectional RankIC",
            base["cross_sectional_rank_ic"],
            fine["cross_sectional_rank_ic"],
        ),
        row("post-cost mean net", bbt["mean"], fbt["mean"]),
        row("post-cost hit-rate", bbt["hit_rate"], fbt["hit_rate"]),
        row("post-cost t-stat", bbt["t_stat"], fbt["t_stat"]),
        f"| post-cost trades | {bbt['n_trades']} | {fbt['n_trades']} | |",
        row("(deprecated) pooled IC", base["ic"]["pooled"], fine["ic"]["pooled"]),
        "",
        "**Decision:** the trading go/no-go is **post-cost mean net + t-stat** (aim |t|≥2). "
        "**Per-symbol IC** and **cross-sectional RankIC** are the skill headline — pooled IC is "
        "Simpson-distorted on a panel, don't use it.",
        "",
    ]
    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("summarize", help="predictions CSV → metrics.json + summary.md")
    s.add_argument("--predictions", required=True)
    s.add_argument("--label", required=True, help="e.g. baseline or finetuned")
    s.add_argument("--costs", default=None, help="JSON overriding default round-trip costs.")
    s.add_argument("--out-dir", default="kronos_ab/results")
    s.set_defaults(func=_cmd_summarize)

    c = sub.add_parser("compare", help="baseline vs fine-tuned metrics → delta table")
    c.add_argument("--baseline", required=True)
    c.add_argument("--finetuned", required=True)
    c.add_argument("--out", default=None)
    c.set_defaults(func=_cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
