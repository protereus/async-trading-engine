"""Phase B — compute the no-leakage train/val/test split for the Kronos fine-tune.

 Reads the per-symbol CSVs produced by
``export_kronos_csv.py`` and emits a reproducible **split manifest** that the Phase C
symbol-aware loader and the Phase E eval harness both consume — so the held-out OOS
tail is byte-identical across the zero-shot baseline and the fine-tuned evaluation.

Split scheme (per symbol, time-ordered — walk-forward per instrument):
  * A training/eval *sample* is a sliding window of ``lookback + predict + 1`` bars:
    a ``lookback`` (400) bar context, then a ``predict`` (120) bar target (+1 overlap
    bar). Within a window starting at row ``s`` the **target region** is rows
    ``[s+lookback, s+lookback+predict]`` (121 bars).
  * A sample is classified by where its **target** lands — context may reach back
    across a split boundary (this is how live inference works; it leaks nothing
    because the model is never *trained* on the future target bars):
      - train: target entirely before ``val_start``   (``s+lookback+predict < val_start``)
      - val:   target within ``[val_start, test_start)``
      - test:  target within ``[test_start, n)``
  * Boundaries are per-symbol fractions of that symbol's bar count:
      ``val_start  = floor(n * train_frac)``
      ``test_start = floor(n * (train_frac + val_frac))``
  * **Tokenizer** is fit on bars ``[0, val_start)`` only. Every train sample's bars
    (context + target) lie below ``val_start``, so no val/test bar enters the codebook.

Why origin-based (not segment-contained): with a 400-bar lookback against our thin
history, requiring a full 521-bar window *inside* a 15 % segment yields zero val/test
samples. Classifying by target position (context free to cross backward) is both
data-efficient and exactly serve-realistic.

Usage::

    uv run python scripts/kronos_split.py --data-dir kronos_ab/data
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

# Share the exact windowing logic with the evaluation harness so the
# manifest's sample counts equal what a loader actually produces.
sys.path.insert(0, str(Path(__file__).parent))

from split_windows import (  # noqa: E402
    split_bounds,
    valid_start_indices,
)

# Defaults — keep lookback/predict identical to serve-time (topk_strategy) so
# train-time and live windows agree (plan decision #3).
_LOOKBACK = 400
_PREDICT = 120
_TRAIN_FRAC = 0.70
_VAL_FRAC = 0.15
_TEST_FRAC = 0.15
_SEED = 42
# Metals carry only ~577 bars and a volume-units discontinuity at the 2026-06-22
# IG-native cutover (Phase A). Excluded from the v1 fine-tune; revisit when they
# accumulate history.
_DEFAULT_EXCLUDE = ("XAU/USD", "XAG/USD")
# Minimum train samples below which a symbol contributes too little to bother.
_MIN_TRAIN_SAMPLES = 50


@dataclass
class SymbolSplit:
    symbol: str
    csv_file: str
    n_bars: int
    start: str | None
    end: str | None
    val_start_idx: int
    test_start_idx: int
    val_start_date: str | None
    test_start_date: str | None
    tokenizer_train_bars: int  # bars [0, val_start_idx)
    n_train_samples: int
    n_val_samples: int
    n_test_samples: int
    included: bool
    flags: list[str]


def _read_timestamps(csv_path: Path) -> list[str]:
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        return [row["timestamps"] for row in reader]


def _compute_split(
    symbol: str,
    csv_path: Path,
    *,
    lookback: int,
    predict: int,
    train_frac: float,
    val_frac: float,
    included: bool,
) -> SymbolSplit:
    ts = _read_timestamps(csv_path)
    n = len(ts)
    val_start, test_start = split_bounds(n, train_frac, val_frac)

    n_train = len(valid_start_indices(n, lookback, predict, 0, val_start))
    n_val = len(valid_start_indices(n, lookback, predict, val_start, test_start))
    n_test = len(valid_start_indices(n, lookback, predict, test_start, n))

    flags: list[str] = []
    if included:
        if n_train < _MIN_TRAIN_SAMPLES:
            flags.append(f"LOW_TRAIN_SAMPLES ({n_train} < {_MIN_TRAIN_SAMPLES})")
        if n_test == 0:
            flags.append("NO_TEST_SAMPLES (cannot be evaluated on its OOS tail)")
        if n_val == 0:
            flags.append("NO_VAL_SAMPLES (no per-symbol early-stop signal; val is FX-led)")
    else:
        flags.append("EXCLUDED_FROM_V1")

    return SymbolSplit(
        symbol=symbol,
        csv_file=csv_path.name,
        n_bars=n,
        start=ts[0] if n else None,
        end=ts[-1] if n else None,
        val_start_idx=val_start,
        test_start_idx=test_start,
        val_start_date=ts[val_start] if val_start < n else None,
        test_start_date=ts[test_start] if test_start < n else None,
        tokenizer_train_bars=val_start,
        n_train_samples=n_train,
        n_val_samples=n_val,
        n_test_samples=n_test,
        included=included,
        flags=flags,
    )


def _symbol_from_filename(name: str) -> str:
    """``EUR_USD.csv`` → ``EUR/USD``; ``KO.csv`` → ``KO``."""
    stem = name[:-4] if name.endswith(".csv") else name
    parts = stem.split("_")
    return "/".join(parts) if len(parts) == 2 else stem


def _write_manifest(splits: list[SymbolSplit], out_dir: Path, args: argparse.Namespace) -> None:
    included = [s for s in splits if s.included]
    totals = {
        "symbols_included": len(included),
        "train_samples": sum(s.n_train_samples for s in included),
        "val_samples": sum(s.n_val_samples for s in included),
        "test_samples": sum(s.n_test_samples for s in included),
    }
    payload = {
        "params": {
            "lookback": args.lookback,
            "predict": args.predict,
            "window": args.lookback + args.predict + 1,
            "train_frac": args.train_frac,
            "val_frac": args.val_frac,
            "test_frac": round(1.0 - args.train_frac - args.val_frac, 4),
            "seed": _SEED,
            "excluded_symbols": list(args.exclude),
        },
        "totals": totals,
        "symbols": [asdict(s) for s in splits],
    }
    (out_dir / "split_manifest.json").write_text(json.dumps(payload, indent=2))

    t = totals
    lines = [
        "# Kronos fine-tune — Phase B split manifest",
        "",
        f"- Window: lookback {args.lookback} + predict {args.predict} (+1 overlap) "
        f"= {args.lookback + args.predict + 1} bars",
        f"- Ratios (per symbol, time-ordered): train {args.train_frac} / "
        f"val {args.val_frac} / test {round(1.0 - args.train_frac - args.val_frac, 4)}",
        f"- Excluded from v1: {', '.join(args.exclude) or 'none'}",
        f"- Seed: {_SEED}",
        "",
        f"**Included: {t['symbols_included']} symbols — "
        f"{t['train_samples']:,} train / {t['val_samples']:,} val / "
        f"{t['test_samples']:,} test samples.**",
        "",
        (
            "| symbol | inc | bars | val_start (date) | test_start (date) | "
            "tok_train_bars | train | val | test | flags |"
        ),
        "|---|:--:|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for s in splits:
        flags = "; ".join(s.flags) if s.flags else "—"
        lines.append(
            f"| {s.symbol} | {'Y' if s.included else 'N'} | {s.n_bars} | "
            f"{s.val_start_idx} ({s.val_start_date or '—'}) | "
            f"{s.test_start_idx} ({s.test_start_date or '—'}) | {s.tokenizer_train_bars} | "
            f"{s.n_train_samples} | {s.n_val_samples} | {s.n_test_samples} | {flags} |"
        )
    flagged = [s for s in splits if s.included and s.flags]
    lines += ["", "## Flagged (included) symbols", ""]
    lines += (
        [f"- **{s.symbol}**: {'; '.join(s.flags)}" for s in flagged]
        if flagged
        else ["None — all included symbols have usable train/val/test sample counts."]
    )
    lines.append("")
    (out_dir / "split_report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", default="kronos_ab/data", help="Directory of per-symbol CSVs.")
    p.add_argument("--lookback", type=int, default=_LOOKBACK)
    p.add_argument("--predict", type=int, default=_PREDICT)
    p.add_argument("--train-frac", type=float, default=_TRAIN_FRAC, dest="train_frac")
    p.add_argument("--val-frac", type=float, default=_VAL_FRAC, dest="val_frac")
    p.add_argument(
        "--exclude",
        nargs="*",
        default=list(_DEFAULT_EXCLUDE),
        help="bot_keys to exclude from the v1 fine-tune (default: metals).",
    )
    args = p.parse_args()

    if args.train_frac + args.val_frac >= 1.0:
        raise SystemExit("train_frac + val_frac must be < 1.0 (leave room for test).")

    data_dir = Path(args.data_dir)
    csvs = sorted(f for f in data_dir.glob("*.csv"))
    if not csvs:
        raise SystemExit(f"No CSVs found in {data_dir} — run export_kronos_csv.py first.")

    exclude = set(args.exclude)
    splits: list[SymbolSplit] = []
    for csv_path in csvs:
        symbol = _symbol_from_filename(csv_path.name)
        splits.append(
            _compute_split(
                symbol,
                csv_path,
                lookback=args.lookback,
                predict=args.predict,
                train_frac=args.train_frac,
                val_frac=args.val_frac,
                included=symbol not in exclude,
            )
        )

    _write_manifest(splits, data_dir, args)

    inc = [s for s in splits if s.included]
    print(f"Split manifest → {data_dir}/split_manifest.json")
    print(
        f"Included {len(inc)} symbols: "
        f"{sum(s.n_train_samples for s in inc):,} train / "
        f"{sum(s.n_val_samples for s in inc):,} val / "
        f"{sum(s.n_test_samples for s in inc):,} test samples"
    )
    flagged = [s for s in inc if s.flags]
    if flagged:
        print(f"\n{len(flagged)} included symbol(s) flagged — see {data_dir}/split_report.md:")
        for s in flagged:
            print(f"  {s.symbol:<9} {'; '.join(s.flags)}")
    else:
        print("No flags on included symbols.")


if __name__ == "__main__":
    main()
