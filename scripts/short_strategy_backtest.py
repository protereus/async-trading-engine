"""CLI wrapper for ``bot.analysis.short_backtest``.

Run a rolling-window paper backtest of Kronos-driven shorting strategies
against the live ``signal_history`` table.  Emits a human-readable text
report and (optionally) a JSON snapshot so consecutive runs can be diffed
to track edge persistence over time.

Usage::

    uv run python scripts/short_strategy_backtest.py
    uv run python scripts/short_strategy_backtest.py --window-days 60
    uv run python scripts/short_strategy_backtest.py --full-history --json

Snapshot output: ``journal/short_strategy_backtest_YYYY-MM-DD.txt`` and
matching ``.json`` when ``--json`` is set.  Do NOT auto-tune from these
numbers — they're directional-edge measurements, not P&L forecasts.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.analysis.short_backtest import (  # noqa: E402
    BacktestConfig,
    format_report,
    load_signals,
    report_to_json,
    run_all,
)

_DEFAULT_DB = "candles.db"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--db", default=_DEFAULT_DB, help=f"Path to candles.db (default: {_DEFAULT_DB})"
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=30,
        help="Rolling lookback window in days (default: 30)",
    )
    parser.add_argument(
        "--full-history",
        action="store_true",
        help="Override --window-days and use every resolved signal",
    )
    parser.add_argument("--min-return", type=float, default=0.003, help="Threshold |mean_return|")
    parser.add_argument("--max-uncertainty", type=float, default=10.0)
    parser.add_argument(
        "--output-dir",
        default="journal",
        help="Directory for date-stamped report (default: journal/)",
    )
    parser.add_argument("--json", action="store_true", help="Also emit a JSON snapshot")
    parser.add_argument(
        "--no-write", action="store_true", help="Print report only, don't write file"
    )
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Error: DB not found at {args.db}", file=sys.stderr)
        sys.exit(1)

    if args.full_history:
        since_ms = None
        window_label: int | None = None
    else:
        cutoff = datetime.now(UTC) - timedelta(days=args.window_days)
        since_ms = int(cutoff.timestamp() * 1000)
        window_label = args.window_days

    signals = load_signals(args.db, since_ms=since_ms)
    cfg = BacktestConfig(min_return=args.min_return, max_uncertainty=args.max_uncertainty)
    results = run_all(signals, cfg)

    report = format_report(
        results,
        n_signals=len(signals),
        window_days=window_label,
        cfg=cfg,
    )
    print(report)

    if args.no_write:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    txt_path = output_dir / f"short_strategy_backtest_{date_str}.txt"
    txt_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten: {txt_path}")

    if args.json:
        json_path = output_dir / f"short_strategy_backtest_{date_str}.json"
        json_path.write_text(
            json.dumps(
                report_to_json(results, n_signals=len(signals), window_days=window_label, cfg=cfg),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"Written: {json_path}")


if __name__ == "__main__":
    main()
