"""CLI wrapper for ``bot.analysis.sentiment_edge``.

Measures whether the sentiment overlay would carry trading edge before
we enable the gate.  Reads the live ``signal_history`` table — rows
written by the rerank loop with the sentiment fields captured at
signal time — and partitions resolved signals three ways:

* AGREE     sentiment matches the reading (momentum: sentiment_dir ==
            Kronos_dir; contrarian: sentiment_dir != Kronos_dir)
* DISAGREE  sentiment opposes the reading
* ABSENT    no sentiment coverage at signal time

Per partition, prints hit rate / mean realised return / Sharpe-proxy,
bucketed by asset class (fx_major / metals / equity_index / other) and
by sentiment magnitude (weak / moderate / strong / extreme).  Cells
below ``MIN_CELL_N=30`` are flagged ⚠.

Usage::

    uv run python scripts/sentiment_edge_analysis.py
    uv run python scripts/sentiment_edge_analysis.py --window-days 60
    uv run python scripts/sentiment_edge_analysis.py --full-history --json

Snapshot output: ``journal/sentiment_edge_analysis_YYYY-MM-DD.txt`` and
matching ``.json`` when ``--json`` is set.  Decision protocol lives in
 — read that before drawing any
conclusion from a single weekly run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.analysis.sentiment_edge import (  # noqa: E402
    compute_harness,
    format_report,
    load_signals,
    report_to_json,
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
        since_label: str | None = "full history"
    else:
        cutoff = datetime.now(UTC) - timedelta(days=args.window_days)
        since_ms = int(cutoff.timestamp() * 1000)
        since_label = f"last {args.window_days} days"

    signals = load_signals(args.db, since_ms=since_ms)
    result = compute_harness(signals)

    report = format_report(result, since_label=since_label)
    print(report)

    if args.no_write:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    txt_path = output_dir / f"sentiment_edge_analysis_{date_str}.txt"
    txt_path.write_text(report + "\n", encoding="utf-8")
    print(f"\nWritten: {txt_path}")

    if args.json:
        json_path = output_dir / f"sentiment_edge_analysis_{date_str}.json"
        json_path.write_text(
            json.dumps(report_to_json(result, since_label=since_label), indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Written: {json_path}")


if __name__ == "__main__":
    main()
