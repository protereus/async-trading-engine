"""Empirical threshold calibration for the TopK entry filters.

Sweeps ``min_confidence`` × ``min_predicted_return`` over the resolved,
non-gap signal_history cohort accumulated by a running bot and reports the
entry-subset stats for each grid cell — use it to replace the wide shipped
defaults with values grounded in your own data.  ``max_uncertainty`` is held
at the configured cap (the storage filter never records rows above it, so
sweeping it adds no data).

Scope: forex by default (typically the class that accumulates clean rows
fastest on a 24/5 calendar).  Pass ``--symbols-like`` to override.

Reports per cell:
  - n: entries that would have been admitted
  - rate: entries/day across the data window
  - hit%: fraction with realized_return > 0
  - RankIC: Spearman between predicted and realised return
  - mean_ret, median_ret: realised return after admission

Not a live-config change — outputs a text report under ``journal/`` for you
to read before editing ``.env``.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median

import numpy as np

# Grid axes bracket the plausible operating range for each filter; widen or
# shift them if your calibration report shows the optimum at an edge cell.
_CONFIDENCE_GRID = (0.55, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95)
_RETURN_GRID = (0.000, 0.001, 0.003, 0.005, 0.008, 0.012)
_MAX_UNCERTAINTY_LIVE = 5.0
_FOREX_LIKE = "%/%"  # all currency pairs (and XAU/USD, filtered separately)


def _spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation; ties break by row position (fine for the
    effectively-continuous predicted/realised returns this sweeps)."""
    if len(x) < 3:
        return float("nan")
    # Guard on the raw values: double-argsort always yields ranks 0..n-1
    # (even for a constant series), so a rank-std check can never fire and
    # a constant input would otherwise return a spurious ±1.0.
    if float(np.std(x)) == 0.0 or float(np.std(y)) == 0.0:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return float(np.corrcoef(rx, ry)[0, 1])


def _load_cohort(
    db_path: str, symbols_like: str, exclude_symbols: tuple[str, ...]
) -> list[sqlite3.Row]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    # Emit the NOT IN clause only when there is something to exclude —
    # `symbol NOT IN (NULL)` is never true in SQL and would silently drop
    # the entire cohort.
    exclude_clause = ""
    if exclude_symbols:
        placeholders = ",".join("?" * len(exclude_symbols))
        exclude_clause = f"AND symbol NOT IN ({placeholders})"
    rows = con.execute(
        f"""
        SELECT scored_at, symbol, mean_return, direction_confidence, uncertainty,
               realized_return_at_horizon
        FROM signal_history
        WHERE realized_return_at_horizon IS NOT NULL
          AND COALESCE(gap_spanned, 0) = 0
          AND mean_return IS NOT NULL
          AND direction_confidence IS NOT NULL
          AND uncertainty IS NOT NULL
          AND uncertainty <= ?
          AND symbol LIKE ?
          {exclude_clause}
        ORDER BY scored_at ASC
        """,
        (_MAX_UNCERTAINTY_LIVE, symbols_like, *exclude_symbols),
    ).fetchall()
    con.close()
    return rows


def _grid_search(rows: list[sqlite3.Row]) -> list[dict]:
    if not rows:
        return []
    first_ms = rows[0]["scored_at"]
    last_ms = rows[-1]["scored_at"]
    span_days = max((last_ms - first_ms) / (1000 * 86400), 1.0)
    cells: list[dict] = []
    for conf in _CONFIDENCE_GRID:
        for ret in _RETURN_GRID:
            preds = []
            reals = []
            for r in rows:
                if r["direction_confidence"] < conf:
                    continue
                if r["mean_return"] < ret:
                    continue
                preds.append(r["mean_return"])
                reals.append(r["realized_return_at_horizon"])
            n = len(preds)
            if n == 0:
                cells.append(
                    {
                        "conf": conf,
                        "ret": ret,
                        "n": 0,
                        "rate": 0.0,
                        "hit": float("nan"),
                        "ic": float("nan"),
                        "mean_ret": float("nan"),
                        "median_ret": float("nan"),
                    }
                )
                continue
            hits = sum(1 for v in reals if v > 0)
            cells.append(
                {
                    "conf": conf,
                    "ret": ret,
                    "n": n,
                    "rate": n / span_days,
                    "hit": hits / n,
                    "ic": _spearman(preds, reals),
                    "mean_ret": mean(reals),
                    "median_ret": median(reals),
                }
            )
    return cells


def _format_report(rows: list[sqlite3.Row], cells: list[dict], cohort_label: str) -> str:
    lines = [
        f"TopK Threshold Calibration — {cohort_label}",
        f"DB cohort: {len(rows)} resolved non-gap rows, "
        f"{datetime.fromtimestamp(rows[0]['scored_at'] / 1000, UTC):%Y-%m-%d} → "
        f"{datetime.fromtimestamp(rows[-1]['scored_at'] / 1000, UTC):%Y-%m-%d}",
        f"Generated: {datetime.now(tz=UTC):%Y-%m-%d %H:%M UTC}",
        f"Held fixed: max_uncertainty = {_MAX_UNCERTAINTY_LIVE:.1f} (live default)",
        "",
        "Live defaults: min_confidence = 0.70, min_predicted_return = 0.003 → see ★",
        "",
    ]

    # Grid: rows = confidence, cols = return.  Show n / hit% / RankIC stack per cell.
    lines.append("── Grid: n / hit% / RankIC ──")
    hdr = f"{'conf \\ ret':>11} | " + " | ".join(f"{r:>7.4f}" for r in _RETURN_GRID)
    lines.append(hdr)
    lines.append("-" * len(hdr))
    by_cell = {(c["conf"], c["ret"]): c for c in cells}
    for conf in _CONFIDENCE_GRID:
        row1 = [f"{conf:>11.2f}"]
        row2 = [" " * 11]
        row3 = [" " * 11]
        for ret in _RETURN_GRID:
            c = by_cell[(conf, ret)]
            marker = "★" if (conf == 0.70 and ret == 0.003) else " "
            if c["n"] == 0:
                row1.append(f"  {marker}    -")
                row2.append("        -")
                row3.append("        -")
            else:
                row1.append(f" {marker}n={c['n']:>4}")
                row2.append(f" {c['hit'] * 100:>6.1f}%")
                row3.append(f" IC={c['ic']:>+5.3f}")
        lines.append(" | ".join(row1))
        lines.append(" | ".join(row2))
        lines.append(" | ".join(row3))
        lines.append("")

    # Best-cells shortlist (sorted by RankIC × hit-rate × log(n))
    lines.append("── Top 10 cells by score = RankIC × (hit% − 0.50) × log10(n+10) ──")
    scored = []
    for c in cells:
        if c["n"] < 30:
            continue
        score = c["ic"] * max(c["hit"] - 0.50, 0) * np.log10(c["n"] + 10)
        scored.append((score, c))
    scored.sort(reverse=True, key=lambda t: t[0])
    lines.append(
        f"  {'conf':>6} {'ret':>7}  {'n':>5} {'rate/d':>7} "
        f"{'hit%':>6} {'RankIC':>7} {'mean':>8} {'median':>8} {'score':>7}"
    )
    for score, c in scored[:10]:
        lines.append(
            f"  {c['conf']:>6.2f} {c['ret']:>7.4f}  {c['n']:>5} {c['rate']:>7.1f} "
            f"{c['hit'] * 100:>5.1f}% {c['ic']:>+7.3f} "
            f"{c['mean_ret']:>+8.4f} {c['median_ret']:>+8.4f} {score:>+7.3f}"
        )
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n", maxsplit=1)[0])
    p.add_argument("--db", default="candles.db")
    p.add_argument(
        "--symbols-like",
        default=_FOREX_LIKE,
        help="SQL LIKE pattern (default %% / %% — currency pairs)",
    )
    p.add_argument(
        "--exclude-symbols",
        default="XAU/USD",
        help="Comma-separated exclusions (default: XAU/USD — metals are scan-only)",
    )
    p.add_argument("--output-dir", default="journal")
    p.add_argument("--label", default="forex", help="Cohort label for the report header")
    args = p.parse_args()

    if not Path(args.db).exists():
        print(f"DB not found: {args.db}", file=sys.stderr)
        sys.exit(1)

    exclude = tuple(s for s in args.exclude_symbols.split(",") if s)
    rows = _load_cohort(args.db, args.symbols_like, exclude)
    if not rows:
        print("No resolved rows match the cohort filter.", file=sys.stderr)
        sys.exit(1)

    cells = _grid_search(rows)
    report = _format_report(rows, cells, args.label)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(exist_ok=True)
    date_str = datetime.now(tz=UTC).strftime("%Y-%m-%d")
    out_path = out_dir / f"phase_10c_calibration_{args.label}_{date_str}.txt"
    out_path.write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
