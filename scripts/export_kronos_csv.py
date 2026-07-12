"""Phase A — export per-symbol 1h OHLCV from the candle DB to Kronos CSV format.

Feeds offline Kronos experiments (fine-tuning / evaluation) with per-symbol CSVs.


Output: one CSV per symbol with the columns the ``finetune_csv`` loader
(``CustomKlineDataset``) requires::

    timestamps,open,high,low,close,volume,amount

The loader reads each CSV as a single contiguous time series and slides
windows across rows ordered by time — it has **no symbol column**.  A
naively-concatenated multi-symbol CSV would let a training window straddle two
instruments, so we emit one file per symbol.  The global-vs-per-symbol training
decision (and any loader change to combine symbols safely) is Phase C/D.

Volume convention mirrors serve-time inference (``TopKStrategy._prepare_asset``):
  * volume assets (14 US equities + 2 IG-native metals): ``volume`` = stored
    volume, ``amount`` = ``volume * close``;
  * FX (``has_volume=False``): ``volume = amount = 0`` (the no-volume Kronos
    path feeds no volume channel; the CSV loader always reads 6 features, so FX
    trains with the channel zeroed — a known train/serve nuance).

Safety:
  * **Never read the live ``candles.db`` directly.** ``VACUUM INTO`` a snapshot
    first.  This script refuses a DB path that looks live (a ``-wal``/``-shm``
    sibling is present) and opens read-only.

Usage::

    # 1. snapshot the live DB (safe hot-copy — never cp):
    sqlite3 candles.db "VACUUM INTO 'candles_snapshot.db'"
    # 2. export:
    uv run python scripts/export_kronos_csv.py \
        --db kronos_ab/candles_snapshot.db --out kronos_ab/data
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.data.eodhd_symbols import EODHD_UNIVERSE  # noqa: E402

_HOUR_MS = 3_600_000
# A window of lookback(400) + predict(120) + 1 is the minimum to form ONE
# training sample; flag symbols below it (and warn on thin history generally).
_MIN_SAMPLE_BARS = 400 + 120 + 1
_HEALTHY_BARS = 2_000
# Gap larger than this is bigger than a normal Fri→Sun FX weekend (~65 h) and
# worth a human look (feed outage, instrument switch).
_SUSPICIOUS_GAP_H = 80.0
# Ratio between the last- and first-quartile median non-zero volume above which
# we flag a volume-units discontinuity (e.g. the metals IG-native cutover, where
# backfilled IG-historical bars carry far less volume than live-streamed LTV).
_VOLUME_STEP_RATIO = 8.0
_LIVE_DB_PATH = "candles.db"

_CSV_HEADER = ["timestamps", "open", "high", "low", "close", "volume", "amount"]


@dataclass
class SymbolReport:
    symbol: str
    asset_class: str
    has_volume: bool
    csv_file: str
    n_rows: int
    start: str | None
    end: str | None
    span_days: float
    n_gaps_gt_90min: int
    max_gap_hours: float
    n_suspicious_gaps: int  # > _SUSPICIOUS_GAP_H
    n_nonpositive_price: int
    n_high_lt_low: int
    n_nonmonotonic_ts: int
    pct_zero_volume: float
    volume_step_ratio: float  # last-quartile / first-quartile median non-zero volume
    enough_for_training: bool
    flags: list[str]


def _open_readonly(db_path: Path) -> sqlite3.Connection:
    """Open *db_path* read-only, refusing anything that looks like the live DB."""
    resolved = db_path.resolve()
    if str(resolved) == str(Path(_LIVE_DB_PATH).resolve()):
        raise SystemExit(
            f"Refusing to read the live DB directly: {resolved}\n"
            f"VACUUM INTO a snapshot first (see this script's module docstring)."
        )
    for sibling in (f"{resolved}-wal", f"{resolved}-shm"):
        if Path(sibling).exists():
            raise SystemExit(
                f"{resolved} has a {Path(sibling).suffix} sibling — it looks like a "
                f"live/in-use DB. Snapshot it with VACUUM INTO and point --db at the copy."
            )
    # mode=ro: never create -wal/-shm or mutate the file.
    return sqlite3.connect(f"file:{resolved}?mode=ro", uri=True)


def _fetch_rows(
    con: sqlite3.Connection, symbol: str
) -> list[tuple[int, float, float, float, float, float]]:
    return con.execute(
        "SELECT timestamp, open, high, low, close, volume FROM candles "
        "WHERE symbol = ? ORDER BY timestamp ASC",
        (symbol,),
    ).fetchall()


def _iso(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")


def _median(xs: list[float]) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    n = len(s)
    mid = n // 2
    return s[mid] if n % 2 else (s[mid - 1] + s[mid]) / 2.0


def _volume_step_ratio(nonzero_vols: list[float]) -> float:
    """Last-quartile / first-quartile median of time-ordered non-zero volumes.

    A large ratio flags a within-series volume-units discontinuity (the metals
    IG-native cutover is the live example: backfilled IG-historical bars carry
    far less volume than live-streamed LTV).  Returns 0.0 when there is too
    little non-zero volume to judge.
    """
    n = len(nonzero_vols)
    if n < 8:
        return 0.0
    q = max(1, n // 4)
    first = _median(nonzero_vols[:q])
    last = _median(nonzero_vols[-q:])
    if first <= 0:
        return 0.0
    return last / first


def _export_symbol(
    con: sqlite3.Connection,
    bot_key: str,
    out_dir: Path,
) -> SymbolReport:
    meta = EODHD_UNIVERSE[bot_key]
    has_volume = meta.has_volume
    rows = _fetch_rows(con, bot_key)

    safe = bot_key.replace("/", "_")
    csv_path = out_dir / f"{safe}.csv"

    flags: list[str] = []
    n_nonpositive = 0
    n_high_lt_low = 0
    n_nonmono = 0
    n_zero_vol = 0
    gaps_gt_90 = 0
    suspicious_gaps = 0
    max_gap_h = 0.0
    prev_ts: int | None = None
    nonzero_vols: list[float] = []

    with csv_path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(_CSV_HEADER)
        for ts, o, h, low, c, vol in rows:
            if prev_ts is not None:
                if ts <= prev_ts:
                    n_nonmono += 1
                gap_h = (ts - prev_ts) / _HOUR_MS
                if gap_h > 1.5:
                    gaps_gt_90 += 1
                if gap_h > _SUSPICIOUS_GAP_H:
                    suspicious_gaps += 1
                max_gap_h = max(max_gap_h, gap_h)
            prev_ts = ts

            if min(o, h, low, c) <= 0:
                n_nonpositive += 1
            if h < low:
                n_high_lt_low += 1

            if has_volume:
                v = float(vol)
                amount = v * float(c)
            else:
                v = 0.0
                amount = 0.0
            if v == 0.0:
                n_zero_vol += 1
            elif has_volume:
                nonzero_vols.append(v)

            writer.writerow([_iso(ts), o, h, low, c, v, amount])

    n = len(rows)
    start = _iso(rows[0][0]) if n else None
    end = _iso(rows[-1][0]) if n else None
    span_days = (rows[-1][0] - rows[0][0]) / (_HOUR_MS * 24) if n > 1 else 0.0
    pct_zero_vol = (n_zero_vol / n * 100.0) if n else 0.0
    vol_step = _volume_step_ratio(nonzero_vols)
    enough = n >= _MIN_SAMPLE_BARS

    if n == 0:
        flags.append("EMPTY — no candles for this symbol")
    elif not enough:
        flags.append(f"TOO_FEW_BARS ({n} < {_MIN_SAMPLE_BARS}; cannot form one window)")
    elif n < _HEALTHY_BARS:
        flags.append(f"THIN_HISTORY ({n} < {_HEALTHY_BARS}; few training samples)")
    if n_nonpositive:
        flags.append(f"NONPOSITIVE_PRICE x{n_nonpositive}")
    if n_high_lt_low:
        flags.append(f"HIGH_LT_LOW x{n_high_lt_low}")
    if n_nonmono:
        flags.append(f"NONMONOTONIC_TS x{n_nonmono}")
    if has_volume and n and pct_zero_vol > 50.0:
        flags.append(f"DEAD_VOLUME_CHANNEL ({pct_zero_vol:.0f}% zero on a volume asset)")
    if has_volume and (
        vol_step >= _VOLUME_STEP_RATIO or (0.0 < vol_step <= 1.0 / _VOLUME_STEP_RATIO)
    ):
        flags.append(
            f"VOLUME_STEP ({vol_step:.0f}× last/first-quartile median — units discontinuity)"
        )
    if not has_volume and n_zero_vol < n:
        flags.append("FX_UNEXPECTED_VOLUME (FX row carried nonzero volume; exported as 0)")
    if suspicious_gaps:
        flags.append(
            f"LARGE_GAPS x{suspicious_gaps} (>{_SUSPICIOUS_GAP_H:.0f}h; max {max_gap_h:.0f}h)"
        )

    return SymbolReport(
        symbol=bot_key,
        asset_class=meta.asset_class,
        has_volume=has_volume,
        csv_file=csv_path.name,
        n_rows=n,
        start=start,
        end=end,
        span_days=round(span_days, 1),
        n_gaps_gt_90min=gaps_gt_90,
        max_gap_hours=round(max_gap_h, 1),
        n_suspicious_gaps=suspicious_gaps,
        n_nonpositive_price=n_nonpositive,
        n_high_lt_low=n_high_lt_low,
        n_nonmonotonic_ts=n_nonmono,
        pct_zero_volume=round(pct_zero_vol, 1),
        volume_step_ratio=round(vol_step, 1),
        enough_for_training=enough,
        flags=flags,
    )


def _write_reports(reports: list[SymbolReport], out_dir: Path, db_path: Path) -> None:
    generated = datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S UTC")
    payload = {
        "generated_at": generated,
        "source_db": str(db_path.resolve()),
        "min_sample_bars": _MIN_SAMPLE_BARS,
        "healthy_bars": _HEALTHY_BARS,
        "symbols": [asdict(r) for r in reports],
    }
    (out_dir / "hygiene_report.json").write_text(json.dumps(payload, indent=2))

    lines = [
        "# Kronos fine-tune — Phase A data-export hygiene report",
        "",
        f"- Generated: {generated}",
        f"- Source DB: `{db_path.resolve()}`",
        f"- Symbols exported: {len(reports)}",
        f"- Total rows: {sum(r.n_rows for r in reports):,}",
        f"- Min bars to form one training window (lookback+predict+1): {_MIN_SAMPLE_BARS}",
        "",
        (
            "| symbol | class | vol | rows | start | end | span_d | gaps>90m | "
            "max_gap_h | %zero_vol | vstep× | train? | flags |"
        ),
        "|---|---|---|---:|---|---|---:|---:|---:|---:|---:|:--:|---|",
    ]
    for r in reports:
        flags = "; ".join(r.flags) if r.flags else "—"
        lines.append(
            f"| {r.symbol} | {r.asset_class} | {'Y' if r.has_volume else 'N'} | {r.n_rows} | "
            f"{r.start or '—'} | {r.end or '—'} | {r.span_days} | {r.n_gaps_gt_90min} | "
            f"{r.max_gap_hours} | {r.pct_zero_volume} | {r.volume_step_ratio} | "
            f"{'✓' if r.enough_for_training else '✗'} | {flags} |"
        )
    flagged = [r for r in reports if r.flags]
    lines += ["", "## Flagged symbols", ""]
    if flagged:
        for r in flagged:
            lines.append(f"- **{r.symbol}**: {'; '.join(r.flags)}")
    else:
        lines.append("None — all symbols passed the hygiene checks.")
    lines.append("")
    (out_dir / "hygiene_report.md").write_text("\n".join(lines))


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--db",
        default="kronos_ab/candles_snapshot.db",
        help="Path to a VACUUM INTO snapshot of the candle DB (NOT the live DB).",
    )
    p.add_argument("--out", default="kronos_ab/data", help="Output directory for per-symbol CSVs.")
    p.add_argument(
        "--symbols",
        nargs="*",
        help="Optional subset of bot_keys to export (default: full 28-symbol universe).",
    )
    args = p.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(
            f"DB snapshot not found: {db_path}\n"
            f"Create it first:  sqlite3 {_LIVE_DB_PATH} \"VACUUM INTO '{db_path}'\""
        )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = args.symbols or list(EODHD_UNIVERSE.keys())
    unknown = [s for s in symbols if s not in EODHD_UNIVERSE]
    if unknown:
        raise SystemExit(f"Unknown bot_keys (not in EODHD_UNIVERSE): {unknown}")

    con = _open_readonly(db_path)
    try:
        reports = [_export_symbol(con, s, out_dir) for s in symbols]
    finally:
        con.close()

    _write_reports(reports, out_dir, db_path)

    print(f"Exported {len(reports)} CSVs → {out_dir}/")
    print(f"Total rows: {sum(r.n_rows for r in reports):,}")
    flagged = [r for r in reports if r.flags]
    if flagged:
        print(f"\n{len(flagged)} symbol(s) flagged — see {out_dir}/hygiene_report.md:")
        for r in flagged:
            print(f"  {r.symbol:<9} {'; '.join(r.flags)}")
    else:
        print("No hygiene flags raised.")


if __name__ == "__main__":
    main()
