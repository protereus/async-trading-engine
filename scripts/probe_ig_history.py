"""Probe IG's /prices REST endpoint to answer: how far back can we backfill
USO / UNG candles, and at what resolution?

Question driving the script
---------------------------
For the D2 phase of  we need to know:

1. Does IG's REST history actually return MINUTE-granularity bars for the
   futures-DFB EPICs we care about (WTI Crude, Nat Gas)?
2. How many bars per call?  How far back can a single response reach?
3. How much of the 10 000-points-per-week allowance does an N-bar fetch
   actually cost?
4. Are the bar timestamps consistently UTC, and do they align cleanly to
   hour boundaries (so 60 × MINUTE bars = 1 × HOUR bar)?

This is purely a research probe.  It does NOT modify any bot state and
does NOT trigger any orders.  Run once, eyeball the output, archive the
report.

Usage:
    uv run python scripts/probe_ig_history.py [--limit N] [--epics EPIC1,EPIC2]

By default it probes WTI Crude, Nat Gas, EUR/USD (FX baseline), and the
S&P 500 cash index (cash baseline), at five resolutions, with limit=50.
That spends ~50 × 5 × 4 = 1 000 allowance points.  Tune --limit down
during the day if quota is tight.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.config import BotConfig  # noqa: E402
from bot.core.models import ExchangeError  # noqa: E402
from bot.execution.ig_client import IGClient  # noqa: E402

_DEFAULT_EPICS = {
    "CC.D.CL.USS.IP": "WTI Crude DFB (IG underlying for USO)",
    "CC.D.NG.USS.IP": "Nat Gas DFB (IG underlying for UNG)",
    "CS.D.EURUSD.TODAY.IP": "EUR/USD (FX baseline)",
    "IX.D.SPTRD.DAILY.IP": "S&P 500 cash (cash-index baseline)",
}

# IG resolution strings to probe + the bot's timeframe key for fetch_ohlcv().
_RESOLUTIONS: list[tuple[str, str]] = [
    ("1m", "MINUTE"),
    ("5m", "MINUTE_5"),
    ("15m", "MINUTE_15"),
    ("30m", "MINUTE_30"),
    ("1h", "HOUR"),
    ("1d", "DAY"),
]


async def probe_one(
    client: IGClient,
    epic: str,
    timeframe: str,
    resolution_label: str,
    limit: int,
) -> None:
    """Fetch ``limit`` bars and print a summary row."""
    label = f"{epic:<30} {resolution_label:<10}"
    try:
        candles = await client.fetch_ohlcv(epic, timeframe, limit=limit)
    except ExchangeError as exc:
        print(f"{label}  ERROR: {exc}")
        return
    except Exception as exc:
        print(f"{label}  EXCEPTION: {type(exc).__name__}: {exc}")
        return

    if not candles:
        print(f"{label}  (0 bars returned)  remaining_quota={client.datapoints_remaining}")
        return

    first = candles[0]
    last = candles[-1]
    first_iso = datetime.fromtimestamp(first.timestamp / 1000, UTC).isoformat()
    last_iso = datetime.fromtimestamp(last.timestamp / 1000, UTC).isoformat()
    span_s = (last.timestamp - first.timestamp) / 1000
    median_interval_s = span_s / max(1, len(candles) - 1) if len(candles) > 1 else 0.0
    print(
        f"{label}  n={len(candles):>3}  "
        f"first={first_iso}  last={last_iso}  "
        f"median_Δ={median_interval_s:>6.0f}s  "
        f"last_close={last.close:.4f}  "
        f"remaining_quota={client.datapoints_remaining}"
    )


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--limit",
        type=int,
        default=50,
        help=(
            "Bars per resolution probe (default: 50; total budget = limit × #epics × #resolutions)"
        ),
    )
    parser.add_argument(
        "--epics",
        help="Comma-separated EPIC list (default: WTI, NG, EUR/USD, SPX cash)",
    )
    parser.add_argument(
        "--resolution",
        help="Restrict to a single resolution label, e.g. MINUTE_15 (default: all)",
    )
    args = parser.parse_args()

    epics_to_probe = {e: e for e in args.epics.split(",")} if args.epics else dict(_DEFAULT_EPICS)
    resolutions = [
        (tf, label)
        for tf, label in _RESOLUTIONS
        if args.resolution is None or label == args.resolution
    ]

    config = BotConfig()
    client = IGClient(config)
    await client.connect()
    print(f"Connected: account={client._account_id}  env={config.bot_env}")
    total_budget = len(epics_to_probe) * len(resolutions) * args.limit
    print(
        f"Initial quota: {client.datapoints_remaining} / "
        f"(probing {len(epics_to_probe)} EPICs × {len(resolutions)} resolutions × "
        f"limit={args.limit} → up to {total_budget} points)\n"
    )

    try:
        for epic, description in epics_to_probe.items():
            print(f"━━━ {description}  ({epic}) ━━━")
            for tf, resolution_label in resolutions:
                await probe_one(client, epic, tf, resolution_label, args.limit)
                # Polite delay to spread the per-bucket rate limiter
                await asyncio.sleep(0.5)
            print()
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
