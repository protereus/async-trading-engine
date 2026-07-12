"""Long-running Lightstreamer CHART probe — answers "does LS emit cleanly when the
underlying market is closed?".

Question driving the script
---------------------------
For the D1 phase of  we need to know what
the IG Lightstreamer CHART:{epic}:1MINUTE channel actually emits across
the trading-hours boundary.  Two failure modes the aggregator would have
to handle:

1. **Stale repeats**: LS keeps sending the last 1-minute bar over and
   over while the market is closed.  We'd need to filter on `UTM`
   (update timestamp) or `CONS_END` to avoid writing duplicate hours.
2. **Silent gaps**: LS goes completely quiet — no updates at all until
   the market reopens.  Then the aggregator can just trust whatever
   arrives; no special handling needed.

The script subscribes to a small set of EPICs spanning different trading
calendars (24/5 FX, 23h commodities, 6.5h US cash session) and appends
every chart update to a CSV.  Leave it running overnight or across a
weekend and inspect the CSV to see which behaviour LS actually exhibits.

Usage:
    uv run python scripts/probe_ls_chart.py                       # foreground, Ctrl-C to stop
    uv run python scripts/probe_ls_chart.py --output journal/...  # custom file
    nohup uv run python scripts/probe_ls_chart.py &               # detach

The output file is appended-to, so re-running adds to the same log.
Each row is one chart update with a timestamp, EPIC, and the relevant
fields from the chart channel.  See ``_CSV_COLUMNS`` below.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import logging
import signal
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.config import BotConfig  # noqa: E402
from bot.execution.ig_client import IGClient  # noqa: E402

# Default EPIC mix — one symbol per trading calendar / instrument class so
# the resulting CSV captures the full range of behaviours:
#   - WTI Crude, Nat Gas: futures DFB, ~23h/day Mon-Fri (the D-target symbols)
#   - EUR/USD: 24/5 FX, no daily break (control: should always emit)
#   - FTSE 100: cash index, 8:00-16:30 UTC weekdays only (control: long
#     idle windows expected)
#   - S&P 500 cash: similar US weekday hours
_DEFAULT_EPICS: dict[str, str] = {
    "CC.D.CL.USS.IP": "WTI",
    "CC.D.NG.USS.IP": "NG",
    "CS.D.EURUSD.TODAY.IP": "EURUSD",
    "IX.D.FTSE.DAILY.IP": "FTSE",
    "IX.D.SPTRD.DAILY.IP": "SPX",
}

# CHART fields that matter for the aggregation question.  UTM is the
# Lightstreamer-server-side update timestamp (epoch ms); CONS_END is "1"
# when a 1-minute bar has finished, blank otherwise.  Both are critical
# for distinguishing a real new bar from a stale repeat.
_CHART_FIELDS = [
    "UTM",
    "BID_OPEN",
    "BID_HIGH",
    "BID_LOW",
    "BID_CLOSE",
    "OFR_CLOSE",
    "LTV",
    "CONS_END",
]

_CSV_COLUMNS = [
    "recv_ts_utc",  # ISO timestamp when the script saw the update
    "epic",
    "label",
    "utm_ms",  # LS-server-side update epoch ms (may lag wall clock)
    "utm_iso",
    "bid_open",
    "bid_high",
    "bid_low",
    "bid_close",
    "ofr_close",
    "ltv",
    "cons_end",  # "1" if this is the final tick of the 1-minute bar
    "is_duplicate_of_prev",  # True iff UTM matches the previous row for this EPIC
]

logger = logging.getLogger("probe_ls_chart")


class _Writer:
    """CSV appender + duplicate-UTM tracker.  Single thread; no lock needed."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._fh = None
        self._csv = None
        self._last_utm: dict[str, str] = {}  # epic → last UTM seen

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self._path.exists()
        self._fh = self._path.open("a", buffering=1, newline="")
        self._csv = csv.writer(self._fh)
        if new_file:
            self._csv.writerow(_CSV_COLUMNS)
            self._fh.flush()
        logger.info("Writing chart updates to %s", self._path)

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None
            self._csv = None

    def record(self, epic: str, label: str, fields: dict[str, Any]) -> None:
        utm_raw = fields.get("UTM") or ""
        is_dup = utm_raw == self._last_utm.get(epic, "")
        if utm_raw:
            self._last_utm[epic] = utm_raw
        try:
            utm_ms = int(utm_raw) if utm_raw else 0
        except ValueError:
            utm_ms = 0
        utm_iso = datetime.fromtimestamp(utm_ms / 1000, UTC).isoformat() if utm_ms > 0 else ""
        self._csv.writerow(
            [
                datetime.now(UTC).isoformat(),
                epic,
                label,
                utm_raw,
                utm_iso,
                fields.get("BID_OPEN", ""),
                fields.get("BID_HIGH", ""),
                fields.get("BID_LOW", ""),
                fields.get("BID_CLOSE", ""),
                fields.get("OFR_CLOSE", ""),
                fields.get("LTV", ""),
                fields.get("CONS_END", ""),
                "1" if is_dup else "0",
            ]
        )


class _ChartListener:
    """Lightstreamer SubscriptionListener — pushes parsed updates onto the queue."""

    def __init__(
        self,
        queue: asyncio.Queue[dict[str, Any]],
        loop: asyncio.AbstractEventLoop,
        label_by_epic: dict[str, str],
    ) -> None:
        self._queue = queue
        self._loop = loop
        self._labels = label_by_epic

    def onItemUpdate(self, update: Any) -> None:  # noqa: N802
        try:
            item_name = update.getItemName() or ""
            # Item name shape: "CHART:CC.D.CL.USS.IP:1MINUTE"
            parts = item_name.split(":")
            epic = parts[1] if len(parts) >= 2 else item_name
            fields = dict(update.getFields() or {})
            self._loop.call_soon_threadsafe(
                self._queue.put_nowait,
                {"epic": epic, "label": self._labels.get(epic, epic), "fields": fields},
            )
        except Exception:
            logger.exception("ChartListener error")

    def onSubscription(self) -> None:  # noqa: N802
        logger.info("Lightstreamer CHART subscription active")

    def onSubscriptionError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error("Lightstreamer subscription error %d: %s", code, message)


class _StatusListener:
    """Minimal connection-status logger so we see drop / reconnect events."""

    def onStatusChange(self, status: str) -> None:  # noqa: N802
        logger.info("LS status: %s", status)

    def onServerError(self, code: int, message: str) -> None:  # noqa: N802
        logger.error("LS server error %d: %s", code, message)


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--epics",
        help="Comma-separated EPIC list (default: WTI, NG, EUR/USD, FTSE, SPX)",
    )
    parser.add_argument(
        "--output",
        default=f"journal/ls_chart_probe_{datetime.now(UTC).strftime('%Y-%m-%d')}.csv",
        help="CSV output path (default: journal/ls_chart_probe_<today>.csv)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)sZ %(levelname)s %(name)s %(message)s",
    )

    label_by_epic = {e: e for e in args.epics.split(",")} if args.epics else dict(_DEFAULT_EPICS)

    config = BotConfig()
    client = IGClient(config)
    await client.connect()
    logger.info(
        "Connected: account=%s env=%s — subscribing to %d epics",
        client._account_id,
        config.bot_env,
        len(label_by_epic),
    )

    writer = _Writer(Path(args.output))
    writer.open()

    # Lightstreamer setup mirrors bot.data.ig_feed._connect_lightstreamer but
    # without the candle / trade / account complications.
    import lightstreamer.client as ls  # noqa: E402  (deferred import)

    ls_client = ls.LightstreamerClient(client._ls_endpoint, "DEFAULT")
    ls_client.connectionDetails.setUser(client._account_id)
    ls_client.connectionDetails.setPassword(f"CST-{client._cst}|XST-{client._xst}")
    ls_client.addListener(_StatusListener())

    queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
    loop = asyncio.get_running_loop()
    chart_items = [f"CHART:{epic}:1MINUTE" for epic in label_by_epic]
    chart_sub = ls.Subscription("MERGE", chart_items, _CHART_FIELDS)
    chart_sub.setDataAdapter("DEFAULT")
    chart_sub.addListener(_ChartListener(queue, loop, label_by_epic))
    ls_client.subscribe(chart_sub)
    ls_client.connect()

    stopping = asyncio.Event()

    def _stop(_sig: int = 0, _frame: Any = None) -> None:
        logger.info("Stopping probe (signal=%s)", _sig)
        stopping.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    start_ts = time.monotonic()
    update_count = 0
    duplicate_count = 0

    try:
        while not stopping.is_set():
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=10.0)
            except TimeoutError:
                # No update in 10s — log a heartbeat so a quiet stretch is visible
                logger.info(
                    "Heartbeat: %d updates received, %d duplicates, %.0fs uptime",
                    update_count,
                    duplicate_count,
                    time.monotonic() - start_ts,
                )
                continue
            epic = msg["epic"]
            label = msg["label"]
            fields = msg["fields"]
            writer.record(epic, label, fields)
            update_count += 1
            if update_count % 100 == 0:
                logger.info(
                    "%d updates received (last: %s UTM=%s bid_close=%s cons_end=%s)",
                    update_count,
                    label,
                    fields.get("UTM"),
                    fields.get("BID_CLOSE"),
                    fields.get("CONS_END"),
                )
    finally:
        writer.close()
        try:
            ls_client.disconnect()
        except Exception:
            logger.exception("LS disconnect failed")
        await client.close()
        logger.info(
            "Probe stopped — total updates=%d, uptime=%.0fs, csv=%s",
            update_count,
            time.monotonic() - start_ts,
            args.output,
        )


if __name__ == "__main__":
    asyncio.run(main())
