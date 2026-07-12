"""Operator tool: close live IG positions by dealId, EPIC, or all-of-them.

Covers the case of an orphan dealId on IG that the bot cannot self-close (for
example a ``confirm_order`` 404 that left no local take-profit state to drive
an exit), plus targeted or full manual close-outs.

DESTRUCTIVE — sends DELETE /positions/otc for every match. Use ``--dry-run``
first to confirm the selection before flipping the switch.

Usage (run as the bot's own user so the 0600 IG token cache at
``.ig_session_cache.json`` is readable):

    # List live positions, do nothing else
    uv run python scripts/close_stale_positions.py

    # Preview a targeted close (no DELETE sent)
    uv run python scripts/close_stale_positions.py --deal-id DIAAAAXJLLGSBAA --dry-run

    # Close one specific orphan dealId
    uv run python scripts/close_stale_positions.py --deal-id DIAAAAXJLLGSBAA

    # Close every position with a given EPIC (repeatable)
    uv run python scripts/close_stale_positions.py \\
        --epic CS.D.NZDUSD.TODAY.IP --epic CS.D.AUDUSD.TODAY.IP

    # Clean-slate full close (stop the bot first; use sparingly)
    uv run python scripts/close_stale_positions.py --all

Flag semantics:
  * No flags → list-only; nothing closed.
  * ``--deal-id ID`` (repeatable) and ``--epic EPIC`` (repeatable) compose as
    OR within a category and OR across categories.
  * ``--all`` is mutually exclusive with ``--deal-id`` / ``--epic``.
  * ``--dry-run`` prints the matches and exits before any DELETE.

The bot tolerates a parallel IG session (demo allows multiple); no need to
stop the bot for a single-position targeted close.  Stop the bot only for
``--all`` so the close storm does not race against a live rerank entry.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

from bot.config import BotConfig
from bot.execution.ig_client import IGClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("close_stale_positions")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Close IG positions by dealId, EPIC, or all-of-them.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--deal-id",
        action="append",
        default=[],
        metavar="ID",
        help="Close only this dealId. Repeatable.",
    )
    parser.add_argument(
        "--epic",
        action="append",
        default=[],
        metavar="EPIC",
        help="Close every position with this EPIC. Repeatable.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Close every open position. Mutually exclusive with --deal-id/--epic.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the matches and exit without sending DELETE.",
    )
    args = parser.parse_args()
    if args.all and (args.deal_id or args.epic):
        parser.error("--all is mutually exclusive with --deal-id / --epic")
    return args


async def _fetch_positions(client: IGClient, banner: str) -> list[dict[str, Any]]:
    data = await client._get("/positions", version="2", authenticated=True)
    positions: list[dict[str, Any]] = data.get("positions", [])
    logger.info("%s — %d open position(s):", banner, len(positions))
    for entry in positions:
        pos = entry["position"]
        mkt = entry["market"]
        logger.info(
            "  %s  %s %s size=%s level=%s",
            pos["dealId"],
            mkt.get("epic"),
            pos.get("direction"),
            pos.get("size"),
            pos.get("level"),
        )
    return positions


def _select(
    positions: list[dict[str, Any]],
    *,
    deal_ids: list[str],
    epics: list[str],
    all_: bool,
) -> list[dict[str, Any]]:
    if all_:
        return list(positions)
    if not deal_ids and not epics:
        return []
    selected = []
    for entry in positions:
        pos = entry["position"]
        mkt = entry["market"]
        if deal_ids and pos.get("dealId") in deal_ids:
            selected.append(entry)
            continue
        if epics and mkt.get("epic") in epics:
            selected.append(entry)
    return selected


async def _close_one(client: IGClient, entry: dict[str, Any]) -> None:
    pos = entry["position"]
    mkt = entry["market"]
    deal_id = pos["dealId"]
    epic = mkt.get("epic", "")
    direction = pos.get("direction", "BUY")
    close_dir = "SELL" if direction == "BUY" else "BUY"
    size = float(pos.get("size", 0.0))
    logger.info(
        "Closing dealId=%s epic=%s open=%s close=%s size=%s",
        deal_id,
        epic,
        direction,
        close_dir,
        size,
    )
    try:
        result = await client.close_position(
            deal_id=deal_id, epic=epic, direction=close_dir, size=size
        )
        logger.info(
            "  closed %s @ %.4f (status=%s)",
            epic,
            result.average_price,
            result.status.value,
        )
    except Exception as exc:
        # IG occasionally returns 404 on /confirms/<ref> right after the
        # close DELETE succeeds — the position is closed, the read raced.
        # The AFTER fetch in main() is the authoritative check.
        logger.warning("  close call raised for %s: %s", epic, exc)


async def _run(args: argparse.Namespace) -> None:
    config = BotConfig()
    config.validate_config()
    client = IGClient(config)
    await client.connect()
    try:
        live = await _fetch_positions(client, "BEFORE")
        if not live:
            logger.info("No open positions — exiting.")
            return

        selected = _select(live, deal_ids=args.deal_id, epics=args.epic, all_=args.all)
        if not selected:
            if args.all or args.deal_id or args.epic:
                logger.info("Selection matched 0 of %d open position(s).", len(live))
            else:
                logger.info("List-only mode (no selection flag) — nothing to close.")
            return

        logger.info("Selected %d / %d position(s) to close.", len(selected), len(live))
        if args.dry_run:
            logger.info("Dry-run — no DELETE will be sent.")
            for entry in selected:
                pos = entry["position"]
                mkt = entry["market"]
                logger.info(
                    "  [DRY] would close dealId=%s epic=%s %s size=%s",
                    pos["dealId"],
                    mkt.get("epic"),
                    pos.get("direction"),
                    pos.get("size"),
                )
            return

        for entry in selected:
            await _close_one(client, entry)
        await _fetch_positions(client, "AFTER")
    finally:
        await client.close()


def main() -> None:
    asyncio.run(_run(_parse_args()))


if __name__ == "__main__":
    main()
