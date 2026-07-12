"""Fetch scalingFactor for every EPIC in SYMBOL_EPIC_MAP and print pip_values.

scalingFactor from IG market details: 1 IG point = (1 / scalingFactor) native price units.
So pip_value = 1 / scalingFactor.

Usage:
    uv run python scripts/check_pip_values.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.config import BotConfig
from bot.data.eodhd_symbols import SYMBOL_EPIC_MAP
from bot.execution.ig_client import IGClient


async def check(client: IGClient) -> None:
    header = (
        f"{'Symbol':<12} {'EPIC':<40} {'scalingFactor':>14}"
        f" {'pip_value':>12} {'minSize':>8} {'minIncrement':>14}"
    )
    print(header)
    print("-" * 110)

    for symbol, epic in sorted(SYMBOL_EPIC_MAP.items()):
        try:
            data = await client.fetch_market_details(epic)
            instrument = data.get("instrument", {})
            dealing = data.get("dealingRules", {})
            scaling = instrument.get("scalingFactor", None)
            pip_value = (1.0 / scaling) if scaling else None
            min_size = dealing.get("minDealSize", {}).get("value", "?")
            min_inc = dealing.get("minControlledRiskStopDistance", {}).get("value", "?")
            print(
                f"{symbol:<12} {epic:<40} {str(scaling):>14} "
                f"{str(round(pip_value, 6)) if pip_value else 'N/A':>12} "
                f"{str(min_size):>8} {str(min_inc):>14}"
            )
        except Exception as exc:
            print(f"{symbol:<12} {epic:<40} {'ERROR':>14}  {str(exc)[:60]}")

        await asyncio.sleep(0.3)  # stay well inside rate limits


async def main() -> None:
    config = BotConfig()
    client = IGClient(config)
    try:
        await client.connect()
        print(f"Connected: account={client._account_id}  env={config.bot_env}\n")
        await check(client)
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(main())
