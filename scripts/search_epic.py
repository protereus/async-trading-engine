"""Search IG REST API for EPIC codes by keyword.

Logs into IG demo and prints matching EPICs for each search term.
Use this to find the correct EPIC strings before adding them to
CANDLE_EPIC_MAP and IG_EPICS in the .env.

Usage:
    uv run python scripts/search_epic.py [TERM ...]

Examples:
    uv run python scripts/search_epic.py bitcoin ethereum solana
    uv run python scripts/search_epic.py BTC ETH SOL XRP BNB
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from bot.config import BotConfig
from bot.execution.ig_client import IGClient


async def search(terms: list[str]) -> None:
    config = BotConfig()  # loads from .env
    client = IGClient(config)

    try:
        await client.connect()
        print(f"Connected: account={client._account_id}  env={config.bot_env}\n")

        for term in terms:
            print(f"── Searching: {term!r}")
            try:
                results = await client.search_epic(term)
            except Exception as exc:
                print(f"   ERROR: {exc}\n")
                continue

            if not results:
                print("   No results.\n")
                continue

            for r in results:
                epic = r.get("epic", "?")
                name = r.get("instrumentName", "?")
                itype = r.get("instrumentType", "?")
                expiry = r.get("expiry", "?")
                print(f"   EPIC: {epic:<35} type={itype:<15} expiry={expiry:<12} name={name}")
            print()

    finally:
        await client.close()


def main() -> None:
    terms = sys.argv[1:] or ["bitcoin", "ethereum", "solana", "ripple", "bnb"]
    asyncio.run(search(terms))


if __name__ == "__main__":
    main()
