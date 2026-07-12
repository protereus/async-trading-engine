"""Per-asset-class round-trip trading cost for the post-cost backtest.

The decision metric for the fine-tune A/B is the **post-cost** edge — statistical
lift (IC) that dies after spreads is a no-go (the core external critique). Costs
are expressed as a **round-trip fraction of price** (entry half-spread + exit
half-spread), applied to each long trade's realised return.

⚠️ The defaults below are conservative PLACEHOLDERS. Pin them to the real IG
spread-bet spreads (improvement #2 / ``IG_LIVE_RISK_REFERENCE.md``) before
treating the post-cost numbers as a go/no-go — pass a JSON via ``--costs`` to
override. They are deliberately on the wide side so an edge that survives them
is unlikely to be flattered.
"""

from __future__ import annotations

import json
from pathlib import Path

# Round-trip cost as a fraction of price (e.g. 0.0002 = 2 bps round trip).
# Placeholders — see module docstring.
DEFAULT_COSTS: dict[str, float] = {
    "forex": 0.0002,  # ~1 bp half-spread on majors; minors/JPY crosses wider
    "metal": 0.0006,  # spot gold/silver spread bet
    "equity": 0.0010,  # US share DFB spreads are materially wider than FX
}
# Fallback when an asset_class isn't in the map.
FALLBACK_COST = 0.0010


def load_costs(path: str | None) -> dict[str, float]:
    """Return the cost map, overlaying a JSON file on the defaults when given."""
    costs = dict(DEFAULT_COSTS)
    if path:
        overrides = json.loads(Path(path).read_text())
        if not isinstance(overrides, dict):
            raise ValueError(
                f"--costs JSON must be an object of class->fraction, got {type(overrides)}"
            )
        costs.update({str(k): float(v) for k, v in overrides.items()})
    return costs


def round_trip_cost(asset_class: str, costs: dict[str, float]) -> float:
    """Round-trip cost fraction for ``asset_class`` (falls back when unmapped)."""
    return costs.get(asset_class, FALLBACK_COST)
