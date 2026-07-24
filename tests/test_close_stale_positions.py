"""Tests for scripts/close_stale_positions.py's position-selection logic.

The script is not a package module — load it via importlib so the tests
exercise exactly what ``uv run python scripts/close_stale_positions.py`` runs.

Covers ``_select``'s three selection modes (deal_id match, epic match,
--all) and the no-selector-flags-given empty-list case. ``_select`` is a
pure function (list[dict] in, list[dict] out) with no I/O — the DELETE
call this script sends live is exercised elsewhere (manual ops runbook),
not here.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "close_stale_positions.py"
_spec = importlib.util.spec_from_file_location("close_stale_positions", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
close_stale = importlib.util.module_from_spec(_spec)
sys.modules["close_stale_positions"] = close_stale
_spec.loader.exec_module(close_stale)


def _entry(deal_id: str, epic: str) -> dict[str, Any]:
    return {
        "position": {"dealId": deal_id, "direction": "BUY", "size": "1.0", "level": "1.1"},
        "market": {"epic": epic},
    }


_POSITIONS = [
    _entry("DEAL_1", "CS.D.EURUSD.TODAY.IP"),
    _entry("DEAL_2", "CS.D.GBPUSD.TODAY.IP"),
    _entry("DEAL_3", "CS.D.EURUSD.TODAY.IP"),
]


class TestSelectByDealId:
    def test_single_deal_id_matches_one(self) -> None:
        selected = close_stale._select(_POSITIONS, deal_ids=["DEAL_2"], epics=[], all_=False)
        assert [e["position"]["dealId"] for e in selected] == ["DEAL_2"]

    def test_multiple_deal_ids_match_each(self) -> None:
        selected = close_stale._select(
            _POSITIONS, deal_ids=["DEAL_1", "DEAL_3"], epics=[], all_=False
        )
        assert [e["position"]["dealId"] for e in selected] == ["DEAL_1", "DEAL_3"]

    def test_unknown_deal_id_matches_nothing(self) -> None:
        selected = close_stale._select(_POSITIONS, deal_ids=["NOPE"], epics=[], all_=False)
        assert selected == []


class TestSelectByEpic:
    def test_epic_matches_every_position_on_it(self) -> None:
        selected = close_stale._select(
            _POSITIONS, deal_ids=[], epics=["CS.D.EURUSD.TODAY.IP"], all_=False
        )
        assert [e["position"]["dealId"] for e in selected] == ["DEAL_1", "DEAL_3"]

    def test_deal_id_and_epic_compose_as_or(self) -> None:
        selected = close_stale._select(
            _POSITIONS,
            deal_ids=["DEAL_2"],
            epics=["CS.D.EURUSD.TODAY.IP"],
            all_=False,
        )
        assert [e["position"]["dealId"] for e in selected] == ["DEAL_1", "DEAL_2", "DEAL_3"]


class TestSelectAll:
    def test_all_true_returns_every_position_regardless_of_other_filters(self) -> None:
        selected = close_stale._select(_POSITIONS, deal_ids=[], epics=[], all_=True)
        assert selected == _POSITIONS

    def test_all_true_ignores_deal_id_and_epic(self) -> None:
        selected = close_stale._select(
            _POSITIONS, deal_ids=["DEAL_1"], epics=["CS.D.GBPUSD.TODAY.IP"], all_=True
        )
        assert selected == _POSITIONS


class TestSelectNoFilters:
    def test_no_selector_flags_returns_empty_list(self) -> None:
        selected = close_stale._select(_POSITIONS, deal_ids=[], epics=[], all_=False)
        assert selected == []

    def test_empty_positions_list_returns_empty_regardless_of_selector(self) -> None:
        selected = close_stale._select([], deal_ids=["DEAL_1"], epics=[], all_=True)
        assert selected == []
