"""Tests for bot.macro.fred."""

from __future__ import annotations

import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bot.data.candle_db import CandleDB
from bot.macro.fred import (
    _TRACKED_SERIES,
    FREDClient,
    _date_str_to_ms,
    compute_macro_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def candle_db() -> CandleDB:
    """Fresh CandleDB on a temp file (autoclean by tempfile)."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
        path = tmp.name
    db = CandleDB(path)
    db.init_db()
    yield db
    db.close()
    Path(path).unlink(missing_ok=True)


class _FakeResponse:
    def __init__(self, status: int, body: Any) -> None:
        self.status = status
        self._body = body

    async def json(self) -> Any:
        assert isinstance(self._body, dict)
        return self._body

    async def text(self) -> str:
        return str(self._body)

    async def __aenter__(self) -> _FakeResponse:
        return self

    async def __aexit__(self, *exc: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# _date_str_to_ms + _parse_observations
# ---------------------------------------------------------------------------


def test_date_str_to_ms_is_utc_midnight() -> None:
    ts = _date_str_to_ms("2026-01-15")
    dt = datetime.fromtimestamp(ts / 1000, UTC)
    assert dt.year == 2026 and dt.month == 1 and dt.day == 15
    assert dt.hour == 0 and dt.minute == 0


def test_parse_observations_happy_path() -> None:
    data = {
        "observations": [
            {"date": "2026-05-01", "value": "100.5"},
            {"date": "2026-05-02", "value": "101.2"},
        ]
    }
    out = FREDClient._parse_observations("DTWEXBGS", data)
    assert out is not None
    assert len(out) == 2
    assert out[0][1] == pytest.approx(100.5)
    # observation_date_ms must be ascending in order of fetch
    assert out[0][0] < out[1][0]


def test_parse_observations_drops_missing_values() -> None:
    """FRED uses '.' for missing observations — they must be silently dropped."""
    data = {
        "observations": [
            {"date": "2026-05-01", "value": "100.5"},
            {"date": "2026-05-02", "value": "."},
            {"date": "2026-05-03", "value": "101.0"},
        ]
    }
    out = FREDClient._parse_observations("VIXCLS", data)
    assert out is not None
    assert len(out) == 2
    assert all(v > 0 for _, v in out)


def test_parse_observations_handles_malformed() -> None:
    assert FREDClient._parse_observations("X", "not a dict") is None
    assert FREDClient._parse_observations("X", {}) is None
    assert FREDClient._parse_observations("X", {"observations": "nope"}) is None


def test_parse_observations_skips_bad_rows() -> None:
    """Garbage rows alongside real ones should not crash."""
    data = {
        "observations": [
            {"date": "garbage", "value": "100"},
            {"date": "2026-05-01", "value": "not-a-number"},
            {"date": "2026-05-02", "value": "100.5"},
            {"no_date_key": True},
        ]
    }
    out = FREDClient._parse_observations("DTWEXBGS", data)
    assert out is not None
    assert len(out) == 1
    assert out[0][1] == pytest.approx(100.5)


# ---------------------------------------------------------------------------
# DB integration
# ---------------------------------------------------------------------------


def test_insert_macro_observations_roundtrip(candle_db: CandleDB) -> None:
    obs = [
        (_date_str_to_ms("2026-05-01"), 100.0),
        (_date_str_to_ms("2026-05-02"), 101.5),
        (_date_str_to_ms("2026-05-03"), 99.8),
    ]
    n = candle_db.insert_macro_observations("VIXCLS", obs)
    assert n == 3

    read = candle_db.get_macro_series("VIXCLS")
    assert len(read) == 3
    # Ascending order
    assert read[0][0] < read[1][0] < read[2][0]
    assert read[2][1] == pytest.approx(99.8)


def test_insert_macro_observations_replaces_revisions(candle_db: CandleDB) -> None:
    """FRED revises recent values — re-inserts must overwrite, not duplicate."""
    obs = [(_date_str_to_ms("2026-05-01"), 100.0)]
    candle_db.insert_macro_observations("DTWEXBGS", obs)
    revised = [(_date_str_to_ms("2026-05-01"), 100.5)]
    candle_db.insert_macro_observations("DTWEXBGS", revised)
    read = candle_db.get_macro_series("DTWEXBGS")
    assert len(read) == 1
    assert read[0][1] == pytest.approx(100.5)


def test_get_latest_macro_value(candle_db: CandleDB) -> None:
    obs = [
        (_date_str_to_ms("2026-05-01"), 100.0),
        (_date_str_to_ms("2026-05-02"), 105.0),
    ]
    candle_db.insert_macro_observations("DGS10", obs)
    latest = candle_db.get_latest_macro_value("DGS10")
    assert latest is not None
    assert latest[1] == pytest.approx(105.0)


def test_get_latest_macro_value_empty(candle_db: CandleDB) -> None:
    assert candle_db.get_latest_macro_value("UNKNOWN_SERIES") is None


# ---------------------------------------------------------------------------
# compute_macro_snapshot — derived features
# ---------------------------------------------------------------------------


def test_compute_macro_snapshot_minimal(candle_db: CandleDB) -> None:
    obs = [(_date_str_to_ms(f"2026-05-{day:02d}"), 100.0 + day) for day in range(1, 11)]
    candle_db.insert_macro_observations("DTWEXBGS", obs)
    snapshot = compute_macro_snapshot(candle_db, series_ids=("DTWEXBGS",))
    assert "DTWEXBGS" in snapshot
    assert snapshot["DTWEXBGS"]["value"] == pytest.approx(110.0)
    # 1-day delta: 110 - 109 = 1
    assert snapshot["DTWEXBGS"]["delta_1d"] == pytest.approx(1.0)
    # 5-day delta: 110 - 105 = 5
    assert snapshot["DTWEXBGS"]["delta_5d"] == pytest.approx(5.0)
    # Fewer than 30 points → no zscore
    assert "zscore" not in snapshot["DTWEXBGS"]


def test_compute_macro_snapshot_zscore_with_enough_data(candle_db: CandleDB) -> None:
    # 60 daily observations with the last one a clear outlier
    obs = [
        (_date_str_to_ms(f"2026-{((day - 1) // 28) + 1:02d}-{((day - 1) % 28) + 1:02d}"), 100.0)
        for day in range(1, 60)
    ]
    obs.append((_date_str_to_ms("2026-04-15"), 200.0))  # latest = outlier
    candle_db.insert_macro_observations("VIXCLS", obs)
    snapshot = compute_macro_snapshot(candle_db, series_ids=("VIXCLS",))
    assert "zscore" in snapshot["VIXCLS"]
    # Outlier should give a large positive z-score
    assert snapshot["VIXCLS"]["zscore"] > 3.0


def test_compute_macro_snapshot_skips_missing_series(candle_db: CandleDB) -> None:
    snapshot = compute_macro_snapshot(candle_db, series_ids=("UNKNOWN",))
    assert snapshot == {}


def test_tracked_series_includes_canonical_set() -> None:
    required = {"DTWEXBGS", "VIXCLS", "DGS2", "DGS10", "T5YIE", "DCOILWTICO"}
    assert required.issubset(set(_TRACKED_SERIES))


# ---------------------------------------------------------------------------
# pull_all wiring
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_pull_all_writes_observations_for_each_tracked_series(
    candle_db: CandleDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Skip the per-series sleep
    monkeypatch.setattr("bot.macro.fred._REQUEST_SLEEP_S", 0.0)
    client = FREDClient("test-key", candle_db)
    # Inject a session whose .get() returns the same body for any series
    body = {
        "observations": [
            {"date": "2026-05-01", "value": "1.0"},
            {"date": "2026-05-02", "value": "2.0"},
        ]
    }
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(200, body))
    client._session = session
    counts = await client.pull_all()
    # All tracked series should have rows
    assert set(counts.keys()) == set(_TRACKED_SERIES)
    assert all(v == 2 for v in counts.values())
    # Roundtrip read from DB
    for sid in _TRACKED_SERIES:
        rows = candle_db.get_macro_series(sid)
        assert len(rows) == 2


@pytest.mark.asyncio
async def test_pull_all_handles_non_200_gracefully(
    candle_db: CandleDB, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("bot.macro.fred._REQUEST_SLEEP_S", 0.0)
    client = FREDClient("test-key", candle_db)
    session = MagicMock()
    session.get = MagicMock(return_value=_FakeResponse(429, {"error": "throttle"}))
    client._session = session
    counts = await client.pull_all()
    assert all(v == 0 for v in counts.values())


def test_client_rejects_empty_api_key(candle_db: CandleDB) -> None:
    with pytest.raises(ValueError):
        FREDClient("", candle_db)
