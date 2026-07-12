"""Tests for the webgui endpoint exception handling.

Guards the broad ``try/except Exception`` wrappers added to the three
dashboard endpoints: an unexpected error from ``DashboardData`` must yield
a generic HTTP 500 without leaking the exception message / traceback to the
client, while still logging it server-side with ``exc_info``.
"""

from __future__ import annotations

import asyncio
import logging

import pytest

import webgui.server as server

_SECRET = "leaked-internal-detail-/srv/trading/candles.db"


def _boom(*_args: object, **_kwargs: object) -> object:
    raise RuntimeError(_SECRET)


class TestSnapshotEndpoint:
    def test_returns_500_without_leaking_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server._data, "snapshot", _boom)
        resp = asyncio.run(server.api_snapshot())
        assert resp.status_code == 500
        body = resp.body.decode()
        assert _SECRET not in body
        assert "Internal server error" in body

    def test_logs_with_exc_info(
        self, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
    ) -> None:
        monkeypatch.setattr(server._data, "snapshot", _boom)
        with caplog.at_level(logging.ERROR, logger=server.logger.name):
            asyncio.run(server.api_snapshot())
        # Error is recorded server-side with the original exception attached.
        assert any(r.exc_info for r in caplog.records)
        assert _SECRET in caplog.text  # the real detail lives in the logs, not the response


class TestSignalsEndpoint:
    def test_returns_500_without_leaking_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(server._data, "latest_signals", _boom)
        resp = asyncio.run(server.api_signals())
        assert resp.status_code == 500
        body = resp.body.decode()
        assert _SECRET not in body
        assert "Internal server error" in body


class TestIndexEndpoint:
    def test_returns_500_without_leaking_detail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # ``snapshot()`` raises before ``request`` is touched, so a dummy is fine.
        monkeypatch.setattr(server._data, "snapshot", _boom)
        resp = asyncio.run(server.index(request=None))  # type: ignore[arg-type]
        assert resp.status_code == 500
        body = resp.body.decode()
        assert _SECRET not in body
        assert "Internal server error" in body
