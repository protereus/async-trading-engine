"""FastAPI dashboard server — read-only view of the live trading bot.

Entry point: ``uv run python -m webgui.server`` or via systemd
(``ExecStart=.../.venv/bin/python -m webgui.server``).

Bound to ``127.0.0.1:8080`` — operator accesses via SSH port forward.
Never exposed publicly.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from webgui.data import DashboardData

_HERE = Path(__file__).parent
_DEFAULT_STATE = _HERE.parent / "bot_state.json"
_DEFAULT_DB = _HERE.parent / "candles.db"

_state_path = Path(os.environ.get("WEBGUI_BOT_STATE", _DEFAULT_STATE))
_db_path = Path(os.environ.get("WEBGUI_CANDLE_DB", _DEFAULT_DB))

_data = DashboardData(_state_path, _db_path)
_templates = Jinja2Templates(directory=str(_HERE / "templates"))


def _datetime_filter(value: int | float) -> str:
    """Format a Unix epoch (seconds) as ``YYYY-MM-DD HH:MM UTC`` for templates."""
    return _dt.datetime.fromtimestamp(int(value), tz=_dt.UTC).strftime("%Y-%m-%d %H:%M UTC")


def _duration_filter(value: int | float | None) -> str:
    """Format a seconds count as a compact ``Xd Yh`` / ``Yh Zm`` / ``Zm Ws`` string."""
    if value is None:
        return "n/a"
    s = int(value)
    if s < 60:
        return f"{s}s"
    if s < 3600:
        return f"{s // 60}m {s % 60}s"
    if s < 86400:
        return f"{s // 3600}h {(s % 3600) // 60}m"
    return f"{s // 86400}d {(s % 86400) // 3600}h"


_templates.env.filters["datetime"] = _datetime_filter
_templates.env.filters["duration"] = _duration_filter

app = FastAPI(title="Trading Bot Dashboard")
logger = logging.getLogger(__name__)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    try:
        snap = _data.snapshot()
        return _templates.TemplateResponse(
            request,
            "index.html",
            {"snap": snap, "state_path": str(_state_path), "db_path": str(_db_path)},
        )
    except Exception:
        logger.error("API index failed", exc_info=True)
        return HTMLResponse("Internal server error", status_code=500)


@app.get("/api/snapshot")
async def api_snapshot() -> JSONResponse:
    try:
        return JSONResponse(_data.snapshot())
    except Exception:
        logger.error("API snapshot failed", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


@app.get("/api/signals")
async def api_signals() -> JSONResponse:
    try:
        return JSONResponse(
            {"signals": _data.latest_signals(), "accuracy": _data.signal_accuracy()}
        )
    except Exception:
        logger.error("API signals failed", exc_info=True)
        return JSONResponse({"error": "Internal server error"}, status_code=500)


def main() -> None:
    host = os.environ.get("WEBGUI_HOST", "127.0.0.1")
    port = int(os.environ.get("WEBGUI_PORT", "8080"))
    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    main()
