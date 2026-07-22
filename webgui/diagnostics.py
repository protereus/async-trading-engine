"""Host- and service-level diagnostics for the dashboard.

Reads four sources, all read-only:
  * ``systemctl show trading-bot`` — service uptime, PID, restart count
  * ``/proc/<pid>/status`` — bot process memory (VmRSS)
  * ``/proc/loadavg``, ``/proc/meminfo``, ``/proc/uptime``, ``/proc/stat`` —
    VPS load average, CPU usage, RAM, host uptime
  * ``journalctl -u trading-bot`` — recent events and errors

`journalctl` access requires the running user to be a member of the
``systemd-journal`` (or ``adm``) group.  If permission is denied, the
recent-events/errors sections return an empty list with an ``error`` key
so the UI can render a graceful message instead of failing.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# systemd service info
# ---------------------------------------------------------------------------

_SYSTEMCTL_PROPS = "MainPID,ActiveState,SubState,ActiveEnterTimestamp,NRestarts"


def systemd_status(unit: str) -> dict[str, Any]:
    """Return a dict of ``{pid, active, sub_state, uptime_s, restarts}``.

    Returns sensible defaults on any failure (unit missing, systemctl
    error, parse error) — the UI shows "n/a" instead of crashing.
    """
    out: dict[str, Any] = {
        "unit": unit,
        "pid": 0,
        "active": "unknown",
        "sub_state": "unknown",
        "uptime_s": None,
        "restarts": None,
    }
    try:
        proc = subprocess.run(
            ["systemctl", "show", unit, "--property", _SYSTEMCTL_PROPS],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return out
    if proc.returncode != 0:
        return out
    props: dict[str, str] = {}
    for line in proc.stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        props[k] = v
    with contextlib.suppress(ValueError):
        out["pid"] = int(props.get("MainPID", "0") or 0)
    out["active"] = props.get("ActiveState", "unknown")
    out["sub_state"] = props.get("SubState", "unknown")
    with contextlib.suppress(ValueError):
        out["restarts"] = int(props.get("NRestarts", "0") or 0)
    # ActiveEnterTimestamp: "Thu 2026-05-14 08:14:56 UTC" (empty if never active)
    ts = props.get("ActiveEnterTimestamp", "").strip()
    if ts:
        out["uptime_s"] = _parse_systemd_timestamp(ts)
    return out


def _parse_systemd_timestamp(ts: str) -> int | None:
    """Parse ``"Thu 2026-05-14 08:14:56 UTC"`` → seconds elapsed since.

    Returns None if parsing fails.  systemd may also emit BST/local
    timezone names; we trust ``date -d`` for the heavy lifting.
    """
    try:
        proc = subprocess.run(
            ["date", "-d", ts, "+%s"],
            capture_output=True,
            text=True,
            timeout=2,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    try:
        started_at = int(proc.stdout.strip())
    except ValueError:
        return None
    return max(0, int(time.time()) - started_at)


# ---------------------------------------------------------------------------
# /proc/<pid>/status — bot process memory
# ---------------------------------------------------------------------------


def process_memory_mb(pid: int) -> float | None:
    """VmRSS in MiB for *pid*, or None if /proc/<pid>/status is unreadable."""
    if pid <= 0:
        return None
    path = Path(f"/proc/{pid}/status")
    if not path.exists():
        return None
    try:
        for line in path.read_text().splitlines():
            if line.startswith("VmRSS:"):
                kb = int(line.split()[1])
                return round(kb / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        return None
    return None


# ---------------------------------------------------------------------------
# Host (VPS) metrics
# ---------------------------------------------------------------------------


def _read_meminfo() -> dict[str, int]:
    """Return raw kB values for the keys we care about."""
    out: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            k, _, rest = line.partition(":")
            v = rest.strip().split()
            if v:
                try:
                    out[k] = int(v[0])
                except ValueError:
                    continue
    except OSError:
        pass
    return out


def host_memory() -> dict[str, Any]:
    """RAM usage in GiB + percentage."""
    info = _read_meminfo()
    total_kb = info.get("MemTotal", 0)
    avail_kb = info.get("MemAvailable", 0)
    if total_kb == 0:
        return {"total_gib": 0.0, "used_gib": 0.0, "used_pct": 0.0}
    used_kb = total_kb - avail_kb
    return {
        "total_gib": round(total_kb / 1024 / 1024, 2),
        "used_gib": round(used_kb / 1024 / 1024, 2),
        "used_pct": round(used_kb / total_kb * 100, 1),
    }


def host_loadavg() -> dict[str, Any]:
    """1/5/15-min load averages + a normalised "load per CPU" view."""
    try:
        parts = Path("/proc/loadavg").read_text().split()
    except OSError:
        return {"load1": None, "load5": None, "load15": None, "cpu_count": 1}
    try:
        load1, load5, load15 = float(parts[0]), float(parts[1]), float(parts[2])
    except (IndexError, ValueError):
        return {"load1": None, "load5": None, "load15": None, "cpu_count": 1}
    cpu_count = _cpu_count()
    return {
        "load1": load1,
        "load5": load5,
        "load15": load15,
        "cpu_count": cpu_count,
        "load1_per_cpu_pct": round(load1 / cpu_count * 100, 1),
        "load5_per_cpu_pct": round(load5 / cpu_count * 100, 1),
        "load15_per_cpu_pct": round(load15 / cpu_count * 100, 1),
    }


def _cpu_count() -> int:
    try:
        return sum(
            1
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("cpu") and line[3:4].isdigit()
        )
    except OSError:
        return 1


def host_uptime_s() -> int | None:
    try:
        parts = Path("/proc/uptime").read_text().split()
        return int(float(parts[0]))
    except (OSError, ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# journalctl — recent events + errors
# ---------------------------------------------------------------------------

# Lines we consider "events worth surfacing" on the dashboard.
_EVENT_PATTERNS = (
    "TopK IG BUY",
    "TopK IG SELL",
    "TopK IG order rejected",
    "TopK stop-loss",
    "TopK selection:",
    "IG position closed",
    "Purging stale",
    "reconciled_external_close",
    "deferred — market closed",
    "Order rejected:",
    "Closing ",
)


def journalctl_events(unit: str, limit: int = 20) -> dict[str, Any]:
    """Return a list of ``{ts, message}`` for the last *limit* matching lines.

    Uses ``journalctl -u <unit> -n 2000 --no-pager -o json`` and post-filters
    in Python.  Returns ``{"events": [...], "error": "..."}`` on failure.
    """
    cmd = [
        "journalctl",
        "-u",
        unit,
        "-n",
        "2000",
        "--no-pager",
        "-o",
        "json",
        "--since",
        "24 hours ago",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"events": [], "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"events": [], "error": proc.stderr.strip()[:200] or "journalctl non-zero"}
    events: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = row.get("MESSAGE")
        if not isinstance(msg, str):
            continue
        if not any(p in msg for p in _EVENT_PATTERNS):
            continue
        ts_us = int(row.get("__REALTIME_TIMESTAMP", 0))
        events.append({"ts_ms": ts_us // 1000, "message": _short(msg)})
    events.reverse()  # newest first
    return {"events": events[:limit], "error": None}


# Python log-level tag → journald-style numeric priority for UI colour-coding.
# The bot pipes all logging to stdout, so journald stamps every line PRIORITY=6
# (info) regardless of the Python level. We therefore can't filter with
# ``journalctl -p`` — the real level lives in the message text (``[WARNING]``
# etc.), so we post-filter on the tag (the same approach as journalctl_events).
_LEVEL_PRIORITY = {
    "[CRITICAL]": 2,
    "[ERROR]": 3,
    "[WARNING]": 4,
}


def journalctl_errors(unit: str, limit: int = 15) -> dict[str, Any]:
    """Last *limit* lines tagged as ERROR/WARNING in the unit's journal."""
    try:
        proc = subprocess.run(
            [
                "journalctl",
                "-u",
                unit,
                "--no-pager",
                "-o",
                "json",
                "--since",
                "24 hours ago",
            ],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return {"errors": [], "error": f"{type(exc).__name__}: {exc}"}
    if proc.returncode != 0:
        return {"errors": [], "error": proc.stderr.strip()[:200] or "journalctl non-zero"}
    errors: list[dict[str, Any]] = []
    for line in proc.stdout.splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        msg = row.get("MESSAGE")
        if not isinstance(msg, str):
            continue
        # The bot's Python level is embedded in the message text, not in the
        # journald PRIORITY field — derive it from the tag so the UI can
        # colour-code (and so warnings/errors are matched at all).
        prio = next((p for tag, p in _LEVEL_PRIORITY.items() if tag in msg), None)
        if prio is None:
            continue
        ts_us = int(row.get("__REALTIME_TIMESTAMP", 0))
        errors.append({"ts_ms": ts_us // 1000, "message": _short(msg), "priority": prio})
    errors.reverse()
    return {"errors": errors[:limit], "error": None}


_MAX_MESSAGE_LEN = 220


def _short(msg: str) -> str:
    """Trim huge HEARTBEAT-style messages to keep the UI compact."""
    msg = msg.replace("\n", " ").strip()
    if len(msg) > _MAX_MESSAGE_LEN:
        return msg[: _MAX_MESSAGE_LEN - 1] + "…"
    return msg


# ---------------------------------------------------------------------------
# Rerank progress — parse Kronos progress lines from the journal
# ---------------------------------------------------------------------------


# Matches the tqdm-format substring emitted by ``bot.kronos.progress`` lines, e.g.
# "Kronos 8%|=         | 10/120 [00:18<03:31,  1.92s/it]"
# (Source: src/bot/strategy/_kronos_progress.py; emission is throttled to ~5 s.)
_TQDM_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\["
    r"(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)s/it\]"
)


def read_rerank_status_raw(status_path: Path) -> dict[str, Any] | None:
    """Read ``rerank_status.json`` verbatim, regardless of ``in_progress``.

    ``RerankStatusWriter`` merges fields into one cached payload and rewrites
    the whole file on every update (``src/bot/state/rerank_status.py``), so
    fields like ``next_rerank_at`` set at the end of a rerank persist in the
    file after ``in_progress`` flips back to false. Use this (rather than
    :func:`rerank_status`) for fields that should stay visible between
    reranks, not just during one.

    Returns ``None`` if the file is missing or unreadable (older bot version,
    or webgui started before the bot ever ran a rerank).
    """
    try:
        with status_path.open("rb") as f:
            data: dict[str, Any] = json.loads(f.read())
    except (OSError, json.JSONDecodeError):
        return None
    return data


def rerank_status(status_path: Path) -> dict[str, Any] | None:
    """Read the bot-written ``rerank_status.json`` next to ``bot_state.json``.

    Replaces the old journald-scrape of Kronos tqdm lines (which only covered
    the inference itself, dropping the bar while the post-inference phases —
    signal history, correlation, sentiment, decay/sentiment-reversal evals,
    balance fetch, Telegram alert — were still running).  Now the bot writes
    every phase boundary to the file, so the dashboard always sees the truth.

    Returns ``None`` if the file is missing or unreadable, or when
    ``in_progress`` is false so the progress bar hides between reranks — the
    next-rerank timestamp is still exposed via the snapshot's
    ``service.next_rerank_at_ms`` field (see :func:`read_rerank_status_raw`)
    for the always-on display.
    """
    data = read_rerank_status_raw(status_path)
    if not data or not data.get("in_progress"):
        return None
    return data


def rerank_progress(unit: str) -> dict[str, Any] | None:
    """Legacy fallback — parse Kronos progress lines from the journal.

    Kept so a new webgui talking to an old bot (no ``rerank_status.json``)
    still shows *some* progress signal.  New deployments should rely on
    ``rerank_status`` above.

    Returns ``None`` if no progress line is found in the last 2 minutes (rerank
    not running, or it finished).  Otherwise returns the most-recent snapshot.
    Kronos emits a fresh bar (0/120 → 120/120) for every ``predict_batch`` call,
    of which there are up to 42 per rerank (forex + volume groups × 1 Pass-1 +
    20 Pass-2).  We don't try to count which call we're on — that requires
    knowing the schedule — we just show the current bar so the operator sees
    the bot is making progress.
    """
    cmd = [
        "journalctl",
        "-u",
        unit,
        "--since",
        "2 minutes ago",
        "--no-pager",
        "-o",
        "cat",
    ]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=4, check=False)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if proc.returncode != 0:
        return None
    last_match: re.Match[str] | None = None
    for line in proc.stdout.splitlines():
        # Each progress line is one journald entry post-_kronos_progress (no more
        # `\r`-merged frames), but `.finditer` is still cheap and a defensive
        # belt-and-braces against the kill-switch (KRONOS_TQDM_INTERCEPT=0)
        # reverting to upstream tqdm's overwrite behaviour.
        for m in _TQDM_RE.finditer(line):
            last_match = m
    if last_match is None:
        return None
    pct, cur, total, em, es, etam, etas, rate = last_match.groups()
    return {
        "percent": int(pct),
        "current": int(cur),
        "total": int(total),
        "elapsed_s": int(em) * 60 + int(es),
        "eta_s": int(etam) * 60 + int(etas),
        "rate_s_per_it": float(rate),
    }
