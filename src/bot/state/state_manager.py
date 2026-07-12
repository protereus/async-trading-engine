"""State manager: atomic JSON persistence for bot state.

Enables crash recovery and clean restart by saving BotState to disk.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
import time
from typing import TYPE_CHECKING

import orjson

from bot.core.models import BotState

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class StateManager:
    """Persists BotState to a JSON file using atomic writes.

    Atomic write strategy: write to a *unique* temp file in the same
    directory (``tempfile.mkstemp``), ``fsync`` it, then ``os.replace()``
    onto the real path.  On POSIX ``os.replace`` is atomic so a mid-write
    crash leaves the previous state intact.

    A unique temp name (rather than a fixed ``{state_file}.tmp``) makes the
    save immune to a stale/foreign leftover blocking it: a fixed name owned
    by another user (e.g. a root-owned ``bot_state.json.tmp`` left by a
    deploy) would make ``open(..., "wb")`` fail with EACCES on *every* save
    until manual cleanup.  See the 2026-06-04 ``PermissionError`` incident.
    """

    def __init__(self, state_file: str = "bot_state.json") -> None:
        self._state_file = state_file

    def save(self, state: BotState) -> bool:
        """Serialise and atomically write *state* to disk.

        Never raises (callers persist state on the shutdown/heartbeat paths and
        must not be derailed by a disk error) but returns ``True`` only when the
        atomic write actually succeeded, so callers can detect a failed persist
        instead of assuming success.
        """
        data = state.to_dict()
        try:
            raw = orjson.dumps(data, option=orjson.OPT_INDENT_2)
            target_dir = os.path.dirname(self._state_file) or "."
            fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".bot_state.", suffix=".tmp")
            try:
                # mkstemp forces 0o600; restore the umask-default 0o644 to match
                # the pre-existing file mode (the webgui dashboard reads this).
                os.fchmod(fd, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp, self._state_file)
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp)
                raise
            logger.debug("State saved to %s", self._state_file)
            return True
        except Exception:
            logger.exception("Failed to save state to %s", self._state_file)
            return False

    def load(self) -> BotState | None:
        """Load and deserialise state from disk.

        Returns None on first run (file not found).
        """
        if not os.path.exists(self._state_file):
            logger.info("No state file found at %s — starting fresh", self._state_file)
            return None

        try:
            with open(self._state_file, "rb") as f:
                data = orjson.loads(f.read())

            state = BotState.from_dict(data)

            # Log recovery summary
            age_s = int(time.time()) - (state.last_heartbeat // 1000 or 0)
            logger.info(
                "Recovered state from %s: %d positions, %d open orders, last heartbeat %ds ago",
                self._state_file,
                len(state.positions),
                len(state.open_orders),
                age_s,
            )
            return state

        except Exception:
            logger.exception("Failed to load state from %s — starting fresh", self._state_file)
            return None
