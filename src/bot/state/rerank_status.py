"""Atomic JSON writer for the TopK rerank's live status.

The dashboard parsed Kronos tqdm lines out of journald to draw a progress
bar, which only covered the Kronos inference itself — the rerank loop in
``main._topk_rerank_loop`` then ran 5–6 more sequential phases
(signal_history write, correlation snapshot, sentiment gate, per-position
decay/sentiment-reversal evaluation with close orders, IG balance fetch,
Telegram alert) without emitting any tqdm output.  The dashboard bar
disappeared while the bot was still working.

This writer is the bot-side replacement: ``main`` calls ``update`` at
every phase boundary, and the webgui reads ``rerank_status.json`` once
per snapshot.  Same atomic-rename pattern as ``StateManager``.
"""

from __future__ import annotations

import contextlib
import logging
import os
import tempfile
from pathlib import Path
from typing import Any

import orjson

logger = logging.getLogger(__name__)


class RerankStatusWriter:
    """Caches the last-written payload and rewrites it atomically on each ``update``.

    A torn read from the webgui falls back to its own last-good cache, but
    using ``os.rename`` keeps the on-disk file consistent regardless.
    """

    def __init__(self, status_file: str | Path = "rerank_status.json") -> None:
        self._file = Path(status_file)
        self._payload: dict[str, Any] = {
            "in_progress": False,
            "phase": "idle",
        }

    def update(self, **fields: Any) -> None:
        """Merge *fields* into the cached payload and rewrite the file."""
        self._payload.update(fields)
        try:
            raw = orjson.dumps(self._payload, option=orjson.OPT_INDENT_2)
            # Unique temp name (not a fixed ``{file}.tmp``) so a stale/foreign
            # leftover — e.g. a root-owned tmp from a deploy — can't make the
            # write fail with EACCES on every call.  Same scheme as StateManager.
            target_dir = self._file.parent
            fd, tmp = tempfile.mkstemp(dir=target_dir, prefix=".rerank_status.", suffix=".tmp")
            try:
                # mkstemp forces 0o600; restore the umask-default 0o644 so the
                # webgui (read-only dashboard) can read the status regardless of
                # which user runs it.
                os.fchmod(fd, 0o644)
                with os.fdopen(fd, "wb") as f:
                    f.write(raw)
                os.replace(tmp, self._file)
            except Exception:
                with contextlib.suppress(FileNotFoundError):
                    os.remove(tmp)
                raise
        except Exception:
            logger.exception("Failed to write rerank status to %s", self._file)
