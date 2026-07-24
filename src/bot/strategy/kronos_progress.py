"""Intercept Kronos's ``tqdm.trange`` progress bar; redirect through Python logging.

Kronos's autoregressive inference loop (``model.kronos.auto_regressive_inference``)
uses ``tqdm.trange`` to draw a 0→pred_len progress bar. tqdm overwrites with
``\\r`` carriage returns and unicode block characters. When systemd-journald
captures the bot's stdout, it accumulates every ``\\r``-separated update into a
single ``MESSAGE`` entry containing control bytes, which ``journalctl`` then
renders as ``[N.NK blob data]``. Worse, any other ``logger.info`` firing during
the rerank (notably the 1-minute HEARTBEAT) gets bundled inside the same
binary blob — the entry finally flushes on its first ``\\n``.

This module replaces ``tqdm.trange`` with a wrapper that:

1. Iterates the same underlying ``range`` — Kronos sees no behavioural change.
2. Throttles to one emission per ``mininterval`` seconds (plus first + last).
3. Emits each update through Python ``logging`` — one ``\\n``-terminated journald
   entry per progress step, with no carriage returns.
4. Preserves the exact substring matched by ``webgui/diagnostics._TQDM_RE`` so
   the dashboard's rerank progress bar continues to parse correctly.

Install before Kronos is imported (``from model import Kronos`` resolves
``trange`` at that point). Kill switch: set ``KRONOS_TQDM_INTERCEPT=0`` to skip
patching entirely, restoring upstream tqdm behaviour without a redeploy.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable, Iterator
from typing import Any, Final

import tqdm

logger = logging.getLogger("bot.kronos.progress")

_DEFAULT_MININTERVAL_S: Final[float] = 5.0
_BAR_WIDTH: Final[int] = 10
_DISABLE_ENV_VAR: Final[str] = "KRONOS_TQDM_INTERCEPT"

_original_trange: Any = None

# --- Progress fan-out for the dashboard's rerank_status.json ----------------
# Each ``predict_batch`` call constructs one ``_LoggingTrange`` instance
# (Kronos's ``auto_regressive_inference`` calls ``tqdm.trange(pred_len)``
# exactly once per call).  We count those starts so the dashboard can show
# overall progress across all 21–42 batches of a rerank instead of just the
# inner 0/120 → 120/120 bar of the current call.
_batch_counter: int = 0
_progress_callback: Callable[[int, dict[str, Any] | None], None] | None = None


def reset_counter() -> None:
    """Reset the per-rerank batch counter.  Call once before inference."""
    global _batch_counter
    _batch_counter = 0


def bump_batch(snapshot: dict[str, Any] | None = None) -> None:
    """Advance the per-rerank batch counter for a *non-verbose* ``predict_batch``
    call (the Pass-2 / variance pass, which uses plain ``range`` not ``trange``).

    Pass-1 calls advance the counter via ``_LoggingTrange.__iter__``; without this
    the dashboard's overall *N/total* bar would freeze through the entire silent
    Pass-2 (40 of the ~42 calls). ``snapshot`` lets the variance loop supply its
    own progress detail (sample N/total + ETA) so the dashboard's inner bar keeps
    animating through the variance pass instead of going dark; pass ``None`` to
    clear the inner bar (legacy behaviour).
    """
    global _batch_counter
    _batch_counter += 1
    if _progress_callback is not None:
        try:
            _progress_callback(_batch_counter, snapshot)
        except Exception:
            logger.debug("Kronos bump_batch callback failed", exc_info=True)


def set_progress_callback(cb: Callable[[int, dict[str, Any] | None], None] | None) -> None:
    """Register a callback invoked on every throttled progress emission.

    Callback receives ``(batch_index, snapshot)`` where ``snapshot`` mirrors
    the fields the dashboard expects (``current, total, elapsed_s, eta_s,
    rate_s_per_it``) for a verbose Pass-1 call, or ``None`` for a Pass-2 batch
    tick (no inner bar).  Failures inside the callback are swallowed so a buggy
    consumer can't break inference.
    """
    global _progress_callback
    _progress_callback = cb


class _LoggingTrange:
    """Drop-in replacement for ``tqdm.trange(...)`` that logs instead of drawing."""

    __slots__ = ("_logger", "_mininterval", "_range", "_t0", "_total", "_last_emit")

    def __init__(
        self,
        *args: int,
        logger_: logging.Logger = logger,
        mininterval: float = _DEFAULT_MININTERVAL_S,
        **_kwargs: Any,
    ) -> None:
        self._range = range(*args)
        self._total = len(self._range)
        self._logger = logger_
        self._mininterval = mininterval
        self._t0 = 0.0
        self._last_emit = 0.0

    def __iter__(self) -> Iterator[int]:
        global _batch_counter
        _batch_counter += 1
        self._t0 = time.monotonic()
        self._last_emit = self._t0
        for i, value in enumerate(self._range):
            now = time.monotonic()
            is_first = i == 0
            is_last = i == self._total - 1
            throttled = now - self._last_emit >= self._mininterval
            if is_first or is_last or throttled:
                self._emit(i, now)
                self._last_emit = now
            yield value

    def _emit(self, i: int, now: float) -> None:
        elapsed = now - self._t0
        rate = elapsed / i if i > 0 else 0.0
        eta = rate * (self._total - i) if i > 0 else 0.0
        pct = (100 * i) // self._total if self._total > 0 else 100
        filled = (pct * _BAR_WIDTH) // 100
        bar = "=" * filled + " " * (_BAR_WIDTH - filled)
        em, es = divmod(int(elapsed), 60)
        etam, etas = divmod(int(eta), 60)
        self._logger.info(
            "Kronos %d%%|%s| %d/%d [%02d:%02d<%02d:%02d, %.2fs/it]",
            pct,
            bar,
            i,
            self._total,
            em,
            es,
            etam,
            etas,
            rate,
        )
        if _progress_callback is not None:
            try:
                _progress_callback(
                    _batch_counter,
                    {
                        "current": i,
                        "total": self._total,
                        "elapsed_s": int(elapsed),
                        "eta_s": int(eta),
                        "rate_s_per_it": float(rate),
                        "label": "forecast",
                        "unit": "s/it",
                    },
                )
            except Exception:
                logger.debug("Kronos progress callback failed", exc_info=True)


def install(
    *,
    mininterval: float = _DEFAULT_MININTERVAL_S,
    logger_: logging.Logger | None = None,
) -> None:
    """Patch ``tqdm.trange`` to log instead of drawing. Idempotent.

    Honours ``KRONOS_TQDM_INTERCEPT=0`` (kill switch — leaves tqdm untouched).
    Call before any module imports ``trange`` from tqdm; in particular before
    Kronos's ``model.kronos`` is first imported.
    """
    if os.environ.get(_DISABLE_ENV_VAR, "1") == "0":
        logger.info("Kronos tqdm interceptor disabled via %s=0", _DISABLE_ENV_VAR)
        return

    global _original_trange
    target_logger = logger_ if logger_ is not None else logger
    if _original_trange is None:
        _original_trange = tqdm.trange

    def _factory(*args: int, **_kwargs: Any) -> _LoggingTrange:
        return _LoggingTrange(*args, logger_=target_logger, mininterval=mininterval)

    # Intentional monkey-patch: replace tqdm's overloaded `trange` with our
    # logging-friendly factory.  mypy can't model "swap an overload for a
    # wrapper" so the [assignment] error here is expected.
    tqdm.trange = _factory  # type: ignore[assignment]
    logger.info("Kronos tqdm interceptor installed (mininterval=%.1fs)", mininterval)


def uninstall() -> None:
    """Restore the original ``tqdm.trange``. No-op if ``install`` never ran."""
    global _original_trange
    if _original_trange is not None:
        tqdm.trange = _original_trange
        _original_trange = None
