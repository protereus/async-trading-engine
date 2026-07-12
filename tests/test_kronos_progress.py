"""Tests for the Kronos tqdm.trange interceptor."""

from __future__ import annotations

import logging
import re
from collections.abc import Iterator

import pytest
import tqdm

from bot.strategy import _kronos_progress
from bot.strategy._kronos_progress import _LoggingTrange, install, uninstall

# Mirrors webgui/diagnostics._TQDM_RE — the contract the wrapper must preserve
# so the dashboard's rerank progress bar keeps parsing.
_WEBGUI_TQDM_RE = re.compile(
    r"(\d+)%\|[^|]*\|\s*(\d+)/(\d+)\s*\["
    r"(\d+):(\d+)<(\d+):(\d+),\s*([\d.]+)s/it\]"
)


@pytest.fixture
def captured() -> Iterator[tuple[list[str], logging.Logger]]:
    """Capture records from a private test logger."""
    test_logger = logging.getLogger("bot.kronos.progress.test")
    test_logger.setLevel(logging.INFO)
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(self.format(record))

    handler = _Capture()
    handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger.addHandler(handler)
    try:
        yield records, test_logger
    finally:
        test_logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _restore_tqdm() -> Iterator[None]:
    """Make sure no test leaves tqdm.trange patched for the next one."""
    yield
    uninstall()


@pytest.fixture(autouse=True)
def _reset_progress_state() -> Iterator[None]:
    """Reset the module-global batch counter + callback between tests."""
    _kronos_progress.reset_counter()
    _kronos_progress.set_progress_callback(None)
    yield
    _kronos_progress.reset_counter()
    _kronos_progress.set_progress_callback(None)


class TestLoggingTrange:
    def test_iterates_int_arg(self) -> None:
        assert list(_LoggingTrange(5, mininterval=0.0)) == [0, 1, 2, 3, 4]

    def test_iterates_start_stop(self) -> None:
        assert list(_LoggingTrange(2, 6, mininterval=0.0)) == [2, 3, 4, 5]

    def test_iterates_start_stop_step(self) -> None:
        assert list(_LoggingTrange(0, 10, 2, mininterval=0.0)) == [0, 2, 4, 6, 8]

    def test_first_and_last_always_emit(self, captured: tuple[list[str], logging.Logger]) -> None:
        records, test_logger = captured
        list(_LoggingTrange(3, logger_=test_logger, mininterval=999.0))
        # i=0 (first) + i=2 (last) → 2 emissions, throttle suppresses i=1
        assert len(records) == 2

    def test_high_mininterval_throttles_middle(
        self, captured: tuple[list[str], logging.Logger]
    ) -> None:
        records, test_logger = captured
        list(_LoggingTrange(100, logger_=test_logger, mininterval=999.0))
        # 100 steps, but only first and last emit
        assert len(records) == 2

    def test_zero_mininterval_emits_every_step(
        self, captured: tuple[list[str], logging.Logger]
    ) -> None:
        records, test_logger = captured
        list(_LoggingTrange(10, logger_=test_logger, mininterval=0.0))
        assert len(records) == 10

    def test_emission_format_at_zero(self, captured: tuple[list[str], logging.Logger]) -> None:
        records, test_logger = captured
        next(iter(_LoggingTrange(120, logger_=test_logger, mininterval=0.0)))
        assert records[0].startswith("Kronos 0%|")
        assert "0/120" in records[0]
        assert "0.00s/it" in records[0]

    def test_emission_format_at_last(self, captured: tuple[list[str], logging.Logger]) -> None:
        records, test_logger = captured
        list(_LoggingTrange(3, logger_=test_logger, mininterval=999.0))
        last = records[-1]
        assert "2/3" in last
        # Last emission is at i=2 (out of 3); 100 * 2 // 3 = 66
        assert last.startswith("Kronos 66%|")

    def test_every_emission_matches_webgui_regex(
        self, captured: tuple[list[str], logging.Logger]
    ) -> None:
        """Locks the contract with webgui/diagnostics._TQDM_RE."""
        records, test_logger = captured
        list(_LoggingTrange(50, logger_=test_logger, mininterval=0.0))
        assert len(records) == 50
        for line in records:
            assert _WEBGUI_TQDM_RE.search(line), f"webgui regex didn't match: {line!r}"

    def test_no_carriage_returns_in_output(
        self, captured: tuple[list[str], logging.Logger]
    ) -> None:
        """The whole point of this module: no CR control bytes reach journald."""
        records, test_logger = captured
        list(_LoggingTrange(20, logger_=test_logger, mininterval=0.0))
        for line in records:
            assert "\r" not in line
            assert "\x0d" not in line

    def test_no_unicode_bar_chars(self, captured: tuple[list[str], logging.Logger]) -> None:
        """ASCII-only bar — keeps the journal payload pure 7-bit."""
        records, test_logger = captured
        list(_LoggingTrange(20, logger_=test_logger, mininterval=0.0))
        for line in records:
            assert line.isascii(), f"non-ASCII in: {line!r}"

    def test_zero_total_emits_nothing(self, captured: tuple[list[str], logging.Logger]) -> None:
        records, test_logger = captured
        assert list(_LoggingTrange(0, logger_=test_logger, mininterval=0.0)) == []
        assert records == []

    def test_swallows_tqdm_kwargs(self) -> None:
        """tqdm.trange accepts desc=, leave= etc — wrapper must not blow up."""
        result = list(_LoggingTrange(3, mininterval=0.0, desc="x", leave=False, ncols=80))
        assert result == [0, 1, 2]


class TestInstall:
    def test_install_replaces_tqdm_trange(self) -> None:
        original = tqdm.trange
        install()
        assert tqdm.trange is not original

    def test_install_idempotent_and_restorable(self) -> None:
        original = tqdm.trange
        install()
        install()
        install()
        assert tqdm.trange is not original
        uninstall()
        assert tqdm.trange is original

    def test_uninstall_without_install_is_safe(self) -> None:
        original = tqdm.trange
        uninstall()  # no prior install
        assert tqdm.trange is original

    def test_patched_trange_iterates_normally(self) -> None:
        install(mininterval=0.0)
        assert list(tqdm.trange(5)) == [0, 1, 2, 3, 4]

    def test_patched_trange_swallows_real_tqdm_kwargs(self) -> None:
        """Kronos doesn't pass kwargs, but tqdm.trange in the wild does."""
        install(mininterval=0.0)
        assert list(tqdm.trange(3, desc="test", leave=False, ncols=80)) == [0, 1, 2]

    def test_kill_switch_disables_install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("KRONOS_TQDM_INTERCEPT", "0")
        original = tqdm.trange
        install()
        assert tqdm.trange is original
        assert _kronos_progress._original_trange is None


class TestBatchCounter:
    """The overall N/total dashboard bar must advance through the silent Pass-2
    variance calls, not just the verbose Pass-1 trange calls (the 'stuck at 1/42'
    bug)."""

    def test_reset_and_bump(self) -> None:
        _kronos_progress.reset_counter()
        assert _kronos_progress._batch_counter == 0
        _kronos_progress.bump_batch()
        _kronos_progress.bump_batch()
        assert _kronos_progress._batch_counter == 2

    def test_bump_fires_callback_with_none_snapshot(self) -> None:
        calls: list[tuple[int, dict | None]] = []
        _kronos_progress.set_progress_callback(lambda idx, snap: calls.append((idx, snap)))
        _kronos_progress.bump_batch()
        _kronos_progress.bump_batch()
        assert calls == [(1, None), (2, None)]

    def test_bump_without_callback_is_safe(self) -> None:
        _kronos_progress.set_progress_callback(None)
        _kronos_progress.bump_batch()
        assert _kronos_progress._batch_counter == 1

    def test_trange_and_bump_share_one_counter(self) -> None:
        """Pass-1 (trange) + Pass-2 (bump) increment the same counter, so a full
        group advances it smoothly: 1 (Pass-1) then +1 per Pass-2 call."""
        list(_LoggingTrange(3, mininterval=0.0))  # Pass-1 call → 1
        _kronos_progress.bump_batch()  # Pass-2 call → 2
        _kronos_progress.bump_batch()  # Pass-2 call → 3
        assert _kronos_progress._batch_counter == 3

    def test_bump_callback_failure_is_swallowed(self) -> None:
        def _boom(idx: int, snap: dict | None) -> None:
            raise RuntimeError("consumer bug")

        _kronos_progress.set_progress_callback(_boom)
        _kronos_progress.bump_batch()  # must not raise
        assert _kronos_progress._batch_counter == 1
