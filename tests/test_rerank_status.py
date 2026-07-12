"""Tests for the rerank status writer."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import orjson
import pytest

from bot.state.rerank_status import RerankStatusWriter


def _read(path: Path) -> dict[str, object]:
    return orjson.loads(path.read_bytes())  # type: ignore[no-any-return]


class TestRerankStatusWriter:
    def test_first_update_creates_file_with_defaults(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(in_progress=True, phase="reconcile")
        data = _read(f)
        assert data["in_progress"] is True
        assert data["phase"] == "reconcile"

    def test_update_merges_into_existing_payload(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(
            in_progress=True,
            phase="inference",
            started_at=1_700_000_000.0,
            batches_total=42,
            batches_done=0,
        )
        # Subsequent partial update preserves keys it didn't touch.
        w.update(phase="signal_history", batches_done=42)
        data = _read(f)
        assert data["phase"] == "signal_history"
        assert data["batches_done"] == 42
        # Untouched fields survive the merge.
        assert data["in_progress"] is True
        assert data["started_at"] == 1_700_000_000.0
        assert data["batches_total"] == 42

    def test_update_overwrites_none_explicitly(self, tmp_path: pytest.TempPathFactory) -> None:
        # Phase transitions intentionally null out current_batch — make sure
        # ``None`` is persisted, not silently dropped from the merge.
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(
            current_batch={
                "current": 50,
                "total": 120,
                "elapsed_s": 80,
                "eta_s": 110,
                "rate_s_per_it": 1.6,
            }
        )
        w.update(current_batch=None)
        data = _read(f)
        assert data["current_batch"] is None

    def test_default_payload_marks_idle(self, tmp_path: pytest.TempPathFactory) -> None:
        # An instance that has never been ``update``d should not write a file
        # at all — the dashboard treats a missing file the same as ``idle``.
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        RerankStatusWriter(str(f))
        assert not f.exists()

    def test_atomic_write_no_tmp_file_left(self, tmp_path: pytest.TempPathFactory) -> None:
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(in_progress=True, phase="reconcile")
        assert f.exists()
        # No stray temp file (unique-named .rerank_status.*.tmp) should survive.
        assert list(tmp_path.glob("*.tmp")) == []  # type: ignore[attr-defined]

    def test_failed_replace_cleans_up_tmp(self, tmp_path: pytest.TempPathFactory) -> None:
        # Simulate an os.replace failure (e.g. cross-device link) and verify
        # the tmp file is unlinked rather than left behind to confuse the
        # next operator who tails the directory.
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        with patch("bot.state.rerank_status.os.replace", side_effect=OSError("boom")):
            w.update(in_progress=True, phase="reconcile")
        assert not f.exists()
        assert list(tmp_path.glob("*.tmp")) == []  # type: ignore[attr-defined]

    def test_write_tolerates_foreign_tmp_leftover(self, tmp_path: pytest.TempPathFactory) -> None:
        # A leftover fixed-name tmp from the old scheme must not block writes:
        # the unique-name strategy never collides with it. (2026-06-04 regression.)
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        (tmp_path / "rerank_status.json.tmp").write_text("stale")  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(in_progress=True, phase="reconcile")
        assert _read(f)["phase"] == "reconcile"

    def test_partial_update_does_not_corrupt_previous_payload(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # The file on disk after a failed write should still be the last
        # known-good payload — os.rename is atomic on POSIX.
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(in_progress=True, phase="inference", batches_total=42)
        good = _read(f)
        with patch("bot.state.rerank_status.os.replace", side_effect=OSError("boom")):
            w.update(phase="correlation")
        # File still reflects the prior successful write.
        assert _read(f) == good

    def test_written_payload_is_valid_json(self, tmp_path: pytest.TempPathFactory) -> None:
        # Webgui reads via stdlib json.loads — make sure orjson's output is
        # interoperable.
        import json

        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(
            in_progress=True,
            phase="decay_eval",
            phase_started_at=1.5,
            current_batch={
                "current": 12,
                "total": 120,
                "elapsed_s": 20,
                "eta_s": 180,
                "rate_s_per_it": 1.8,
            },
        )
        with f.open() as fh:
            decoded = json.load(fh)
        assert decoded["phase"] == "decay_eval"
        assert decoded["current_batch"]["current"] == 12

    def test_file_perms_are_readable(self, tmp_path: pytest.TempPathFactory) -> None:
        # The dashboard runs as a different user via the webgui systemd unit;
        # the default umask on Linux gives 0o644 which is fine.  Just assert
        # the file is at least owner-readable so a missing read bit doesn't
        # silently silence the dashboard.
        f = tmp_path / "rerank_status.json"  # type: ignore[attr-defined]
        w = RerankStatusWriter(str(f))
        w.update(in_progress=True, phase="reconcile")
        mode = os.stat(f).st_mode & 0o777
        assert mode & 0o400, f"owner-read bit missing: {oct(mode)}"
        # The dashboard must stay readable even if the webgui ever runs as a
        # different user — mkstemp's 0o600 default would silently break that.
        assert mode & 0o044, f"group/other-read bit missing: {oct(mode)}"
