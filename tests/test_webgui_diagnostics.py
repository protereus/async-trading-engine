"""Tests for webgui/diagnostics.py's systemd/journalctl/proc parsing functions.

All of these are string/dict transforms over subprocess output or /proc
files with no test file exercising them today. Subprocess calls are mocked
with canned systemctl/journalctl output (mocking subprocess.run rather than
skipping it, since the failure/timeout branches are the only branch logic
worth covering); /proc reads are exercised via monkeypatched Path.read_text.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

import pytest

from webgui import diagnostics


def _fake_run(returncode: int = 0, stdout: str = "", stderr: str = "") -> Any:
    def _runner(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=returncode, stdout=stdout, stderr=stderr
        )

    return _runner


class TestSystemdStatus:
    def test_parses_active_service(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = (
            "MainPID=1234\n"
            "ActiveState=active\n"
            "SubState=running\n"
            "ActiveEnterTimestamp=Thu 2026-05-14 08:14:56 UTC\n"
            "NRestarts=2\n"
        )
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        monkeypatch.setattr(diagnostics, "_parse_systemd_timestamp", lambda ts: 3600)
        out = diagnostics.systemd_status("trading-bot")
        assert out["pid"] == 1234
        assert out["active"] == "active"
        assert out["sub_state"] == "running"
        assert out["restarts"] == 2
        assert out["uptime_s"] == 3600

    def test_never_active_leaves_uptime_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = (
            "MainPID=0\nActiveState=inactive\nSubState=dead\nActiveEnterTimestamp=\nNRestarts=0\n"
        )
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.systemd_status("trading-bot")
        assert out["pid"] == 0
        assert out["uptime_s"] is None

    def test_systemctl_nonzero_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1))
        out = diagnostics.systemd_status("trading-bot")
        assert out["active"] == "unknown"
        assert out["pid"] == 0

    def test_missing_systemctl_binary_returns_defaults(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _raise(*_a: object, **_k: object) -> Any:
            raise FileNotFoundError("systemctl")

        monkeypatch.setattr(subprocess, "run", _raise)
        out = diagnostics.systemd_status("trading-bot")
        assert out["active"] == "unknown"

    def test_timeout_returns_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> Any:
            raise subprocess.TimeoutExpired(cmd="systemctl", timeout=3)

        monkeypatch.setattr(subprocess, "run", _raise)
        out = diagnostics.systemd_status("trading-bot")
        assert out["active"] == "unknown"


class TestParseSystemdTimestamp:
    def test_parses_via_date_command(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fixed_now = 1_000_000
        monkeypatch.setattr(time, "time", lambda: fixed_now)
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=str(fixed_now - 3600) + "\n"))
        assert diagnostics._parse_systemd_timestamp("Thu 2026-05-14 08:14:56 UTC") == 3600

    def test_date_command_nonzero_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1))
        assert diagnostics._parse_systemd_timestamp("garbage") is None

    def test_unparseable_stdout_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout="not-a-number\n"))
        assert diagnostics._parse_systemd_timestamp("Thu 2026-05-14 08:14:56 UTC") is None

    def test_missing_date_binary_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> Any:
            raise FileNotFoundError("date")

        monkeypatch.setattr(subprocess, "run", _raise)
        assert diagnostics._parse_systemd_timestamp("Thu 2026-05-14 08:14:56 UTC") is None


class TestProcessMemoryMb:
    def test_reads_vmrss_from_status(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        proc_dir = tmp_path / "proc" / "1234"
        proc_dir.mkdir(parents=True)
        (proc_dir / "status").write_text("Name:\tpython\nVmRSS:\t204800 kB\nVmSize:\t500000 kB\n")

        real_path = Path

        def _fake_path(arg: str) -> Path:
            if arg == "/proc/1234/status":
                return proc_dir / "status"
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _fake_path)
        assert diagnostics.process_memory_mb(1234) == 200.0

    def test_nonpositive_pid_returns_none(self) -> None:
        assert diagnostics.process_memory_mb(0) is None
        assert diagnostics.process_memory_mb(-1) is None

    def test_missing_status_file_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_path = Path

        def _fake_path(arg: str) -> Path:
            if arg == "/proc/9999/status":
                return real_path("/nonexistent/9999/status")
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _fake_path)
        assert diagnostics.process_memory_mb(9999) is None

    def test_status_without_vmrss_line_returns_none(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        proc_dir = tmp_path / "proc" / "5555"
        proc_dir.mkdir(parents=True)
        (proc_dir / "status").write_text("Name:\tpython\n")
        real_path = Path

        def _fake_path(arg: str) -> Path:
            if arg == "/proc/5555/status":
                return proc_dir / "status"
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _fake_path)
        assert diagnostics.process_memory_mb(5555) is None


def _patch_proc_file(monkeypatch: pytest.MonkeyPatch, target: str, content: str) -> None:
    """Redirect diagnostics.Path(target) to a fake in-memory Path serving *content*."""
    real_path = Path

    class _FakeReadPath:
        def __init__(self, text: str) -> None:
            self._text = text

        def read_text(self) -> str:
            return self._text

        def exists(self) -> bool:
            return True

    def _fake_path(arg: str) -> Any:
        if arg == target:
            return _FakeReadPath(content)
        return real_path(arg)

    monkeypatch.setattr("webgui.diagnostics.Path", _fake_path)


class TestReadMeminfo:
    def test_extracts_known_keys(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_proc_file(
            monkeypatch,
            "/proc/meminfo",
            "MemTotal:       16384000 kB\nMemAvailable:    8192000 kB\nSwapTotal:       0 kB\n",
        )
        info = diagnostics._read_meminfo()
        assert info["MemTotal"] == 16384000
        assert info["MemAvailable"] == 8192000

    def test_missing_file_returns_empty_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_path = Path

        def _raise_path(arg: str) -> Any:
            if arg == "/proc/meminfo":

                class _Boom:
                    def read_text(self) -> str:
                        raise OSError("no such file")

                return _Boom()
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _raise_path)
        assert diagnostics._read_meminfo() == {}


class TestHostMemory:
    def test_computes_used_gib_and_pct(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            diagnostics,
            "_read_meminfo",
            lambda: {"MemTotal": 16 * 1024 * 1024, "MemAvailable": 4 * 1024 * 1024},
        )
        mem = diagnostics.host_memory()
        assert mem["total_gib"] == 16.0
        assert mem["used_gib"] == 12.0
        assert mem["used_pct"] == 75.0

    def test_zero_total_returns_zeroed_dict(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(diagnostics, "_read_meminfo", lambda: {})
        assert diagnostics.host_memory() == {"total_gib": 0.0, "used_gib": 0.0, "used_pct": 0.0}


class TestHostLoadavg:
    def test_computes_per_cpu_percentages(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_proc_file(monkeypatch, "/proc/loadavg", "2.0 1.0 0.5 1/200 12345\n")
        monkeypatch.setattr(diagnostics, "_cpu_count", lambda: 4)
        out = diagnostics.host_loadavg()
        assert out["load1"] == 2.0
        assert out["cpu_count"] == 4
        assert out["load1_per_cpu_pct"] == 50.0

    def test_missing_file_returns_none_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_path = Path

        def _raise_path(arg: str) -> Any:
            if arg == "/proc/loadavg":

                class _Boom:
                    def read_text(self) -> str:
                        raise OSError("no such file")

                return _Boom()
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _raise_path)
        out = diagnostics.host_loadavg()
        assert out == {"load1": None, "load5": None, "load15": None, "cpu_count": 1}


class TestCpuCount:
    def test_counts_numbered_cpu_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_proc_file(
            monkeypatch,
            "/proc/stat",
            "cpu  100 0 50 900 0 0 0 0 0 0\ncpu0 50 0 25 450\ncpu1 50 0 25 450\nintr 12345\n",
        )
        assert diagnostics._cpu_count() == 2

    def test_missing_file_returns_one(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_path = Path

        def _raise_path(arg: str) -> Any:
            if arg == "/proc/stat":

                class _Boom:
                    def read_text(self) -> str:
                        raise OSError("no such file")

                return _Boom()
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _raise_path)
        assert diagnostics._cpu_count() == 1


class TestHostUptimeS:
    def test_parses_first_field(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _patch_proc_file(monkeypatch, "/proc/uptime", "123456.78 98765.43\n")
        assert diagnostics.host_uptime_s() == 123456

    def test_missing_file_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        real_path = Path

        def _raise_path(arg: str) -> Any:
            if arg == "/proc/uptime":

                class _Boom:
                    def read_text(self) -> str:
                        raise OSError("no such file")

                return _Boom()
            return real_path(arg)

        monkeypatch.setattr("webgui.diagnostics.Path", _raise_path)
        assert diagnostics.host_uptime_s() is None


def _journal_line(message: str, ts_us: int = 1_700_000_000_000_000) -> str:
    return json.dumps({"MESSAGE": message, "__REALTIME_TIMESTAMP": str(ts_us)})


class TestJournalctlEvents:
    def test_filters_to_known_event_patterns(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = "\n".join(
            [
                _journal_line("TopK IG BUY EUR/USD size=1.0"),
                _journal_line("irrelevant heartbeat line"),
                _journal_line("TopK stop-loss: EUR/USD loss=1.20%"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.journalctl_events("trading-bot")
        assert out["error"] is None
        assert len(out["events"]) == 2
        # newest-first: the stop-loss line (last in the log) comes first
        assert "stop-loss" in out["events"][0]["message"]

    def test_respects_limit(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = "\n".join(_journal_line(f"Closing pos {i}") for i in range(5))
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.journalctl_events("trading-bot", limit=2)
        assert len(out["events"]) == 2

    def test_skips_malformed_json_lines(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = "not json\n" + _journal_line("TopK IG SELL EUR/USD size=1.0")
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.journalctl_events("trading-bot")
        assert len(out["events"]) == 1

    def test_nonzero_returncode_reports_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1, stderr="permission denied"))
        out = diagnostics.journalctl_events("trading-bot")
        assert out["events"] == []
        assert "permission denied" in out["error"]

    def test_missing_journalctl_binary_reports_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _raise(*_a: object, **_k: object) -> Any:
            raise FileNotFoundError("journalctl")

        monkeypatch.setattr(subprocess, "run", _raise)
        out = diagnostics.journalctl_events("trading-bot")
        assert out["events"] == []
        assert "FileNotFoundError" in out["error"]


class TestJournalctlErrors:
    def test_filters_by_level_tag(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = "\n".join(
            [
                _journal_line("[INFO] normal heartbeat"),
                _journal_line("[WARNING] margin ratio low"),
                _journal_line("[ERROR] IG session expired"),
                _journal_line("[CRITICAL] shutdown triggered"),
            ]
        )
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.journalctl_errors("trading-bot")
        assert out["error"] is None
        assert len(out["errors"]) == 3
        priorities = {e["priority"] for e in out["errors"]}
        assert priorities == {2, 3, 4}

    def test_nonzero_returncode_reports_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1, stderr="no journal"))
        out = diagnostics.journalctl_errors("trading-bot")
        assert out["errors"] == []
        assert "no journal" in out["error"]


class TestShort:
    def test_short_message_unchanged(self) -> None:
        assert diagnostics._short("short message") == "short message"

    def test_long_message_truncated_with_ellipsis(self) -> None:
        msg = "x" * 300
        out = diagnostics._short(msg)
        assert len(out) == diagnostics._MAX_MESSAGE_LEN
        assert out.endswith("…")

    def test_newlines_collapsed_to_spaces(self) -> None:
        assert diagnostics._short("line one\nline two") == "line one line two"


class TestRerankStatusRaw:
    def test_reads_json_file(self, tmp_path: Path) -> None:
        status_file = tmp_path / "rerank_status.json"
        status_file.write_text('{"in_progress": true, "phase": "inference"}')
        data = diagnostics.read_rerank_status_raw(status_file)
        assert data == {"in_progress": True, "phase": "inference"}

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert diagnostics.read_rerank_status_raw(tmp_path / "missing.json") is None

    def test_corrupt_json_returns_none(self, tmp_path: Path) -> None:
        status_file = tmp_path / "rerank_status.json"
        status_file.write_text("not json{{{")
        assert diagnostics.read_rerank_status_raw(status_file) is None


class TestRerankStatus:
    def test_returns_data_when_in_progress(self, tmp_path: Path) -> None:
        status_file = tmp_path / "rerank_status.json"
        status_file.write_text('{"in_progress": true, "phase": "inference"}')
        assert diagnostics.rerank_status(status_file) == {
            "in_progress": True,
            "phase": "inference",
        }

    def test_returns_none_when_not_in_progress(self, tmp_path: Path) -> None:
        status_file = tmp_path / "rerank_status.json"
        status_file.write_text('{"in_progress": false, "phase": "idle"}')
        assert diagnostics.rerank_status(status_file) is None

    def test_returns_none_when_file_missing(self, tmp_path: Path) -> None:
        assert diagnostics.rerank_status(tmp_path / "missing.json") is None


class TestRerankProgress:
    def test_parses_latest_tqdm_line(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = "Kronos 8%|=         | 10/120 [00:18<03:31,  1.92s/it]\n"
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.rerank_progress("trading-bot")
        assert out is not None
        assert out["percent"] == 8
        assert out["current"] == 10
        assert out["total"] == 120
        assert out["elapsed_s"] == 18
        assert out["eta_s"] == 3 * 60 + 31
        assert out["rate_s_per_it"] == pytest.approx(1.92)

    def test_no_matching_line_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout="nothing here\n"))
        assert diagnostics.rerank_progress("trading-bot") is None

    def test_nonzero_returncode_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(subprocess, "run", _fake_run(returncode=1))
        assert diagnostics.rerank_progress("trading-bot") is None

    def test_uses_last_match_when_multiple_present(self, monkeypatch: pytest.MonkeyPatch) -> None:
        stdout = (
            "Kronos 8%|=         | 10/120 [00:18<03:31,  1.92s/it]\n"
            "Kronos 50%|=====     | 60/120 [01:00<01:00,  1.00s/it]\n"
        )
        monkeypatch.setattr(subprocess, "run", _fake_run(stdout=stdout))
        out = diagnostics.rerank_progress("trading-bot")
        assert out is not None
        assert out["percent"] == 50
