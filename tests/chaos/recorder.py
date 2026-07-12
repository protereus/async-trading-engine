"""Structured results recorder for the chaos suite.

Every scenario reports metrics — durations, event counts, position deltas —
never raw payloads, tokens, account identifiers, or log lines.  The output
is safe to publish by construction: there is nothing to sanitise because
nothing sensitive is ever recorded.

Artifacts written at session end:

* ``docs/chaos/chaos_results_<UTC>.json`` — machine-readable run record
* ``docs/chaos/LATEST.md``                — human-readable summary table
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ScenarioResult:
    name: str
    fault: str
    params: dict[str, Any]
    passed: bool
    timeline_s: dict[str, float | None] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)
    positions: dict[str, Any] = field(default_factory=dict)
    notes: str = ""


class ChaosRecorder:
    """Accumulates scenario results and renders the run artifacts."""

    def __init__(self, repo_root: Path) -> None:
        self._root = repo_root
        self.environment: dict[str, Any] = {}
        self.scenarios: list[ScenarioResult] = []

    def record(self, result: ScenarioResult) -> None:
        self.scenarios.append(result)

    def _git_commit(self) -> str:
        try:
            out = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=self._root,
                capture_output=True,
                text=True,
                timeout=5,
                check=True,
            )
            return out.stdout.strip()
        except Exception:
            return "unknown"

    def write(self) -> Path | None:
        if not self.scenarios:
            return None
        out_dir = self._root / "docs" / "chaos"
        out_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

        payload = {
            "generated_utc": datetime.now(UTC).isoformat(timespec="seconds"),
            "engine_commit": self._git_commit(),
            "environment": self.environment,
            "scenarios": [vars(s) for s in self.scenarios],
        }
        json_path = out_dir / f"chaos_results_{stamp}.json"
        json_path.write_text(json.dumps(payload, indent=2) + "\n")

        (out_dir / "LATEST.md").write_text(self._render_md(payload))
        return json_path

    def _render_md(self, payload: dict[str, Any]) -> str:
        env = payload["environment"]
        lines = [
            "# Chaos suite — latest run",
            "",
            f"Generated {payload['generated_utc']} against commit "
            f"`{payload['engine_commit']}` (env: `{env.get('bot_env', '?')}`, "
            f"epics: {env.get('epics', [])}).",
            "",
            "Metrics only — no payloads, tokens, or account identifiers are recorded.",
            "See `docs/CHAOS_TESTING.md` for methodology and how to reproduce.",
            "",
            "| Scenario | Fault | Detected (s) | Recovered (s) | Ticks after | "
            "HB trips | Positions unchanged | Result |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for s in payload["scenarios"]:
            t = s["timeline_s"]
            det = t.get("detected")
            rec = t.get("recovered")
            lines.append(
                "| {name} | {fault} | {det} | {rec} | {ticks} | {trips} | {pos} | {res} |".format(
                    name=s["name"],
                    fault=s["fault"],
                    det="—" if det is None else f"{det:.1f}",
                    rec="—" if rec is None else f"{rec:.1f}",
                    ticks=s["counts"].get("ticks_after", 0),
                    trips=s["counts"].get("heartbeat_trips", 0),
                    pos="yes" if s["positions"].get("unchanged") else "NO",
                    res="PASS" if s["passed"] else "FAIL",
                )
            )
        notes = [f"- **{s['name']}**: {s['notes']}" for s in payload["scenarios"] if s["notes"]]
        if notes:
            lines += ["", "## Notes", *notes]
        lines.append("")
        return "\n".join(lines)
