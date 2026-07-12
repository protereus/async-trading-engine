# Chaos Testing the Streaming Layer

Automates the streaming-resilience validation in
[`IG_LIVE_RISK_REFERENCE.md`](IG_LIVE_RISK_REFERENCE.md) §3 / §9.1. The unit
suite proves the recovery *logic* against mocks; this suite proves the real
`lightstreamer-client-lib` + production `IGFeed` recover from real network
faults, injected deterministically instead of via hand-run `iptables` rules.

## Topology

```text
IGFeed (production code, unmodified)
  └─ TLS ──► toxiproxy 127.0.0.1:18443 ──► <IG demo Lightstreamer host>:443
```

- **Streaming only** goes through [toxiproxy](https://github.com/Shopify/toxiproxy);
  REST (login, token refresh, position queries) stays direct, so heartbeat
  recovery's re-auth step keeps working while the stream is faulted.
- toxiproxy is a Layer-4 proxy: the TLS session runs end-to-end between the
  SDK and IG. The harness relaxes only certificate *hostname* matching
  (IG's genuine certificate cannot name `127.0.0.1`); chain verification
  stays on. This lives in test fixtures only — never in product code.
- The harness pins `IGClient.lightstreamer_endpoint` to the proxy for the
  test session, because heartbeat recovery re-reads the endpoint after its
  REST re-auth would otherwise route reconnects around the proxy.

## Scenarios

| Scenario | Fault | Real-world analogue | Must hold |
|---|---|---|---|
| `baseline` | none | — | stream flows through the proxy; ambient heartbeat trips are recorded as context (sparse instruments can gap past the threshold naturally) and must all recover |
| `silent_stall` | `timeout` toxic (data blackholed, sockets open) | server goes quiet — the "silent thread death" trigger the heartbeat exists for | detected ≤ heartbeat threshold, full §3.2 recovery, stream resumes |
| `hard_cut` | proxy disabled (TCP severed) | RST / infrastructure drop | SDK `DISCONNECTED` or heartbeat path detects; reconnect + resubscribe |
| `latency_spike` | 3 s ± 0.5 s downstream latency, 20 s | congested path | no crash; stream resumes (heartbeat trip allowed, must recover) |
| `slow_trickle` | slicer: ~64-byte fragments, 20 s | dribbling socket, fragmented frames | no errors, updates parse intact, stream resumes |

Every scenario additionally asserts the broker's **open-position count is
unchanged** — the "lose zero positions" criterion from §9.1.

## Running

Prerequisites: `toxiproxy-server` on `PATH` (or `TOXIPROXY_BIN`), IG **demo**
credentials in an env file, network access to IG demo.

```bash
uv run pytest -m chaos                          # credentials from ./.env
CHAOS_ENV_FILE=/path/to/.env uv run pytest -m chaos
```

The suite refuses to run when `BOT_ENV != demo`. It is excluded from default
`pytest` runs and from CI by `addopts = '-m "not chaos"'`; an explicit
`-m chaos` overrides that. A full run takes ~6 minutes and costs a few
`POST /session` refreshes plus a 30-candle backfill per epic — negligible
against IG demo quotas, but don't loop it unattended.

## Results

Each run writes metrics-only artifacts (no payloads, tokens, account
identifiers, or raw log lines — there is nothing to sanitise by design):

- `docs/chaos/chaos_results_<UTC>.json` — machine-readable record: per-scenario
  timelines (detection / recovery seconds), tick counts, heartbeat trips,
  reconnects, position-count deltas, plus the engine commit it ran against.
- `docs/chaos/LATEST.md` — the same as a summary table.

## What this does not cover

- **The ~2 h server force-drop** documented in §3.1 is time-dependent; it
  needs a soak run (hours, randomized kill schedule) rather than a scenario.
  The suite's faults reproduce the *mechanism* (silent stall / hard cut),
  not the schedule.
- **Host-level chaos** (`iptables` DROP/REJECT, `ss -K`) still exercises the
  OS TCP stack in ways a userspace proxy cannot; §9.1's manual run remains
  the final pre-live gate.
- Markets: run against 24/7 instruments (the default epics are crypto) or
  during market hours — a closed, tickless market starves the heartbeat and
  the suite cannot distinguish injected faults from natural silence.
