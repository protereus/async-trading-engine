# Chaos suite — latest run

Generated 2026-07-12T09:04:07+00:00 against commit `ddac700` (env: `demo`, epics: ['CS.D.BITCOIN.TODAY.IP', 'CS.D.ETHUSD.TODAY.IP']).

Metrics only — no payloads, tokens, or account identifiers are recorded.
See `docs/CHAOS_TESTING.md` for methodology and how to reproduce.

| Scenario | Fault | Detected (s) | Recovered (s) | Ticks after | HB trips | Positions unchanged | Result |
|---|---|---|---|---|---|---|---|
| baseline | none | — | — | 0 | 0 | yes | PASS |
| silent_stall | timeout toxic (downstream, indefinite) | 5.5 | 5.8 | 1 | 1 | yes | PASS |
| hard_cut | proxy disabled 12s (TCP severed) | 9.4 | 9.6 | 1 | 1 | yes | PASS |
| latency_spike | latency toxic 3000ms ± 500ms (downstream, 20s) | — | — | 2 | 0 | yes | PASS |
| slow_trickle | slicer toxic ~64B slices, 2ms gaps (downstream, 20s) | — | — | 5 | 0 | yes | PASS |

## Notes
- **baseline**: 7 updates in 30s through the proxy; 0 ambient heartbeat trip(s), 0 recovered — natural tick gaps on these instruments at this hour
- **hard_cut**: detected via heartbeat path
- **latency_spike**: rode out the latency without tripping
- **slow_trickle**: 10 updates arrived through sliced frames
