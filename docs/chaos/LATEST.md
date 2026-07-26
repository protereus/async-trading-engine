# Chaos suite — latest run

Generated 2026-07-12T09:04:07+00:00 against commit `ddac700` (env: `demo`, epics: ['CS.D.BITCOIN.TODAY.IP', 'CS.D.ETHUSD.TODAY.IP']).

Metrics only — no payloads, tokens, or account identifiers are recorded.
See `docs/CHAOS_TESTING.md` for methodology and how to reproduce.

> **Corrections to this record (added after the run).** The measurements below
> are unmodified, but two labelling defects in the harness that produced them
> have since been fixed, and this run predates the fix:
>
> 1. The `hard_cut` fault is recorded as *"proxy disabled 12s"*. It was not.
>    The scenario passed `fault_hold_s=0.0` and the harness clears the fault
>    the instant detection fires, so the proxy was disabled for **9.4 s** here
>    (the recorded `detected` time), not a fixed 12 s. The `12` was an unused
>    parameter that reached the label. The scenario is now labelled
>    *"proxy disabled until heartbeat detection"*.
> 2. The `baseline` row reads *"Ticks after 0"*. That scenario records
>    `ticks_baseline_10s`, not `ticks_after`, and the renderer defaulted the
>    missing key to `0` — its own note (7 updates in 30 s) is the correct
>    figure. Absent counts now render as `—`.
>
> Neither defect changes a pass/fail outcome, and no recorded number has been
> edited. See `docs/CHAOS_TESTING.md` → "What this does not cover" for what
> these scenarios do and do not establish.

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
