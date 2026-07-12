"""Kronos fine-tune evaluation harness (Phases C & E).

Produces the IC/RankIC + post-cost metrics that gate the fine-tune A/B decision:
the zero-shot baseline and the fine-tuned model are each scored on the *same*
held-out test origins from the Phase B split manifest.

- ``predict``  — GPU: run a Kronos model over the manifest's test origins,
                 deriving the same signal the live bot derives, → predictions CSV.
- ``metrics``  — pure CPU: IC / RankIC / hit-rate / post-cost signal backtest.
- ``costs``    — per-asset-class round-trip cost config (realistic IG spreads).
- ``run_eval`` — orchestrate predict → metrics → results/<label>/.


"""
