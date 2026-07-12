"""Pure (no-torch) windowing logic shared by the split-manifest builder
(`scripts/kronos_split.py`) and the offline evaluation harness
(`scripts/kronos_eval/`).

Keeping this in one place guarantees the manifest's sample counts equal the
counts a training/evaluation loader actually produces — the whole point of
pinning the split.

A *sample* is a sliding window of ``lookback + predict + 1`` bars starting at
row ``s``: a ``lookback`` bar context, then a ``predict`` bar target plus one
overlap bar. The window therefore spans rows ``[s, s + lookback + predict]``
(inclusive) and its **target region** is rows ``[s + lookback, s + lookback +
predict]``. A sample is assigned to a split by where its *target* lands; the
context may reach back across a boundary (serve-realistic, and leaks nothing
because the model never trains on the future target bars).
"""

from __future__ import annotations


def split_bounds(n: int, train_ratio: float, val_ratio: float) -> tuple[int, int]:
    """Return ``(val_start, test_start)`` row indices for an ``n``-bar series.

    Train = ``[0, val_start)``, val = ``[val_start, test_start)``,
    test = ``[test_start, n)``. Integer-floored so the manifest and loader agree.
    """
    val_start = int(n * train_ratio)
    test_start = int(n * (train_ratio + val_ratio))
    return val_start, test_start


def valid_start_indices(n: int, lookback: int, predict: int, lo: int, hi: int) -> list[int]:
    """Window start indices ``s`` whose target region lies within ``[lo, hi)``.

    A window fits the series when ``s + lookback + predict <= n - 1``; its target
    region ``[s + lookback, s + lookback + predict]`` must satisfy
    ``lo <= s + lookback`` and ``s + lookback + predict < hi``.
    """
    last_target = lookback + predict  # offset of the final (overlap) target bar from s
    max_s = n - last_target - 1  # window must fit: s + last_target <= n - 1
    out: list[int] = []
    for s in range(0, max_s + 1):
        tgt_start = s + lookback
        tgt_end = s + last_target
        if lo <= tgt_start and tgt_end < hi:
            out.append(s)
    return out
