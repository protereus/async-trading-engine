#!/usr/bin/env python
"""Benchmark: sequential vs concurrent two-group Kronos inference.

Question this answers
---------------------
A rerank runs Kronos over two independent homogeneous groups (no-volume FX and
volume-bearing shares) *sequentially* in ``TopKStrategy._run_batch_inference``.
A single group's inference rarely saturates every core, so the groups are a
candidate for running *concurrently* to reclaim the idle cores and cut rerank
wall-time — **without changing the statistical output** (same calls, same
temperatures, same sample counts; only the execution order/cores change).

This harness drives the *real* inference path (``_prepare_asset`` →
``_build_timestamps`` → ``_run_inference_group`` → ``predict_batch``) two ways and
reports:

  * wall-time per mode (mean over ``--reps``),
  * **effective cores** = CPU-seconds / wall-seconds (via ``getrusage``), which
    is the thing that decides whether concurrency actually reclaims idle cores,
  * a **statistical** signal-equivalence check.  Kronos sampling is stochastic
    and unseeded in production, so sequential and concurrent (and two sequential
    runs) never match bit-for-bit.  We therefore run each mode ``--reps`` times
    and check that the per-symbol ``mean_return`` distributions agree within
    Monte-Carlo noise — i.e. concurrency introduces no *bias*.

Correctness note
----------------
``auto_regressive_inference`` runs under ``torch.no_grad()`` with all mutable
buffers local to the call, reading the shared model weights read-only — so two
threads calling ``predict_batch`` on one predictor is safe.  The only shared
mutable state is the global RNG (``torch.multinomial``); concurrent draws
interleave non-deterministically, which is exactly why we compare distributions,
not values.

Usage
-----
Quick scaling check (a couple of minutes)::

    uv run python scripts/bench_parallel_groups.py --fx 4 --shares 4 \
        --pred-len 24 --variance-samples 4 --context-bars 256 --reps 2

Full-size confirmation (representative of a real rerank; ~tens of minutes —
run it on an otherwise idle machine so it isn't fighting a running bot for
cores)::

    uv run python scripts/bench_parallel_groups.py --fx 12 --shares 16 \
        --pred-len 120 --variance-samples 20 --context-bars 400 --reps 3

``--threads N`` sets torch intra-op threads *per group* in concurrent mode (the
sequential baseline always runs at the torch default so it mirrors production).
A natural setting on a 4-core host is ``--threads 2`` (2 groups × 2 = 4 cores).
"""

from __future__ import annotations

import argparse
import logging
import os
import resource
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

# Make ``from model import Kronos`` resolve before importing the strategy's
# predictor loader.  Mirrors lifecycle.py; defaults to a ./Kronos checkout.
_KRONOS_DIR = os.environ.get("KRONOS_DIR", "./Kronos")
if _KRONOS_DIR not in sys.path:
    sys.path.insert(0, _KRONOS_DIR)

import torch  # noqa: E402  (after sys.path tweak)

from bot.core.models import Candle  # noqa: E402
from bot.strategy.topk_strategy import TopKConfig, TopKStrategy  # noqa: E402

_DEFAULT_THREADS = torch.get_num_threads()

# Synthetic share / FX names — kept off the real EODHD universe so nothing reads
# them as live symbols; grouping is decided by ``has_volume`` here, not by the
# strategy's _VOLUME_SYMBOLS table.
_FX_NAMES = [f"FX{i:02d}/USD" for i in range(32)]
_SHARE_NAMES = [f"EQ{i:02d}" for i in range(32)]


def _make_candles(symbol: str, n: int, has_volume: bool, seed: int) -> list[Candle]:
    """Geometric-random-walk 1h OHLC candles of length ``n``.

    Values are irrelevant to timing (which depends on shapes) and only need to be
    well-formed for the tokenizer; a GBM keeps prices positive and realistic.
    """
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0, 0.004, size=n)
    close = 100.0 * np.exp(np.cumsum(rets))
    # Build OHLC around each close with a small intrabar range.
    spread = np.abs(rng.normal(0.0, 0.002, size=n)) * close
    high = close + spread
    low = close - spread
    open_ = np.concatenate([[close[0]], close[:-1]])
    vol = np.abs(rng.normal(1_000_000, 200_000, size=n)) if has_volume else np.zeros(n)
    start_ms = 1_700_000_000_000  # fixed epoch; hourly spacing
    step_ms = 3_600_000
    return [
        Candle(
            timestamp=start_ms + i * step_ms,
            open=float(open_[i]),
            high=float(high[i]),
            low=float(low[i]),
            close=float(close[i]),
            volume=float(vol[i]),
            symbol=symbol,
            is_confirmed=True,
        )
        for i in range(n)
    ]


@dataclass
class Group:
    symbols: list[str]
    dfs: list[pd.DataFrame]
    x_ts: list[pd.Series]
    y_ts: list[pd.Series]
    pred_len: int
    entries: dict[str, float]


def _build_groups(strategy: TopKStrategy, fx: int, shares: int, n_bars: int) -> list[Group]:
    """Prepare the two homogeneous groups exactly as ``_run_batch_inference``."""
    cfg = strategy._config
    fx_syms = _FX_NAMES[:fx]
    share_syms = _SHARE_NAMES[:shares]
    groups: list[Group] = []
    for syms, has_vol, seed0 in ((fx_syms, False, 0), (share_syms, True, 1000)):
        if not syms:
            continue
        prepared: dict[str, tuple[pd.DataFrame, float]] = {}
        for j, s in enumerate(syms):
            candles = _make_candles(s, n_bars, has_vol, seed=seed0 + j)
            res = strategy._prepare_asset(candles, s, has_volume=has_vol)
            if res is None:
                raise SystemExit(f"_prepare_asset returned None for {s} (need ≥{n_bars} bars)")
            prepared[s] = res
        dfs = [prepared[s][0] for s in syms]
        min_len = min(len(d) for d in dfs)
        dfs = [d.iloc[-min_len:] for d in dfs]
        grp_pred_len = min(cfg.pred_len, min_len)
        ts_pairs = [strategy._build_timestamps(d, grp_pred_len) for d in dfs]
        x_ts = [pair[0] for pair in ts_pairs]
        y_ts = [pair[1] for pair in ts_pairs]
        groups.append(
            Group(
                symbols=syms,
                dfs=dfs,
                x_ts=x_ts,
                y_ts=y_ts,
                pred_len=grp_pred_len,
                entries={s: prepared[s][1] for s in syms},
            )
        )
    return groups


def _run_group(strategy: TopKStrategy, g: Group) -> dict[str, float]:
    """Run one group through the real inference path; return per-symbol mean_return."""
    result = strategy._run_inference_group(g.symbols, g.dfs, g.x_ts, g.y_ts, g.pred_len)
    if result is None:
        raise SystemExit("_run_inference_group returned None (Pass 1 failed)")
    point_preds, _var_closes = result
    out: dict[str, float] = {}
    for s in g.symbols:
        entry = g.entries[s]
        close = float(point_preds[s]["close"].iloc[-1])
        out[s] = (close - entry) / entry
    return out


def _cpu_seconds() -> float:
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_utime + ru.ru_stime


def _run_sequential(
    strategy: TopKStrategy, groups: list[Group], threads: int | None = None
) -> tuple[float, float, dict[str, float]]:
    """Run both groups serially.  ``threads=None`` keeps the torch default (the
    production baseline); passing ``threads`` runs the same serial path at a
    tuned intra-op count — the isolation arm that separates the thread-tuning
    effect from the parallelism of the concurrent mode."""
    if threads is not None:
        torch.set_num_threads(threads)
    try:
        c0, w0 = _cpu_seconds(), time.perf_counter()
        sigs: dict[str, float] = {}
        for g in groups:
            sigs.update(_run_group(strategy, g))
        wall, cpu = time.perf_counter() - w0, _cpu_seconds() - c0
    finally:
        if threads is not None:
            torch.set_num_threads(_DEFAULT_THREADS)
    return wall, cpu, sigs


def _run_concurrent(
    strategy: TopKStrategy, groups: list[Group], threads: int
) -> tuple[float, float, dict[str, float]]:
    torch.set_num_threads(threads)
    try:
        c0, w0 = _cpu_seconds(), time.perf_counter()
        with ThreadPoolExecutor(max_workers=len(groups)) as ex:
            results = list(ex.map(lambda g: _run_group(strategy, g), groups))
        wall, cpu = time.perf_counter() - w0, _cpu_seconds() - c0
    finally:
        torch.set_num_threads(_DEFAULT_THREADS)
    sigs: dict[str, float] = {}
    for r in results:
        sigs.update(r)
    return wall, cpu, sigs


def _summarise(label: str, walls: list[float], cpus: list[float]) -> dict[str, float]:
    w = float(np.mean(walls))
    c = float(np.mean(cpus))
    cores = c / w if w > 0 else 0.0
    print(
        f"  {label:<26} wall={w:7.1f}s (±{np.std(walls):4.1f})  "
        f"cpu={c:8.1f}s  eff_cores={cores:4.2f}"
    )
    return {"wall": w, "cpu": c, "eff_cores": cores}


def _equivalence(seq_runs: list[dict[str, float]], conc_runs: list[dict[str, float]]) -> None:
    """Check concurrency introduces no *bias* in the per-symbol mean_return.

    The signals are stochastic and unseeded, so sequential and concurrent never
    match bit-for-bit (nor do two sequential runs).  A real bias would have to
    survive Monte-Carlo noise, so we flag a symbol only when the across-rep mean
    gap is significant **both** in absolute terms (> 2× the pooled standard
    error) **and** relative terms (> 15 % of the signal magnitude) — that
    conjunction is what stops the test crying wolf at low rep counts.  We also
    hard-fail on any non-finite output, which would betray a broken concurrent
    orchestration rather than mere noise.
    """
    syms = sorted(seq_runs[0].keys())
    # Finiteness guard — a real bug (NaN/inf) is not "within MC noise".
    nonfinite = [
        s for s in syms for runs in (seq_runs, conc_runs) for r in runs if not np.isfinite(r[s])
    ]
    if nonfinite:
        print(f"\n  Signal equivalence: NON-FINITE outputs for {sorted(set(nonfinite))} — FAIL")
        return

    print("\n  Signal equivalence (mean_return, per symbol, across reps):")
    print(f"    {'symbol':<10} {'seq_mean':>10} {'conc_mean':>10} {'Δ':>10} {'tol':>10}  flag")
    n_flag = 0
    for s in syms:
        sv = np.array([r[s] for r in seq_runs])
        cv = np.array([r[s] for r in conc_runs])
        d = float(sv.mean() - cv.mean())
        se = (
            float(np.sqrt(sv.var(ddof=1) / len(sv) + cv.var(ddof=1) / len(cv)))
            if len(sv) > 1
            else 0.0
        )
        mag = max(abs(sv.mean()), abs(cv.mean()), 1e-6)
        tol = max(2.0 * se, 1e-4)
        # Require BOTH absolute (> tol) and relative (> 15 % of magnitude)
        # significance before calling a symbol biased.
        biased = abs(d) > tol and abs(d) > 0.15 * mag
        flag = "  <-- DIFFERS" if biased else ""
        if biased:
            n_flag += 1
        print(f"    {s:<10} {sv.mean():>10.5f} {cv.mean():>10.5f} {d:>10.5f} {tol:>10.5f}{flag}")
    verdict = (
        "PASS — concurrent signals within MC noise of sequential"
        if n_flag == 0
        else f"REVIEW — {n_flag} symbol(s) significant in both abs & rel terms"
    )
    print(f"\n  Equivalence verdict: {verdict}")
    if len(seq_runs) < 3:
        print("  (note: <3 reps — noise estimate is rough; use --reps 3+ to be confident)")


def _run_sweep(strategy: TopKStrategy, groups: list[Group], args: argparse.Namespace) -> None:
    """Sweep concurrent thread-counts/group to find the core-allocation sweet spot.

    One seq@default baseline, then concurrent at each thread count in --sweep.
    All share the single model load + warmup already done by the caller.
    """
    counts = [int(x) for x in args.sweep.split(",") if x.strip()]
    print(f"\nSWEEP: seq@{_DEFAULT_THREADS}thr baseline + concurrent at threads/group ∈ {counts}\n")

    def _mean(fn: Any) -> tuple[float, float]:
        walls, cpus = [], []
        for _ in range(args.reps):
            w, c, _s = fn()
            walls.append(w)
            cpus.append(c)
        return float(np.mean(walls)), float(np.mean(cpus))

    base_w, base_c = _mean(lambda: _run_sequential(strategy, groups))
    print(
        f"  sequential @{_DEFAULT_THREADS}thr (prod): "
        f"wall={base_w:7.1f}s  eff_cores={base_c / base_w:4.2f}"
    )

    rows: list[tuple[int, float, float]] = []
    for t in counts:
        w, c = _mean(lambda t=t: _run_concurrent(strategy, groups, t))
        rows.append((t, w, c))
        print(
            f"  concurrent @{t}thr/grp     : wall={w:7.1f}s  eff_cores={c / w:4.2f}  "
            f"speedup={base_w / w:4.2f}×"
        )

    best_t, best_w, _ = min(rows, key=lambda r: r[1])
    print("\n" + "=" * 72)
    print(
        f"BEST: {best_t} thr/group → {base_w / best_w:.2f}× vs the sequential "
        f"baseline (wall {base_w:.0f}s → {best_w:.0f}s at this bench size)"
    )
    print("=" * 72)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--fx", type=int, default=4, help="no-volume (FX) symbols in group A")
    ap.add_argument("--shares", type=int, default=4, help="volume symbols in group B")
    ap.add_argument("--pred-len", type=int, default=24)
    ap.add_argument("--variance-samples", type=int, default=4)
    ap.add_argument("--forecast-samples", type=int, default=10)
    ap.add_argument("--context-bars", type=int, default=256)
    ap.add_argument("--reps", type=int, default=2)
    ap.add_argument(
        "--threads", type=int, default=2, help="torch intra-op threads per group (concurrent mode)"
    )
    ap.add_argument("--mode", choices=["both", "sequential", "concurrent"], default="both")
    ap.add_argument(
        "--seq-isolation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also time serial execution at --threads to isolate thread-tuning from parallelism",
    )
    ap.add_argument(
        "--sweep",
        type=str,
        default="",
        help="comma-separated concurrent thread-counts/group to sweep, e.g. '1,2,3,4'. "
        "When set, runs one seq@default baseline then concurrent at each count "
        "(skips the isolation arm); finds the core-allocation sweet spot.",
    )
    ap.add_argument(
        "--quiet-kronos",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="silence per-call Kronos progress logging (--no-quiet-kronos to keep it)",
    )
    args = ap.parse_args()

    if args.quiet_kronos:
        logging.getLogger("bot.kronos.progress").setLevel(logging.WARNING)
        logging.getLogger("bot.strategy.topk_strategy").setLevel(logging.WARNING)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    cfg = TopKConfig(
        kronos_dir=_KRONOS_DIR,
        kronos_context_bars=args.context_bars,
        pred_len=args.pred_len,
        forecast_sample_count=args.forecast_samples,
        variance_pass_enabled=True,
        variance_sample_count=args.variance_samples,
    )
    strategy = TopKStrategy(cfg)

    print("Loading Kronos predictor…")
    strategy._load_predictor()

    print(
        f"\nConfig: fx={args.fx} shares={args.shares} pred_len={args.pred_len} "
        f"variance_samples={args.variance_samples} forecast_samples={args.forecast_samples} "
        f"context_bars={args.context_bars} reps={args.reps}\n"
        f"Box: {os.cpu_count()} cores, torch default threads={_DEFAULT_THREADS}, "
        f"concurrent threads/group={args.threads}\n"
    )

    groups = _build_groups(strategy, args.fx, args.shares, n_bars=args.context_bars + 8)
    print(
        "Groups: "
        + ", ".join(f"{['A(no-vol)', 'B(vol)'][i]}={len(g.symbols)}" for i, g in enumerate(groups))
    )

    print("\nWarmup (1 sequential pass, untimed)…")
    _run_sequential(strategy, groups)

    if args.sweep:
        _run_sweep(strategy, groups, args)
        return

    seq_walls: list[float] = []
    seq_cpus: list[float] = []
    seq_runs: list[dict[str, float]] = []
    seqt_walls: list[float] = []
    seqt_cpus: list[float] = []
    conc_walls: list[float] = []
    conc_cpus: list[float] = []
    conc_runs: list[dict[str, float]] = []
    run_seq = args.mode in ("both", "sequential")
    run_conc = args.mode in ("both", "concurrent")
    # Isolation arm: serial execution at the tuned thread count, only meaningful
    # when it differs from the default and we're already running the serial path.
    run_seqt = run_seq and args.seq_isolation and args.threads != _DEFAULT_THREADS

    for r in range(args.reps):
        print(f"\n— rep {r + 1}/{args.reps} —")
        if run_seq:
            w, c, s = _run_sequential(strategy, groups)
            seq_walls.append(w)
            seq_cpus.append(c)
            seq_runs.append(s)
            print(f"  sequential @{_DEFAULT_THREADS}thr : wall={w:7.1f}s  eff_cores={c / w:4.2f}")
        if run_seqt:
            w, c, _ = _run_sequential(strategy, groups, threads=args.threads)
            seqt_walls.append(w)
            seqt_cpus.append(c)
            print(f"  sequential @{args.threads}thr : wall={w:7.1f}s  eff_cores={c / w:4.2f}")
        if run_conc:
            w, c, s = _run_concurrent(strategy, groups, args.threads)
            conc_walls.append(w)
            conc_cpus.append(c)
            conc_runs.append(s)
            print(f"  concurrent @{args.threads}/grp : wall={w:7.1f}s  eff_cores={c / w:4.2f}")

    print("\n" + "=" * 72 + "\nRESULTS (mean over reps)")
    seq = (
        _summarise(f"sequential @{_DEFAULT_THREADS}thr (prod)", seq_walls, seq_cpus)
        if seq_walls
        else None
    )
    seqt = (
        _summarise(f"sequential @{args.threads}thr (tuned)", seqt_walls, seqt_cpus)
        if seqt_walls
        else None
    )
    conc = (
        _summarise(f"concurrent @{args.threads}/grp", conc_walls, conc_cpus) if conc_walls else None
    )
    if seq and conc:
        speedup = seq["wall"] / conc["wall"] if conc["wall"] > 0 else 0.0
        print(f"\n  TOTAL SPEEDUP (concurrent vs prod sequential): {speedup:.2f}×")
        # Attribute the win between thread-tuning and parallelism when we have
        # the isolation arm.
        if seqt:
            tune = seq["wall"] / seqt["wall"] if seqt["wall"] > 0 else 0.0
            par = seqt["wall"] / conc["wall"] if conc["wall"] > 0 else 0.0
            print(
                f"    ├─ thread-tuning ({_DEFAULT_THREADS}→{args.threads} thr, serial): {tune:.2f}×"
            )
            print(f"    └─ parallelism (concurrent at {args.threads} thr/grp):  {par:.2f}×")
        print(
            f"  → a full-size rerank scales roughly linearly: whatever your "
            f"sequential wall-time is, expect ≈ 1/{speedup:.2f} of it concurrent."
        )
        _equivalence(seq_runs, conc_runs)
    print("=" * 72)


if __name__ == "__main__":
    main()
