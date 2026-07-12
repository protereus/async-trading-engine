"""GPU: run a Kronos model over the Phase B manifest's test origins → predictions CSV.

Mirrors the live bot's dual-pass inference and signal derivation
(``TopKStrategy._run_batch_inference`` / ``_run_inference_group``) so the
offline A/B measures what the bot would actually trade:

* Pass 1 (point): one ``predict_batch`` at ``forecast_temperature`` (0.6),
  ``forecast_sample_count`` (10) averaged → the predicted OHLC path.
* Pass 2 (variance): ``variance_sample_count`` (20) single draws at
  ``variance_temperature`` (1.0) → std_return + direction_confidence.

For each test origin (window start ``s`` from the manifest): context = the
``lookback`` (400) bars ``[s, s+lookback)``, current price = the last context
close, and the realised target is measured ``H`` bars ahead (``H`` = ranking
horizon, default = ``pred_len`` 120). Origins of a symbol share the same context
length, so they batch through ``predict_batch`` together (chunked by
``--batch-size``).

Run on the RunPod pod (needs the model on GPU). Both the zero-shot baseline and
the fine-tuned model are scored with this script on the *same* manifest →
``predictions.csv`` for each, consumed by ``metrics.py``.

Simplification vs live (v1): the path-signal volatility filter
(``extract_path_signal`` → ``vol_ok``) is omitted, so ``tradeable`` uses the four
core gates (LONG, min_predicted_return, min_confidence, max_uncertainty). This is
slightly more permissive; noted in .
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from split_windows import valid_start_indices  # noqa: E402

_PRICE_COLS = ["open", "high", "low", "close"]


def _slice_index(horizon: int, pred_len: int) -> int:
    """Bar index sliced for ranking: H-1 when 0<H<=pred_len, else terminal bar."""
    if 0 < horizon <= pred_len:
        return horizon - 1
    return pred_len - 1


def _build_y_timestamps(ctx_ts: pd.Series, pred_len: int) -> pd.Series:
    """Future timestamps from the modal bar interval (mirrors _build_timestamps)."""
    diffs = ctx_ts.iloc[1:].reset_index(drop=True) - ctx_ts.iloc[:-1].reset_index(drop=True)
    interval = diffs.mode().iloc[0] if len(diffs) > 0 else (ctx_ts.iloc[-1] - ctx_ts.iloc[-2])
    return pd.Series(
        pd.date_range(start=ctx_ts.iloc[-1] + interval, periods=pred_len, freq=interval, tz="UTC"),
        name="timestamp",
    )


def _load_symbol_meta(data_dir: Path) -> dict[str, dict]:
    """Read asset_class + has_volume per symbol from the export's hygiene report."""
    report = data_dir / "hygiene_report.json"
    meta: dict[str, dict] = {}
    if report.exists():
        payload = json.loads(report.read_text())
        for s in payload.get("symbols", []):
            meta[s["symbol"]] = {"asset_class": s["asset_class"], "has_volume": s["has_volume"]}
    return meta


def _infer_meta(symbol: str) -> dict:
    if symbol in ("XAU/USD", "XAG/USD"):
        return {"asset_class": "metal", "has_volume": True}
    if "/" in symbol:
        return {"asset_class": "forex", "has_volume": False}
    return {"asset_class": "equity", "has_volume": True}


def _load_predictor(args: argparse.Namespace):
    sys.path.insert(0, str(Path(args.kronos_src).resolve()))
    from model.kronos import Kronos, KronosPredictor, KronosTokenizer  # type: ignore

    tokenizer = KronosTokenizer.from_pretrained(args.tokenizer)
    model = Kronos.from_pretrained(args.predictor)
    return KronosPredictor(
        model, tokenizer, device=args.device, max_context=args.max_context, clip=args.clip
    )


def _origins_for_symbol(n: int, manifest_sym: dict, lookback: int, predict: int) -> list[int]:
    test_start = manifest_sym["test_start_idx"]
    return valid_start_indices(n, lookback, predict, test_start, n)


def _build_inputs(
    df: pd.DataFrame, starts: list[int], lookback: int, pred_len: int, has_volume: bool
) -> tuple[list[pd.DataFrame], list[pd.Series], list[pd.Series]]:
    cols = _PRICE_COLS + (["volume", "amount"] if has_volume else [])
    df_list, x_ts_list, y_ts_list = [], [], []
    for s in starts:
        ctx = df.iloc[s : s + lookback]
        df_list.append(ctx[cols].reset_index(drop=True))
        ctx_ts = ctx["timestamps"].reset_index(drop=True)
        x_ts_list.append(ctx_ts)
        y_ts_list.append(_build_y_timestamps(ctx_ts, pred_len))
    return df_list, x_ts_list, y_ts_list


def _predict_symbol(predictor, args, df_list, x_ts_list, y_ts_list, slice_idx):
    """Return (mean_closes, var_closes_lists) per origin for one symbol, chunked."""
    n = len(df_list)
    mean_closes: list[float] = [float("nan")] * n
    var_closes: list[list[float]] = [[] for _ in range(n)]
    for lo in range(0, n, args.batch_size):
        hi = min(lo + args.batch_size, n)
        dfs, xts, yts = df_list[lo:hi], x_ts_list[lo:hi], y_ts_list[lo:hi]
        # Pass 1 — point estimate
        p1 = predictor.predict_batch(
            df_list=dfs,
            x_timestamp_list=xts,
            y_timestamp_list=yts,
            pred_len=args.pred_len,
            T=args.forecast_temperature,
            top_p=args.forecast_top_p,
            sample_count=args.forecast_sample_count,
            verbose=False,
        )
        for j, pred_df in enumerate(p1):
            mean_closes[lo + j] = float(pred_df["close"].iloc[slice_idx])
        # Pass 2 — variance draws
        for _ in range(args.variance_sample_count):
            p2 = predictor.predict_batch(
                df_list=dfs,
                x_timestamp_list=xts,
                y_timestamp_list=yts,
                pred_len=args.pred_len,
                T=args.variance_temperature,
                top_p=args.forecast_top_p,
                sample_count=1,
                verbose=False,
            )
            for j, pred_df in enumerate(p2):
                var_closes[lo + j].append(float(pred_df["close"].iloc[slice_idx]))
    return mean_closes, var_closes


def _derive_row(
    args,
    symbol,
    meta,
    origin_ts,
    horizon_ts,
    current_price,
    mean_close,
    vc: list[float],
    realized_close,
    realized_high,
    realized_low,
) -> dict:
    mean_return = (mean_close - current_price) / current_price
    if vc:
        var_returns = [(c - current_price) / current_price for c in vc]
        std_return = float(np.std(var_returns))
        if mean_return >= 0:
            direction_confidence = float(np.mean([r >= 0 for r in var_returns]))
        else:
            direction_confidence = float(np.mean([r < 0 for r in var_returns]))
    else:
        std_return = 0.0
        direction_confidence = 1.0 if mean_return >= 0 else 0.0
    uncertainty = std_return / (abs(mean_return) + 1e-8)
    stop_pct = max(std_return * args.vol_stop_multiplier, args.min_stop_pct)
    direction = "LONG" if mean_return >= 0 else "SHORT"
    tradeable = (
        direction == "LONG"
        and mean_return >= args.min_predicted_return
        and direction_confidence >= args.min_confidence
        and uncertainty <= args.max_uncertainty
    )
    realized_return = (realized_close - current_price) / current_price
    return {
        "symbol": symbol,
        "asset_class": meta["asset_class"],
        "origin_ts": origin_ts,
        "horizon_ts": horizon_ts,
        "current_price": current_price,
        "predicted_close": mean_close,
        "mean_return": mean_return,
        "std_return": std_return,
        "direction_confidence": direction_confidence,
        "uncertainty": uncertainty,
        "stop_pct": stop_pct,
        "direction": direction,
        "tradeable": tradeable,
        "realized_return": realized_return,
        "realized_mfe": (realized_high - current_price) / current_price,
        "realized_mae": (current_price - realized_low) / current_price,
    }


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--data-dir", default="kronos_ab/data")
    p.add_argument("--manifest", default="kronos_ab/data/split_manifest.json")
    p.add_argument("--tokenizer", required=True, help="Path/repo for the Kronos tokenizer.")
    p.add_argument("--predictor", required=True, help="Path/repo for the Kronos predictor.")
    p.add_argument("--out", required=True, help="Output predictions CSV.")
    p.add_argument(
        "--kronos-src",
        default=os.environ.get("KRONOS_DIR", ""),
        help="Kronos checkout holding the model/ package (defaults to $KRONOS_DIR).",
    )
    p.add_argument("--device", default=None)
    p.add_argument("--max-context", type=int, default=2048)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--pred-len", type=int, default=120)
    p.add_argument("--ranking-horizon", type=int, default=0, help="0 = terminal bar (pred_len).")
    p.add_argument("--batch-size", type=int, default=64, help="Origins per predict_batch chunk.")
    p.add_argument(
        "--limit-origins", type=int, default=0, help="Cap origins/symbol (0=all; smoke)."
    )
    # Inference hyperparameters — defaults mirror TopKConfig.
    p.add_argument("--forecast-temperature", type=float, default=0.6)
    p.add_argument("--forecast-top-p", type=float, default=0.90)
    p.add_argument("--forecast-sample-count", type=int, default=10)
    p.add_argument("--variance-temperature", type=float, default=1.0)
    p.add_argument("--variance-sample-count", type=int, default=20)
    # Tradeable thresholds — defaults mirror TopKConfig.
    p.add_argument("--min-confidence", type=float, default=0.80)
    p.add_argument("--max-uncertainty", type=float, default=10.0)
    p.add_argument("--min-predicted-return", type=float, default=0.003)
    p.add_argument("--vol-stop-multiplier", type=float, default=2.0)
    p.add_argument("--min-stop-pct", type=float, default=0.005)
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    manifest = json.loads(Path(args.manifest).read_text())
    params = manifest["params"]
    lookback, predict = params["lookback"], params["predict"]
    horizon = args.ranking_horizon if args.ranking_horizon > 0 else args.pred_len
    slice_idx = _slice_index(horizon, args.pred_len)
    sym_meta = _load_symbol_meta(data_dir)

    predictor = _load_predictor(args)

    rows: list[dict] = []
    for s_entry in manifest["symbols"]:
        if not s_entry["included"]:
            continue
        symbol = s_entry["symbol"]
        meta = sym_meta.get(symbol) or _infer_meta(symbol)
        csv_path = data_dir / s_entry["csv_file"]
        df = pd.read_csv(csv_path)
        df["timestamps"] = pd.to_datetime(df["timestamps"], utc=True)
        n = len(df)

        starts = _origins_for_symbol(n, s_entry, lookback, predict)
        if args.limit_origins > 0:
            starts = starts[: args.limit_origins]
        if not starts:
            continue

        df_list, x_ts_list, y_ts_list = _build_inputs(
            df, starts, lookback, args.pred_len, meta["has_volume"]
        )
        mean_closes, var_closes = _predict_symbol(
            predictor, args, df_list, x_ts_list, y_ts_list, slice_idx
        )

        closes = df["close"].to_numpy()
        highs = df["high"].to_numpy()
        lows = df["low"].to_numpy()
        ts = df["timestamps"]
        for j, s in enumerate(starts):
            origin_idx = s + lookback - 1
            horizon_idx = s + lookback + horizon - 1
            current_price = float(closes[origin_idx])
            realized_close = float(closes[horizon_idx])
            realized_high = float(highs[origin_idx + 1 : horizon_idx + 1].max())
            realized_low = float(lows[origin_idx + 1 : horizon_idx + 1].min())
            rows.append(
                _derive_row(
                    args,
                    symbol,
                    meta,
                    ts.iloc[origin_idx].isoformat(),
                    ts.iloc[horizon_idx].isoformat(),
                    current_price,
                    mean_closes[j],
                    var_closes[j],
                    realized_close,
                    realized_high,
                    realized_low,
                )
            )
        print(f"{symbol}: scored {len(starts)} origins")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nWrote {len(rows)} predictions → {out}")


if __name__ == "__main__":
    main()
