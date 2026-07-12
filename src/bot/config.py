"""Bot configuration loaded from environment / .env via pydantic-settings."""

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class BotConfig(BaseSettings):
    # Broker selection. IG is the only supported broker; kept as a field so a
    # non-"ig" value fails fast with a clear message rather than silently.
    broker: str = "ig"  # "ig" (only supported value)

    # IG Group credentials (demo and live share same key/username/password;
    # account ID selects demo vs live account)
    ig_demo_api: str = ""
    ig_demo_username: str = ""
    ig_demo_password: str = ""
    ig_live_api: str = ""
    ig_live_username: str = ""
    ig_live_password: str = ""

    # Telegram
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""

    # Gemini API key (used for sentiment escalation)
    gemini_api_key: str = ""

    # Groq API key (used for bulk sentiment inference)
    groq_api: str = ""

    # FRED API key (macro overlay: DXY, VIX, yields, CPI, NFP)
    fred_api_key: str = ""

    # Finnhub API key (used for news, social sentiment, economic calendar)
    finhub_io_api: str = ""

    # Twelve Data API key (required when candle_exchange="twelvedata")
    twelve_data_api: str = ""

    # EODHD API key (required when candle_exchange="eodhd").  Primary market
    # data vendor: intraday-historical backfill + real-time WebSocket ticks
    # aggregated to 1h locally.
    eodhd_api: str = ""

    # FinBERT local headline pre-filter — runs ProsusAI/finbert via ONNX to
    # triage news headlines before LLM dispatch.  Gated by default because it
    # contends with model inference for CPU.
    finbert_enabled: bool = False

    # Bot settings
    bot_env: str = "demo"  # "demo" or "live"
    log_level: str = "INFO"

    # Monitoring
    healthcheck_url: str = ""

    # Trading defaults
    candle_timeframe: str = "1m"
    candle_buffer_size: int = 3000  # candles per pair in memory (must exceed strategy warmup)

    # IG-specific: list of EPICs used for order placement (e.g. ["CS.D.AVXUSD.TODAY.IP"])
    # If empty when broker="ig" and candle_exchange="ig", the bot will raise at startup.
    ig_epics: list[str] = []

    # Candle data source. Valid values: "ig", "twelvedata", "eodhd".
    # "eodhd"      — EODHD intraday backfill + WebSocket aggregated to 1h (primary;
    #                sources the 28-asset universe)
    # "twelvedata" — Twelve Data REST hourly OHLC (FX-only warm-standby failover)
    # "ig"         — IG Lightstreamer (EPICs used as candle keys; requires IG_EPICS)
    candle_exchange: str = "ig"

    # Maps a trading symbol → DB key / IG EPIC for order routing.  Used by the
    # twelvedata/eodhd feeds' SYMBOL_EPIC_MAP indirection; rarely set directly
    # in .env (JSON-encoded: CANDLE_EPIC_MAP={"EUR/USD":"CS.D.EURUSD.TODAY.IP"}).
    candle_epic_map: dict[str, str] = {}

    # Top-K multi-asset Kronos strategy (IG only)
    topk_enabled: bool = False
    topk_watchlist: list[str] = []  # EPICs to scan; defaults to ig_epics if empty
    # Symbols still scanned by Kronos and logged to signal_history, but never
    # selected for entry.  Use to monitor an instrument's model performance
    # without putting capital at risk.  Empty = nothing held out of selection.
    topk_exclude_from_selection: list[str] = []
    topk_k: int = 3  # Max simultaneous positions
    topk_rerank_interval_minutes: int = 60
    topk_aggregate_to_minutes: int = 1  # Resample input candles before Kronos; 1 = no resampling
    topk_pred_len: int = 120  # Prediction horizon in bars (120 × 1h = 5-day horizon)
    # Dual-pass inference (see the Kronos paper's sampling-parameter table):
    # Pass 1 draws a sharp low-temperature point estimate; Pass 2 re-samples at
    # high temperature to measure Monte-Carlo spread.
    topk_forecast_temperature: float = 0.6  # Pass 1 — low T for sharp point estimate
    topk_forecast_top_p: float = 0.9
    topk_forecast_sample_count: int = 10  # Internal draws averaged by library
    topk_variance_pass_enabled: bool = True  # Pass 2 for std/confidence
    topk_variance_temperature: float = 1.0  # High T preserves MC spread
    topk_variance_sample_count: int = 20  # Loops; total calls = 1 + 20 = 21
    # Run the no-volume and volume groups concurrently during inference to
    # reclaim idle cores on multi-core hosts (signals unchanged).  Off by
    # default — set TOPK_PARALLEL_GROUPS=true to enable.
    topk_parallel_groups: bool = False
    # --- Signal-quality entry filters -------------------------------------
    # The defaults below are deliberately WIDE, uncalibrated starting points.
    # They gate entries only against degenerate signals; they are not tuned
    # to any particular market, account, or data window.  Calibrate them
    # against your own accumulated signal_history with
    # scripts/calibrate_thresholds.py (run scripts/setup.py for a guided
    # first-time configuration), and re-run the calibration periodically —
    # threshold quality decays as market regime shifts.
    #
    #   topk_min_confidence       [0.50 .. 0.95]  fraction of variance-pass
    #                             paths that must agree with the mean direction
    #   topk_max_uncertainty      [1.0 .. 100.0]  cap on std_return relative to
    #                             |mean_return|; high values ≈ safety bound only
    #   topk_min_predicted_return [0.0 .. 0.02]   floor on predicted return
    #                             over the prediction horizon
    topk_min_confidence: float = 0.60
    topk_max_uncertainty: float = 5.0
    topk_min_predicted_return: float = 0.001
    topk_vol_stop_multiplier: float = 2.0
    topk_min_stop_pct: float = 0.005
    # Correlation filter: skip candidates highly correlated with an
    # already-selected symbol.  1.0 disables filtering; typical values sit in
    # [0.5 .. 0.9] depending on how concentrated the universe is.
    topk_correlation_enabled: bool = True
    topk_correlation_max: float = 0.75  # |Pearson| threshold; 1.0 disables filtering
    topk_correlation_lookback_bars: int = 200  # ~8 days of 1h returns
    # Long-only book: bump only on high *positive* correlation, keeping
    # negatively-correlated (hedging) candidates.  Off by default — on a
    # long-only universe it is usually a no-op, since a strongly
    # anti-correlated counterpart to a LONG signal is itself predicted DOWN
    # and already fails the LONG entry filter before correlation runs.
    topk_correlation_long_only: bool = False
    # Ranking-horizon slice into the existing Pass-1 path.
    # 0 → terminal bar (ranking pegged to pred_len). Set to a positive
    # bar count (e.g. 24) to slice ranking at that bar — the surrounding path
    # is already generated by Pass 1, so no extra inference is required.
    topk_ranking_horizon_bars: int = 0
    # Per-asset-class ranking horizon overriding the global scalar, format
    # "forex:24,us_equity:12,metal:24" (classes from the symbol universe;
    # equities are labelled us_equity).  Parsed + validated by
    # bot.strategy.topk_strategy.parse_ranking_horizon_by_class at
    # validate_config() — unknown classes or H > pred_len fail at startup.
    # Empty (default) = global topk_ranking_horizon_bars applies to all.
    # Changing the ranking horizon shifts the signal distribution, so couple
    # any change with a threshold recalibration at the chosen horizon.
    topk_ranking_horizon_by_class: str = ""
    kronos_dir: str = ""  # Path added to sys.path so `from model import` works
    kronos_model: str = "NeoQuasar/Kronos-mini"
    kronos_tokenizer: str = "NeoQuasar/Kronos-Tokenizer-2k"
    kronos_context_bars: int = 400  # 2k tokenizer supports up to 2048; 400 bars is practical

    # Sentiment overlay (optional)
    sentiment_enabled: bool = False
    sentiment_scan_interval_minutes: int = 30
    sentiment_escalation_max_per_hour: int = 20
    # If True, TopK reranks are gated by sentiment score (requires sentiment_enabled=True).
    # Direction-aware: LONG signals require sentiment >= long_threshold, SHORT signals
    # require sentiment <= short_threshold. Assets with no sentiment pass through.
    sentiment_gate_enabled: bool = False
    sentiment_gate_long_threshold: float = 0.3
    sentiment_gate_short_threshold: float = -0.3

    # Take-profit strategy — five independently toggleable exit components
    tp_static_enabled: bool = True
    tp_trailing_enabled: bool = True
    tp_signal_decay_enabled: bool = True
    tp_time_enabled: bool = True
    tp_sentiment_reversal_enabled: bool = False  # requires sentiment_enabled=True

    tp_min_rr_multiplier: float = 1.5
    tp_kronos_target_fraction: float = 0.80

    tp_breakeven_activation_mult: float = 1.0
    tp_breakeven_buffer: float = 0.001
    tp_trail_activation_mult: float = 2.0
    tp_trail_multiplier: float = 0.5

    tp_signal_decay_min_confidence: float = 0.55
    tp_signal_decay_max_uncertainty: float = 3.0
    tp_signal_decay_max_strikes: int = 2
    tp_signal_decay_max_topk_misses: int = 6

    tp_time_horizon_multiplier: float = 1.0

    tp_sentiment_reversal_threshold: float = -0.3
    tp_sentiment_reversal_min_confidence: float = 0.6

    # ``extra="ignore"`` lets env vars be retired safely without forcing the
    # operator to delete the key from ``.env`` before the bot will start.
    # Pydantic's default ``extra="forbid"`` would crash on any unrecognised line.
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("bot_env")
    @classmethod
    def validate_bot_env(cls, v: str) -> str:
        if v not in ("demo", "live"):
            raise ValueError(f"bot_env must be 'demo' or 'live', got: {v!r}")
        return v

    def validate_config(self) -> None:
        """Fail fast at startup if required fields are missing."""
        if self.broker != "ig":
            raise ValueError(f"broker must be 'ig', got: {self.broker!r}")

        if self.bot_env == "demo":
            ig_required = {
                "ig_demo_api": self.ig_demo_api,
                "ig_demo_username": self.ig_demo_username,
                "ig_demo_password": self.ig_demo_password,
            }
        else:
            ig_required = {
                "ig_live_api": self.ig_live_api,
                "ig_live_username": self.ig_live_username,
                "ig_live_password": self.ig_live_password,
            }
        missing = [name for name, val in ig_required.items() if not val]
        if missing:
            raise ValueError(f"Missing required IG config fields: {', '.join(missing)}")

        if self.candle_exchange == "ig":
            if not self.ig_epics:
                raise ValueError(
                    "candle_exchange='ig' requires ig_epics to be set "
                    '(e.g. IG_EPICS=["CS.D.AVXUSD.TODAY.IP"])'
                )
        elif self.candle_exchange == "twelvedata":
            if not self.twelve_data_api:
                raise ValueError("candle_exchange='twelvedata' requires TWELVE_DATA_API to be set")
            if self.topk_aggregate_to_minutes != 1:
                raise ValueError(
                    "candle_exchange='twelvedata' delivers pre-built 1h bars; "
                    "TOPK_AGGREGATE_TO_MINUTES must be 1 (no resampling). "
                    f"Got {self.topk_aggregate_to_minutes}."
                )
        elif self.candle_exchange == "eodhd":
            if not self.eodhd_api:
                raise ValueError("candle_exchange='eodhd' requires EODHD_API to be set")
            if self.topk_aggregate_to_minutes != 1:
                raise ValueError(
                    "candle_exchange='eodhd' emits 1h bars (intraday backfill + "
                    "WebSocket aggregation); TOPK_AGGREGATE_TO_MINUTES must be 1. "
                    f"Got {self.topk_aggregate_to_minutes}."
                )
        else:
            raise ValueError(
                f"candle_exchange must be 'ig', 'twelvedata', or 'eodhd', "
                f"got: {self.candle_exchange!r}"
            )

        if self.topk_enabled and not self.kronos_dir:
            raise ValueError(
                "topk_enabled=True requires KRONOS_DIR to be set "
                "(path to a Kronos checkout containing the model/ package, "
                "e.g. KRONOS_DIR=/path/to/Kronos)"
            )

        if self.topk_ranking_horizon_by_class:
            # Imported lazily — the strategy module pulls pandas/numpy, which
            # config import-time must not depend on.  Raises ValueError on
            # unknown class names or horizons outside [1, pred_len].
            from bot.strategy.topk_strategy import parse_ranking_horizon_by_class

            parse_ranking_horizon_by_class(self.topk_ranking_horizon_by_class, self.topk_pred_len)

        if self.tp_sentiment_reversal_enabled and not self.sentiment_enabled:
            import warnings

            warnings.warn(
                "TP_SENTIMENT_REVERSAL_ENABLED=true has no effect without SENTIMENT_ENABLED=true",
                stacklevel=2,
            )
