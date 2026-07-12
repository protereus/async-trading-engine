"""Tests for BotConfig.validate_config()."""

from __future__ import annotations

import pytest

from bot.config import BotConfig


def _ig_base(**overrides: object) -> BotConfig:
    """Minimal valid IG+twelvedata config (env file disabled to isolate from .env)."""
    defaults: dict[str, object] = {
        "_env_file": None,
        "broker": "ig",
        "candle_exchange": "twelvedata",
        "twelve_data_api": "test_key",
        "ig_demo_api": "test_api",
        "ig_demo_username": "user",
        "ig_demo_password": "pass",
        "bot_env": "demo",
        "topk_enabled": False,
    }
    defaults.update(overrides)
    return BotConfig(**defaults)  # type: ignore[arg-type]


class TestAggregateToMinutesValidation:
    def test_default_passes_twelvedata(self) -> None:
        cfg = _ig_base()
        cfg.validate_config()  # must not raise

    def test_explicit_one_passes(self) -> None:
        cfg = _ig_base(topk_aggregate_to_minutes=1)
        cfg.validate_config()

    def test_nonone_raises_for_twelvedata(self) -> None:
        cfg = _ig_base(topk_aggregate_to_minutes=5)
        with pytest.raises(ValueError, match="TOPK_AGGREGATE_TO_MINUTES must be 1"):
            cfg.validate_config()

    def test_binance_candle_exchange_rejected(self) -> None:
        """The ccxt binance/okx candle feeds were archived 2026-06-24; only
        'ig', 'twelvedata', 'eodhd' are valid candle_exchange values now."""
        cfg = BotConfig(
            _env_file=None,
            broker="ig",
            candle_exchange="binance",
            candle_epic_map={"BTC/USDT": "CS.D.BITCOIN.TODAY.IP"},
            ig_demo_api="k",
            ig_demo_username="u",
            ig_demo_password="p",
            bot_env="demo",
            topk_enabled=False,
            topk_aggregate_to_minutes=5,
        )
        with pytest.raises(ValueError, match="candle_exchange must be"):
            cfg.validate_config()


class TestCerebrasFullyRemoved:
    """The 2026-06-01 head-to-head bench showed neither Cerebras model on the
    account produced usable signals for our sentiment prompt; Cerebras was
    removed entirely.  These tests pin
    the post-removal state so a future change that re-introduces a
    ``cerebras_*`` field on BotConfig fails loud rather than silently
    re-enabling a dead provider chain."""

    def test_botconfig_has_no_cerebras_fields(self) -> None:
        cfg = _ig_base()
        assert not hasattr(cfg, "cerebras_api")
        assert not hasattr(cfg, "cerebras_model")

    def test_stale_cerebras_env_var_is_ignored(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """An operator who set ``CEREBRAS_API`` in ``.env`` before the
        removal shouldn't have to delete it for the bot to start.
        ``model_config=extra=ignore`` makes the field a silent no-op rather
        than a startup crash on ``extra_forbidden``."""
        monkeypatch.setenv("CEREBRAS_API", "stale-key")
        monkeypatch.setenv("CEREBRAS_MODEL", "stale-model")
        # Must not raise pydantic ValidationError
        cfg = _ig_base()
        assert cfg.bot_env == "demo"  # construction succeeded


class TestRankingHorizonByClassValidation:
    """TOPK_RANKING_HORIZON_BY_CLASS must fail at validate_config(), not
    silently mis-slice."""

    def test_empty_default_passes(self) -> None:
        cfg = _ig_base()
        cfg.validate_config()  # must not raise
        assert cfg.topk_ranking_horizon_by_class == ""

    def test_valid_map_passes(self) -> None:
        cfg = _ig_base(topk_ranking_horizon_by_class="forex:48,us_equity:24,metal:24")
        cfg.validate_config()  # must not raise

    def test_unknown_class_fails_at_startup(self) -> None:
        cfg = _ig_base(topk_ranking_horizon_by_class="crypto:24")
        with pytest.raises(ValueError, match="unknown class"):
            cfg.validate_config()

    def test_horizon_above_pred_len_fails_at_startup(self) -> None:
        cfg = _ig_base(topk_ranking_horizon_by_class="forex:121")  # pred_len default 120
        with pytest.raises(ValueError, match="outside"):
            cfg.validate_config()
