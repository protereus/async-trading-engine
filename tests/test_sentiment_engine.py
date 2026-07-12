"""Tests for SentimentEngine — rerank-busy deferral and scan dataflow."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from bot.sentiment.config import SentimentConfig
from bot.sentiment.engine import SentimentEngine
from bot.sentiment.models import ConsensusSignal, RawSignal


def _make_engine(tmp_path: Path) -> SentimentEngine:
    cfg = SentimentConfig(groq_api="test", scan_interval_minutes=30)
    return SentimentEngine(cfg, db_path=tmp_path / "sentiment.db")


class TestRerankDeferral:
    @pytest.mark.asyncio
    async def test_scan_skipped_when_rerank_busy(self, tmp_path: Path) -> None:
        eng = _make_engine(tmp_path)
        eng.set_rerank_busy_callback(lambda: True)

        agents = [MagicMock()]
        agents[0].analyze = AsyncMock(return_value=[])
        agents[0].name = "stub"
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={})

        await eng._scan_once(agents, aggregator)
        agents[0].analyze.assert_not_called()
        aggregator.aggregate.assert_not_called()

    @pytest.mark.asyncio
    async def test_scan_proceeds_when_rerank_idle(self, tmp_path: Path) -> None:
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng.set_rerank_busy_callback(lambda: False)
        eng._assets = ["SPY"]

        agents = [MagicMock()]
        agents[0].analyze = AsyncMock(return_value=[])
        agents[0].name = "stub"
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={})

        await eng._scan_once(agents, aggregator)
        agents[0].analyze.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_scan_proceeds_without_callback(self, tmp_path: Path) -> None:
        """When no callback is set, scans always run (back-compat path)."""
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY"]

        agents = [MagicMock()]
        agents[0].analyze = AsyncMock(return_value=[])
        agents[0].name = "stub"
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={})

        await eng._scan_once(agents, aggregator)
        agents[0].analyze.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_callback_evaluated_at_scan_time(self, tmp_path: Path) -> None:
        """Callback is checked per-scan, not just once at injection."""
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY"]
        state = {"busy": True}
        eng.set_rerank_busy_callback(lambda: state["busy"])

        agents = [MagicMock()]
        agents[0].analyze = AsyncMock(return_value=[])
        agents[0].name = "stub"
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={})

        await eng._scan_once(agents, aggregator)
        agents[0].analyze.assert_not_called()

        state["busy"] = False
        await eng._scan_once(agents, aggregator)
        agents[0].analyze.assert_awaited_once()


def _raw(asset: str, sentiment: float, confidence: float, source: str) -> RawSignal:
    return RawSignal(
        asset=asset,
        sentiment=sentiment,
        confidence=confidence,
        reasoning="stub",
        source=source,
    )


def _consensus(asset: str, sentiment: float, confidence: float = 0.8) -> ConsensusSignal:
    return ConsensusSignal(
        asset=asset,
        sentiment=sentiment,
        confidence=confidence,
        agreement=1.0,
        sources=["news"],
        escalated=False,
        reasoning="stub consensus",
    )


class TestScanDataflow:
    """Value-level assertions on what a scan actually produces — the scores
    the strategy overlay reads via ``get_sentiment_scores`` and the rows
    persisted to the sentiment DB."""

    @pytest.mark.asyncio
    async def test_consensus_values_reach_current_scores(self, tmp_path: Path) -> None:
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY", "EUR/USD"]

        agent = MagicMock()
        agent.name = "news"
        agent.analyze = AsyncMock(
            return_value=[
                _raw("SPY", 0.6, 0.9, "news"),
                _raw("EUR/USD", -0.4, 0.7, "news"),
            ]
        )
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(
            return_value={
                "SPY": _consensus("SPY", 0.6, confidence=0.9),
                "EUR/USD": _consensus("EUR/USD", -0.4, confidence=0.7),
            }
        )

        await eng._scan_once([agent], aggregator)

        # The aggregator received exactly the raw signals the agent produced.
        raw_arg, assets_arg = aggregator.aggregate.await_args.args
        assert [(r.asset, r.sentiment) for r in raw_arg] == [("SPY", 0.6), ("EUR/USD", -0.4)]
        assert assets_arg == ["SPY", "EUR/USD"]

        # And the consensus values are what the overlay will read.
        scores = eng.get_sentiment_scores()
        assert scores["SPY"].sentiment == pytest.approx(0.6)
        assert scores["SPY"].confidence == pytest.approx(0.9)
        assert scores["EUR/USD"].sentiment == pytest.approx(-0.4)

    @pytest.mark.asyncio
    async def test_consensus_rows_persisted_to_db(self, tmp_path: Path) -> None:
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY"]

        agent = MagicMock()
        agent.name = "news"
        agent.analyze = AsyncMock(return_value=[_raw("SPY", 0.25, 0.8, "news")])
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={"SPY": _consensus("SPY", 0.25)})

        await eng._scan_once([agent], aggregator)

        stored = eng._db.get_latest("SPY")
        assert stored is not None
        assert stored["sentiment"] == pytest.approx(0.25)
        assert stored["asset"] == "SPY"

    @pytest.mark.asyncio
    async def test_failing_agent_degrades_not_aborts(self, tmp_path: Path) -> None:
        """One agent raising must not lose the other agents' signals — the
        scan aggregates what it got."""
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY"]

        good = MagicMock()
        good.name = "news"
        good.analyze = AsyncMock(return_value=[_raw("SPY", 0.5, 0.8, "news")])
        bad = MagicMock()
        bad.name = "macro"
        bad.analyze = AsyncMock(side_effect=RuntimeError("boom"))
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={"SPY": _consensus("SPY", 0.5)})

        await eng._scan_once([good, bad], aggregator)

        raw_arg, _ = aggregator.aggregate.await_args.args
        assert [(r.asset, r.source) for r in raw_arg] == [("SPY", "news")]
        assert eng.get_sentiment_scores()["SPY"].sentiment == pytest.approx(0.5)

    @pytest.mark.asyncio
    async def test_no_raw_signals_leaves_scores_untouched(self, tmp_path: Path) -> None:
        eng = _make_engine(tmp_path)
        eng._db.init_db()
        eng._assets = ["SPY"]
        eng._current_scores["SPY"] = _consensus("SPY", 0.9)  # pre-existing score

        agent = MagicMock()
        agent.name = "news"
        agent.analyze = AsyncMock(return_value=[])
        aggregator = MagicMock()
        aggregator.aggregate = AsyncMock(return_value={})

        await eng._scan_once([agent], aggregator)

        aggregator.aggregate.assert_not_awaited()
        # The stale score survives (honest: absence of news isn't neutral news).
        assert eng.get_sentiment_scores()["SPY"].sentiment == pytest.approx(0.9)
