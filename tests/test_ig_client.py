"""Tests for IGClient REST client.

Mocks the aiohttp session so no real network calls are made.
"""

from __future__ import annotations

import contextlib
import json
import time
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from bot.core.models import (
    ErrorType,
    ExchangeError,
    IGOrderRequest,
    MarketClosedError,
    OrderSide,
    OrderStatus,
)
from bot.execution.ig_client import IGClient
from bot.execution.ig_http import Bucket, IGHttp, TokenBucket, bucket_for_path
from bot.execution.ig_parsers import mid_price, parse_ig_timestamp

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(bot_env: str = "demo") -> Any:
    cfg = MagicMock()
    cfg.bot_env = bot_env
    cfg.ig_demo_api = "demo-api-key"
    cfg.ig_demo_username = "user"
    cfg.ig_demo_password = "pass"
    cfg.ig_live_api = "live-api-key"
    cfg.ig_live_username = "live-user"
    cfg.ig_live_password = "live-pass"
    return cfg


def make_client(bot_env: str = "demo") -> IGClient:
    client = IGClient(make_config(bot_env))
    client._cst = "test-cst"
    client._xst = "test-xst"
    client._account_id = "Z6ACA2"
    client._ls_endpoint = "https://mock-ls-endpoint"
    return client


def mock_response(status: int = 200, json_body: Any = None, text_body: str = "") -> MagicMock:
    """Build an async context manager mock that looks like an aiohttp response."""
    resp = MagicMock()
    resp.status = status
    resp.json = AsyncMock(return_value=json_body or {})
    resp.text = AsyncMock(return_value=text_body)
    resp.__aenter__ = AsyncMock(return_value=resp)
    resp.__aexit__ = AsyncMock(return_value=False)
    return resp


def attach_session(client: IGClient, method: str, resp: MagicMock) -> MagicMock:
    """Wire a mock session onto the client, returning the session mock."""
    session = MagicMock()
    setattr(session, method, MagicMock(return_value=resp))
    client._session = session
    return session


def _swallow_create_task(coro: Any, *_args: Any, **_kwargs: Any) -> MagicMock:
    """Substitute for ``asyncio.create_task`` in tests that don't want a live task.

    The mocked replacement must explicitly ``close()`` the coroutine it receives
    or the GC eventually emits ``RuntimeWarning: coroutine ... was never awaited``
    against whichever test happens to be running at the time — classic test
    pollution that obscures the real source.
    """
    if hasattr(coro, "close"):
        coro.close()
    return MagicMock()


# ===========================================================================
# Pure helpers
# ===========================================================================


class TestParseIgTimestamp:
    def test_slash_format_with_ms(self) -> None:
        ts = parse_ig_timestamp("2024/10/01 05:28:00:000")
        assert ts == 1727760480000

    def test_slash_format_without_ms(self) -> None:
        ts = parse_ig_timestamp("2026/04/10 19:00:00")
        assert ts > 0

    def test_iso_format(self) -> None:
        ts = parse_ig_timestamp("2024-10-01T05:28:00")
        assert ts == 1727760480000

    def test_iso_format_with_ms(self) -> None:
        ts = parse_ig_timestamp("2024-10-01T05:28:00.000")
        assert ts == 1727760480000

    def test_empty_string_returns_zero(self) -> None:
        assert parse_ig_timestamp("") == 0

    def test_unparseable_returns_zero(self) -> None:
        assert parse_ig_timestamp("not-a-date") == 0


class TestMid:
    def test_both_bid_and_ask(self) -> None:
        assert mid_price({"bid": 10.0, "ask": 12.0}) == 11.0

    def test_bid_only(self) -> None:
        assert mid_price({"bid": 10.0}) == 10.0

    def test_ask_only(self) -> None:
        assert mid_price({"ask": 12.0}) == 12.0

    def test_neither_returns_zero(self) -> None:
        assert mid_price({}) == 0.0

    def test_string_values_converted(self) -> None:
        assert mid_price({"bid": "9.5", "ask": "10.5"}) == 10.0


# ===========================================================================
# _handle_response
# ===========================================================================


class TestHandleResponse:
    @pytest.mark.asyncio
    async def test_200_returns_json(self) -> None:
        resp = mock_response(200, {"key": "value"})
        resp.__aenter__ = AsyncMock(return_value=resp)
        result = await IGHttp._handle_response(resp)
        assert result == {"key": "value"}

    @pytest.mark.asyncio
    async def test_401_raises_auth_error(self) -> None:
        resp = mock_response(401, text_body="Unauthorized")
        with pytest.raises(ExchangeError) as exc_info:
            await IGHttp._handle_response(resp)
        assert exc_info.value.error_type == ErrorType.AUTHENTICATION_FAILED

    @pytest.mark.asyncio
    async def test_403_raises_rate_limit(self) -> None:
        resp = mock_response(403, text_body="Forbidden")
        with pytest.raises(ExchangeError) as exc_info:
            await IGHttp._handle_response(resp)
        assert exc_info.value.error_type == ErrorType.RATE_LIMIT

    @pytest.mark.asyncio
    async def test_500_raises_service_unavailable(self) -> None:
        resp = mock_response(500, text_body="Server error")
        with pytest.raises(ExchangeError) as exc_info:
            await IGHttp._handle_response(resp)
        assert exc_info.value.error_type == ErrorType.SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_503_raises_service_unavailable(self) -> None:
        resp = mock_response(503, text_body="Service unavailable")
        with pytest.raises(ExchangeError) as exc_info:
            await IGHttp._handle_response(resp)
        assert exc_info.value.error_type == ErrorType.SERVICE_UNAVAILABLE

    @pytest.mark.asyncio
    async def test_other_error_raises_exchange_error(self) -> None:
        resp = mock_response(422, text_body="Unprocessable")
        with pytest.raises(ExchangeError) as exc_info:
            await IGHttp._handle_response(resp)
        assert exc_info.value.error_type == ErrorType.EXCHANGE_ERROR


# ===========================================================================
# Token cache
# ===========================================================================


class TestTokenCache:
    @pytest.mark.asyncio
    async def test_save_and_load_round_trip(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        client = make_client()
        client._cst = "cst-abc"
        client._xst = "xst-xyz"
        client._account_id = "Z6ACA2"
        client._ls_endpoint = "https://ls.example.com"

        with patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file):
            client._sess.save_tokens()
            assert cache_file.exists()
            # Verify permissions
            assert oct(cache_file.stat().st_mode)[-3:] == "600"

            # Load into a fresh client
            client2 = make_client()
            client2._cst = ""
            client2._xst = ""

            result = await client2._sess.load_cached_tokens()
        assert result is True
        assert client2._cst == "cst-abc"
        assert client2._xst == "xst-xyz"
        assert client2._account_id == "Z6ACA2"

    @pytest.mark.asyncio
    async def test_load_returns_false_when_no_file(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        client = make_client()

        with patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file):
            result = await client._sess.load_cached_tokens()
        assert result is False

    @pytest.mark.asyncio
    async def test_load_returns_false_when_stale(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        payload = {
            "cst": "old-cst",
            "xst": "old-xst",
            "account_id": "Z6ACA2",
            "ls_endpoint": "https://ls.example.com",
            "env": "demo",
            "saved_at": time.time() - (6 * 3600),  # 6 hours ago — stale
        }
        cache_file.write_text(json.dumps(payload))
        client = make_client()

        with patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file):
            result = await client._sess.load_cached_tokens()
        assert result is False

    @pytest.mark.asyncio
    async def test_load_returns_false_for_wrong_env(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        payload = {
            "cst": "cst",
            "xst": "xst",
            "account_id": "Z6ACA2",
            "ls_endpoint": "https://ls.example.com",
            "env": "live",  # cache was saved for live
            "saved_at": time.time(),
        }
        cache_file.write_text(json.dumps(payload))
        client = make_client("demo")  # but we're in demo

        with patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file):
            result = await client._sess.load_cached_tokens()
        assert result is False

    @pytest.mark.asyncio
    async def test_load_returns_false_for_corrupt_json(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        cache_file.write_text("not valid json{{{")
        client = make_client()

        with patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file):
            result = await client._sess.load_cached_tokens()
        assert result is False


# ===========================================================================
# fetch_balance
# ===========================================================================


class TestFetchBalance:
    @pytest.mark.asyncio
    async def test_returns_balance_for_matching_account(self) -> None:
        client = make_client()
        accounts_resp = mock_response(
            200,
            {
                "accounts": [
                    {
                        "accountId": "Z6ACA2",
                        "currency": "GBP",
                        "balance": {
                            "balance": 1200.0,
                            "deposit": 34.56,
                            "profitLoss": 34.56,
                            "available": 800.0,
                        },
                    }
                ]
            },
        )
        attach_session(client, "get", accounts_resp)

        result = await client.fetch_balance()

        assert result["equity"] == pytest.approx(1234.56)
        assert result["available"] == pytest.approx(800.0)
        assert result["balance"] == pytest.approx(1200.0)
        assert result["open_pnl"] == pytest.approx(34.56)
        assert result["margin"] == pytest.approx(34.56)
        assert result["currency"] == "GBP"

    @pytest.mark.asyncio
    async def test_raises_when_account_not_found(self) -> None:
        client = make_client()
        accounts_resp = mock_response(200, {"accounts": [{"accountId": "OTHER", "balance": {}}]})
        attach_session(client, "get", accounts_resp)

        with pytest.raises(ExchangeError) as exc_info:
            await client.fetch_balance()
        assert "not found" in str(exc_info.value)


# ===========================================================================
# place_order
# ===========================================================================


class TestPlaceOrder:
    @pytest.mark.asyncio
    async def test_buy_order_returns_pending_result(self) -> None:
        client = make_client()
        resp = mock_response(200, {"dealReference": "DEAL-123"})
        attach_session(client, "post", resp)

        order = IGOrderRequest(
            epic="CS.D.AVXUSD.TODAY.IP",
            direction="BUY",
            size=2.5,
            order_type="MARKET",
        )
        result = await client.place_order(order)

        assert result.order_id == "DEAL-123"
        assert result.status == OrderStatus.PENDING
        assert result.side == OrderSide.BUY
        assert result.requested_quantity == pytest.approx(2.5)

    @pytest.mark.asyncio
    async def test_sell_order_sets_sell_side(self) -> None:
        client = make_client()
        resp = mock_response(200, {"dealReference": "DEAL-SELL"})
        attach_session(client, "post", resp)

        order = IGOrderRequest(
            epic="CS.D.AVXUSD.TODAY.IP",
            direction="SELL",
            size=1.0,
        )
        result = await client.place_order(order)
        assert result.side == OrderSide.SELL

    @pytest.mark.asyncio
    async def test_stop_distance_included_in_body(self) -> None:
        client = make_client()
        resp = mock_response(200, {"dealReference": "DEAL-STOP"})
        session = attach_session(client, "post", resp)

        order = IGOrderRequest(
            epic="CS.D.AVXUSD.TODAY.IP",
            direction="BUY",
            size=1.0,
            stop_distance=0.5,
        )
        await client.place_order(order)

        call_kwargs = session.post.call_args
        sent_body = call_kwargs.kwargs.get("json") or call_kwargs.args[1]
        assert sent_body["stopDistance"] == 0.5


class TestOrderRetrySafety:
    """Order placement is non-idempotent: a lost ACK on a timeout/5xx must NOT
    be blind-retried (IG may have already opened the position, and dealReference
    is not an idempotency key) — that would create a duplicate position.
    Rate-limit / market-closed still retry (request rejected, not executed),
    and idempotent GETs are unaffected."""

    @staticmethod
    def _order() -> IGOrderRequest:
        return IGOrderRequest(epic="CS.D.AVXUSD.TODAY.IP", direction="BUY", size=1.0)

    @pytest.mark.asyncio
    async def test_order_not_retried_on_timeout(self) -> None:
        client = make_client()
        session = MagicMock()
        session.post = MagicMock(side_effect=TimeoutError())
        client._session = session
        with pytest.raises(TimeoutError):
            await client.place_order(self._order())
        assert session.post.call_count == 1  # failed closed, no retry

    @pytest.mark.asyncio
    async def test_order_not_retried_on_5xx(self) -> None:
        client = make_client()
        session = attach_session(client, "post", mock_response(500, {}))
        with pytest.raises(ExchangeError) as ei:
            await client.place_order(self._order())
        assert ei.value.error_type == ErrorType.SERVICE_UNAVAILABLE
        assert session.post.call_count == 1

    @pytest.mark.asyncio
    async def test_order_still_retries_on_rate_limit(self) -> None:
        # 403 → RATE_LIMIT: the order was rejected, not executed, so retrying is
        # safe even for a non-idempotent POST.
        client = make_client()
        session = MagicMock()
        session.post = MagicMock(
            side_effect=[mock_response(403, {}), mock_response(200, {"dealReference": "OK"})]
        )
        client._session = session
        with patch("asyncio.sleep", new=AsyncMock()):
            result = await client.place_order(self._order())
        assert result.order_id == "OK"
        assert session.post.call_count == 2  # retried through the rate-limit

    @pytest.mark.asyncio
    async def test_idempotent_get_still_retries_on_timeout(self) -> None:
        # The fix must not change GET behaviour — idempotent reads still retry.
        client = make_client()
        session = MagicMock()
        session.get = MagicMock(side_effect=[TimeoutError(), mock_response(200, {"ok": True})])
        client._session = session
        with patch("asyncio.sleep", new=AsyncMock()):
            data = await client._http.request(
                "GET", "/markets/CS.D.AVXUSD.TODAY.IP", version="1", authenticated=True
            )
        assert data == {"ok": True}
        assert session.get.call_count == 2


# ===========================================================================
# confirm_order
# ===========================================================================


class TestConfirmOrder:
    @pytest.mark.asyncio
    async def test_accepted_returns_filled_result(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "dealStatus": "ACCEPTED",
                "dealId": "DID-999",
                "epic": "CS.D.AVXUSD.TODAY.IP",
                "direction": "BUY",
                "size": 2.0,
                "level": 31.50,
            },
        )
        attach_session(client, "get", resp)

        result = await client.confirm_order("DEAL-REF")
        assert result.status == OrderStatus.FILLED
        assert result.order_id == "DID-999"
        assert result.average_price == pytest.approx(31.50)
        assert result.filled_quantity == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_rejected_raises_exchange_error(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "dealStatus": "REJECTED",
                "reason": "MARGIN_CLOSE_ONLY",
            },
        )
        attach_session(client, "get", resp)

        with pytest.raises(ExchangeError) as exc_info:
            await client.confirm_order("DEAL-REF")
        assert exc_info.value.error_type == ErrorType.INVALID_ORDER
        assert "MARGIN_CLOSE_ONLY" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_open_status_mapped(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "dealStatus": "OPEN",
                "dealId": "DID-OPEN",
                "epic": "CS.D.AVXUSD.TODAY.IP",
                "direction": "BUY",
                "size": 1.0,
                "level": 30.0,
            },
        )
        attach_session(client, "get", resp)

        result = await client.confirm_order("DEAL-REF")
        assert result.status == OrderStatus.OPEN

    @pytest.mark.preflight
    @pytest.mark.asyncio
    async def test_market_closed_raises_market_closed_error(self) -> None:
        """``MARKET_CLOSED_WITH_EDITS`` must raise MarketClosedError so callers
        can defer retry instead of treating it as a hard rejection."""
        client = make_client()
        resp = mock_response(
            200,
            {
                "dealStatus": "REJECTED",
                "reason": "MARKET_CLOSED_WITH_EDITS",
            },
        )
        attach_session(client, "get", resp)

        with pytest.raises(MarketClosedError) as exc_info:
            await client.confirm_order("DEAL-REF")
        assert exc_info.value.error_type == ErrorType.MARKET_CLOSED
        assert "MARKET_CLOSED_WITH_EDITS" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_market_offline_also_raises_market_closed_error(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {"dealStatus": "REJECTED", "reason": "MARKET_OFFLINE"},
        )
        attach_session(client, "get", resp)

        with pytest.raises(MarketClosedError):
            await client.confirm_order("DEAL-REF")


# ===========================================================================
# confirm_order — retry on deal-not-found 404
#
# IG's /confirms cache lags placement; an immediate query can 404 with
# error.confirms.deal-not-found even though the order has filled.  Without
# retry every laggy confirm produced an orphan position (May 28 2026
# GBP/NZD incident).  These tests pin the retry semantics.
# ===========================================================================


class TestConfirmOrderRetry:
    @staticmethod
    def _success_payload() -> dict[str, Any]:
        return {
            "dealId": "DEAL-123",
            "dealStatus": "ACCEPTED",
            "epic": "CS.D.EURUSD.TODAY.IP",
            "direction": "BUY",
            "size": 1.0,
            "level": 1.10,
        }

    @pytest.mark.asyncio
    async def test_first_attempt_succeeds_no_retry(self) -> None:
        """Happy path: confirm returns immediately. asyncio.sleep should not fire."""
        client = make_client()
        attach_session(client, "get", mock_response(200, self._success_payload()))

        with patch("bot.execution.ig_client.asyncio.sleep", AsyncMock()) as sleep_mock:
            result = await client.confirm_order("DEAL-REF")

        assert result.status == OrderStatus.FILLED
        sleep_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_404_deal_not_found_retries_then_succeeds(self) -> None:
        """First call 404s with deal-not-found; second call returns ACCEPTED.
        The retry must observe exactly one backoff sleep and the returned
        OrderResult must reflect the successful second response."""
        client = make_client()
        responses = [
            mock_response(
                404,
                json_body={"errorCode": "error.confirms.deal-not-found"},
            ),
            mock_response(200, self._success_payload()),
        ]
        session = MagicMock()
        session.get = MagicMock(side_effect=responses)
        client._session = session

        with patch("bot.execution.ig_client.asyncio.sleep", AsyncMock()) as sleep_mock:
            result = await client.confirm_order("DEAL-REF")

        assert result.status == OrderStatus.FILLED
        assert result.order_id == "DEAL-123"
        assert session.get.call_count == 2
        # Exactly one retry-backoff sleep at the first configured delay
        sleep_mock.assert_awaited_once_with(1.5)

    @pytest.mark.asyncio
    async def test_all_attempts_exhausted_raises_last_exception(self) -> None:
        """If every retry 404s with deal-not-found, the last ExchangeError
        propagates so the entry path can fall back to its orphan-detect flow."""
        client = make_client()
        # 4 attempts total (initial + 3 retries) — all 404
        session = MagicMock()
        session.get = MagicMock(
            return_value=mock_response(
                404,
                json_body={"errorCode": "error.confirms.deal-not-found"},
            )
        )
        client._session = session

        with (
            patch("bot.execution.ig_client.asyncio.sleep", AsyncMock()) as sleep_mock,
            pytest.raises(ExchangeError) as exc_info,
        ):
            await client.confirm_order("DEAL-REF")

        assert "deal-not-found" in str(exc_info.value)
        assert session.get.call_count == 4
        # Three sleeps at the three configured delays
        sleep_calls = [c.args[0] for c in sleep_mock.await_args_list]
        assert sleep_calls == [1.5, 3.0, 6.0]

    @pytest.mark.asyncio
    async def test_non_deal_not_found_404_propagates_without_retry(self) -> None:
        """A 404 with a *different* error code is a real not-found and must
        surface immediately — we only retry the cache-lag case."""
        client = make_client()
        session = MagicMock()
        session.get = MagicMock(
            return_value=mock_response(
                404,
                json_body={"errorCode": "error.public-api.failure.kyc.required"},
            )
        )
        client._session = session

        with (
            patch("bot.execution.ig_client.asyncio.sleep", AsyncMock()) as sleep_mock,
            pytest.raises(ExchangeError) as exc_info,
        ):
            await client.confirm_order("DEAL-REF")

        assert "deal-not-found" not in str(exc_info.value)
        assert session.get.call_count == 1
        sleep_mock.assert_not_called()

    @pytest.mark.asyncio
    async def test_rejected_response_after_retry_raises_invalid_order(self) -> None:
        """If the second attempt succeeds but the deal was REJECTED, the
        retry must NOT swallow that — surface as ExchangeError exactly as
        the pre-retry path did."""
        client = make_client()
        responses = [
            mock_response(404, json_body={"errorCode": "error.confirms.deal-not-found"}),
            mock_response(
                200,
                {
                    "dealStatus": "REJECTED",
                    "reason": "INSUFFICIENT_FUNDS",
                },
            ),
        ]
        session = MagicMock()
        session.get = MagicMock(side_effect=responses)
        client._session = session

        with (
            patch("bot.execution.ig_client.asyncio.sleep", AsyncMock()),
            pytest.raises(ExchangeError) as exc_info,
        ):
            await client.confirm_order("DEAL-REF")

        assert exc_info.value.error_type == ErrorType.INVALID_ORDER
        assert "INSUFFICIENT_FUNDS" in str(exc_info.value)


# ===========================================================================
# close_position
# ===========================================================================


class TestClosePosition:
    @pytest.mark.asyncio
    async def test_close_happy_path(self) -> None:
        client = make_client()
        # First call (DELETE with body) → returns dealReference
        delete_resp = mock_response(200, {"dealReference": "CLOSE-REF"})
        # Second call (GET /confirms/CLOSE-REF) → accepted
        confirm_resp = mock_response(
            200,
            {
                "dealStatus": "ACCEPTED",
                "dealId": "DID-CLOSE",
                "epic": "CS.D.AVXUSD.TODAY.IP",
                "direction": "SELL",
                "size": 2.0,
                "level": 32.0,
            },
        )

        session = MagicMock()
        # place_order uses POST for the DELETE-with-body; confirm uses GET
        session.post = MagicMock(return_value=delete_resp)
        session.get = MagicMock(return_value=confirm_resp)
        client._session = session

        result = await client.close_position(
            deal_id="DID-999",
            epic="CS.D.AVXUSD.TODAY.IP",
            direction="SELL",
            size=2.0,
        )
        assert result.status == OrderStatus.FILLED
        assert result.average_price == pytest.approx(32.0)

    @pytest.mark.asyncio
    async def test_close_raises_if_no_deal_reference(self) -> None:
        client = make_client()
        # IG returns response without dealReference
        delete_resp = mock_response(200, {})
        attach_session(client, "post", delete_resp)

        with pytest.raises(RuntimeError, match="no dealReference"):
            await client.close_position(
                deal_id="DID-999",
                epic="CS.D.AVXUSD.TODAY.IP",
                direction="SELL",
                size=1.0,
            )

    @pytest.mark.asyncio
    async def test_close_sends_delete_method_header(self) -> None:
        """IG requires _method: DELETE override when sending body."""
        client = make_client()
        delete_resp = mock_response(200, {"dealReference": "CLOSE-REF"})
        confirm_resp = mock_response(
            200,
            {
                "dealStatus": "ACCEPTED",
                "dealId": "DID-CLOSE",
                "epic": "CS.D.AVXUSD.TODAY.IP",
                "direction": "SELL",
                "size": 1.0,
                "level": 30.0,
            },
        )
        session = MagicMock()
        session.post = MagicMock(return_value=delete_resp)
        session.get = MagicMock(return_value=confirm_resp)
        client._session = session

        await client.close_position(
            deal_id="DID-999",
            epic="CS.D.AVXUSD.TODAY.IP",
            direction="SELL",
            size=1.0,
        )
        post_call = session.post.call_args
        sent_headers = post_call.kwargs.get("headers") or {}
        assert sent_headers.get("_method") == "DELETE"


# ===========================================================================
# fetch_closed_transaction
# ===========================================================================


class TestFetchClosedTransaction:
    """Lookup of the GET /history/transactions record matching a closed position."""

    @pytest.mark.asyncio
    async def test_matches_by_open_timestamp(self) -> None:
        client = make_client()
        # Position opened at 2026-05-10 23:08:04 UTC (1778591284000 ms)
        opened_at_ms = 1778454484000
        resp = mock_response(
            200,
            {
                "transactions": [
                    {
                        "instrumentName": "GBP/CHF",
                        "openDateUtc": "2026-05-10T23:08:04",
                        "closeLevel": "1.0521",
                        "openLevel": "1.0573",
                        "profitAndLoss": "E-12.30",
                        "size": "+2.07",
                        "period": "DFB",
                    },
                    {
                        "instrumentName": "EUR/USD",
                        "openDateUtc": "2026-05-10T21:00:00",
                        "closeLevel": "1.1700",
                        "profitAndLoss": "E+5.00",
                    },
                ]
            },
        )
        attach_session(client, "get", resp)

        txn = await client.fetch_closed_transaction(opened_at_ms=opened_at_ms)
        assert txn is not None
        assert txn["instrumentName"] == "GBP/CHF"
        assert txn["closeLevel"] == "1.0521"

    @pytest.mark.asyncio
    async def test_returns_none_when_no_match(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "transactions": [
                    {
                        "instrumentName": "EUR/USD",
                        "openDateUtc": "2026-05-10T21:00:00",
                        "closeLevel": "1.1700",
                    }
                ]
            },
        )
        attach_session(client, "get", resp)

        txn = await client.fetch_closed_transaction(opened_at_ms=1778454484000)
        assert txn is None

    @pytest.mark.asyncio
    async def test_returns_none_on_zero_opened_at(self) -> None:
        client = make_client()
        # No HTTP call should be made for opened_at_ms=0
        client._session = MagicMock()
        txn = await client.fetch_closed_transaction(opened_at_ms=0)
        assert txn is None

    @pytest.mark.asyncio
    async def test_instrument_filter_narrows_match(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "transactions": [
                    {
                        "instrumentName": "GBP/CHF spread bet",
                        "openDateUtc": "2026-05-10T23:08:04",
                        "closeLevel": "1.0521",
                    },
                    {
                        "instrumentName": "Spot Gold",
                        "openDateUtc": "2026-05-10T23:08:04",
                        "closeLevel": "2400",
                    },
                ]
            },
        )
        attach_session(client, "get", resp)

        txn = await client.fetch_closed_transaction(
            opened_at_ms=1778454484000,
            instrument_name_contains="Spot Gold",
        )
        assert txn is not None
        assert txn["instrumentName"] == "Spot Gold"

    @pytest.mark.asyncio
    async def test_swallows_http_errors(self) -> None:
        """A failing /history/transactions call must not raise; the caller
        falls back to a candle-based estimate."""
        client = make_client()
        session = MagicMock()
        # 500 returns will trigger ExchangeError in _handle_response
        session.get = MagicMock(return_value=mock_response(500, {}, "boom"))
        client._session = session
        txn = await client.fetch_closed_transaction(opened_at_ms=1778454484000)
        assert txn is None


# ===========================================================================
# fetch_ohlcv
# ===========================================================================


class TestFetchOhlcv:
    def _make_price_item(
        self,
        ts: str = "2026/04/10 19:00:00",
        o: float = 30.0,
        h: float = 31.0,
        lo: float = 29.0,
        c: float = 30.5,
        vol: float = 100.0,
    ) -> dict[str, Any]:
        def price(bid: float, ask: float) -> dict[str, str]:
            return {"bid": str(bid), "ask": str(ask)}

        return {
            "snapshotTimeUTC": ts,
            "openPrice": price(o - 0.1, o + 0.1),
            "highPrice": price(h - 0.1, h + 0.1),
            "lowPrice": price(lo - 0.1, lo + 0.1),
            "closePrice": price(c - 0.1, c + 0.1),
            "lastTradedVolume": str(vol),
        }

    @pytest.mark.asyncio
    async def test_candles_parsed_correctly(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "allowance": {
                    "remainingAllowance": 9900,
                    "totalAllowance": 10000,
                    "allowanceExpiry": 604800,
                },
                "prices": [
                    self._make_price_item("2026/04/10 19:00:00", o=30.0, h=31.0, lo=29.0, c=30.5)
                ],
            },
        )
        attach_session(client, "get", resp)

        candles = await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "1m", limit=1)
        assert len(candles) == 1
        assert candles[0].symbol == "CS.D.AVXUSD.TODAY.IP"
        assert candles[0].is_confirmed is True
        assert candles[0].close == pytest.approx(30.5)

    @pytest.mark.asyncio
    async def test_quota_tracked(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "allowance": {
                    "remainingAllowance": 7500,
                    "totalAllowance": 10000,
                    "allowanceExpiry": 604800,
                },  # noqa: E501
                "prices": [],
            },
        )
        attach_session(client, "get", resp)

        await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "1m", limit=1)
        assert client.datapoints_remaining == 7500

    @pytest.mark.asyncio
    async def test_low_quota_logged_as_error(self, caplog: Any) -> None:
        import logging

        client = make_client()
        resp = mock_response(
            200,
            {
                "allowance": {
                    "remainingAllowance": 1500,
                    "totalAllowance": 10000,
                    "allowanceExpiry": 604800,
                },
                "prices": [],
            },
        )
        attach_session(client, "get", resp)

        with caplog.at_level(logging.ERROR, logger="bot.execution.ig_client"):
            await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "1m", limit=1)
        assert any("LOW" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_warning_quota_logged(self, caplog: Any) -> None:
        import logging

        client = make_client()
        resp = mock_response(
            200,
            {
                "allowance": {
                    "remainingAllowance": 3000,
                    "totalAllowance": 10000,
                    "allowanceExpiry": 604800,
                },
                "prices": [],
            },
        )
        attach_session(client, "get", resp)

        with caplog.at_level(logging.WARNING, logger="bot.execution.ig_client"):
            await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "1m", limit=1)
        assert any("warning" in r.message.lower() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_unsupported_timeframe_raises(self) -> None:
        client = make_client()
        client._session = MagicMock()
        with pytest.raises(ValueError, match="Unsupported timeframe"):
            await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "45m", limit=10)

    @pytest.mark.asyncio
    async def test_candles_sorted_ascending(self) -> None:
        client = make_client()
        # Two candles in reverse order
        resp = mock_response(
            200,
            {
                "allowance": {},
                "prices": [
                    self._make_price_item("2026/04/10 19:01:00"),
                    self._make_price_item("2026/04/10 19:00:00"),
                ],
            },
        )
        attach_session(client, "get", resp)

        candles = await client.fetch_ohlcv("CS.D.AVXUSD.TODAY.IP", "1m", limit=2)
        assert candles[0].timestamp < candles[1].timestamp


# ===========================================================================
# connect — fresh login and cache paths
# ===========================================================================


class TestConnect:
    @pytest.mark.asyncio
    async def test_fresh_login_when_no_cache(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        client = IGClient(make_config())

        session_resp = MagicMock()
        session_resp.status = 200
        session_resp.headers = {"CST": "new-cst", "X-SECURITY-TOKEN": "new-xst"}
        session_resp.json = AsyncMock(
            return_value={
                "currentAccountId": "Z6ACA1",
                "lightstreamerEndpoint": "https://ls.ig.com",
            }
        )
        session_resp.__aenter__ = AsyncMock(return_value=session_resp)
        session_resp.__aexit__ = AsyncMock(return_value=False)

        accounts_resp = mock_response(
            200,
            {
                "accounts": [
                    {"accountId": "Z6ACA1", "accountType": "CFD"},
                    {"accountId": "Z6ACA2", "accountType": "SPREADBET"},
                ]
            },
        )

        switch_resp = MagicMock()
        switch_resp.status = 200
        switch_resp.headers = {"X-SECURITY-TOKEN": "switched-xst"}
        switch_resp.json = AsyncMock(return_value={"lightstreamerEndpoint": "https://ls.ig.com"})
        switch_resp.__aenter__ = AsyncMock(return_value=switch_resp)
        switch_resp.__aexit__ = AsyncMock(return_value=False)

        mock_session = MagicMock()
        mock_session.post = MagicMock(return_value=session_resp)
        mock_session.get = MagicMock(return_value=accounts_resp)
        mock_session.put = MagicMock(return_value=switch_resp)

        with (
            patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file),
            patch("aiohttp.ClientSession", return_value=mock_session),
            patch.object(client._sess, "save_tokens"),
            patch.object(client._sess, "refresh_loop", new_callable=AsyncMock),
        ):
            # Patch create_task to avoid background tasks in tests
            import asyncio

            with patch.object(asyncio, "create_task", side_effect=_swallow_create_task):
                await client.connect()

        assert client._cst == "new-cst"
        assert client._account_id == "Z6ACA2"  # switched to spreadbet

    @pytest.mark.asyncio
    async def test_cached_tokens_used_when_fresh(self, tmp_path: Path) -> None:
        cache_file = tmp_path / ".ig_session_cache.json"
        payload = {
            "cst": "cached-cst",
            "xst": "cached-xst",
            "account_id": "Z6ACA2",
            "ls_endpoint": "https://ls.ig.com",
            "env": "demo",
            "saved_at": time.time(),  # fresh
        }
        cache_file.write_text(json.dumps(payload))
        cache_file.chmod(0o600)

        client = IGClient(make_config())
        accounts_ok = mock_response(200, {"accounts": []})
        mock_session = MagicMock()
        mock_session.get = MagicMock(return_value=accounts_ok)

        with (
            patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file),
            patch("aiohttp.ClientSession", return_value=mock_session),
        ):
            import asyncio

            with patch.object(asyncio, "create_task", side_effect=_swallow_create_task):
                await client.connect()

        assert client._cst == "cached-cst"
        assert client._xst == "cached-xst"
        # No POST /session should have been called
        mock_session.post.assert_not_called()

    @pytest.mark.asyncio
    async def test_session_constructed_with_request_timeout(self, tmp_path: Path) -> None:
        """Regression for the 2026-05-07 startup hang: the session must be
        created with a per-request ``ClientTimeout`` so a single stalled IG
        endpoint cannot freeze the whole event loop indefinitely.
        """
        import aiohttp

        from bot.execution.ig_client import _REQUEST_TIMEOUT_S

        cache_file = tmp_path / ".ig_session_cache.json"
        client = IGClient(make_config())

        # Fresh-login path is exercised; we just need to inspect the session
        # construction call, not the full happy path.
        captured: dict[str, Any] = {}

        def _capture(*args: Any, **kwargs: Any) -> MagicMock:
            captured.update(kwargs)
            mock = MagicMock()
            mock.post = MagicMock(return_value=mock_response(401, text_body="reject"))
            mock.get = MagicMock(return_value=mock_response(401, text_body="reject"))
            return mock

        with (
            patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file),
            patch("aiohttp.ClientSession", side_effect=_capture),
            pytest.raises(ExchangeError),
        ):
            await client.connect()

        assert "timeout" in captured, "ClientSession must be constructed with timeout="
        timeout_obj = captured["timeout"]
        assert isinstance(timeout_obj, aiohttp.ClientTimeout)
        assert timeout_obj.total == _REQUEST_TIMEOUT_S

    @pytest.mark.asyncio
    async def test_cached_token_timeout_falls_back_to_reauth(self, tmp_path: Path) -> None:
        """If the cached-token verification call times out (the production
        scenario observed on 2026-05-07 — IG demo stalled on ``GET /accounts``
        for ~4 minutes), the bot must catch the timeout and re-authenticate
        rather than propagate it and crash startup.
        """
        cache_file = tmp_path / ".ig_session_cache.json"
        cache_file.write_text(
            json.dumps(
                {
                    "cst": "cached-cst",
                    "xst": "cached-xst",
                    "account_id": "Z6ACA2",
                    "ls_endpoint": "https://ls.ig.com",
                    "env": "demo",
                    "saved_at": time.time(),
                }
            )
        )
        cache_file.chmod(0o600)

        client = IGClient(make_config())

        with (
            patch("bot.execution.ig_session._TOKEN_CACHE_FILE", cache_file),
            patch("aiohttp.ClientSession", return_value=MagicMock()),
            patch.object(client._http, "get", AsyncMock(side_effect=TimeoutError())),
            patch.object(client._sess, "create_session", AsyncMock()) as mock_create,
            patch.object(client._sess, "switch_to_spreadbet", AsyncMock()) as mock_switch,
            patch.object(client._sess, "save_tokens"),
            patch.object(client._sess, "refresh_loop", new_callable=AsyncMock),
        ):
            import asyncio

            with patch.object(asyncio, "create_task", side_effect=_swallow_create_task):
                await client.connect()

        mock_create.assert_awaited_once()
        mock_switch.assert_awaited_once()


# ===========================================================================
# _switch_to_spreadbet
# ===========================================================================


class TestSwitchToSpreadbet:
    @pytest.mark.asyncio
    async def test_already_on_spreadbet_no_switch(self) -> None:
        client = make_client()
        accounts_resp = mock_response(
            200,
            {
                "accounts": [
                    {"accountId": "Z6ACA2", "accountType": "SPREADBET"},
                ]
            },
        )
        session = MagicMock()
        session.get = MagicMock(return_value=accounts_resp)
        client._session = session

        await client._sess.switch_to_spreadbet()
        # No PUT call should have been made
        session.put.assert_not_called()

    @pytest.mark.asyncio
    async def test_no_spreadbet_account_raises(self) -> None:
        client = make_client()
        accounts_resp = mock_response(
            200,
            {
                "accounts": [
                    {"accountId": "Z6ACA2", "accountType": "CFD"},
                ]
            },
        )
        attach_session(client, "get", accounts_resp)

        with pytest.raises(ExchangeError, match="No SPREADBET account"):
            await client._sess.switch_to_spreadbet()

    @pytest.mark.asyncio
    async def test_switch_updates_account_id_and_xst(self) -> None:
        client = make_client()
        client._account_id = "Z6ACA1"

        accounts_resp = mock_response(
            200,
            {
                "accounts": [
                    {"accountId": "Z6ACA1", "accountType": "CFD"},
                    {"accountId": "Z6ACA2", "accountType": "SPREADBET"},
                ]
            },
        )

        switch_resp = MagicMock()
        switch_resp.status = 200
        switch_resp.headers = {"X-SECURITY-TOKEN": "new-xst-after-switch"}
        switch_resp.json = AsyncMock(
            return_value={"lightstreamerEndpoint": "https://ls.ig.com/new"}
        )
        switch_resp.__aenter__ = AsyncMock(return_value=switch_resp)
        switch_resp.__aexit__ = AsyncMock(return_value=False)

        session = MagicMock()
        session.get = MagicMock(return_value=accounts_resp)
        session.put = MagicMock(return_value=switch_resp)
        client._session = session

        await client._sess.switch_to_spreadbet()

        assert client._account_id == "Z6ACA2"
        assert client._xst == "new-xst-after-switch"
        assert client._ls_endpoint == "https://ls.ig.com/new"


# ===========================================================================
# fetch_positions
# ===========================================================================


class TestFetchPositions:
    @pytest.mark.asyncio
    async def test_parses_position_list(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "positions": [
                    {
                        "position": {
                            "direction": "BUY",
                            "level": 31.5,
                            "size": 2.0,
                            "upl": 3.0,
                            "createdDateUTC": "2026-04-10T19:00:00",
                        },
                        "market": {"epic": "CS.D.AVXUSD.TODAY.IP", "bid": 31.8},
                    }
                ]
            },
        )
        attach_session(client, "get", resp)

        positions = await client.fetch_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p.symbol == "CS.D.AVXUSD.TODAY.IP"
        assert p.side == OrderSide.BUY
        assert p.entry_price == pytest.approx(31.5)
        assert p.quantity == pytest.approx(2.0)

    @pytest.mark.asyncio
    async def test_symbol_filter(self) -> None:
        client = make_client()
        resp = mock_response(
            200,
            {
                "positions": [
                    {
                        "position": {
                            "direction": "BUY",
                            "level": 30.0,
                            "size": 1.0,
                            "upl": 0.0,
                            "createdDateUTC": "",
                        },
                        "market": {"epic": "CS.D.AVXUSD.TODAY.IP", "bid": 30.0},
                    },
                    {
                        "position": {
                            "direction": "BUY",
                            "level": 45000.0,
                            "size": 0.5,
                            "upl": 0.0,
                            "createdDateUTC": "",
                        },
                        "market": {"epic": "CS.D.BITCOIN.TODAY.IP", "bid": 45000.0},
                    },
                ]
            },
        )
        attach_session(client, "get", resp)

        positions = await client.fetch_positions(symbol="CS.D.AVXUSD.TODAY.IP")
        assert len(positions) == 1
        assert positions[0].symbol == "CS.D.AVXUSD.TODAY.IP"

    @pytest.mark.asyncio
    async def test_empty_positions_list(self) -> None:
        client = make_client()
        resp = mock_response(200, {"positions": []})
        attach_session(client, "get", resp)

        positions = await client.fetch_positions()
        assert positions == []


# ===========================================================================
# is_connected property
# ===========================================================================


class TestIsConnected:
    def test_connected_when_session_and_cst_present(self) -> None:
        client = make_client()
        client._session = MagicMock()
        assert client.is_connected is True

    def test_not_connected_when_no_session(self) -> None:
        client = make_client()
        client._session = None
        assert client.is_connected is False

    def test_not_connected_when_cst_empty(self) -> None:
        client = make_client()
        client._session = MagicMock()
        client._cst = ""
        assert client.is_connected is False


# ===========================================================================
# REST resilience layer (IG_LIVE_RISK_REFERENCE.md §2) — keep-alive,
# rate-limit buckets, retry/backoff,
# 401 force-refresh.
# ===========================================================================


@pytest.mark.preflight
class TestRequestRetry:
    """The ``_request`` central HTTP path applies exponential backoff with jitter
    on transient errors (HTTP 403 / 429 / 5xx) and a single forced session refresh
    on HTTP 401."""

    @pytest.mark.asyncio
    async def test_5xx_retries_then_succeeds(self) -> None:
        client = make_client()
        # First call: 503. Second call: 200.
        responses = [
            mock_response(503, text_body="service unavailable"),
            mock_response(200, {"ok": True}),
        ]
        session = MagicMock()
        session.get = MagicMock(side_effect=responses)
        client._session = session

        with patch("bot.execution.ig_http.asyncio.sleep", AsyncMock()):
            result = await client._http.get("/accounts", version="1", authenticated=True)
        assert result == {"ok": True}
        assert session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_403_retries_with_backoff(self) -> None:
        client = make_client()
        responses = [
            mock_response(
                403, json_body={"errorCode": "error.public-api.exceeded-account-trading-allowance"}
            ),
            mock_response(200, {"ok": True}),
        ]
        session = MagicMock()
        session.post = MagicMock(side_effect=responses)
        client._session = session

        with patch("bot.execution.ig_http.asyncio.sleep", AsyncMock()) as sleep_mock:
            result = await client._post(
                "/positions/otc", body={"x": 1}, version="2", authenticated=True
            )
        assert result == {"ok": True}
        assert session.post.call_count == 2
        sleep_mock.assert_awaited()  # backoff happened

    @pytest.mark.asyncio
    async def test_401_forces_refresh_then_retries_once(self) -> None:
        client = make_client()
        responses = [
            mock_response(401, text_body="auth expired"),
            mock_response(200, {"ok": True}),
        ]
        session = MagicMock()
        session.get = MagicMock(side_effect=responses)
        client._session = session

        refresh = AsyncMock()
        with (
            patch.object(client, "refresh_session", refresh),
            patch("bot.execution.ig_http.asyncio.sleep", AsyncMock()),
        ):
            result = await client._http.get("/accounts", version="1", authenticated=True)

        assert result == {"ok": True}
        refresh.assert_awaited_once()
        assert session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_second_401_after_refresh_raises(self) -> None:
        client = make_client()
        # Both calls 401 — second should propagate without further retries.
        session = MagicMock()
        session.get = MagicMock(return_value=mock_response(401, text_body="still bad"))
        client._session = session

        refresh = AsyncMock()
        with (
            patch.object(client, "refresh_session", refresh),
            patch("bot.execution.ig_http.asyncio.sleep", AsyncMock()),
            pytest.raises(ExchangeError) as exc_info,
        ):
            await client._http.get("/accounts", version="1", authenticated=True)

        assert exc_info.value.error_type == ErrorType.AUTHENTICATION_FAILED
        refresh.assert_awaited_once()
        # Two HTTP attempts: original + post-refresh retry.
        assert session.get.call_count == 2

    @pytest.mark.asyncio
    async def test_400_does_not_retry(self) -> None:
        """HTTP 400 is a payload bug — must surface immediately without retries."""
        client = make_client()
        session = MagicMock()
        session.post = MagicMock(return_value=mock_response(400, text_body="bad payload"))
        client._session = session

        with (
            patch("bot.execution.ig_http.asyncio.sleep", AsyncMock()) as sleep_mock,
            pytest.raises(ExchangeError),
        ):
            await client._post("/positions/otc", body={"x": 1}, version="2", authenticated=True)
        assert session.post.call_count == 1
        sleep_mock.assert_not_awaited()


class TestRateLimitBuckets:
    """Each REST call type has its own per-minute token bucket so a trade-burst
    can't starve historical-price or account-info calls (and vice-versa)."""

    def test_bucket_for_path_classifies_paths(self) -> None:

        assert bucket_for_path("GET", "/prices/EPIC/HOUR/100") is Bucket.HISTORICAL
        assert bucket_for_path("POST", "/positions/otc") is Bucket.TRADE
        assert bucket_for_path("DELETE", "/workingorders/otc/123") is Bucket.TRADE
        assert bucket_for_path("GET", "/accounts") is Bucket.ACCOUNT
        assert bucket_for_path("GET", "/markets/EPIC") is Bucket.ACCOUNT

    @pytest.mark.asyncio
    async def test_token_bucket_emits_at_configured_rate(self) -> None:
        """A 60/min bucket should hand out tokens roughly every 1 s once
        the initial burst is consumed.  We mock event-loop time to make this
        deterministic."""

        bucket = TokenBucket(per_minute=60)  # 1 token / sec
        # Drain the initial burst (capacity = 60 tokens)
        for _ in range(60):
            await bucket.acquire()

        # 61st token must wait — patch sleep to detect the wait without delay
        sleep_calls: list[float] = []

        async def _record_sleep(s: float) -> None:
            sleep_calls.append(s)

        with patch("bot.execution.ig_client.asyncio.sleep", _record_sleep):
            await bucket.acquire()

        assert sleep_calls, "61st acquire on a drained bucket must have slept"
        # Should sleep close to 1 second to refill one token at 1/sec
        assert 0.5 <= sleep_calls[0] <= 1.5


class TestKeepalive:
    """The keep-alive loop pings IG every 45 s to prevent the v2 session-token
    inactivity timeout (IG_LIVE_RISK_REFERENCE.md §2.2)."""

    @pytest.mark.asyncio
    async def test_keepalive_pings_accounts_endpoint(self) -> None:
        import asyncio as _asyncio

        from bot.execution.ig_session import _KEEPALIVE_PATH

        client = make_client()
        client._session = MagicMock()
        ping = AsyncMock(return_value={"accounts": []})

        # Patch the interval to a tick — avoid mocking asyncio.sleep itself
        # (that breaks the test's own scheduling primitives).
        with (
            patch("bot.execution.ig_session._KEEPALIVE_INTERVAL_S", 0.001),
            patch.object(client._http, "get", ping),
        ):
            task = _asyncio.create_task(client._sess.keepalive_loop())
            await _asyncio.sleep(0.05)  # let several iterations execute
            task.cancel()
            with contextlib.suppress(_asyncio.CancelledError):
                await task

        assert ping.call_count >= 1
        args, kwargs = ping.call_args
        assert args[0] == _KEEPALIVE_PATH
        assert kwargs["authenticated"] is True

    @pytest.mark.asyncio
    async def test_keepalive_skips_when_disconnected(self) -> None:
        """Don't ping if there's no live session — would just generate noise."""
        import asyncio as _asyncio

        client = make_client()
        client._session = None  # disconnected
        ping = AsyncMock()

        with (
            patch("bot.execution.ig_session._KEEPALIVE_INTERVAL_S", 0.001),
            patch.object(client._http, "get", ping),
        ):
            task = _asyncio.create_task(client._sess.keepalive_loop())
            await _asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(_asyncio.CancelledError):
                await task

        ping.assert_not_called()

    @pytest.mark.asyncio
    async def test_keepalive_swallows_ping_failure(self) -> None:
        """A transient ping error must not kill the loop — we want it to keep
        trying on the next tick rather than silently leaving tokens to expire."""
        import asyncio as _asyncio

        client = make_client()
        client._session = MagicMock()
        # First ping raises, second succeeds
        ping = AsyncMock(side_effect=[RuntimeError("transient"), {"accounts": []}])

        with (
            patch("bot.execution.ig_session._KEEPALIVE_INTERVAL_S", 0.001),
            patch.object(client._http, "get", ping),
        ):
            task = _asyncio.create_task(client._sess.keepalive_loop())
            await _asyncio.sleep(0.05)
            task.cancel()
            with contextlib.suppress(_asyncio.CancelledError):
                await task

        assert ping.call_count >= 2  # didn't die on the first error


# ===========================================================================
# Pre-trade market-status gate (require_tradeable).
# IG_LIVE_RISK_REFERENCE.md §1.3 — refuse entries on non-TRADEABLE markets.
# ===========================================================================


class TestAssertTradeable:
    @pytest.mark.asyncio
    async def test_tradeable_no_rules_returns_none(self) -> None:
        client = make_client()
        resp = mock_response(200, {"snapshot": {"marketStatus": "TRADEABLE"}})
        attach_session(client, "get", resp)
        assert await client.require_tradeable("CS.D.EURUSD.MINI.IP") is None  # no exception

    @pytest.mark.asyncio
    async def test_tradeable_returns_min_deal_size(self) -> None:
        """On a tradeable market, surface dealingRules.minDealSize so the caller
        can skip a sub-minimum stake (the MINIMUM_ORDER_SIZE_ERROR seen on
        higher-priced US shares)."""
        client = make_client()
        resp = mock_response(
            200,
            {
                "snapshot": {"marketStatus": "TRADEABLE"},
                "dealingRules": {"minDealSize": {"unit": "POINTS", "value": 0.24}},
            },
        )
        attach_session(client, "get", resp)
        assert await client.require_tradeable("SH.D.XOM.DAILY.IP") == pytest.approx(0.24)

    @pytest.mark.asyncio
    async def test_closed_raises_market_closed(self) -> None:
        client = make_client()
        resp = mock_response(200, {"snapshot": {"marketStatus": "CLOSED"}})
        attach_session(client, "get", resp)
        with pytest.raises(MarketClosedError) as exc_info:
            await client.require_tradeable("CS.D.EURUSD.MINI.IP")
        assert exc_info.value.error_type == ErrorType.MARKET_CLOSED
        assert "CLOSED" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_market_closed_with_edits_raises(self) -> None:
        """The 22:00 UTC daily funding-tick state that hit forex on 2026-04-28
        — must be caught before placing an order, not after the broker rejects."""
        client = make_client()
        resp = mock_response(200, {"snapshot": {"marketStatus": "MARKET_CLOSED_WITH_EDITS"}})
        attach_session(client, "get", resp)
        with pytest.raises(MarketClosedError):
            await client.require_tradeable("CS.D.EURUSD.MINI.IP")

    @pytest.mark.asyncio
    async def test_each_restricted_state_blocks(self) -> None:
        for status in ("EDITS_ONLY", "OFFLINE", "ON_AUCTION", "SUSPENDED"):
            client = make_client()
            resp = mock_response(200, {"snapshot": {"marketStatus": status}})
            attach_session(client, "get", resp)
            with pytest.raises(MarketClosedError):
                await client.require_tradeable("CS.D.EURUSD.MINI.IP")

    @pytest.mark.asyncio
    async def test_missing_snapshot_treated_as_unknown(self) -> None:
        """If IG returns no snapshot block we can't prove it's tradeable —
        better to skip than to fire a blind order."""
        client = make_client()
        resp = mock_response(200, {})
        attach_session(client, "get", resp)
        with pytest.raises(MarketClosedError) as exc_info:
            await client.require_tradeable("CS.D.EURUSD.MINI.IP")
        assert "UNKNOWN" in str(exc_info.value)
