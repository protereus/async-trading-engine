"""HTTP transport layer for IG REST.

Extracted from ``ig_client.py``.  Owns:

* The per-bucket token-bucket rate limiter
  (IG_LIVE_RISK_REFERENCE.md §2.1).
* Header construction (``X-IG-API-KEY``, ``CST``, ``X-SECURITY-TOKEN``,
  ``VERSION``).
* The retry-with-backoff central HTTP entry point and the response → dict
  / ``ExchangeError`` translator.

The aiohttp ``ClientSession`` itself stays on ``IGClient`` (so the
lifecycle and the LS endpoint introspection stay together).  ``IGHttp``
holds a back-reference to its parent ``IGClient`` and reads the live
session/cst/xst off it; on 401 it calls back into the parent's
``refresh_session`` (which delegates to ``IGSession``).
"""

from __future__ import annotations

import asyncio
import logging
import random
from enum import StrEnum
from typing import TYPE_CHECKING, Any

import aiohttp

from bot.core.models import ErrorType, ExchangeError

if TYPE_CHECKING:
    from bot.execution.ig_client import IGClient

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Resilience knobs (IG_LIVE_RISK_REFERENCE.md §2).
#
# Rate-limit buffers follow §2.1: configured at ``allowance - 2`` per minute.
# IG's published defaults are 40/min trade, 10/min historical-price,
# ~30/min account/info.  Buckets are independent — a trade call does not eat
# from the historical or info budget.
# ---------------------------------------------------------------------------
_MAX_RETRIES = 6
_RETRY_INITIAL_DELAY_S = 1.0
_RETRY_MAX_DELAY_S = 30.0
_RETRY_JITTER_S = 2.0

# Failures where the request may have *already executed* server-side before the
# response was lost.  Blind-retrying these on a non-idempotent request (order
# placement) risks a duplicate position, so they fail closed instead.  Rate-limit
# and market-closed are NOT here: they mean the request was rejected, not
# executed, so they stay retryable even for non-idempotent calls.
_AMBIGUOUS_ON_RETRY = frozenset(
    {
        ErrorType.NETWORK_TIMEOUT,
        ErrorType.CONNECTION_ERROR,
        ErrorType.SERVICE_UNAVAILABLE,
    }
)

_RATE_LIMIT_TRADE_PER_MIN = 38  # allowance 40 − 2
_RATE_LIMIT_HISTORICAL_PER_MIN = 8  # allowance 10 − 2
_RATE_LIMIT_ACCOUNT_PER_MIN = 28  # allowance 30 − 2


class Bucket(StrEnum):
    TRADE = "trade"
    HISTORICAL = "historical"
    ACCOUNT = "account"


# IG REST error-code prefixes that indicate rate-limit pressure rather than a
# payload bug.  The full taxonomy is in IG_LIVE_RISK_REFERENCE.md §2.1.
_RATE_LIMIT_ERROR_CODES = (
    "error.public-api.exceeded-api-key-allowance",
    "error.public-api.exceeded-account-allowance",
    "error.public-api.exceeded-account-trading-allowance",
    "error.public-api.exceeded-account-historical-data-allowance",
)


class TokenBucket:
    """Async-safe per-minute token bucket.

    Single-event-loop only — no threading primitives.  ``acquire()`` blocks
    until at least one token is available, refilling continuously at the
    configured rate.
    """

    def __init__(self, per_minute: int) -> None:
        self._capacity = float(per_minute)
        self._refill_per_sec = per_minute / 60.0
        self._tokens = float(per_minute)
        self._last: float = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        while True:
            wait_s = 0.0
            async with self._lock:
                now = asyncio.get_event_loop().time()
                if self._last == 0.0:
                    self._last = now
                elapsed = now - self._last
                self._tokens = min(self._capacity, self._tokens + elapsed * self._refill_per_sec)
                self._last = now
                if self._tokens >= 1.0:
                    self._tokens -= 1.0
                    return
                deficit = 1.0 - self._tokens
                wait_s = deficit / self._refill_per_sec
            await asyncio.sleep(wait_s)


def bucket_for_path(method: str, path: str) -> Bucket:
    """Classify an IG REST path into a rate-limit bucket.

    Trade-path includes order/position/working-order mutations.  Historical
    is the /prices endpoint.  Everything else is account/info.
    """
    if path.startswith("/prices"):
        return Bucket.HISTORICAL
    if method.upper() == "GET":
        return Bucket.ACCOUNT
    if path.startswith(("/positions", "/workingorders")):
        return Bucket.TRADE
    return Bucket.ACCOUNT


class IGHttp:
    """HTTP transport for IG REST.

    Holds a back-reference to its parent ``IGClient`` so it can read live
    session state (``_session``, ``_cst``, ``_xst``, ``_base``, ``_config``,
    ``_refreshing``) and trigger ``refresh_session`` on 401.
    """

    def __init__(self, client: IGClient) -> None:
        self._client = client
        # Per-bucket rate limiters — IG_LIVE_RISK_REFERENCE.md §2.1
        self._buckets: dict[Bucket, TokenBucket] = {
            Bucket.TRADE: TokenBucket(_RATE_LIMIT_TRADE_PER_MIN),
            Bucket.HISTORICAL: TokenBucket(_RATE_LIMIT_HISTORICAL_PER_MIN),
            Bucket.ACCOUNT: TokenBucket(_RATE_LIMIT_ACCOUNT_PER_MIN),
        }

    def headers(self, version: str, authenticated: bool) -> dict[str, str]:
        c = self._client
        cfg = c._config
        api_key = cfg.ig_demo_api if cfg.bot_env == "demo" else cfg.ig_live_api
        h: dict[str, str] = {
            "X-IG-API-KEY": api_key,
            "VERSION": version,
            "Content-Type": "application/json",
            "Accept": "application/json; charset=UTF-8",
        }
        if authenticated:
            h["CST"] = c._cst
            h["X-SECURITY-TOKEN"] = c._xst
        return h

    async def get(self, path: str, version: str, authenticated: bool) -> dict[str, Any]:
        return await self.request("GET", path, version=version, authenticated=authenticated)

    async def post(
        self,
        path: str,
        body: dict[str, Any],
        version: str,
        authenticated: bool,
        *,
        idempotent: bool = True,
    ) -> dict[str, Any]:
        return await self.request(
            "POST",
            path,
            body=body,
            version=version,
            authenticated=authenticated,
            idempotent=idempotent,
        )

    async def delete(self, path: str, version: str, authenticated: bool) -> dict[str, Any]:
        return await self.request("DELETE", path, version=version, authenticated=authenticated)

    async def delete_with_body(
        self, path: str, body: dict[str, Any], version: str, authenticated: bool
    ) -> dict[str, Any]:
        """DELETE with a JSON body (used by /positions/otc close).

        IG requires the ``_method: DELETE`` override header when sending
        a body with what is technically a POST request.
        """
        return await self.request(
            "POST",
            path,
            body=body,
            version=version,
            authenticated=authenticated,
            extra_headers={"_method": "DELETE"},
        )

    async def request(
        self,
        method: str,
        path: str,
        *,
        body: dict[str, Any] | None = None,
        version: str,
        authenticated: bool,
        extra_headers: dict[str, str] | None = None,
        idempotent: bool = True,
    ) -> dict[str, Any]:
        """Central HTTP entry-point.

        Applies the per-bucket rate limiter, exponential backoff + jitter on
        transient errors (HTTP 403 / 429 / 5xx and ``exceeded-*-allowance``),
        and a single forced session refresh + retry on HTTP 401.  Permanent
        errors (HTTP 400, payload validation, second 401) propagate without
        retry as per IG_LIVE_RISK_REFERENCE.md §2.1.

        ``idempotent=False`` (order placement) makes ambiguous failures —
        timeout / connection drop / 5xx, where IG may have already executed the
        request before the response was lost — fail closed instead of retrying,
        so a lost ACK can't become a duplicate position.  The caller reconciles
        (the resulting position, if any, surfaces as an orphan).  Rate-limit and
        market-closed still retry: those reject the request rather than execute
        it.  IG does not honour ``dealReference`` as an idempotency key.
        """
        c = self._client
        assert c._session is not None
        bucket = bucket_for_path(method, path)
        url = f"{c._base}{path}"

        attempt = 0
        auth_retry_used = False
        last_exc: BaseException | None = None

        while attempt <= _MAX_RETRIES:
            await self._buckets[bucket].acquire()
            token_at_send = c._cst
            headers = self.headers(version, authenticated)
            if extra_headers:
                headers.update(extra_headers)

            try:
                if method == "GET":
                    cm = c._session.get(url, headers=headers)
                elif method == "POST":
                    cm = c._session.post(url, json=body, headers=headers)
                elif method == "DELETE":
                    cm = c._session.delete(url, headers=headers)
                else:
                    raise ValueError(f"Unsupported HTTP method: {method}")
                async with cm as resp:
                    return await self._handle_response(resp)
            except ExchangeError as exc:
                last_exc = exc
                if exc.error_type == ErrorType.AUTHENTICATION_FAILED:
                    if auth_retry_used or c._refreshing:
                        # Already retried once, or we're inside a refresh path —
                        # halt and propagate per §2.2 ("halt trading and alert").
                        logger.error("IG 401 on %s %s after refresh-retry — halting", method, path)
                        raise
                    auth_retry_used = True
                    try:
                        if c._cst == token_at_send:
                            await c.refresh_session()
                        # else: another task already refreshed; just retry with the
                        # new tokens already in c._cst / c._xst
                    except Exception:
                        logger.exception("Forced refresh after 401 failed")
                        raise exc from None
                    attempt += 1
                    continue
                if not exc.error_type.is_retryable:
                    raise
                if not idempotent and exc.error_type in _AMBIGUOUS_ON_RETRY:
                    logger.error(
                        "IG %s %s failed with %s on a non-idempotent request — "
                        "NOT retrying (may have executed; caller must reconcile)",
                        method,
                        path,
                        exc.error_type.value,
                    )
                    raise
            except (TimeoutError, aiohttp.ClientConnectionError) as exc:
                last_exc = exc
                if not idempotent:
                    logger.error(
                        "IG %s %s timed out / connection-dropped on a non-idempotent "
                        "request — NOT retrying (may have executed; caller must reconcile)",
                        method,
                        path,
                    )
                    raise
            except asyncio.CancelledError:
                raise

            if attempt >= _MAX_RETRIES:
                break
            delay = min(_RETRY_MAX_DELAY_S, _RETRY_INITIAL_DELAY_S * (2**attempt))
            jitter = random.uniform(0.0, _RETRY_JITTER_S)
            wait_s = delay + jitter
            logger.warning(
                "IG %s %s transient failure (attempt %d/%d): %s — backing off %.1fs",
                method,
                path,
                attempt + 1,
                _MAX_RETRIES + 1,
                last_exc,
                wait_s,
            )
            await asyncio.sleep(wait_s)
            attempt += 1

        assert last_exc is not None
        raise last_exc

    @staticmethod
    async def _handle_response(resp: aiohttp.ClientResponse) -> dict[str, Any]:
        """Map an aiohttp response onto a dict or an ``ExchangeError``.

        Reclassifies a 403 to ``RATE_LIMIT`` even when the body carries an
        ``exceeded-*-allowance`` error code (so the retry layer in
        ``request`` knows to back off rather than fail fast).
        """
        if resp.status == 200:
            result: dict[str, Any] = await resp.json()
            return result
        text = await resp.text()
        # Try to extract IG's structured error code from the body
        error_code = ""
        try:
            payload = await resp.json()
            if isinstance(payload, dict):
                error_code = str(payload.get("errorCode", ""))
        except Exception:  # noqa: BLE001 — body may not be JSON
            pass

        if resp.status == 401:
            raise ExchangeError(
                f"IG auth error: {error_code or text}", ErrorType.AUTHENTICATION_FAILED
            )
        if resp.status == 403:
            # 403 is rate-limit territory on IG (per-key, per-account, per-trade
            # quota or session-token-expired-mid-request).  Always retryable.
            raise ExchangeError(
                f"IG rate limit / forbidden: {error_code or text}", ErrorType.RATE_LIMIT
            )
        if resp.status == 429:
            raise ExchangeError(
                f"IG rate limit HTTP 429: {error_code or text}", ErrorType.RATE_LIMIT
            )
        if resp.status in (500, 501, 502, 503, 504):
            raise ExchangeError(
                f"IG server error HTTP {resp.status}: {error_code or text}",
                ErrorType.SERVICE_UNAVAILABLE,
            )
        # Body-level allowance-exceeded that managed to come back with a 200-ish
        # status (rare but documented) — surface as retryable.
        if error_code.startswith(_RATE_LIMIT_ERROR_CODES):
            raise ExchangeError(f"IG allowance exceeded: {error_code}", ErrorType.RATE_LIMIT)
        raise ExchangeError(
            f"IG HTTP {resp.status}: {error_code or text}", ErrorType.EXCHANGE_ERROR
        )
