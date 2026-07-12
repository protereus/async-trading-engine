"""IG session / auth lifecycle.

Extracted from ``ig_client.py``.  Owns:

* POST /session (initial auth) and PUT /session (SPREADBET switch).
* Background refresh loop (5.5 h hard re-auth before the 6 h token expiry).
* Background keep-alive loop (45 s authenticated ping so v2 tokens
  auto-extend before the 60 s inactivity window invalidates them —
  IG_LIVE_RISK_REFERENCE.md §2.2).
* Token cache file (``.ig_session_cache.json``) — load + save with 0o600.

Auth state (CST/XST/account_id/ls_endpoint) is mutated *on the parent
IGClient* so direct attribute reads from external callers (e.g. the LS
feed and main.py's HEARTBEAT check) keep working.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING

from bot.core.models import ErrorType, ExchangeError

if TYPE_CHECKING:
    from bot.execution.ig_client import IGClient

logger = logging.getLogger(__name__)

# IG v2 session tokens auto-extend on every authenticated call.  The live risk
# reference (IG_LIVE_RISK_REFERENCE.md §2.2) cites a 60 s inactivity invalidation
# window.  A lightweight authenticated ping every 45 s keeps the rolling
# extension alive during quiet trading periods.
_KEEPALIVE_INTERVAL_S = 45.0
_KEEPALIVE_PATH = "/accounts"
_KEEPALIVE_VERSION = "1"

# Session refresh: re-authenticate 30 minutes before the 6-hour expiry.
_SESSION_REFRESH_INTERVAL = 5.5 * 3600  # seconds

# Token cache: only re-login if the saved tokens are older than this.
# IG tokens auto-extend on each API call; 5.5 h gives a comfortable buffer
# before the 6 h hard expiry.
_TOKEN_CACHE_MAX_AGE_S = 5.5 * 3600
_TOKEN_CACHE_FILE = Path(__file__).parent.parent.parent.parent / ".ig_session_cache.json"


class IGSession:
    """Auth + session lifecycle for IG REST.

    Composed onto ``IGClient``; mutates ``client._cst`` / ``client._xst`` /
    ``client._account_id`` / ``client._ls_endpoint`` so direct attribute
    reads on the client keep working.
    """

    def __init__(self, client: IGClient) -> None:
        self._client = client

    async def create_session(self) -> None:
        """POST /session v2 to obtain fresh CST and XST tokens."""
        c = self._client
        cfg = c._config
        api_key = cfg.ig_demo_api if cfg.bot_env == "demo" else cfg.ig_live_api
        username = cfg.ig_demo_username if cfg.bot_env == "demo" else cfg.ig_live_username
        password = cfg.ig_demo_password if cfg.bot_env == "demo" else cfg.ig_live_password

        assert c._session is not None
        async with c._session.post(
            f"{c._base}/session",
            json={"identifier": username, "password": password},
            headers={
                "X-IG-API-KEY": api_key,
                "VERSION": "2",
                "Content-Type": "application/json",
            },
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                raise ExchangeError(
                    f"IG session creation failed HTTP {resp.status}: {text}",
                    ErrorType.AUTHENTICATION_FAILED,
                )
            c._cst = resp.headers["CST"]
            c._xst = resp.headers["X-SECURITY-TOKEN"]
            body = await resp.json()
            c._account_id = body["currentAccountId"]
            c._ls_endpoint = body["lightstreamerEndpoint"]

    async def switch_to_spreadbet(self) -> None:
        """Switch active account to the SPREADBET account if not already on it.

        IG demo rate-limits auth calls, so both the GET /accounts probe and
        the PUT /session switch use retry-with-backoff.
        """
        c = self._client
        # Probe current account list — retry if token hasn't propagated yet
        for attempt in range(3):
            try:
                data = await c._http.get("/accounts", version="1", authenticated=True)
                break
            except ExchangeError as exc:
                if "client-token-invalid" in str(exc) and attempt < 2:
                    await asyncio.sleep(1.5 * (attempt + 1))
                    continue
                raise
        else:
            raise ExchangeError(
                "GET /accounts failed after retries", ErrorType.AUTHENTICATION_FAILED
            )

        spreadbet_id: str | None = None
        current_type: str = ""
        for acc in data.get("accounts", []):
            if acc["accountId"] == c._account_id:
                current_type = acc.get("accountType", "")
            if acc.get("accountType") == "SPREADBET":
                spreadbet_id = acc["accountId"]

        if current_type == "SPREADBET":
            logger.debug("Already on SPREADBET account %s", c._account_id)
            return

        if spreadbet_id is None:
            raise ExchangeError(
                "No SPREADBET account found on this IG login. "
                "Ensure a spread betting account is linked.",
                ErrorType.AUTHENTICATION_FAILED,
            )

        # PUT /session to switch — retry with increasing sleep on rate-limit
        cfg = c._config
        api_key = cfg.ig_demo_api if cfg.bot_env == "demo" else cfg.ig_live_api
        assert c._session is not None

        for attempt in range(3):
            await asyncio.sleep(1.0 * (attempt + 1))
            async with c._session.put(
                f"{c._base}/session",
                json={"accountId": spreadbet_id, "lightstreamerEndpoint": None},
                headers={
                    "X-IG-API-KEY": api_key,
                    "CST": c._cst,
                    "X-SECURITY-TOKEN": c._xst,
                    "VERSION": "1",
                    "Content-Type": "application/json",
                },
            ) as resp:
                if resp.status == 200:
                    new_xst = resp.headers.get("X-SECURITY-TOKEN", "")
                    if new_xst:
                        c._xst = new_xst
                    body = await resp.json()
                    c._account_id = spreadbet_id
                    c._ls_endpoint = body.get("lightstreamerEndpoint", c._ls_endpoint)
                    logger.info("Switched to SPREADBET account %s", c._account_id)
                    return
                text = await resp.text()
                if attempt < 2:
                    logger.warning(
                        "Account switch attempt %d failed HTTP %d, retrying: %s",
                        attempt + 1,
                        resp.status,
                        text,
                    )
                else:
                    raise ExchangeError(
                        f"IG account switch failed after retries HTTP {resp.status}: {text}",
                        ErrorType.AUTHENTICATION_FAILED,
                    )

    async def refresh_session(self) -> None:
        """Re-authenticate to get fresh CST/XST tokens before expiry.

        Serialised via ``client._auth_lock`` so concurrent 401-driven retries
        don't trigger duplicate POST /session calls (which IG demo rate-limits
        aggressively).
        """
        c = self._client
        async with c._auth_lock:
            logger.info("Refreshing IG session tokens")
            c._refreshing = True
            try:
                await self.create_session()
                await self.switch_to_spreadbet()
                self.save_tokens()
            finally:
                c._refreshing = False
        logger.info("IG session refreshed  account=%s", c._account_id)

    async def refresh_loop(self) -> None:
        """Background task: refresh session every 5.5 hours.

        Hard re-auth fallback for the absolute 72 h token cap.  Activity-based
        token extension is handled by ``keepalive_loop`` on a 45 s cadence
        (see IG_LIVE_RISK_REFERENCE.md §2.2).
        """
        while True:
            await asyncio.sleep(_SESSION_REFRESH_INTERVAL)
            try:
                await self.refresh_session()
            except Exception:
                logger.exception("IG session refresh failed — will retry next cycle")

    async def keepalive_loop(self) -> None:
        """Background task: ping IG every 45 s to keep CST/XST tokens warm.

        IG v2 session tokens auto-extend on every authenticated call but invalidate
        after 60 s of inactivity (IG_LIVE_RISK_REFERENCE.md §2.2).  Calling a
        cheap authenticated endpoint inside that window guarantees the next real
        order won't 401 just because the rerank loop was idle overnight.

        Any 401 here triggers the ``request`` auth-retry path which transparently
        re-auths and re-issues the ping.  We only log unexpected exceptions —
        cancellation is propagated normally so ``close()`` finishes cleanly.
        """
        c = self._client
        while True:
            try:
                await asyncio.sleep(_KEEPALIVE_INTERVAL_S)
                if not c.is_connected:
                    continue
                await c._http.get(_KEEPALIVE_PATH, version=_KEEPALIVE_VERSION, authenticated=True)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.warning("IG session keep-alive ping failed", exc_info=True)

    def save_tokens(self) -> None:
        """Persist current CST/XST tokens to the cache file."""
        c = self._client
        payload = {
            "cst": c._cst,
            "xst": c._xst,
            "account_id": c._account_id,
            "ls_endpoint": c._ls_endpoint,
            "env": c._config.bot_env,
            "saved_at": time.time(),
        }
        try:
            # Write atomically (temp + rename) so a crash mid-write can't leave a
            # corrupt cache, and create the temp owner-only from the start so the
            # session tokens are never briefly world-readable before a chmod.
            tmp = _TOKEN_CACHE_FILE.with_name(_TOKEN_CACHE_FILE.name + ".tmp")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, json.dumps(payload).encode())
            finally:
                os.close(fd)
            os.replace(tmp, _TOKEN_CACHE_FILE)  # atomic swap on the same filesystem
            logger.debug("IG session tokens cached to %s", _TOKEN_CACHE_FILE)
        except OSError as exc:
            logger.warning("Could not write IG token cache: %s", exc)

    async def load_cached_tokens(self) -> bool:
        """Load tokens from cache file if present and not stale.

        Returns True if valid tokens were loaded, False if a fresh login
        is required.
        """
        c = self._client
        if not _TOKEN_CACHE_FILE.exists():
            return False
        try:
            payload = json.loads(_TOKEN_CACHE_FILE.read_text())
        except (OSError, json.JSONDecodeError):
            return False

        # Reject if saved for a different environment
        if payload.get("env") != c._config.bot_env:
            return False

        age_s = time.time() - float(payload.get("saved_at", 0))
        if age_s > _TOKEN_CACHE_MAX_AGE_S:
            logger.info("IG token cache stale (%.0f h), re-authenticating", age_s / 3600)
            return False

        c._cst = payload["cst"]
        c._xst = payload["xst"]
        c._account_id = payload["account_id"]
        c._ls_endpoint = payload["ls_endpoint"]
        logger.info(
            "Loaded cached IG tokens (%.0f min old)  account=%s",
            age_s / 60,
            c._account_id,
        )
        return True
