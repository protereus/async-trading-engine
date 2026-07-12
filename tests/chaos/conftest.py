"""Fixtures for the live chaos suite (`pytest -m chaos`).

Topology::

    IGFeed (production code, unmodified)
      └─ TLS ──► toxiproxy 127.0.0.1:18443 ──► <real IG Lightstreamer host>:443

The REST path (login, token refresh, position queries) stays direct — only
the streaming connection is proxied, so faults hit exactly the layer under
test while heartbeat recovery's REST re-auth remains functional.

Two process-wide adjustments, both test-side only:

* ``LightstreamerClient.setTrustManagerFactory`` receives a context with
  full chain verification but ``check_hostname=False`` — the genuine IG
  certificate cannot match ``127.0.0.1``.  Verification of the certificate
  chain is NOT disabled.
* ``IGClient.lightstreamer_endpoint`` (the property) is pinned to the proxy
  address for the session, because heartbeat recovery re-reads it after a
  REST re-auth rewrites the underlying attribute.

Prerequisites (each absence skips the suite): a ``toxiproxy-server`` binary
on PATH, IG **demo** credentials in the env file (``CHAOS_ENV_FILE``,
default ``.env``), and network access to IG demo.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import shutil
import ssl
import subprocess
import time
import urllib.parse
from collections.abc import AsyncIterator, Iterator
from pathlib import Path
from typing import Any

import pytest

from bot.config import BotConfig
from bot.core.event_bus import EventBus
from bot.data.store import DataStore

from .drivers import EventLog
from .recorder import ChaosRecorder
from .toxiproxy_api import ToxiproxyClient

REPO_ROOT = Path(__file__).resolve().parents[2]
TOXIPROXY_URL = "http://127.0.0.1:8474"
PROXY_NAME = "ig_ls"
PROXY_LISTEN = "127.0.0.1:18443"
PROXY_URL = "https://127.0.0.1:18443"

# Chaos scenarios drive a live (demo) broker session — keep them out of any
# run that didn't ask for them explicitly, whatever the -m expression.
_session_state: dict[str, Any] = {}


@pytest.fixture(scope="session")
def chaos_config() -> BotConfig:
    env_file = os.environ.get("CHAOS_ENV_FILE", ".env")
    if not Path(env_file).exists():
        pytest.skip(f"chaos: env file not found: {env_file}")
    config = BotConfig(_env_file=env_file)  # type: ignore[call-arg]
    if config.bot_env != "demo":
        pytest.skip("chaos: refusing to run against a non-demo account")
    if not (config.ig_demo_api and config.ig_demo_username and config.ig_demo_password):
        pytest.skip("chaos: IG demo credentials missing from env file")
    # Streaming is the layer under test — keep the REST backfill tiny so a
    # full suite run costs a negligible slice of the historical-data quota.
    config.candle_buffer_size = 30
    if not config.ig_epics:
        config.ig_epics = ["CS.D.BITCOIN.TODAY.IP"]
    return config


@pytest.fixture(scope="session")
def toxiproxy(chaos_config: BotConfig) -> Iterator[ToxiproxyClient]:
    binary = os.environ.get("TOXIPROXY_BIN") or shutil.which("toxiproxy-server")
    if binary is None:
        pytest.skip("chaos: toxiproxy-server not found on PATH (TOXIPROXY_BIN to override)")
    proc = subprocess.Popen(  # noqa: S603
        [binary, "-host", "127.0.0.1", "-port", "8474"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    client = ToxiproxyClient(TOXIPROXY_URL)
    try:
        deadline = time.monotonic() + 10
        while True:
            try:
                client.version()
                break
            except Exception:
                if time.monotonic() > deadline:
                    proc.terminate()
                    pytest.skip("chaos: toxiproxy-server did not become ready")
                time.sleep(0.2)
        yield client
    finally:
        with contextlib.suppress(Exception):
            client.delete_proxy(PROXY_NAME)
        proc.terminate()
        proc.wait(timeout=10)


@pytest.fixture(scope="session")
def recorder(chaos_config: BotConfig) -> Iterator[ChaosRecorder]:
    from bot.data.ig_ls_connection import _HEARTBEAT_INITIAL_GRACE_S, _HEARTBEAT_TIMEOUT_SEC

    rec = ChaosRecorder(REPO_ROOT)
    rec.environment = {
        "bot_env": chaos_config.bot_env,
        "epics": chaos_config.ig_epics,
        "heartbeat_timeout_s": _HEARTBEAT_TIMEOUT_SEC,
        "heartbeat_initial_grace_s": _HEARTBEAT_INITIAL_GRACE_S,
        "proxy": "toxiproxy (TLS pass-through, streaming only)",
    }
    yield rec
    path = rec.write()
    if path is not None:
        print(f"\nchaos: results written to {path} and docs/chaos/LATEST.md")


def _ensure_proxy(client: Any, toxiproxy: ToxiproxyClient) -> None:
    """First-connection setup: proxy to the real LS host + endpoint pinning."""
    if "proxied" in _session_state:
        return
    import lightstreamer.client as ls

    from bot.execution.ig_client import IGClient

    real = urllib.parse.urlparse(client._ls_endpoint)
    upstream = f"{real.hostname}:{real.port or 443}"
    toxiproxy.create_proxy(PROXY_NAME, PROXY_LISTEN, upstream)

    # Chain verification stays on; only hostname matching is relaxed, because
    # IG's genuine certificate cannot name 127.0.0.1.  Never mirror this in
    # product code.
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ls.LightstreamerClient.setTrustManagerFactory(ctx)

    _session_state["orig_endpoint_prop"] = IGClient.lightstreamer_endpoint
    IGClient.lightstreamer_endpoint = property(  # type: ignore[assignment,method-assign]
        lambda self: PROXY_URL
    )
    _session_state["proxied"] = True


@pytest.fixture(scope="session", autouse=True)
def _restore_endpoint_property() -> Iterator[None]:
    yield
    orig = _session_state.pop("orig_endpoint_prop", None)
    if orig is not None:
        from bot.execution.ig_client import IGClient

        IGClient.lightstreamer_endpoint = orig  # type: ignore[method-assign]


@pytest.fixture
def event_log() -> Iterator[EventLog]:
    handler = EventLog()
    bot_logger = logging.getLogger("bot")
    bot_logger.addHandler(handler)
    bot_logger.setLevel(logging.INFO)
    yield handler
    bot_logger.removeHandler(handler)


@pytest.fixture
async def chaos_feed(
    chaos_config: BotConfig,
    toxiproxy: ToxiproxyClient,
    event_log: EventLog,
) -> AsyncIterator[Any]:
    """A running production IGFeed streaming through the fault proxy."""
    from bot.data.ig_feed import IGFeed
    from bot.execution.ig_client import IGClient

    from .drivers import wait_tick

    toxiproxy.reset()  # clean slate: proxies enabled, no toxics
    client = IGClient(chaos_config)
    await client.connect()
    _ensure_proxy(client, toxiproxy)

    feed = IGFeed(
        client=client,
        store=DataStore(buffer_size=chaos_config.candle_buffer_size),
        event_bus=EventBus(),
        config=chaos_config,
    )
    run_task = asyncio.create_task(feed.run(), name="chaos_feed_run")
    first = await wait_tick(feed, after=0.0, within_s=90)
    if first is None:
        run_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await run_task
        await client.close()
        pytest.fail("chaos: no update within 90s of subscribing — cannot establish baseline")

    yield feed

    toxiproxy.reset()
    await feed.close()
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task
    # Deliberately NOT client.close(): its DELETE /session logout would
    # invalidate the cached CST/XST and force every scenario through
    # POST /session — IG's most rate-limited endpoint (an early run tripped
    # ``invalid-client-security-token`` exactly this way).  Cancel the
    # background tasks, close the transport, keep the tokens cached; the
    # next scenario resumes the same broker session via GET /accounts.
    for task in (client._refresh_task, client._keepalive_task):
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
    if client._session is not None:
        await client._session.close()
