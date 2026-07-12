"""Minimal synchronous client for the toxiproxy HTTP control API.

Control-plane only (create/delete proxies, add/remove toxics) — the data
plane is the proxied TCP stream itself.  Synchronous ``urllib`` is fine
here: control calls happen between scenario phases, never on the hot path.

API reference: https://github.com/Shopify/toxiproxy#http-api
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any


class ToxiproxyClient:
    """Talks to a running toxiproxy-server control endpoint."""

    def __init__(self, base_url: str = "http://127.0.0.1:8474") -> None:
        self._base = base_url.rstrip("/")

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> Any:
        body = json.dumps(payload).encode() if payload is not None else None
        req = urllib.request.Request(
            f"{self._base}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read()
        return json.loads(raw) if raw else None

    def version(self) -> str:
        req = urllib.request.Request(f"{self._base}/version")
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.read().decode()

    def create_proxy(self, name: str, listen: str, upstream: str) -> None:
        # Idempotent: tear down any leftover proxy with the same name first.
        self.delete_proxy(name)
        self._request("POST", "/proxies", {"name": name, "listen": listen, "upstream": upstream})

    def delete_proxy(self, name: str) -> None:
        try:
            self._request("DELETE", f"/proxies/{name}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

    def set_enabled(self, name: str, enabled: bool) -> None:
        """Disable closes the listener and severs live connections (hard cut)."""
        self._request("POST", f"/proxies/{name}", {"enabled": enabled})

    def add_toxic(
        self,
        proxy: str,
        name: str,
        toxic_type: str,
        stream: str = "downstream",
        attributes: dict[str, Any] | None = None,
    ) -> None:
        self._request(
            "POST",
            f"/proxies/{proxy}/toxics",
            {
                "name": name,
                "type": toxic_type,
                "stream": stream,
                "attributes": attributes or {},
            },
        )

    def remove_toxic(self, proxy: str, name: str) -> None:
        try:
            self._request("DELETE", f"/proxies/{proxy}/toxics/{name}")
        except urllib.error.HTTPError as exc:
            if exc.code != 404:
                raise

    def reset(self) -> None:
        """Re-enable all proxies and remove every toxic."""
        self._request("POST", "/reset")
