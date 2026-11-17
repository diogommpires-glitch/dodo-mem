"""A cancelled request cannot release a model route until its backend is idle."""

import asyncio
import time
from contextlib import asynccontextmanager

import anyio
import httpx


class RouteBlocked(RuntimeError):
    pass


class RouteGate:
    def __init__(self, *, transport=None):
        self.transport = transport
        self.locks = {}
        self.blocked = {}
        self.active = 0

    @asynccontextmanager
    async def track(self):
        self.active += 1
        try:
            yield
        finally:
            self.active -= 1

    async def _idle(self, route, *, autoload=False):
        headers = (
            {"Authorization": f"Bearer {route.upstream_token}"} if route.upstream_token else {}
        )
        params = (
            {"model": route.alias, "autoload": "true" if autoload else "false"}
            if route.raw_router
            else None
        )
        timeout = 180 if route.raw_router and autoload else 1
        deadline = time.monotonic() + 10
        async with httpx.AsyncClient(
            trust_env=False, follow_redirects=False, transport=self.transport, timeout=timeout
        ) as client:
            while time.monotonic() < deadline:
                try:
                    if route.raw_router and not autoload:
                        response = await client.get(route.url + "/models", headers=headers)
                        response.raise_for_status()
                        catalog = response.json()
                        rows = catalog.get("data") if isinstance(catalog, dict) else None
                        if not isinstance(rows, list):
                            raise ValueError("Router model status is unavailable.")
                        selected = [
                            row
                            for row in rows
                            if isinstance(row, dict) and row.get("id") == route.alias
                        ]
                        if len(selected) != 1 or not isinstance(selected[0].get("status"), dict):
                            raise ValueError("The original backend is absent from router status.")
                        state = selected[0]["status"].get("value")
                        if not isinstance(state, str):
                            raise ValueError("Router model state must be a string.")
                        if state in {"sleeping", "unloaded"}:
                            return
                        if state != "loaded":
                            raise ValueError("The original backend is still changing state.")
                    response = await client.get(
                        route.url + "/slots", headers=headers, params=params
                    )
                    response.raise_for_status()
                    slots = response.json()
                    if (
                        isinstance(slots, list)
                        and slots
                        and all(
                            isinstance(slot, dict) and slot.get("is_processing") is False
                            for slot in slots
                        )
                    ):
                        return
                except (httpx.HTTPError, ValueError):
                    pass
                await asyncio.sleep(0.1)
        raise RouteBlocked(
            "Backend cancellation/idle state could not be confirmed within ten seconds."
        )

    @asynccontextmanager
    async def route(self, route):
        key = route.url
        lock = self.locks.setdefault(key, asyncio.Lock())
        async with lock:
            if key in self.blocked:
                await self._idle(self.blocked[key])
                del self.blocked[key]
            # Recovery checks the original model before a different selection can replace it.
            await self._idle(route, autoload=True)
            try:
                yield
            finally:
                # Task.cancel() can interrupt an AnyIO shield, so retain the owner before waiting.
                self.blocked[key] = route
                with anyio.CancelScope(shield=True):
                    await self._idle(route)
                del self.blocked[key]
