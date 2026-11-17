import asyncio
from types import SimpleNamespace

import httpx
import pytest

from dodo_mem_neural.contracts import ModelRoute
from dodo_mem_neural.route_gate import RouteBlocked, RouteGate


async def test_failed_preflight_can_retry_when_backend_becomes_ready(monkeypatch):
    clock = [0]
    ready = False

    def upstream(request):
        if not ready:
            clock[0] += 11
            raise httpx.ConnectError("router is starting", request=request)
        if request.url.path == "/models":
            return httpx.Response(
                200, json={"data": [{"id": "alias", "status": {"value": "loaded"}}]}
            )
        return httpx.Response(200, json=[{"is_processing": False}])

    monkeypatch.setattr(
        "dodo_mem_neural.route_gate.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    gate = RouteGate(transport=httpx.MockTransport(upstream))
    route = ModelRoute("coder", "alias", "http://127.0.0.1:8110", raw_router=True)
    with pytest.raises(RouteBlocked):
        async with gate.route(route):
            pytest.fail("An unavailable backend must not receive evidence or generation.")

    ready = True
    entered = False
    async with gate.route(route):
        entered = True
    assert entered


@pytest.mark.parametrize("inactive_state", ["sleeping", "unloaded"])
async def test_cleanup_accepts_inactive_router_models_without_waking_them(inactive_state):
    state = "loaded"

    def upstream(request):
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "alias", "status": {"value": state}}]})
        assert request.url.path == "/slots"
        if state != "loaded":
            pytest.fail("Cleanup must not wake or reload an inactive backend.")
        return httpx.Response(200, json=[{"is_processing": False}])

    gate = RouteGate(transport=httpx.MockTransport(upstream))
    route = ModelRoute("coder", "alias", "http://127.0.0.1:8110", raw_router=True)
    async with gate.route(route):
        state = inactive_state


async def test_blocked_route_checks_original_backend_before_allowing_another_model(monkeypatch):
    clock = [0]
    busy = False
    states = {"old": "loaded", "new": "unloaded"}
    inspected = []

    def upstream(request):
        if request.url.path == "/models":
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": alias, "status": {"value": state}} for alias, state in states.items()
                    ]
                },
            )
        alias = request.url.params["model"]
        inspected.append(alias)
        if alias == "old" and busy:
            clock[0] += 11
            return httpx.Response(200, json=[{"is_processing": True}])
        if alias == "new":
            states.update(old="unloaded", new="loaded")
        return httpx.Response(200, json=[{"is_processing": False}])

    monkeypatch.setattr(
        "dodo_mem_neural.route_gate.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    gate = RouteGate(transport=httpx.MockTransport(upstream))
    original = ModelRoute("first", "old", "http://127.0.0.1:8110", raw_router=True)
    selected = ModelRoute("second", "new", original.url, raw_router=True)
    with pytest.raises(RouteBlocked):
        async with gate.route(original):
            busy = True
    with pytest.raises(RouteBlocked):
        async with gate.route(selected):
            pytest.fail("The original backend is still processing.")
    assert "new" not in inspected

    busy = False
    entered = False
    async with gate.route(selected):
        entered = True
    assert entered
    assert "new" in inspected


async def test_missing_router_model_cannot_be_treated_as_confirmed_idle(monkeypatch):
    clock = [0]

    def upstream(request):
        if request.url.path == "/models":
            clock[0] += 11
            return httpx.Response(200, json={"data": []})
        return httpx.Response(200, json=[{"is_processing": False}])

    monkeypatch.setattr(
        "dodo_mem_neural.route_gate.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    gate = RouteGate(transport=httpx.MockTransport(upstream))
    route = ModelRoute("coder", "alias", "http://127.0.0.1:8110", raw_router=True)
    with pytest.raises(RouteBlocked):
        async with gate.route(route):
            pass


@pytest.mark.parametrize("invalid_state", [{}, []])
async def test_invalid_status_keeps_the_original_backend_blocked(monkeypatch, invalid_state):
    clock = [0]
    inspected = []

    def upstream(request):
        if request.url.path == "/models":
            clock[0] += 11
            return httpx.Response(
                200, json={"data": [{"id": "old", "status": {"value": invalid_state}}]}
            )
        inspected.append(request.url.params["model"])
        return httpx.Response(200, json=[{"is_processing": False}])

    monkeypatch.setattr(
        "dodo_mem_neural.route_gate.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    gate = RouteGate(transport=httpx.MockTransport(upstream))
    original = ModelRoute("first", "old", "http://127.0.0.1:8110", raw_router=True)
    selected = ModelRoute("second", "new", original.url, raw_router=True)
    with pytest.raises(RouteBlocked):
        async with gate.route(original):
            pass
    with pytest.raises(RouteBlocked):
        async with gate.route(selected):
            pytest.fail("Malformed original-model status must not allow another selection.")
    assert "new" not in inspected


async def test_native_cancellation_during_cleanup_retains_the_busy_backend(monkeypatch):
    clock = [0]
    busy = False
    hold_cleanup = True
    checking = asyncio.Event()
    inspected = []

    async def upstream(request):
        if request.url.path == "/models":
            clock[0] += 1
            return httpx.Response(
                200,
                json={
                    "data": [
                        {"id": "old", "status": {"value": "loaded"}},
                        {"id": "new", "status": {"value": "loaded"}},
                    ]
                },
            )
        alias = request.url.params["model"]
        inspected.append(alias)
        if alias == "old" and busy:
            if hold_cleanup:
                checking.set()
                await asyncio.Event().wait()
            clock[0] += 11
            return httpx.Response(200, json=[{"is_processing": True}])
        return httpx.Response(200, json=[{"is_processing": False}])

    monkeypatch.setattr(
        "dodo_mem_neural.route_gate.time", SimpleNamespace(monotonic=lambda: clock[0])
    )
    gate = RouteGate(transport=httpx.MockTransport(upstream))
    original = ModelRoute("first", "old", "http://127.0.0.1:8110", raw_router=True)
    selected = ModelRoute("second", "new", original.url, raw_router=True)

    async def generation():
        nonlocal busy
        async with gate.route(original):
            busy = True

    task = asyncio.create_task(generation())
    await asyncio.wait_for(checking.wait(), 2)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    hold_cleanup = False
    with pytest.raises(RouteBlocked):
        async with gate.route(selected):
            pytest.fail("The cancelled request's backend is still busy.")
    assert "new" not in inspected
