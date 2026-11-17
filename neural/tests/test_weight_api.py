import asyncio
import hashlib
import threading
from types import SimpleNamespace

import httpx
import pytest

from dodo_mem_neural.activity import NeuralActivity
from dodo_mem_neural.broker import create_broker_app
from dodo_mem_neural.contracts import Credential
from dodo_mem_neural.resources import ResourcePressure


def app_for(source):
    credentials = tuple(
        Credential(hashlib.sha256(token.encode()).hexdigest(), client, scope)
        for token, client, scope in (
            ("owner", "dodo", "full_private"),
            ("shared", "dodo", "shared"),
        )
    )
    return create_broker_app(
        object(), credentials, routes=(), activity=NeuralActivity(), projection_source=source
    )


def test_read_only_parameters_keep_ram_gate_without_background_disk_admission(monkeypatch):
    from dodo_mem_neural import service

    calls = []
    reader = SimpleNamespace(
        weights=lambda *args, **kwargs: calls.append((args, kwargs)) or {"edges": []}
    )
    instance = service.OrganisationService.__new__(service.OrganisationService)
    instance._projection_lock = threading.RLock()
    instance._weight_reader = lambda: reader

    def background_disk_refusal(*_args, **_kwargs):
        raise ResourcePressure("C: free disk is below 100 GiB.")

    monkeypatch.setattr(service, "check_resources", background_disk_refusal)
    monkeypatch.setattr(
        service.psutil, "virtual_memory", lambda: SimpleNamespace(available=5 * 1024**3)
    )
    assert instance.projection_weights(11, "ff_down", 512, None) == {"edges": []}
    assert calls == [((11, "ff_down"), {"limit": 512, "output_index": None})]
    calls.clear()
    monkeypatch.setattr(
        service.psutil, "virtual_memory", lambda: SimpleNamespace(available=5 * 1024**3 - 1)
    )
    with pytest.raises(ResourcePressure, match="5 GiB"):
        instance.projection_weights(11, "ff_down", 512, None)
    assert not calls


async def test_verified_static_architecture_remains_available_without_an_encoder(monkeypatch):
    from dodo_mem_neural import service, static_weights

    info = {"name": "e5", "revision": "pinned", "space": "fixture-space", "device": []}
    reader = SimpleNamespace(
        descriptors=lambda: {"model_revision": "pinned", "model_space": "fixture-space"}
    )
    instance = service.OrganisationService.__new__(service.OrganisationService)
    instance.runtime = None
    instance.selection = {"encoder": "e5"}
    instance._projection_lock = threading.RLock()
    instance._projection_source = None
    instance._projection_space = None
    instance.neural_activity = NeuralActivity()
    instance.neural_unavailable = "Disk headroom is below 100 GiB."
    monkeypatch.setattr(static_weights, "static_model_info", lambda: info)
    monkeypatch.setattr(static_weights, "StaticWeights", lambda model: reader)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(instance)), base_url="http://test"
    ) as client:
        headers = {"Authorization": "Bearer owner"}
        response = await client.get("/activity/architecture", headers=headers)
        assert response.status_code == 200
        architecture = response.json()
        assert architecture["verified"] is True
        assert architecture["model"] == info
        assert architecture["runtime_available"] is False
        assert architecture["runtime_reason"] == instance.neural_unavailable
        assert "No live activations" in architecture["qualification_reason"]
        assert instance.runtime is None
        assert (await client.get("/activity/snapshot", headers=headers)).json()["running"] == []


async def test_parameter_views_are_owner_only_and_bounded_before_reading_weights():
    class Source:
        def __init__(self):
            self.calls = []

        def projection_descriptors(self):
            return {"static": True, "projections": []}

        def projection_weights(self, block, projection, limit, output_index):
            self.calls.append((block, projection, limit, output_index))
            return {"static": True, "shown_count": 1, "edges": [{"weight": -0.25}]}

    source = Source()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(source)), base_url="http://test"
    ) as client:
        for route in ("/activity/projections", "/activity/weights?block=11&projection=ff_down"):
            assert (await client.get(route)).status_code == 401
            assert (
                await client.get(route, headers={"Authorization": "Bearer shared"})
            ).status_code == 403
        assert not source.calls
        headers = {"Authorization": "Bearer owner"}
        assert (await client.get("/activity/projections", headers=headers)).json()["static"]
        for query in ("block=12", "projection=unknown", "limit=513", "output_index=1536"):
            assert (
                await client.get("/activity/weights?" + query, headers=headers)
            ).status_code == 422
        assert not source.calls
        result = await client.get(
            "/activity/weights?block=11&projection=ff_down&limit=256&output_index=7",
            headers=headers,
        )
        assert result.status_code == 200 and result.json()["edges"][0]["weight"] == -0.25
        assert source.calls == [(11, "ff_down", 256, 7)]


async def test_cancelled_parameter_request_cannot_admit_another_background_reader():
    started, release = threading.Event(), threading.Event()

    class Source:
        calls = 0

        def projection_weights(self, *_args):
            self.calls += 1
            started.set()
            assert release.wait(3)
            return {"static": True}

    source = Source()
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(source)), base_url="http://test"
    ) as client:
        headers = {"Authorization": "Bearer owner"}
        request = asyncio.create_task(client.get("/activity/weights", headers=headers))
        try:
            assert await asyncio.to_thread(started.wait, 2)
            request.cancel()
            await asyncio.gather(request, return_exceptions=True)
            assert (await client.get("/activity/weights", headers=headers)).status_code == 429
            assert source.calls == 1
        finally:
            release.set()
            await asyncio.gather(request, return_exceptions=True)


async def test_resource_refusal_is_visible_and_does_not_call_model_inference():
    class Source:
        def projection_weights(self, *_args):
            raise ResourcePressure("Parameter view is waiting for 5 GiB available RAM.")

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app_for(Source())), base_url="http://test"
    ) as client:
        result = await client.get("/activity/weights", headers={"Authorization": "Bearer owner"})
        assert result.status_code == 503
        assert "5 GiB" in result.json()["detail"]
