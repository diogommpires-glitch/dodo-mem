import hashlib
import json
from types import SimpleNamespace

import httpx

from dodo_mem_neural import service
from dodo_mem_neural.broker import create_broker_app
from dodo_mem_neural.contracts import Credential
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.retrieval import RetrievalEngine


async def test_missing_derived_drive_keeps_authenticated_lexical_context_available(
    tmp_path, monkeypatch
):
    (tmp_path / "selection.json").write_text(json.dumps({"encoder": "e5", "device": "NPU"}))
    node = {"id": "public", "digest": "v1", "state": "indexed"}
    catalogue = SimpleNamespace(
        search=lambda *args, **kwargs: [node],
        authorized=lambda *args: True,
        node=lambda *args: node,
        neighbours=lambda *args, **kwargs: [],
        read_evidence=lambda *args, **kwargs: {
            "path": "shared/public.md",
            "digest": "v1",
            "locator": {"line": 1},
            "text": "public fact",
        },
        profile_sources={},
        profile_error=None,
    )
    monkeypatch.setattr(service, "local_sources", lambda: (catalogue, None))
    monkeypatch.setattr(service, "AuthorisedCatalogue", lambda source, vault, **kwargs: source)

    def disconnected():
        raise ResourcePressure("Derived index drive unavailable: E:\\")

    monkeypatch.setattr(service, "check_resources", disconnected)
    organisation = service.OrganisationService(tmp_path)
    organisation.progress["private_source_count"] = 123456
    assert organisation.pause_background(True) == {"paused": True}

    async def tokens(text, model):
        return len(text)

    engine = RetrievalEngine(catalogue, organisation.store, organisation.vectors, None, tokens)
    token = "public-status-fixture"
    credential = Credential(hashlib.sha256(token.encode()).hexdigest(), "dodo", "shared")
    app = create_broker_app(engine, (credential,), routes=(), status=organisation.status)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/status")).status_code == 401
        headers = {"Authorization": "Bearer " + token}
        status = (await client.get("/status", headers=headers)).json()
        packet = (await client.post("/context", headers=headers, json={"query": "fact"})).json()
    assert status["ready"] is True
    assert status["state"] == status["retrieval"] == "lexical_only"
    assert status["scope"] == "shared" and status["device"] == []
    assert "private_source_count" not in status
    assert "local_private_preferences_available" not in status
    assert organisation.runtime is organisation.store is organisation.vectors is None
    assert packet["sources"][0]["excerpt"] == "public fact"
    assert organisation.background_paused.is_set()
    assert "dense_unavailable" in packet["coverage"]["coverage_gaps"]
    assert not (tmp_path / "spaces").exists()


async def test_reconnecting_derived_storage_restores_same_broker_and_route_locks(
    tmp_path, monkeypatch
):
    service_object = service.OrganisationService.__new__(service.OrganisationService)
    service_object.runtime = None
    service_object.progress = {"state": "lexical_only"}
    service_object.neural_unavailable = "offline"
    service_object.worker_thread = None
    service_object.activity = SimpleNamespace(blocked={"http://protected": "cancel pending"})
    original_gate = service_object.activity
    index, vectors = object(), object()
    runtime = SimpleNamespace(encode=lambda texts, query=False: [], space_version="test-space")
    monkeypatch.setattr(
        service_object,
        "_open_neural",
        lambda: {
            "runtime": runtime,
            "reranker": None,
            "store": index,
            "vectors": vectors,
            "device_fallback": None,
            "space": "test",
            "space_home": tmp_path,
        },
    )
    engine = SimpleNamespace(index=None, vectors=None, runtime=None)
    assert await service_object.recover_neural(engine, index=False)
    assert service_object.runtime is runtime and engine.index is index and engine.vectors is vectors
    assert service_object.activity is original_gate
    assert service_object.activity.blocked == {"http://protected": "cancel pending"}
    assert engine.runtime.encode(["query"], query=True) == []
    assert engine.runtime.rerank is None
    assert service_object.neural_unavailable is None
