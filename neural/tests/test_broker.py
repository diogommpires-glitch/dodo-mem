import asyncio
import hashlib
import threading
from time import monotonic

import httpx

from dodo_mem_neural.broker import create_broker_app
from dodo_mem_neural.contracts import Credential, EvidencePacket, ModelRoute


async def test_topic_timeouts_share_one_worker_per_scope_and_cancel_without_restarting():
    release = threading.Event()
    calls = []

    def topics(scope):
        calls.append(scope)
        assert release.wait(3)
        return {"scope": scope, "topics": []}

    credential = Credential(hashlib.sha256(b"secret").hexdigest(), "dodo", "full_private")
    app = create_broker_app(Engine(), (credential,), routes=(), topics=topics)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        requests = [
            asyncio.create_task(client.get("/topics", headers={"Authorization": "Bearer secret"}))
            for _ in range(8)
        ]
        try:
            await asyncio.sleep(0.05)
            requests[0].cancel()
            await asyncio.gather(requests[0], return_exceptions=True)
            requests.append(
                asyncio.create_task(
                    client.get("/topics", headers={"Authorization": "Bearer secret"})
                )
            )
            await asyncio.sleep(0.05)
            assert calls == ["full_private"]
        finally:
            release.set()
            responses = await asyncio.gather(*requests[1:])
        assert all(response.json()["scope"] == "full_private" for response in responses)


class Engine:
    def __init__(self):
        self.latest_paths = {"paths": [], "coverage": {}}

    async def retrieve(self, query, scope, model):
        return EvidencePacket("memory", (), {"scope": scope, "model": model, "query": query})

    def verify(self, packet, scope):
        assert packet.memory_text == "memory"
        assert scope == "shared"


async def test_broker_derives_scope_and_route_from_credential_and_rejects_scope_override():
    credential = Credential(hashlib.sha256(b"secret").hexdigest(), "jan", "shared", ("coder",))
    app = create_broker_app(
        Engine(), (credential,), routes=(ModelRoute("coder", "alias", "http://127.0.0.1:8110"),)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        assert (await client.get("/status")).status_code == 401
        headers = {"Authorization": "Bearer secret"}
        response = await client.post(
            "/retrieve", headers=headers, json={"query": "q", "model": "coder"}
        )
        assert response.json()["coverage"] == {"scope": "shared", "model": "coder", "query": "q"}
        packet_id = response.json()["packet_id"]
        assert (
            await client.post(
                "/validate", headers=headers, json={"packet_id": packet_id, "model": "coder"}
            )
        ).status_code == 200
        assert (
            await client.post(
                "/validate", headers=headers, json={"packet_id": "unknown", "model": "coder"}
            )
        ).status_code == 409
        assert (
            await client.post(
                "/retrieve",
                headers=headers,
                json={"query": "q", "model": "coder", "scope": "full_private"},
            )
        ).status_code == 422
        assert (
            await client.post("/retrieve", headers=headers, json={"query": "q", "model": "unknown"})
        ).status_code == 403


async def test_context_inspection_has_credential_bound_scope_and_no_dispatch_route():
    credential = Credential(hashlib.sha256(b"secret").hexdigest(), "dodo", "shared", ())
    app = create_broker_app(Engine(), (credential,), routes=())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/context", headers={"Authorization": "Bearer secret"}, json={"query": "q"}
        )
        assert response.status_code == 200
        assert response.json()["coverage"] == {
            "scope": "shared",
            "model": "local-inspection",
            "query": "q",
            "token_count_mode": "utf8-byte-upper-bound",
            "dispatch_allowed": False,
        }
        assert (
            await client.post(
                "/context",
                headers={"Authorization": "Bearer secret"},
                json={"query": "q", "scope": "full_private"},
            )
        ).status_code == 422


async def test_packet_receipt_is_credential_bound_and_tracks_dispatched_source_subset():
    class Sources:
        latest_paths = None

        async def retrieve(self, query, scope, model):
            return EvidencePacket(
                "original",
                ({"node_id": "a", "excerpt": "a"}, {"node_id": "b", "excerpt": "b"}),
                {"returned": 2},
            )

        def verify(self, packet, scope):
            assert scope == "shared"

    credentials = tuple(
        Credential(hashlib.sha256(token.encode()).hexdigest(), "jan", "shared", ("coder",))
        for token in ("first", "other")
    )
    app = create_broker_app(
        Sources(), credentials, routes=(ModelRoute("coder", "alias", "http://127.0.0.1:8098"),)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/retrieve",
            headers={"Authorization": "Bearer first"},
            json={"query": "q", "model": "coder"},
        )
        receipt = {"packet_id": response.json()["packet_id"], "model": "coder"}
        assert (
            await client.post("/packet", headers={"Authorization": "Bearer other"}, json=receipt)
        ).status_code == 409
        response = await client.post(
            "/validate",
            headers={"Authorization": "Bearer first"},
            json={**receipt, "source_count": 1},
        )
        assert response.json() == {"valid": True}
        response = await client.post(
            "/packet", headers={"Authorization": "Bearer first"}, json=receipt
        )
        assert [source["node_id"] for source in response.json()["sources"]] == ["a"]
        assert response.json()["coverage"]["returned"] == 1
        assert '"node_id":"b"' not in response.json()["memory_text"]


async def test_failed_context_diagnostics_expose_type_only_to_local_owner():
    class Failed(Engine):
        def __init__(self):
            super().__init__()
            self.diagnostics = {"stage": "embedding", "started": monotonic(), "error_type": None}

        async def retrieve(self, query, scope, model):
            raise ValueError("synthetic-private-detail")

    credentials = tuple(
        Credential(hashlib.sha256(token.encode()).hexdigest(), "dodo", scope, ())
        for token, scope in (("shared", "shared"), ("owner", "full_private"))
    )
    app = create_broker_app(Failed(), credentials, routes=())
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/context", headers={"Authorization": "Bearer shared"}, json={"query": "q"}
        )
        assert response.status_code == 503
        assert "synthetic-private-detail" not in response.text
        shared = await client.get("/status", headers={"Authorization": "Bearer shared"})
        assert "retrieval_diagnostic" not in shared.json()
        owner = await client.get("/status", headers={"Authorization": "Bearer owner"})
        assert owner.json()["retrieval_diagnostic"]["error_type"] == "ValueError"
        assert "synthetic-private-detail" not in owner.text
