import hashlib

import httpx

from dodo_mem_neural.broker import create_broker_app
from dodo_mem_neural.contracts import Credential


async def test_only_controlled_private_client_can_pause_background_jobs():
    public = Credential(hashlib.sha256(b"public").hexdigest(), "jan", "shared")
    owner = Credential(hashlib.sha256(b"owner").hexdigest(), "dodo", "full_private")
    calls = []
    app = create_broker_app(
        None,
        (public, owner),
        routes=(),
        background_control=lambda paused: calls.append(paused) or {"paused": paused},
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        assert (
            await client.post(
                "/index/control", json={"paused": True}, headers={"Authorization": "Bearer public"}
            )
        ).status_code == 403
        assert calls == []
        assert (
            await client.post(
                "/index/control", json={"paused": True}, headers={"Authorization": "Bearer owner"}
            )
        ).json() == {"paused": True}
        assert calls == [True]
