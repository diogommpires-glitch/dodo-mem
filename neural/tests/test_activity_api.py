import asyncio
import hashlib

import httpx
import numpy as np

from dodo_mem_neural.activity import NeuralActivity
from dodo_mem_neural.broker import create_broker_app
from dodo_mem_neural.contracts import Credential


async def test_activation_reads_and_controls_are_local_owner_only_and_do_not_run_inference():
    activity = NeuralActivity()
    model = {
        "name": "e5",
        "revision": "pinned",
        "space": "fixture",
        "device": ["NPU"],
        "architecture": {"verified": True},
    }
    activity.set_model(model)
    credentials = tuple(
        Credential(hashlib.sha256(token.encode()).hexdigest(), client, scope)
        for token, client, scope in (
            ("owner", "dodo", "full_private"),
            ("shared", "dodo", "shared"),
            ("native", "opencode", "shared"),
        )
    )
    app = create_broker_app(object(), credentials, routes=(), activity=activity)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        for token in (None, "shared", "native"):
            headers = {"Authorization": "Bearer " + token} if token else {}
            assert (await client.get("/activity/architecture", headers=headers)).status_code == (
                403 if token else 401
            )
        owner = {"Authorization": "Bearer owner"}
        architecture = (await client.get("/activity/architecture", headers=owner)).json()
        assert architecture["num_hidden_layers"] == 12 and architecture["hidden_size"] == 384
        assert (
            await client.post(
                "/activity/control",
                headers=owner,
                json={"visible": True, "paused": False, "token_index": 256},
            )
        ).status_code == 422
        assert (
            await client.post(
                "/activity/control",
                headers=owner,
                json={"visible": True, "paused": False, "token_index": None},
            )
        ).status_code == 200
        assert (await client.get("/activity/snapshot", headers=owner)).json()["running"] == []
        ticket = activity.begin([1, 1], True, model)
        activity.complete(ticket, {"output": np.ones(384)}, "final", model)
        capture = await client.get("/activity/activations/" + ticket["id"], headers=owner)
        assert capture.status_code == 200 and capture.json()["tensors"]["output"] == [1] * 384
        assert (
            await client.get(
                "/activity/activations/" + ticket["id"], headers={"Authorization": "Bearer shared"}
            )
        ).status_code == 403
        assert (await client.get("/activity/activations/missing", headers=owner)).status_code == 404
        events = activity.events_since()
        assert all("tensors" not in event for event in events["events"])


async def test_activity_event_stream_is_metadata_only_and_disconnects_cleanly():
    activity = NeuralActivity()
    credential = Credential(hashlib.sha256(b"owner").hexdigest(), "dodo", "full_private")
    app = create_broker_app(object(), (credential,), routes=(), activity=activity)
    disconnected = asyncio.Event()
    messages = []

    async def receive():
        if disconnected.is_set():
            return {"type": "http.disconnect"}
        await disconnected.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        messages.append(message)
        if message["type"] == "http.response.body" and message.get("body"):
            disconnected.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.4"},
        "method": "GET",
        "scheme": "http",
        "path": "/activity/events",
        "raw_path": b"/activity/events",
        "query_string": b"after=0",
        "root_path": "",
        "http_version": "1.1",
        "server": ("127.0.0.1", 8770),
        "client": ("127.0.0.1", 1234),
        "headers": [(b"authorization", b"Bearer owner")],
    }
    await asyncio.wait_for(app(scope, receive, send), 2)
    assert messages[0]["status"] == 200
    body = b"".join(message.get("body", b"") for message in messages)
    assert b"event: snapshot" in body and b"tensors" not in body
    assert messages[-1].get("more_body") is False
