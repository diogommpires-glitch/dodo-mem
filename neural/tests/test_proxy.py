import asyncio
import hashlib
import json
from contextlib import asynccontextmanager

import httpx
import pytest

from dodo_mem_neural.client_routes import JAN_TITLE
from dodo_mem_neural.contracts import Credential, ModelRoute
from dodo_mem_neural.proxy import create_proxy_app, model_token_count
from dodo_mem_neural.route_gate import RouteGate


class Bytes(httpx.AsyncByteStream):
    def __init__(self, content):
        self.content = content

    async def __aiter__(self):
        yield self.content


def props(route, context=8192):
    return {
        "model_alias": route.alias,
        "model_path": route.model_path,
        "default_generation_settings": {"n_ctx": context},
    }


def prompt_response(request):
    if request.url.path == "/apply-template":
        return httpx.Response(200, json={"prompt": "templated conversation"})
    if request.url.path == "/tokenize":
        return httpx.Response(200, json={"tokens": [1] * 100})
    if request.url.path == "/validate":
        receipt = json.loads(request.content)
        assert receipt["packet_id"] == "receipt"
        assert receipt["model"] == "coder"
        assert 0 <= receipt["source_count"] <= 8
        return httpx.Response(200, json={"valid": True})
    return None


def configuration():
    route = ModelRoute(
        "coder", "alias", "http://127.0.0.1:8110", "upstream-secret", True, "C:/models/model.gguf"
    )
    credential = Credential(
        hashlib.sha256(b"client-secret").hexdigest(), "cline", "shared", ("coder",)
    )
    return route, credential


@pytest.mark.parametrize("initial_state", ["unloaded", "sleeping"])
async def test_chat_loads_the_selected_router_model_before_readiness_checks(initial_state):
    route, credential = configuration()
    state = initial_state
    loads = []

    def upstream(request):
        nonlocal state
        assert request.headers["authorization"] == "Bearer upstream-secret"
        if request.url.path == "/models":
            return httpx.Response(200, json={"data": [{"id": "alias", "status": {"value": state}}]})
        if request.url.path == "/slots":
            assert request.url.params["model"] == "alias"
            if state != "loaded":
                if state == "unloaded" and request.url.params["autoload"] != "true":
                    return httpx.Response(400, json={"error": "model is not loaded"})
                assert request.extensions["timeout"]["read"] >= 120
                loads.append(state)
                state = "loaded"
            else:
                assert request.url.params["autoload"] == "false"
            return httpx.Response(200, json=[{"is_processing": False}])
        assert state == "loaded"
        assert request.url.params["autoload"] == "false"
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route))
        if response := prompt_response(request):
            return response
        assert request.url.path == "/v1/chat/completions"
        assert json.loads(request.content)["model"] == route.alias
        return httpx.Response(200, json={"choices": [{"message": {"content": "LOCAL_OK"}}]})

    transport = httpx.MockTransport(upstream)
    app = create_proxy_app(
        (route,), (credential,), transport=transport, activity=RouteGate(transport=transport)
    )
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={
                "model": "coder",
                "messages": [{"role": "user", "content": "Reply LOCAL_OK"}],
                "max_tokens": 16,
                "dodo_memory": False,
            },
        )
    assert response.status_code == 200, response.text
    assert response.json()["choices"][0]["message"]["content"] == "LOCAL_OK"
    assert loads == [initial_state]


async def test_jan_title_call_preserves_local_generation_without_new_retrieval():
    route, cline = configuration()
    credential = Credential(cline.token_sha256, "jan", "shared", ("coder",))
    requests = []

    async def upstream(request):
        requests.append(request.url.path)
        if response := prompt_response(request):
            return response
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(200, json={"choices": [{"message": {"content": "Local title"}}]})

    app = create_proxy_app((route,), (credential,), transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={"model": "coder", "messages": [{"role": "user", "content": JAN_TITLE + "Task"}]},
        )
    assert response.status_code == 200
    assert response.headers["X-Dodo-Memory"] == "auxiliary"
    assert "/retrieve" not in requests and "/validate" not in requests


async def test_proxy_preserves_payload_stream_and_auth_separation_while_using_true_query():
    route, credential = configuration()
    requests = []
    stream = b'data: {"choices":[{"delta":{"reasoning_content":"plan"}}]}\n\ndata: [DONE]\n\n'

    async def upstream(request):
        requests.append(request)
        if response := prompt_response(request):
            return response
        if request.url.path == "/props":
            assert request.url.params["autoload"] == "false"
            assert request.url.params["model"] == "alias"
            return httpx.Response(200, json=props(route))
        if request.url.path == "/retrieve":
            assert request.headers["authorization"] == "Bearer client-secret"
            assert json.loads(request.content) == {"query": "Fix parser", "model": "coder"}
            return httpx.Response(
                200,
                json={
                    "memory_text": "untrusted memory [1]",
                    "coverage": {},
                    "sources": [],
                    "packet_id": "receipt",
                },
            )
        assert request.headers["authorization"] == "Bearer upstream-secret"
        assert request.url.params["autoload"] == "false"
        payload = json.loads(request.content)
        assert payload["model"] == "alias"
        assert payload["messages"][1:] == messages
        assert payload["tools"] == [{"type": "function"}]
        assert payload["temperature"] == 0.2
        return httpx.Response(
            200, stream=Bytes(stream), headers={"content-type": "text/event-stream"}
        )

    messages = [
        {"role": "user", "content": "<task>Fix parser</task>"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "think",
            "tool_calls": [{"id": "a"}],
        },
        {"role": "user", "content": "[read_file for 'p.py'] Result:\nsource"},
    ]
    app = create_proxy_app((route,), (credential,), transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={
                "model": "coder",
                "messages": messages,
                "stream": True,
                "tools": [{"type": "function"}],
                "temperature": 0.2,
            },
        )
    assert response.content == stream
    assert response.headers["x-dodo-memory"] == "injected"
    assert len(requests) == 8


async def test_proxy_off_bypasses_broker_but_unknown_route_and_unavailable_broker_fail_closed():
    route, credential = configuration()
    calls = []

    async def upstream(request):
        calls.append(request.url.path)
        if response := prompt_response(request):
            return response
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route))
        if request.url.path == "/retrieve":
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json={"choices": []})

    app = create_proxy_app((route,), (credential,), transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
        headers={"Authorization": "Bearer client-secret"},
    ) as client:
        body = {"model": "coder", "messages": [{"role": "user", "content": "question"}]}
        response = await client.post("/v1/chat/completions", json=body)
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "memory_unavailable"
        assert "/v1/chat/completions" not in calls
        calls.clear()
        response = await client.post("/v1/chat/completions", json={**body, "dodo_memory": False})
        assert response.status_code == 200
        assert response.headers["x-dodo-memory"] == "off"
        assert calls == ["/props", "/apply-template", "/tokenize", "/v1/chat/completions"]
        calls.clear()
        assert (
            await client.post("/v1/chat/completions", json={**body, "model": "unknown"})
        ).status_code == 403
        assert calls == []


async def test_tokenizer_uses_selected_loaded_model_and_never_autoloads():
    route, _ = configuration()
    calls = []

    def upstream(request):
        calls.append(request)
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"model_alias": "alias", "model_path": route.model_path}
            )
        assert json.loads(request.content) == {
            "model": "alias",
            "content": "hello",
            "add_special": False,
        }
        assert request.url.params["autoload"] == "false"
        return httpx.Response(200, json={"tokens": [1, 2]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        assert await model_token_count(route, "hello", client) == 2
    assert calls[0].url.params["autoload"] == "false"


@pytest.mark.parametrize("model_alias", ["wrong", None])
async def test_wrong_backend_identity_is_rejected_before_model_post(model_alias):
    route, _ = configuration()
    calls = []

    def upstream(request):
        calls.append(request.method)
        return httpx.Response(200, json={"model_alias": model_alias})

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        with pytest.raises(ValueError, match="identity"):
            await model_token_count(route, "hello", client)
    assert calls == ["GET"]


async def test_wrong_weights_fail_closed_even_when_alias_matches():
    route, _ = configuration()

    def upstream(request):
        return httpx.Response(
            200, json={"model_alias": "alias", "model_path": "C:/models/other.gguf"}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(upstream)) as client:
        with pytest.raises(ValueError, match="weights"):
            await model_token_count(route, "hello", client)


async def test_downstream_disconnect_closes_upstream_and_releases_activity():
    route, credential = configuration()
    first_chunk = asyncio.Event()
    closed = asyncio.Event()

    class Stream(httpx.AsyncByteStream):
        async def __aiter__(self):
            yield b"data: first\n\n"
            await asyncio.Event().wait()

        async def aclose(self):
            closed.set()

    class Activity:
        active = 0

        @asynccontextmanager
        async def route(self, route):
            yield

        @asynccontextmanager
        async def track(self):
            self.active += 1
            try:
                yield
            finally:
                self.active -= 1

    activity = Activity()

    def upstream(request):
        if response := prompt_response(request):
            return response
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route))
        return httpx.Response(200, stream=Stream(), headers={"content-type": "text/event-stream"})

    app = create_proxy_app(
        (route,), (credential,), transport=httpx.MockTransport(upstream), activity=activity
    )
    body = json.dumps(
        {
            "model": "coder",
            "messages": [{"role": "user", "content": "q"}],
            "stream": True,
            "dodo_memory": False,
        }
    ).encode()
    received = False

    async def receive():
        nonlocal received
        if not received:
            received = True
            return {"type": "http.request", "body": body}
        await first_chunk.wait()
        return {"type": "http.disconnect"}

    async def send(message):
        if message["type"] == "http.response.body" and message.get("body"):
            first_chunk.set()

    scope = {
        "type": "http",
        "asgi": {"version": "3.0", "spec_version": "2.0"},
        "method": "POST",
        "scheme": "http",
        "path": "/v1/chat/completions",
        "raw_path": b"/v1/chat/completions",
        "query_string": b"",
        "root_path": "",
        "headers": [
            (b"authorization", b"Bearer client-secret"),
            (b"content-type", b"application/json"),
        ],
        "client": ("127.0.0.1", 1),
        "server": ("127.0.0.1", 8771),
    }
    await asyncio.wait_for(app(scope, receive, send), 2)
    assert first_chunk.is_set()
    assert closed.is_set()
    assert activity.active == 0


async def test_complete_prompt_budget_shrinks_evidence_and_validates_immediately_before_dispatch():
    route, credential = configuration()
    calls = []
    sources = [
        {"node_id": value, "path": f"/{value}.md", "excerpt": value * 80, "cite": f"[{number}]"}
        for number, value in enumerate(("a", "b"), 1)
    ]
    original = [{"role": "user", "content": "question"}]

    def upstream(request):
        calls.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route, 1000))
        if request.url.path == "/retrieve":
            return httpx.Response(
                200,
                json={
                    "packet_id": "receipt",
                    "memory_text": "untrusted memory with /a.md /b.md",
                    "sources": sources,
                },
            )
        if request.url.path == "/apply-template":
            payload = json.loads(request.content)
            if payload["messages"] == original:
                return httpx.Response(200, json={"prompt": "base"})
            assert payload["messages"][1:] == original
            return httpx.Response(
                200,
                json={"prompt": "two" if "/b.md" in payload["messages"][0]["content"] else "one"},
            )
        if request.url.path == "/tokenize":
            payload = json.loads(request.content)
            assert payload["add_special"] is True
            return httpx.Response(
                200, json={"tokens": [1] * (900 if payload["content"] == "two" else 800)}
            )
        if response := prompt_response(request):
            return response
        payload = json.loads(request.content)
        assert "/a.md" in payload["messages"][0]["content"]
        assert "/b.md" not in payload["messages"][0]["content"]
        assert payload["max_tokens"] == 100
        return httpx.Response(200, json={"choices": []})

    app = create_proxy_app((route,), (credential,), transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={"model": "coder", "messages": original, "max_tokens": 100},
        )
    assert response.status_code == 200
    assert response.headers["x-dodo-memory"] == "shrunk"
    assert calls[-2:] == ["/validate", "/v1/chat/completions"]


@pytest.mark.parametrize("limits", [{"max_tokens": 100}, {"max_tokens": 1, "n_predict": 100}])
async def test_original_conversation_over_context_is_rejected_without_dispatch(limits):
    route, credential = configuration()
    calls = []

    def upstream(request):
        calls.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json=props(route, 1000))
        if request.url.path == "/apply-template":
            return httpx.Response(200, json={"prompt": "long conversation"})
        return httpx.Response(200, json={"tokens": [1] * 950})

    app = create_proxy_app((route,), (credential,), transport=httpx.MockTransport(upstream))
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer client-secret"},
            json={
                "model": "coder",
                "messages": [{"role": "user", "content": "long"}],
                **limits,
                "dodo_memory": False,
            },
        )
    assert response.status_code == 400
    assert response.json()["error"]["code"] == "context_exceeded"
    assert "/v1/chat/completions" not in calls
