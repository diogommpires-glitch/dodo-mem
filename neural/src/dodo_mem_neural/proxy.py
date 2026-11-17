"""Local OpenAI chat proxy with credential-bound memory and model routing."""

import json
from contextlib import AsyncExitStack
from pathlib import Path

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

from .client_routes import auxiliary_request, inject_memory, latest_user_text
from .contracts import Credential, ModelRoute, authenticate
from .retrieval import _render

MAX_REQUEST_BYTES = 8 * 1024 * 1024


def _error(status, code, message):
    return JSONResponse(
        {"error": {"type": "dodo_memory_error", "code": code, "message": message}},
        status_code=status,
    )


def _upstream_headers(route):
    headers = {"Accept-Encoding": "identity"}
    if route.upstream_token:
        headers["Authorization"] = f"Bearer {route.upstream_token}"
    return headers


async def check_model_ready(route: ModelRoute, client: httpx.AsyncClient):
    if not route.model_path:
        raise ValueError("upstream weights path is not configured")
    params = {"model": route.alias, "autoload": "false"} if route.raw_router else None
    response = await client.get(
        f"{route.url.rstrip('/')}/props", params=params, headers=_upstream_headers(route), timeout=2
    )
    response.raise_for_status()
    props = response.json()
    if not isinstance(props, dict) or props.get("model_alias") != route.alias:
        raise ValueError("upstream model identity does not match the configured route")
    if not isinstance(props.get("model_path"), str) or Path(props["model_path"]) != Path(
        route.model_path
    ):
        raise ValueError("upstream weights do not match the configured route")
    return props


async def _prompt_tokens(route, payload, client):
    response = await client.post(
        f"{route.url.rstrip('/')}/apply-template",
        params={"autoload": "false"} if route.raw_router else None,
        json=payload,
        headers=_upstream_headers(route),
        timeout=5,
    )
    response.raise_for_status()
    prompt = response.json().get("prompt")
    if not isinstance(prompt, str):
        raise TypeError("the selected model did not return a text chat template")
    response = await client.post(
        f"{route.url.rstrip('/')}/tokenize",
        params={"autoload": "false"} if route.raw_router else None,
        json={"model": route.alias, "content": prompt, "add_special": True},
        headers=_upstream_headers(route),
        timeout=5,
    )
    response.raise_for_status()
    tokens = response.json().get("tokens")
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise ValueError("the selected model did not return valid prompt tokens")
    return len(tokens)


async def model_token_count(route: ModelRoute, text: str, client: httpx.AsyncClient) -> int:
    await check_model_ready(route, client)
    response = await client.post(
        f"{route.url.rstrip('/')}/tokenize",
        params={"autoload": "false"} if route.raw_router else None,
        json={"model": route.alias, "content": text, "add_special": False},
        headers=_upstream_headers(route),
        timeout=5,
    )
    response.raise_for_status()
    result = response.json()
    tokens = result.get("tokens") if isinstance(result, dict) else None
    if not isinstance(tokens, list) or any(type(token) is not int for token in tokens):
        raise ValueError("selected model tokenizer returned invalid tokens")
    return len(tokens)


def create_proxy_app(
    routes: tuple[ModelRoute, ...],
    credentials: tuple[Credential, ...],
    *,
    broker_url="http://127.0.0.1:8770",
    transport=None,
    activity=None,
):
    app = FastAPI(title="Dodo memory OpenAI proxy", docs_url=None, redoc_url=None)
    models = {route.id: route for route in routes}
    if len(models) != len(routes):
        raise ValueError("model route IDs must be unique")

    @app.get("/v1/models")
    async def list_models(request: Request):
        try:
            credential = authenticate(request.headers.get("authorization"), credentials)
        except PermissionError as error:
            return _error(401, "unauthorised", str(error))
        return {
            "object": "list",
            "data": [
                {"id": route.id, "object": "model", "owned_by": "local"}
                for route in routes
                if route.id in credential.routes
            ],
        }

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        try:
            credential = authenticate(request.headers.get("authorization"), credentials)
        except PermissionError as error:
            return _error(401, "unauthorised", str(error))
        raw = bytearray()
        async for chunk in request.stream():
            raw.extend(chunk)
            if len(raw) > MAX_REQUEST_BYTES:
                return _error(413, "request_too_large", "request exceeds 8 MiB")
        try:
            payload = json.loads(raw)
        except (ValueError, UnicodeDecodeError):
            return _error(400, "invalid_request", "request must contain a JSON object")
        if not isinstance(payload, dict):
            return _error(400, "invalid_request", "request must contain a JSON object")
        route_id = payload.get("model")
        if (
            not isinstance(route_id, str)
            or route_id not in models
            or route_id not in credential.routes
        ):
            return _error(403, "unknown_route", "model route is not authorised for this credential")
        if any(
            key in payload
            for key in ("upstream", "upstream_url", "upstream_token", "memory_scope", "client")
        ):
            return _error(
                400, "invalid_request", "routing and memory scope are fixed by the credential"
            )
        messages = payload.get("messages")
        if (
            not isinstance(messages, list)
            or not messages
            or any(not isinstance(message, dict) for message in messages)
        ):
            return _error(400, "invalid_request", "messages must be a nonempty array of objects")
        context = payload.pop("dodo_context", {})
        if (
            not isinstance(context, dict)
            or set(context) - {"node_id"}
            or (
                context
                and (
                    not isinstance(context.get("node_id"), str)
                    or not 1 <= len(context["node_id"]) <= 256
                )
            )
        ):
            return _error(
                400, "invalid_context", "local context must contain only a catalogue node_id"
            )
        enabled = payload.pop("dodo_memory", True)
        control = request.headers.get("x-dodo-memory")
        if type(enabled) is not bool or control not in {None, "on", "off"}:
            return _error(
                400,
                "invalid_request",
                "dodo_memory must be boolean; X-Dodo-Memory must be on or off",
            )
        auxiliary = auxiliary_request(messages, credential.client)
        enabled = enabled and control != "off" and not auxiliary
        route = models[route_id]
        stack = AsyncExitStack()
        await stack.__aenter__()
        streaming = False
        try:
            if activity is not None:
                await stack.enter_async_context(activity.track())
                try:
                    await stack.enter_async_context(activity.route(route))
                except RuntimeError:
                    return _error(
                        503,
                        "backend_busy",
                        "selected backend idle state could not be confirmed; retry",
                    )
            client = await stack.enter_async_context(
                httpx.AsyncClient(
                    transport=transport, timeout=httpx.Timeout(120, read=None), trust_env=False
                )
            )
            try:
                backend_props = await check_model_ready(route, client)
            except (httpx.HTTPError, ValueError):
                return _error(
                    503, "backend_unavailable", "selected model is not already loaded and ready"
                )
            memory_state = "auxiliary" if auxiliary else "off"
            packet = None
            text = ""
            if enabled:
                query = latest_user_text(messages, credential.client)
                if not query:
                    return _error(
                        400,
                        "missing_user_query",
                        "no genuine user query was found for this client adapter",
                    )
                try:
                    memory = await client.post(
                        f"{broker_url.rstrip('/')}/retrieve",
                        json={"query": query, "model": route_id, **context},
                        headers={"Authorization": request.headers["authorization"]},
                        timeout=30,
                    )
                    memory.raise_for_status()
                    packet = memory.json()
                    text = packet["memory_text"]
                    if (
                        not isinstance(text, str)
                        or len(text) > 6000
                        or not isinstance(packet.get("packet_id"), str)
                        or not isinstance(packet.get("sources"), list)
                    ):
                        raise ValueError("invalid broker memory packet")
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    return _error(
                        503,
                        "memory_unavailable",
                        "memory broker is unavailable; retry or explicitly disable memory",
                    )
                payload["messages"] = inject_memory(messages, text, credential.client)
                memory_state = "injected" if text else "no_matches"
            payload["model"] = route.alias
            context = backend_props.get("default_generation_settings", {}).get("n_ctx")
            if type(context) is not int or context <= 32:
                return _error(
                    503, "backend_unavailable", "loaded model context size could not be verified"
                )
            limits = [
                payload[key]
                for key in ("n_predict", "max_completion_tokens", "max_tokens")
                if payload.get(key) is not None
            ]
            if any(type(limit) is not int or limit <= 0 for limit in limits):
                return _error(
                    400, "invalid_output_limit", "all output limits must be positive integers"
                )
            if limits:
                # llama.cpp aliases can override one another, so reserve the largest supplied cap.
                output = max(limits)
            else:
                output = min(route.max_output_tokens, context // 4)
                payload["max_tokens"] = output
            if type(output) is not int or output <= 0 or output + 32 >= context:
                return _error(
                    400,
                    "invalid_output_limit",
                    "reserved output must be positive and fit the loaded model context",
                )
            sources = list(packet["sources"]) if packet else []
            base_tokens = None
            if text:
                try:
                    base_tokens = await _prompt_tokens(
                        route, {**payload, "messages": messages}, client
                    )
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    return _error(
                        503,
                        "tokenizer_unavailable",
                        "original conversation model tokens could not be verified",
                    )
            while True:
                try:
                    prompt_tokens = await _prompt_tokens(route, payload, client)
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    return _error(
                        503,
                        "tokenizer_unavailable",
                        "complete prompt template and model tokens could not be verified",
                    )
                memory_tokens = prompt_tokens - base_tokens if base_tokens is not None else 0
                if prompt_tokens + output + 32 <= context and memory_tokens <= 1536:
                    break
                if not text:
                    return _error(
                        400,
                        "context_exceeded",
                        "conversation and reserved output exceed the loaded model context",
                    )
                if sources:
                    sources.pop()
                text = _render(sources)
                payload["messages"] = inject_memory(messages, text, credential.client)
                memory_state = "shrunk" if text else "omitted_for_context"
            if packet is not None:
                try:
                    validation = await client.post(
                        f"{broker_url.rstrip('/')}/validate",
                        json={
                            "packet_id": packet["packet_id"],
                            "model": route_id,
                            "source_count": len(sources),
                        },
                        headers={"Authorization": request.headers["authorization"]},
                        timeout=5,
                    )
                    validation.raise_for_status()
                    if validation.json().get("valid") is not True:
                        raise ValueError("source revalidation was not confirmed")
                except (httpx.HTTPError, ValueError, KeyError, TypeError):
                    return _error(
                        503,
                        "memory_changed",
                        "memory evidence changed or could not be revalidated; retry",
                    )
            upstream_request = client.build_request(
                "POST",
                f"{route.url.rstrip('/')}/v1/chat/completions",
                params={"autoload": "false"} if route.raw_router else None,
                json=payload,
                headers=_upstream_headers(route),
            )
            try:
                upstream = await client.send(upstream_request, stream=True)
            except httpx.HTTPError:
                return _error(502, "backend_unavailable", "selected model request failed")
            stack.push_async_callback(upstream.aclose)
            headers = {"X-Dodo-Memory": memory_state}
            if packet is not None:
                headers["X-Dodo-Packet"] = packet["packet_id"]
            if "content-type" in upstream.headers:
                headers["Content-Type"] = upstream.headers["content-type"]
            if payload.get("stream") and upstream.is_success:

                async def chunks():
                    try:
                        async for part in upstream.aiter_raw():
                            yield part
                    finally:
                        await stack.aclose()

                streaming = True
                return StreamingResponse(
                    chunks(), status_code=upstream.status_code, headers=headers
                )
            return Response(
                await upstream.aread(), status_code=upstream.status_code, headers=headers
            )
        finally:
            if not streaming:
                await stack.aclose()

    return app
