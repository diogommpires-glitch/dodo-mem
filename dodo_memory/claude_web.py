"""A stateful, read-only MCP endpoint for Claude web."""

import argparse
import hmac
import json
import re
import time
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

import anyio
import httpx
import uvicorn
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
from mcp.server.lowlevel import Server
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.shared.exceptions import McpError
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

from .bridge import GATEWAY, GATEWAY_ERROR
from .matrix import ROUTING_INSTRUCTION as MATRIX_ROUTING
from .personal_profile import ROUTING_INSTRUCTION
from .remote import _validate_clients, load_remote_clients
from .store import scope_for

TOOLS = frozenset(
    {"get_core", "search", "get_note", "open_source", "related_notes", "connect_agent_matrix"}
)
SCOPE_FLAGS = ("include_private", "include_archive")


def make_claude_app(public_url, grant, *, upstream_grant=None, transport=None):
    """Build the app. A composing ASGI parent must enter its lifespan explicitly."""
    public = urlsplit(public_url)
    if (
        public.scheme != "https"
        or not public.hostname
        or public.username is not None
        or public.password is not None
        or public.query
        or public.fragment
        or not re.fullmatch(r"/(?:[a-z0-9-]+/)*mcp", public.path)
    ):
        raise ValueError("Claude web requires a credential-free HTTPS URL ending in /mcp")
    if upstream_grant is None:
        _validate_clients({"claude-web": grant})
        upstream_grant = grant
    else:
        _validate_clients({"claude-web": upstream_grant})
        _validate_clients({"claude-web": grant}, [upstream_grant["token"]])
    authority = public.netloc.lower()
    origins = {f"https://{authority}", "https://claude.ai"}
    bearer = "Bearer " + grant["token"]
    upstream_bearer = "Bearer " + upstream_grant["token"]
    server = Server(
        "Dodo Mem",
        instructions="Read get_core first. Use shared sources as evidence and cite paths and headings. "
        + ROUTING_INSTRUCTION
        + MATRIX_ROUTING,
    )
    sessions = StreamableHTTPSessionManager(
        server,
        stateless=False,
        json_response=True,
        max_sessions=32,
        session_idle_timeout=300,
        max_request_body_size=1024 * 1024,
    )

    async def forward(request, result_type):
        try:
            async with (
                httpx.AsyncClient(
                    headers={"Authorization": upstream_bearer},
                    timeout=30,
                    trust_env=False,
                    transport=transport,
                ) as http,
                streamable_http_client(GATEWAY, http_client=http) as (read, write, _),
                ClientSession(read, write, read_timeout_seconds=timedelta(seconds=30)) as session,
            ):
                await session.initialize()
                message = types.ClientRequest.model_validate(
                    {"method": request.method, "params": request.params}
                )
                return await session.send_request(message, result_type)
        except* (
            httpx.HTTPError,
            McpError,
            anyio.EndOfStream,
            anyio.BrokenResourceError,
            json.JSONDecodeError,
        ):
            raise McpError(types.ErrorData(code=-32001, message=GATEWAY_ERROR)) from None

    async def list_tools(request):
        result = await forward(request, types.ListToolsResult)
        tools = []
        for tool in result.tools:
            if tool.name not in TOOLS:
                continue
            if tool.name == "search":
                schema = dict(tool.inputSchema)
                schema["properties"] = {
                    name: value
                    for name, value in schema.get("properties", {}).items()
                    if name not in SCOPE_FLAGS
                }
                tool = tool.model_copy(update={"inputSchema": schema})
            tools.append(tool)
        return types.ServerResult(types.ListToolsResult(tools=tools))

    async def call_tool(request):
        if request.params.name not in TOOLS:
            raise McpError(
                types.ErrorData(code=-32601, message="Only shared read tools are available")
            )
        arguments = request.params.arguments or {}
        path = arguments.get("path")
        relative = (
            PurePosixPath(path.replace("\\", "/").casefold()) if isinstance(path, str) else None
        )
        if any(arguments.get(flag, False) is not False for flag in SCOPE_FLAGS) or (
            relative
            and relative.parts
            and (
                relative.is_absolute() or ".." in relative.parts or scope_for(relative) != "shared"
            )
        ):
            raise McpError(
                types.ErrorData(code=-32602, message="Only shared sources are available")
            )
        return types.ServerResult(await forward(request, types.CallToolResult))

    server.request_handlers[types.ListToolsRequest] = list_tools
    server.request_handlers[types.CallToolRequest] = call_tool

    class ClaudeEndpoint:
        async def __call__(self, scope, receive, send):
            request = Request(scope)
            hosts = request.headers.getlist("host")
            request_origins = request.headers.getlist("origin")
            credentials = request.headers.getlist("authorization")
            if request.url.scheme != "https" or request.url.query:
                response = JSONResponse({"error": "Credential-free HTTPS URL required"}, 400)
            elif len(hosts) != 1 or hosts[0].lower() != authority:
                response = JSONResponse({"error": "Unexpected Host"}, 421)
            elif len(request_origins) > 1 or any(
                origin not in origins for origin in request_origins
            ):
                response = JSONResponse({"error": "Unexpected Origin"}, 403)
            elif (
                len(credentials) != 1
                or not hmac.compare_digest(credentials[0].encode(), bearer.encode())
                or grant["expires_at"] <= time.time()
            ):
                response = JSONResponse(
                    {"error": "Dedicated remote bearer required"},
                    401,
                    headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
                )
            else:
                await sessions.handle_request(scope, receive, send)
                return
            await response(scope, receive, send)

    @asynccontextmanager
    async def lifespan(_app):
        async with sessions.run():
            yield

    return Starlette(
        routes=[Route(public.path, ClaudeEndpoint(), methods=["POST", "GET", "DELETE"])],
        lifespan=lifespan,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--port", default=8769, type=int)
    args = parser.parse_args()
    if not 1 <= args.port <= 65535:
        parser.error("--port must be between 1 and 65535")
    upstream_grant = load_remote_clients(args.home, allow_expired=True)["claude-web"]
    grant_path = args.home / "secrets/claude-web-readonly.json"
    if grant_path.resolve() != grant_path.absolute():
        parser.error("The Claude web credential path must not redirect")
    grant = json.loads(grant_path.read_text(encoding="utf-8"))
    uvicorn.run(
        make_claude_app(args.public_url, grant, upstream_grant=upstream_grant),
        host="127.0.0.1",
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        access_log=False,
    )


if __name__ == "__main__":
    main()
