"""Restrict a TLS remote connector to dedicated credentials and the local gateway."""

import argparse
import hmac
import json
import time
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import urlsplit

import anyio
import httpx
import uvicorn
from mcp.server.transport_security import RequestBodyLimitMiddleware
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

from .bridge import CLIENTS, GATEWAY

REMOTE_CLIENTS = {"chatgpt-tunnel", "claude-web"}
REQUEST_HEADERS = {
    "accept",
    "content-type",
    "mcp-session-id",
    "mcp-protocol-version",
    "last-event-id",
}
RESPONSE_HEADERS = {"content-type", "mcp-session-id", "mcp-protocol-version", "retry-after"}


def _url(value, remote):
    parsed = urlsplit(value)
    if (
        parsed.scheme != ("https" if remote else "http")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path != "/mcp"
        or (not remote and parsed.hostname not in {"127.0.0.1", "localhost", "::1"})
    ):
        kind = "HTTPS public" if remote else "HTTP loopback"
        raise ValueError(f"Expected a credential-free {kind} /mcp URL")
    return parsed


def load_remote_clients(home, *, allow_expired=False):
    """Reject native or private credentials before any remote listener is created."""
    home = Path(home)
    clients = json.loads((home / "secrets/remote-clients.json").read_text(encoding="utf-8"))
    forbidden = []
    for filename, names in [("client-tokens.json", CLIENTS), ("service-tokens.json", None)]:
        path = home / "secrets" / filename
        if path.exists():
            values = json.loads(path.read_text(encoding="utf-8"))
            forbidden.extend(values[name] for name in values if names is None or name in names)
    _validate_clients(clients, forbidden, allow_expired=allow_expired)
    return clients


def _validate_clients(clients, forbidden=(), *, allow_expired=False):
    if not isinstance(clients, dict) or not clients or set(clients) - REMOTE_CLIENTS:
        raise ValueError(
            "Remote credentials require explicit chatgpt-tunnel or claude-web identities"
        )
    seen = list(forbidden)
    for name, grant in clients.items():
        if not isinstance(grant, dict) or set(grant) != {"token", "expires_at"}:
            raise ValueError(f"Remote client {name!r} requires token and expires_at")
        token, expires_at = grant["token"], grant["expires_at"]
        if (
            not isinstance(token, str)
            or len(token) < 32
            or not token.isascii()
            or any(ord(char) < 33 or ord(char) > 126 for char in token)
            or token in seen
            or type(expires_at) is not int
            or (not allow_expired and expires_at <= time.time())
        ):
            raise ValueError(
                f"Remote client {name!r} needs a distinct, unexpired dedicated credential"
            )
        seen.append(token)


def make_remote_app(public_url, clients, gateway=GATEWAY, transport=None):
    """Forward authenticated requests without retries, private grants or arbitrary destinations."""
    public = _url(public_url, remote=True)
    _url(gateway, remote=False)
    _validate_clients(clients)
    authority = public.netloc.lower()
    origins = {f"https://{authority}", "https://claude.ai", "https://chatgpt.com"}
    http = httpx.AsyncClient(
        timeout=120, trust_env=False, follow_redirects=False, transport=transport
    )

    @asynccontextmanager
    async def lifespan(app):
        async with http:
            yield

    async def forward(request: Request):
        # Only the TLS terminator on loopback may supply the ASGI HTTPS scheme.
        if request.url.scheme != "https":
            return JSONResponse({"error": "HTTPS required"}, status_code=400)
        hosts = request.headers.getlist("host")
        if len(hosts) != 1 or hosts[0].lower() != authority:
            return JSONResponse({"error": "Unexpected Host"}, status_code=421)
        request_origins = request.headers.getlist("origin")
        if len(request_origins) > 1 or any(origin not in origins for origin in request_origins):
            return JSONResponse({"error": "Unexpected Origin"}, status_code=403)
        if request.url.query:
            return JSONResponse(
                {"error": "Query credentials and parameters are not accepted"}, status_code=400
            )
        authorisations = request.headers.getlist("authorization")
        token = None
        for grant in clients.values():
            expected = "Bearer " + grant["token"]
            if (
                len(authorisations) == 1
                and hmac.compare_digest(authorisations[0].encode(), expected.encode())
                and grant["expires_at"] > time.time()
            ):
                token = grant["token"]
        if token is None:
            return JSONResponse(
                {"error": "Dedicated remote bearer required"},
                status_code=401,
                headers={"WWW-Authenticate": "Bearer", "Cache-Control": "no-store"},
            )
        headers = {key: value for key, value in request.headers.items() if key in REQUEST_HEADERS}
        headers["authorization"] = "Bearer " + token
        try:
            upstream = await http.send(
                http.build_request(
                    request.method, gateway, headers=headers, content=await request.body()
                ),
                stream=True,
            )
        except httpx.HTTPError:
            return JSONResponse({"error": "Dodo Mem gateway unavailable"}, status_code=503)
        if upstream.status_code in {301, 302, 303, 307, 308, 401, 403}:
            await upstream.aclose()
            return JSONResponse({"error": "Gateway rejected remote configuration"}, status_code=502)

        async def body():
            try:
                async for chunk in upstream.aiter_bytes():
                    yield chunk
            finally:
                with anyio.CancelScope(shield=True):
                    await upstream.aclose()

        return StreamingResponse(
            body(),
            status_code=upstream.status_code,
            headers={
                **{
                    key: value for key, value in upstream.headers.items() if key in RESPONSE_HEADERS
                },
                "cache-control": "no-store",
            },
        )

    app = Starlette(
        routes=[Route("/mcp", forward, methods=["POST", "GET", "DELETE"])], lifespan=lifespan
    )
    return RequestBodyLimitMiddleware(app, max_body_size=1024 * 1024)


def main():
    parser = argparse.ArgumentParser(
        description="TLS connector ingress for the loopback memory gateway"
    )
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--public-url", required=True)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    app = make_remote_app(args.public_url, load_remote_clients(args.home))
    # A tunnel terminates public TLS; trust forwarded scheme only from its loopback connection.
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=args.port,
        proxy_headers=True,
        forwarded_allow_ips="127.0.0.1",
        log_level="warning",
        access_log=False,
    )


if __name__ == "__main__":
    main()
