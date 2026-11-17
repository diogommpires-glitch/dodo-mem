"""The desktop's authenticated loopback connection and separate owner action."""

import json
import os
import subprocess
from pathlib import Path
from uuid import uuid4

import httpx

ENDPOINT = "http://127.0.0.1:8765/mcp"
from .configuration import runtime_path

MEMORY_HOME = runtime_path("gateway")
DESKTOP_TOOLS = frozenset(
    {
        "browse_notes",
        "related_notes",
        "search",
        "get_note",
        "read_note_for_edit",
        "ordinary_note_history",
        "read_ordinary_note_revision",
        "resolve_note_link",
        "suggest_note_links",
        "graph_snapshot",
        "note_card_refs",
        "open_source",
        "write_note",
        "list_knowledge_proposals",
        "read_knowledge_proposal",
        "propose_working_knowledge",
        "read_working_knowledge",
        "working_knowledge_history",
        "connect_agent_matrix",
        "matrix_submit",
        "matrix_status",
        "matrix_list",
        "matrix_control",
    }
)
MAX_RESPONSE_BYTES = 8 * 1024**2
OWNER_BRIDGE = """
import json
import sys
from dodo_memory.owner import call_owner

try:
    arguments = json.load(sys.stdin)
    operation = sys.argv[2]
    matrix = {"matrix_approve", "matrix_status", "matrix_list", "matrix_control"}
    if operation not in matrix | {"accept_shared_proposal", "save_ordinary_note"}:
        raise ValueError("Unsupported desktop owner operation")
    value = call_owner(sys.argv[1], operation, arguments)
    if operation in matrix:
        output = json.dumps(value, ensure_ascii=True, allow_nan=False)
        if len(output.encode()) > 8 * 1024**2:
            raise ValueError("Matrix owner response exceeds 8 MiB")
        print(output)
except (OSError, ValueError) as error:
    print(type(error).__name__, file=sys.stderr)
    sys.exit(1)
"""


class GatewayError(Exception):
    def __init__(self, message: str, *, connection_lost: bool = False):
        super().__init__(message)
        self.connection_lost = connection_lost


class Gateway:
    def __init__(self, home: Path = MEMORY_HOME, *, transport: httpx.BaseTransport | None = None):
        self.home = home
        self.transport = transport

    def _token(self) -> str:
        path = self.home / "secrets" / "client-tokens.json"
        try:
            if path.resolve(strict=True) != path.absolute():
                raise ValueError("Redirected credential")
            with path.open("rb") as stream:
                body = stream.read(65537)
            if len(body) > 65536:
                raise ValueError("Credential file too large")
            token = json.loads(body)["desktop"]
            if (
                not isinstance(token, str)
                or len(token) < 32
                or not token.isascii()
                or any(not 33 <= ord(character) <= 126 for character in token)
            ):
                raise ValueError("Invalid desktop credential")
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise GatewayError(
                "Desktop credential is missing or invalid.", connection_lost=True
            ) from error
        return token

    def call(self, name: str, arguments: dict | None = None):
        if name not in DESKTOP_TOOLS:
            raise GatewayError("This operation is outside the desktop's shared access.")
        identity = uuid4().hex
        try:
            with (
                httpx.Client(
                    trust_env=False,
                    follow_redirects=False,
                    timeout=10,
                    transport=self.transport,
                ) as client,
                client.stream(
                    "POST",
                    ENDPOINT,
                    headers={
                        "Authorization": f"Bearer {self._token()}",
                        "Accept": "application/json, text/event-stream",
                        "Mcp-Protocol-Version": "2025-11-25",
                    },
                    json={
                        "jsonrpc": "2.0",
                        "id": identity,
                        "method": "tools/call",
                        "params": {"name": name, "arguments": arguments or {}},
                    },
                ) as response,
            ):
                if response.status_code in {401, 403}:
                    raise GatewayError(
                        "Dodo Mem rejected the desktop credential.",
                        connection_lost=True,
                    )
                response.raise_for_status()
                body = bytearray()
                for part in response.iter_bytes():
                    if len(body) + len(part) > MAX_RESPONSE_BYTES:
                        raise GatewayError("The Dodo Mem response is too large to display.")
                    body.extend(part)
            envelope = json.loads(body)
        except httpx.RequestError as error:
            raise GatewayError(
                "Dodo Mem is offline or did not respond.", connection_lost=True
            ) from error
        except httpx.HTTPStatusError as error:
            raise GatewayError(
                "Dodo Mem returned a service error.", connection_lost=True
            ) from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise GatewayError(
                "Dodo Mem returned an invalid response.", connection_lost=True
            ) from error
        if not isinstance(envelope, dict) or envelope.get("id") != identity:
            raise GatewayError("Dodo Mem returned a mismatched response.", connection_lost=True)
        result = envelope.get("result")
        if not isinstance(result, dict) or result.get("isError"):
            if name == "open_source":
                raise GatewayError("Original unavailable or changed.")
            if name == "get_note":
                raise GatewayError("This note is no longer available.")
            raise GatewayError("Dodo Mem refused the operation. Refresh and try again.")
        value = result.get("structuredContent")
        if isinstance(value, dict) and set(value) == {"result"}:
            value = value["result"]
        if not isinstance(value, (dict, list)):
            raise GatewayError("Dodo Mem returned an unsupported result.", connection_lost=True)
        return value

    def approve(self, proposal_id: str, expected_revision: int, review_note: str) -> None:
        self._owner(
            "accept_shared_proposal",
            {
                "proposal_id": proposal_id,
                "expected_revision": expected_revision,
                "review_note": review_note,
            },
            "Approval was not confirmed. Refresh and check the revision and evidence.",
        )

    def save_note(self, path: str, text: str, expected_sha256: str | None) -> None:
        self._owner(
            "save_ordinary_note",
            {"path": path, "text": text, "expected_sha256": expected_sha256},
            "Save was not confirmed. Reopen the current revision before trying again.",
        )

    def matrix_owner(self, operation, arguments):
        if operation not in {"matrix_approve", "matrix_status", "matrix_list", "matrix_control"}:
            raise GatewayError("This Matrix operation is outside the desktop owner bridge.")
        return self._owner(
            operation, arguments, "Matrix owner operation was not confirmed.", read_result=True
        )

    def _owner(self, operation, arguments, error_message, *, read_result=False):
        executable = self.home / ".venv" / "Scripts" / "python.exe"
        try:
            result = subprocess.run(
                [str(executable), "-I", "-B", "-c", OWNER_BRIDGE, str(self.home), operation],
                input=json.dumps(arguments, ensure_ascii=True),
                capture_output=True,
                text=True,
                timeout=30,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GatewayError(error_message) from error
        if result.returncode:
            raise GatewayError(error_message)
        if read_result:
            if len(result.stdout.encode("utf-8")) > MAX_RESPONSE_BYTES:
                raise GatewayError("Matrix owner response exceeds the display limit.")
            try:
                value = json.loads(result.stdout)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise GatewayError("Matrix owner returned invalid JSON.") from error
            if isinstance(value, dict) and set(value) == {"result"}:
                value = value["result"]
            if not isinstance(value, dict):
                raise GatewayError("Matrix owner returned an unsupported result.")
            return value
        return None

    def service_status(self) -> dict:
        try:
            self.call("browse_notes", {"limit": 1})
            gateway = "Connected"
        except GatewayError as error:
            gateway = str(error)
        try:
            with httpx.Client(trust_env=False, follow_redirects=False, timeout=2) as client:
                response = client.get("http://127.0.0.1:18092/health")
                embeddings = "Ready" if response.is_success else "Listener present"
        except httpx.HTTPError:
            embeddings = "Unavailable"
        return {
            "Dodo Mem": gateway,
            "Embeddings": embeddings,
            "External connectors": "Managed separately",
        }

    def start(self) -> None:
        launcher = self.home / "Start-Shared-Memory.cmd"
        command = Path(os.environ.get("SYSTEMROOT", "C:/Windows")) / "System32" / "cmd.exe"
        try:
            result = subprocess.run(
                [str(command), "/d", "/c", str(launcher)],
                cwd=self.home,
                timeout=180,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                creationflags=subprocess.CREATE_NO_WINDOW,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as error:
            raise GatewayError("The Dodo Mem launcher did not confirm startup.") from error
        if result.returncode:
            raise GatewayError("The Dodo Mem launcher could not start the service.")
