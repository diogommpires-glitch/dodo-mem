"""Owner console calls the local gateway without becoming a second database writer."""

import json
from pathlib import Path
from uuid import uuid4

import httpx

ENDPOINT = "http://127.0.0.1:8765/mcp"
OWNER_TOOLS = frozenset(
    {
        "get_core",
        "connect_agent_matrix",
        "matrix_submit",
        "matrix_status",
        "matrix_list",
        "matrix_approve",
        "matrix_control",
        "search",
        "get_note",
        "save_ordinary_note",
        "open_source",
        "investigate",
        "review_collection",
        "list_sources",
        "ingestion_status",
        "enqueue_import",
        "enqueue_repository",
        "pause_ingestion",
        "cancel_import",
        "retry_import",
        "propose_milestone_procedure",
        "propose_working_knowledge",
        "reparse_source",
        "check_live_code",
        "refresh_live_code",
        "list_knowledge_proposals",
        "read_knowledge_proposal",
        "accept_working_knowledge",
        "accept_shared_proposal",
        "read_working_knowledge",
        "working_knowledge_history",
        "list_working_knowledge",
        "preview_purge",
        "execute_purge",
        "resume_purge",
    }
)


def call_owner(home, tool, arguments):
    if tool not in OWNER_TOOLS:
        raise ValueError("Unsupported owner operation")
    if (
        not isinstance(arguments, dict)
        or len(json.dumps(arguments, allow_nan=False).encode()) > 1024**2
    ):
        raise ValueError("Owner arguments must be an object of at most 1 MiB")
    path = Path(home) / "secrets/service-tokens.json"
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("Owner credential must be a real local file without redirects")
    with path.open("rb") as stream:
        content = stream.read(65537)
    if len(content) > 65536:
        raise ValueError("Owner credential configuration exceeds 64 KiB")
    config = json.loads(content)
    token = config.get("private_token")
    if not isinstance(token, str) or len(token) < 32 or any(ord(c) < 32 for c in token):
        raise ValueError("Owner credential is missing or invalid")
    identity = uuid4().hex
    with httpx.Client(trust_env=False, follow_redirects=False, timeout=120) as client:
        response = client.post(
            ENDPOINT,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/json, text/event-stream",
                "Mcp-Protocol-Version": "2025-11-25",
            },
            json={
                "jsonrpc": "2.0",
                "id": identity,
                "method": "tools/call",
                "params": {"name": tool, "arguments": arguments},
            },
        )
        response.raise_for_status()
        envelope = response.json()
    if envelope.get("id") != identity or "error" in envelope or "result" not in envelope:
        raise ValueError("Owner gateway returned an invalid protocol response")
    result = envelope["result"]
    if result.get("isError"):
        raise ValueError("Owner gateway refused the operation: " + json.dumps(result["content"]))
    return result.get("structuredContent", result)
