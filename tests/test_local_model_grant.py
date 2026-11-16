import pytest
from starlette.testclient import TestClient

from dodo_memory.server import make_app
from dodo_memory.store import Vault


@pytest.fixture
def local_model_request(tmp_path):
    vault = Vault(tmp_path)
    for name in ("core/CORE.md", "private/source.md"):
        note = vault.resolve(name)
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text("# Fixture\n\nExpected evidence.", encoding="utf-8")
    app = make_app(vault, {}, "s" * 40, "o" * 40, {"local-models": "m" * 40})
    headers = {
        "Authorization": "Bearer " + "m" * 40,
        "Accept": "application/json, text/event-stream",
        "Mcp-Protocol-Version": "2025-11-25",
    }
    with TestClient(app, base_url="http://127.0.0.1:8765", headers=headers) as client:

        def request(method, params=None):
            return client.post(
                "/mcp", json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
            ).json()["result"]

        yield request


def test_local_model_exposes_the_explicit_shared_contract(local_model_request):
    assert {tool["name"] for tool in local_model_request("tools/list")["tools"]} == {
        "get_core",
        "search",
        "get_note",
        "open_source",
        "related_notes",
        "retrieve_context",
        "connect_agent_matrix",
        "matrix_status",
    }
    core = local_model_request("tools/call", {"name": "get_core", "arguments": {}})
    assert "Expected evidence" in core["structuredContent"]["text"]


def test_local_model_cannot_read_private_sources(local_model_request):
    result = local_model_request(
        "tools/call", {"name": "get_note", "arguments": {"path": "private/source.md"}}
    )
    assert result["isError"]
    assert "not granted" in result["content"][0]["text"]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("write_note", {"title": "x", "text": "y"}),
        (
            "matrix_submit",
            {"session_id": "7" * 32, "request_id": "acl-check", "question": "Synthetic check"},
        ),
        ("matrix_approve", {"run_id": "8" * 32, "expected_request_sha256": "9" * 64}),
    ],
)
def test_local_model_cannot_write_dispatch_or_approve(local_model_request, tool, arguments):
    result = local_model_request("tools/call", {"name": tool, "arguments": arguments})
    assert result["isError"]
    assert "permits shared reads" in result["content"][0]["text"]
