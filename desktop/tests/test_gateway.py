import json
from types import SimpleNamespace

import httpx
import pytest

from dodo_memory_desktop.gateway import Gateway, GatewayError


def test_gateway_pins_endpoint_and_requests_exact_source(tmp_path):
    folder = tmp_path / "secrets"
    folder.mkdir()
    (folder / "client-tokens.json").write_text(json.dumps({"desktop": "a" * 32}))

    def respond(request):
        assert str(request.url) == "http://127.0.0.1:8765/mcp"
        assert request.headers["Authorization"] == "Bearer " + "a" * 32
        body = json.loads(request.content)
        assert body["params"] == {
            "name": "open_source",
            "arguments": {"path": "notes/data.md", "lines": [2, 3], "preview": True},
        }
        return httpx.Response(
            200,
            json={"id": body["id"], "result": {"structuredContent": {"preview_text": "example"}}},
        )

    gateway = Gateway(tmp_path, transport=httpx.MockTransport(respond))
    result = gateway.call(
        "open_source", {"path": "notes/data.md", "lines": [2, 3], "preview": True}
    )
    assert result["preview_text"] == "example"
    with pytest.raises(GatewayError):
        gateway.call("accept_shared_proposal")


def test_gateway_authentication_failure_is_distinct(tmp_path):
    folder = tmp_path / "secrets"
    folder.mkdir()
    (folder / "client-tokens.json").write_text(json.dumps({"desktop": "b" * 32}))
    gateway = Gateway(tmp_path, transport=httpx.MockTransport(lambda _: httpx.Response(401)))
    with pytest.raises(GatewayError, match="credential") as failure:
        gateway.call("browse_notes")
    assert failure.value.connection_lost


def test_matrix_preparation_uses_only_desktop_tools_and_never_owner_approval(tmp_path):
    folder = tmp_path / "secrets"
    folder.mkdir()
    (folder / "client-tokens.json").write_text(json.dumps({"desktop": "c" * 32}))
    calls = []

    def respond(request):
        body = json.loads(request.content)
        calls.append(body["params"])
        return httpx.Response(
            200,
            json={"id": body["id"], "result": {"structuredContent": {"state": "ready"}}},
        )

    gateway = Gateway(tmp_path, transport=httpx.MockTransport(respond))
    for tool in (
        "connect_agent_matrix",
        "matrix_submit",
        "matrix_status",
        "matrix_list",
        "matrix_control",
    ):
        assert gateway.call(tool, {"request_id": "fixture"}) == {"state": "ready"}
    assert [call["name"] for call in calls] == [
        "connect_agent_matrix",
        "matrix_submit",
        "matrix_status",
        "matrix_list",
        "matrix_control",
    ]
    with pytest.raises(GatewayError):
        gateway.call("matrix_approve")


def test_matrix_owner_bridge_preserves_exact_approval_and_refuses_unrelated_tools(
    tmp_path, monkeypatch
):
    from dodo_memory_desktop import gateway as module

    calls = []
    expected = {"run_id": "fixture", "expected_request_sha256": "d" * 64}

    def run(command, **options):
        calls.append((command, options))
        return SimpleNamespace(returncode=0, stdout=json.dumps({"result": {"state": "queued"}}))

    monkeypatch.setattr(module.subprocess, "run", run)
    gateway = Gateway(tmp_path)
    assert gateway.matrix_owner("matrix_approve", expected) == {"state": "queued"}
    command, options = calls[0]
    assert command[-1] == "matrix_approve"
    assert json.loads(options["input"]) == expected
    assert options["creationflags"] == module.subprocess.CREATE_NO_WINDOW
    with pytest.raises(GatewayError):
        gateway.matrix_owner("get_note", {"path": "fixture"})
    assert len(calls) == 1


@pytest.mark.parametrize("output", ["not json", "[]", " " * (8 * 1024**2 + 1)])
def test_matrix_owner_bridge_rejects_invalid_or_oversized_output(tmp_path, monkeypatch, output):
    from dodo_memory_desktop import gateway as module

    monkeypatch.setattr(
        module.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout=output),
    )
    with pytest.raises(GatewayError, match="Matrix"):
        Gateway(tmp_path).matrix_owner("matrix_status", {"run_id": "fixture"})
