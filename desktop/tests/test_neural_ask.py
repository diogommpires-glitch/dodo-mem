import json
from types import SimpleNamespace

import httpx
import pytest

from dodo_memory_desktop.ask_evidence import AskError, Context
from dodo_memory_desktop.neural_ask import activated, ask_neural, ask_without_memory


def setup_state(tmp_path):
    (tmp_path / "broker.token").write_text("dodo-secret", encoding="utf-8")
    route = {
        "id": "coder",
        "alias": "alias",
        "url": "http://127.0.0.1:8098",
        "model_path": "C:/models/coder.gguf",
        "raw_router": False,
    }
    (tmp_path / "routes.json").write_text(json.dumps({"routes": [route]}), encoding="utf-8")
    model = SimpleNamespace(alias="alias", port=8098, file="C:/models/coder.gguf", name="Coder")
    return route, model


def test_neural_ask_sends_plain_question_once_and_displays_exact_proxy_packet(tmp_path):
    route, model = setup_state(tmp_path)
    calls = []

    def upstream(request):
        calls.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"model_alias": "alias", "model_path": route["model_path"]}
            )
        assert request.headers["authorization"] == "Bearer dodo-secret"
        if request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["model"] == "coder"
            assert body["messages"][-1]["content"] == "question"
            assert "sources" not in body["messages"][-1]["content"]
            return httpx.Response(
                200,
                json={"choices": [{"message": {"content": "answer [1]"}}]},
                headers={"X-Dodo-Packet": "receipt"},
            )
        if request.url.path == "/packet":
            return httpx.Response(
                200,
                json={
                    "sources": [
                        {
                            "node_id": "n",
                            "path": "fixture.md",
                            "locator": {"kind": "lines", "start": 1, "end": 1},
                            "excerpt": "verified",
                            "cite": "[1]",
                            "relationship_path": [
                                {
                                    "source": "a",
                                    "target": "n",
                                    "kind": "reference",
                                    "evidence": {"line": 1},
                                }
                            ],
                        }
                    ],
                    "coverage": {"returned": 1},
                },
            )
        return httpx.Response(200, json={"valid": True})

    answer, snapshot, alias = ask_neural(
        "question",
        Context(),
        None,
        (model,),
        home=tmp_path,
        transport=httpx.MockTransport(upstream),
    )
    assert (answer, alias) == ("answer [1]", "alias")
    assert snapshot.description["sources"][0]["excerpt"] == "verified"
    assert snapshot.description["sources"][0]["relationship_path"][0]["from"] == "a"
    snapshot.verify(None)
    assert calls == ["/props", "/v1/chat/completions", "/packet", "/validate"]


def test_memory_off_uses_selected_backend_without_memory_service_or_source_reads(tmp_path):
    _, model = setup_state(tmp_path)
    calls = []

    def upstream(request):
        calls.append(request.url.path)
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_alias": "alias", "model_path": str(model.file)})
        if request.url.path == "/slots":
            return httpx.Response(200, json=[{"is_processing": False}])
        body = json.loads(request.content)
        assert body["model"] == "alias"
        assert body["messages"][-1]["content"] == "question"
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain answer"}}]})

    answer, snapshot, alias = ask_without_memory(
        "question", None, (model,), transport=httpx.MockTransport(upstream)
    )
    assert answer == "plain answer"
    assert alias == "alias"
    assert snapshot.sources == ()
    assert snapshot.description["coverage"]["memory_enabled"] is False
    assert calls == ["/props", "/slots", "/v1/chat/completions", "/slots"]


def test_rollout_defaults_off_and_unavailable_proxy_never_falls_back(tmp_path):
    _, model = setup_state(tmp_path)
    assert activated(tmp_path) is False
    (tmp_path / "selection.json").write_text('{"activated":true}', encoding="utf-8")
    assert activated(tmp_path) is True

    def upstream(request):
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_alias": "alias", "model_path": str(model.file)})
        raise httpx.ConnectError("offline")

    with pytest.raises(AskError, match="disable memory"):
        ask_neural(
            "question",
            Context(),
            None,
            (model,),
            home=tmp_path,
            transport=httpx.MockTransport(upstream),
        )


def test_public_ask_local_memory_off_does_not_read_catalogue(tmp_path):
    from dodo_memory_desktop.ask_models import ask_local

    _, model = setup_state(tmp_path)

    def upstream(request):
        if request.url.path == "/props":
            return httpx.Response(
                200, json={"model_alias": model.alias, "model_path": str(model.file)}
            )
        if request.url.path == "/slots":
            return httpx.Response(200, json=[{"is_processing": False}])
        return httpx.Response(200, json={"choices": [{"message": {"content": "plain answer"}}]})

    answer, snapshot, alias = ask_local(
        None,
        "question",
        Context(),
        model=model,
        memory_enabled=False,
        include_snapshot=True,
        transport=httpx.MockTransport(upstream),
    )
    assert (answer, alias) == ("plain answer", "alias")
    assert snapshot.sources == ()
