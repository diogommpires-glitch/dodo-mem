import hashlib
import json
import shutil
import subprocess
import time
from dataclasses import replace
from functools import partial
from types import SimpleNamespace

import httpx
import pytest

from dodo_memory_desktop.ask_evidence import AskError, Context, EvidenceSession, Handoffs
from dodo_memory_desktop.ask_models import ask_local, local_models, send_hosted
from dodo_memory_desktop.neural_ask import activated
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


@pytest.fixture(autouse=True)
def isolated_neural_rollout(tmp_path, monkeypatch):
    monkeypatch.setattr(
        "dodo_memory_desktop.ask_models.neural_activated", partial(activated, home=tmp_path)
    )


class FixtureCatalogue:
    def __init__(self):
        self.files = {
            "commit": "Streaming design landed in the reviewed merge.",
            "file": "Merge records without loading the whole file.\nUse deterministic ordering.",
            "decision": "Stream merges for low peak RAM.",
            "spec": "The import budget is 256 MiB.",
            "guess": "Inferred link.",
            "secret": "Private key fixture.",
        }
        self.edges = {
            "commit": [("file", "changed_file")],
            "file": [("decision", "reference"), ("guess", "topic"), ("secret", "contains")],
            "decision": [("spec", "citation")],
        }

    def status(self, *, summary=False):
        return {"generation": 2, "scanning": False}

    def node(self, node_id):
        if node_id in self.files:
            return {
                "id": node_id,
                "name": node_id,
                "kind": "file",
                "digest": self.digest(node_id),
                "project_id": "fixture",
            }
        return None

    def resolve_path(self, _path):
        return None

    def search(self, query, limit=20, scope=None):
        if scope is not None:
            assert scope == "fixture"
        return [self.node("file")] if query == "streaming" else []

    def neighbours(self, node_id, limit=40, include_inferred=False):
        assert not include_inferred
        return [
            {"node": self.node(linked), "kind": kind, "evidence": "verified fixture"}
            for linked, kind in self.edges.get(node_id, [])[:limit]
            if linked in self.files
        ]

    def digest(self, node_id):
        return hashlib.sha256(self.files[node_id].encode()).hexdigest()

    def read_evidence(self, node_id, limit=1000, start_line=1):
        if node_id not in self.files:
            return None
        return {
            "path": f"C:/fixture/{node_id}.md",
            "digest": self.digest(node_id),
            "locator": {"kind": "lines", "start": start_line, "end": start_line},
            "text": "\n".join(self.files[node_id].splitlines()[start_line - 1 :])[:limit],
        }


def test_local_evidence_follows_git_file_decision_and_source():
    session = EvidenceSession(
        FixtureCatalogue(), "Why does the streaming design work?", Context(node_id="commit")
    )
    session.fill()
    items = {item["path"]: item for item in session.describe()["sources"]}
    assert set(items) == {
        "C:/fixture/commit.md",
        "C:/fixture/file.md",
        "C:/fixture/decision.md",
        "C:/fixture/spec.md",
    }
    assert [edge["kind"] for edge in items["C:/fixture/spec.md"]["relationship_path"]] == [
        "reference",
        "citation",
    ]
    assert items["C:/fixture/spec.md"]["excerpt"] == "The import budget is 256 MiB."
    assert [edge["kind"] for edge in session.describe()["connections"]] == ["changed_file"]
    session.verify()
    assert session.read("secret") is None


@pytest.mark.parametrize("provider", ["Codex", "Claude"])
def test_hosted_preview_requires_exact_single_use_approval(provider):
    catalogue, calls = FixtureCatalogue(), []
    handoff = Handoffs(catalogue, lambda *payload: calls.append(payload) or "Evidence [1]")
    cancelled = handoff.prepare(provider, "streaming")
    assert handoff.submit(cancelled, approved=False) is None
    assert not calls
    packet = handoff.prepare(provider, "streaming")
    forged = replace(packet, preview="unapproved bytes")
    with pytest.raises(AskError, match="expired"):
        handoff.submit(forged, approved=True)
    assert not calls
    assert handoff.submit(packet, approved=True) == "Evidence [1]"
    assert calls == [(provider, packet.preview)]
    with pytest.raises(AskError, match="expired"):
        handoff.submit(packet, approved=True)
    assert len(calls) == 1


def test_hosted_preview_is_invalidated_by_edge_and_source_changes():
    catalogue, calls = FixtureCatalogue(), []
    handoff = Handoffs(catalogue, lambda *payload: calls.append(payload))
    packet = handoff.prepare("Codex", "streaming")
    catalogue.edges["decision"] = []
    with pytest.raises(AskError, match="relationship"):
        handoff.submit(packet, approved=True)
    packet = handoff.prepare("Claude", "streaming")
    catalogue.files.pop("file")
    with pytest.raises(AskError, match="became unavailable"):
        handoff.submit(packet, approved=True)
    assert not calls


@pytest.fixture
def fallback_model_routes(tmp_path, monkeypatch):
    from dodo_memory_desktop.ask_models import LocalModel

    models = (
        LocalModel("Synthetic coder", 8098, "local-coder-heretic"),
        LocalModel("Synthetic model", 8091, "qwen27-q8"),
    )
    monkeypatch.setattr("dodo_memory_desktop.ask_models.local_models", lambda: models)


def test_qwen_uses_exact_ready_alias_and_only_reached_evidence(fallback_model_routes):
    posts = []

    def route(request):
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            alias = "qwen27-q8" if request.url.port == 8091 else "unexpected-model"
            return httpx.Response(200, json={"data": [{"id": alias}]})
        assert request.url.port == 8091
        data = json.loads(request.content)
        posts.append(data)
        assert data["model"] == "qwen27-q8"
        if len(posts) == 1:
            message = {
                "content": None,
                "tool_calls": [
                    {
                        "id": "denied",
                        "function": {
                            "name": "read_evidence",
                            "arguments": '{"node_id":"secret"}',
                        },
                    },
                    {
                        "id": "extra",
                        "function": {
                            "name": "read_evidence",
                            "arguments": '{"node_id":"file","start_line":2}',
                        },
                    },
                ],
            }
        else:
            assert "unavailable" in data["messages"][-2]["content"]
            assert "deterministic ordering" in data["messages"][-1]["content"]
            message = {"content": "Use deterministic ordering [4]."}
        return httpx.Response(200, json={"model": "qwen27-q8", "choices": [{"message": message}]})

    answer, evidence, alias = ask_local(
        FixtureCatalogue(),
        "streaming",
        Context(),
        ("Dodo guidance", "Check assumptions"),
        transport=httpx.MockTransport(route),
    )
    assert alias == "qwen27-q8"
    assert answer.endswith("[4].")
    assert len(evidence["sources"]) == 4
    assert "Dodo guidance" in posts[0]["messages"][0]["content"]
    assert "Private key fixture" not in json.dumps(posts)


def test_contextualize_uses_private_profile_only_for_local_opt_in(
    tmp_path, monkeypatch, fallback_model_routes
):
    monkeypatch.setenv("DODO_MEM_RUNTIME_ROOT", str(tmp_path))
    source = tmp_path / "Writing voice.md"
    source.write_text(
        "My casual drafts use concise sentences and dry humour.\n"
        + "\n" * 30
        + "z" * 2500
        + "\nClose institutional emails with Warm regards.\n",
        encoding="utf-8",
    )
    runtime = tmp_path / "desktop"
    runtime.mkdir(parents=True)
    (runtime / "personal-profile.json").write_text(
        json.dumps(
            {
                "sources": [
                    {
                        "path": str(source),
                        "section": "Writing",
                        "basis": "observed",
                        "enabled": True,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    posts = []

    def route(request):
        assert request.url.host == "127.0.0.1"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "qwen27-q8"}]})
        posts.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "model": "qwen27-q8",
                "choices": [{"message": {"content": "Draft [1]."}}],
            },
        )

    catalogue = FixtureCatalogue()
    ask_local(catalogue, "Draft a message", Context(), transport=httpx.MockTransport(route))
    assert posts == []
    _, snapshot, alias = ask_local(
        catalogue,
        "Draft a message. CONTEXTUALISE WITH MEM",
        Context(),
        transport=httpx.MockTransport(route),
        include_snapshot=True,
    )
    assert alias == "qwen27-q8"
    assert posts[0]["reasoning_effort"] == "none"
    packet = json.loads(posts[0]["messages"][-1]["content"])
    assert packet["question"] == "Draft a message."
    assert packet["sources"][0]["path"] == str(source)
    assert packet["sources"][0]["profile"] == {"section": "Writing", "basis": "observed"}
    assert "dry humour" in packet["sources"][0]["excerpt"]

    overview, _, overview_model = ask_local(
        catalogue, "Contextualize with mem.", Context(), transport=httpx.MockTransport(route)
    )
    assert overview_model is None and "Writing voice (observed) [1]" in overview
    assert "My casual drafts use concise sentences and dry humour." in overview
    assert len(posts) == 1

    closing = EvidenceSession(
        catalogue,
        "Contextualize with mem. What closing fits an institutional email?",
        personal=True,
    )
    assert "Warm regards" in closing.describe()["sources"][0]["excerpt"]

    for question in ("streaming", "Contextualize with mem: debug streaming"):
        ask_local(catalogue, question, Context(), transport=httpx.MockTransport(route))
        assert "reasoning_effort" not in posts[-1]

    hosted = Handoffs(catalogue, lambda *_: pytest.fail("An unapproved request was sent."))
    preview = hosted.prepare("Codex", "streaming, contextualize with mem")
    assert str(source) not in preview.preview
    source.write_text("Use a different voice.", encoding="utf-8")
    with pytest.raises(AskError, match="changed"):
        snapshot.verify(catalogue)


def test_model_failure_after_post_never_uses_another_provider(fallback_model_routes):
    called = []

    def route(request):
        called.append((request.url.port, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": "local-coder-heretic"}]})
        return httpx.Response(503)

    with pytest.raises(AskError, match="stopped responding"):
        ask_local(FixtureCatalogue(), "streaming", Context(), transport=httpx.MockTransport(route))
    assert {port for port, _ in called} == {8098}


def test_registry_selected_model_requires_the_matching_weights_before_receiving_evidence(tmp_path):
    registry = tmp_path / "models.json"
    registry.write_text(
        json.dumps(
            {
                "qwythos-heretic": {
                    "launcher": "Qwythos Heretic",
                    "alias": "local-qwythos-heretic",
                    "file": "models/qwythos.gguf",
                    "port": 8097,
                }
            }
        ),
        encoding="utf-8",
    )
    (model,) = local_models(registry)
    posts = []
    loaded = str(model.file.with_name("wrong.gguf"))

    def route(request):
        assert request.url.host == "127.0.0.1" and request.url.port == 8097
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        if request.url.path == "/v1/models":
            return httpx.Response(200, json={"data": [{"id": model.alias}]})
        if request.url.path == "/props":
            return httpx.Response(200, json={"model_alias": model.alias, "model_path": loaded})
        posts.append(json.loads(request.content))
        return httpx.Response(
            200, json={"model": model.alias, "choices": [{"message": {"content": "Stream [1]."}}]}
        )

    arguments = {"transport": httpx.MockTransport(route), "model": model}
    answer, evidence, alias = ask_local(FixtureCatalogue(), "streaming", Context(), **arguments)
    assert alias is None and "Qwythos Heretic is not ready" in answer and evidence["sources"]
    assert not posts

    loaded = str(model.file)
    answer, _, alias = ask_local(FixtureCatalogue(), "streaming", Context(), **arguments)
    assert answer == "Stream [1]." and alias == model.alias
    assert posts[0]["model"] == model.alias
    assert json.loads(posts[0]["messages"][-1]["content"])["sources"]


@pytest.mark.parametrize("provider", ["Codex", "Claude"])
def test_hosted_cli_receives_packet_via_stdin_with_tools_disabled(provider, monkeypatch):
    preview = '{"question":"fixture","sources":[{"excerpt":"approved bytes"}]}'
    monkeypatch.setattr(
        "dodo_memory_desktop.ask_models._provider_executable",
        lambda _provider: "C:/Program Files/Fixture/provider.exe",
    )

    def run(arguments, *, input, stdout, cwd, env, **kwargs):
        assert preview not in arguments
        assert "https://" not in input.decode()
        assert not {"OPENAI_API_KEY", "ANTHROPIC_API_KEY", "HTTP_PROXY"} & env.keys()
        if provider == "Codex":
            assert "features.shell_tool=false" in arguments
            assert "mcp_servers={}" in arguments
            assert input.decode().endswith("\n\n" + preview)
            (cwd / "answer.txt").write_text("Approved evidence [1]", encoding="utf-8")
        else:
            assert arguments[arguments.index("--tools") + 1] == ""
            assert "--strict-mcp-config" in arguments
            assert input.decode() == preview
            stdout.write(b'{"result":"Approved evidence [1]"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr("dodo_memory_desktop.ask_models.subprocess.run", run)
    assert send_hosted(provider, preview) == "Approved evidence [1]"


def test_desktop_exact_preview_cancel_and_approve(monkeypatch):
    from PySide6.QtCore import QTimer
    from PySide6.QtWidgets import QApplication, QDialog, QPlainTextEdit, QPushButton

    from dodo_memory_desktop.ask_dodo import AskDodo

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    calls = []
    catalogue = FixtureCatalogue()
    panel = AskDodo(
        catalogue,
        send=lambda *data: calls.append(data) or "Done [1]",
        local=lambda *_args, **_kwargs: pytest.fail("An unavailable model fell back silently."),
    )
    panel.set_skill("Local workflow", "Use private skill guidance.")
    panel.select_model("unlisted-model")
    panel.question.setText("streaming")
    panel.ask()
    assert "no longer lists unlisted-model" in panel.status.text() and not panel._busy
    panel.provider.setCurrentText("Claude")
    panel.show()

    previews = []
    approved = False
    timer = QTimer(interval=10)

    def review():
        dialog = app.activeModalWidget()
        if not isinstance(dialog, QDialog):
            return
        previews.append(dialog.findChild(QPlainTextEdit).toPlainText())
        if approved:
            send = next(
                item
                for item in dialog.findChildren(QPushButton)
                if item.text() == "Send once to Claude"
            )
            send.click()
        else:
            dialog.reject()

    def until(predicate):
        deadline = time.monotonic() + 4
        while not predicate():
            assert time.monotonic() < deadline, panel.status.text()
            app.processEvents()
            time.sleep(0.01)

    def wait(status):
        until(lambda: status in panel.status.text())

    timer.timeout.connect(review)
    timer.start()
    try:
        panel.hosted_button.click()
        wait("cancelled")
        assert len(previews) == 1
        assert not calls
        approved = True
        panel.hosted_button.click()
        wait("approved request")
        assert len(previews) == 2
        assert calls == [("Claude", previews[1])]
        assert "private skill guidance" not in previews[1]
        assert "The import budget is 256 MiB." in previews[1]
        evidence = panel.evidence.toPlainText()
        assert "C:/fixture/spec.md\nLine 1" in evidence
        catalogue.files["unrelated"] = "Unrelated index revision"
        panel.refresh_visible()
        until(lambda: not panel._refreshing)
        assert panel.evidence.toPlainText() == evidence
        catalogue.files["spec"] = "The source changed"
        panel.refresh_visible()
        wait("The library changed")
        assert not panel.evidence.toPlainText()
        assert not panel.answer.toPlainText()
    finally:
        timer.stop()
        panel.close()


def test_real_index_exposes_commit_file_decision_document_chain(tmp_path):
    git = shutil.which("git")
    if not git:
        pytest.skip("Git is required for the local evidence fixture.")
    project = tmp_path / "project"
    project.mkdir()

    def command(*arguments):
        return (
            subprocess.run(
                [git, "-C", str(project), *arguments],
                check=True,
                capture_output=True,
            )
            .stdout.decode()
            .strip()
        )

    command("init")
    changed = project / "parser.md"
    changed.write_text(
        "Unrelated setup line.\n" * 48 + "Crocodile parsing. See [decision](decision.md).\n",
        encoding="utf-8",
    )
    (project / "decision.md").write_text(
        "Parse in bounded chunks. See the [measurement](spec.md).\n", encoding="utf-8"
    )
    spec = project / "spec.md"
    spec.write_text("Peak RAM is 256 MiB.\n", encoding="utf-8")
    command("add", "parser.md")
    command(
        "-c",
        "user.email=239468521+diogommpires-glitch@users.noreply.github.com",
        "-c",
        "user.name=Fixture",
        "commit",
        "-m",
        "Crocodile parsing design",
    )
    commit = command("rev-parse", "HEAD")
    database = tmp_path / "index.sqlite3"
    writer = Indexer(database)
    try:
        writer.reconcile([project])
    finally:
        writer.close()
    catalogue = Catalogue(database)
    session = EvidenceSession(catalogue, "Crocodile parsing", Context(workspace=project))
    session.fill()
    session.verify()
    result = session.describe()
    sources = {item["path"]: item for item in result["sources"]}
    assert sources[f"{project}#{commit}"]["locator"] == {"kind": "commit", "commit": commit}
    assert sources[str(changed)]["locator"] == {"kind": "lines", "start": 49, "end": 49}
    assert [link["kind"] for link in result["connections"]] == ["changed_file"]
    assert [link["kind"] for link in sources[str(spec)]["relationship_path"]] == [
        "reference",
        "reference",
    ]
    assert "Peak RAM is 256 MiB." in sources[str(spec)]["excerpt"]
    spec.unlink()
    with pytest.raises(AskError, match="became unavailable"):
        session.verify()
