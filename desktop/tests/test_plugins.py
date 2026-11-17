import base64
import json
import os
import socket
import zipfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest
from wasmtime import wat2wasm

from dodo_memory_desktop.plugin_sandbox import Sandbox, SandboxError
from dodo_memory_desktop.plugins import EXAMPLES, PluginError, PluginStore

WINDOWS = pytest.mark.skipif(os.name != "nt", reason="Requires a Windows AppContainer")
RELATED = EXAMPLES / "relationship-tools.dodo-pack.zip"
RESEARCH = EXAMPLES / "research-starter.dodo-pack.zip"


def rewrite(source, destination, update, extras=()):
    with zipfile.ZipFile(source) as original, zipfile.ZipFile(destination, "w") as archive:
        manifest = json.loads(original.read("manifest.json"))
        manifest.update(update)
        for item in original.infolist():
            if item.filename != "manifest.json":
                archive.writestr(item.filename, original.read(item))
        archive.writestr("manifest.json", json.dumps(manifest))
        for path, content in extras:
            archive.writestr(path, content)


@pytest.fixture
def store(tmp_path):
    plugin_store = PluginStore(tmp_path / "plugins")
    try:
        yield plugin_store
    finally:
        plugin_store.close()


@pytest.mark.parametrize("example", [RELATED, RESEARCH])
def test_explicit_approval_version_rollback_and_disable(store, example):
    earlier = example.with_name(example.name.replace(".dodo-pack.zip", "-1.0.0.dodo-pack.zip"))
    installed = store.install(earlier)
    assert not store.active()
    newest = store.install(example)
    with pytest.raises(PluginError, match="exact permissions"):
        store.enable(installed.id, installed.version, ["write:owner"])
    store.enable(installed.id, installed.version, installed.permissions)
    first_generation = store.active()[0].generation
    with pytest.raises(PluginError, match="bytes changed"):
        store.enable(
            newest.id, newest.version, newest.permissions, expected_sha256=installed.sha256
        )
    store.enable(newest.id, newest.version, newest.permissions, expected_sha256=newest.sha256)
    assert [plugin.version for plugin in store.active()] == ["1.1.0"]
    assert store.previous(installed.id).version == "1.0.0"
    store.rollback(installed.id)
    assert [plugin.version for plugin in store.active()] == ["1.0.0"]
    assert store.active()[0].generation != first_generation
    store.disable(installed.id)
    assert not store.active()


def test_pack_rejects_owner_capability_and_undeclared_path(store, tmp_path):
    malicious = tmp_path / "malicious.zip"
    rewrite(RELATED, malicious, {"permissions": ["write:owner"]})
    with pytest.raises(PluginError, match="recognised"):
        store.install(malicious)
    rewrite(RESEARCH, malicious, {}, extras=[("../secrets/token.txt", b"unsafe")])
    with pytest.raises(PluginError, match="path"):
        store.install(malicious)
    rewrite(RELATED, malicious, {"files": {"help.wasm": "0" * 64, "related.wasm": "0" * 64}})
    with pytest.raises(PluginError, match="does not match"):
        store.install(malicious)
    assert not store.versions()


def test_installed_archive_bytes_are_bound_to_approval(store, tmp_path):
    installed = store.install(RELATED)
    store.enable(installed.id, installed.version, installed.permissions)
    changed = tmp_path / "changed.zip"
    rewrite(RELATED, changed, {"name": "Changed bytes"})
    destination = next(store.root.glob("*.zip"))
    destination.write_bytes(changed.read_bytes())
    assert not store.active()
    with pytest.raises(PluginError, match="changed after approval"):
        store.run(installed.id, "instructions", {})


@WINDOWS
def test_examples_execute_under_real_appcontainer(store):
    plugin = store.install(RELATED)
    with pytest.raises(PluginError, match="Enable this plugin"):
        store.run(plugin.id, "instructions", {})
    store.enable(plugin.id, plugin.version, plugin.permissions)
    with pytest.raises(PluginError, match="selected plugin version changed"):
        store.run(plugin.id, "instructions", {}, expected_sha256="0" * 64)
    message = store.run(plugin.id, "instructions", {}, expected_sha256=plugin.sha256)
    assert "Open a note" in message["text"]
    context = {"items": [{"title": "A note", "path": "notes/example.md", "kind": "link"}]}
    panel = store.run(plugin.id, "related", context, expected_sha256=plugin.sha256)
    assert panel["rows"] == [{"label": "A note", "value": "link", "path": "notes/example.md"}]
    context["items"][0]["path"] = "private/example.md"
    with pytest.raises(PluginError, match="outside shared memory"):
        store.run(plugin.id, "related", context)


@WINDOWS
def test_disable_and_reenable_rejects_queued_invocation_before_resume(store, monkeypatch):
    installed = store.install(RELATED)
    store.enable(installed.id, installed.version, installed.permissions)
    active = store.active()[0]
    sandbox = store._sandbox
    sandbox._prepare()
    queued = Event()
    resumptions = []
    execute = sandbox.execute
    resume = sandbox._api.resume

    def record_queue(*args, **kwargs):
        queued.set()
        return execute(*args, **kwargs)

    def record_resume(thread):
        resumptions.append(thread)
        return resume(thread)

    monkeypatch.setattr(sandbox, "execute", record_queue)
    monkeypatch.setattr(sandbox._api, "resume", record_resume)
    with ThreadPoolExecutor(max_workers=1) as worker:
        with sandbox._lock:
            running = worker.submit(store.run, active.id, "instructions", {})
            assert queued.wait(3)
            other = PluginStore(store.root)
            try:
                other.disable(active.id)
                other.enable(active.id, active.version, active.permissions)
            finally:
                other.close()
        with pytest.raises(PluginError, match="disabled or changed"):
            running.result(timeout=8)
    assert resumptions == []
    with pytest.raises(PluginError, match="selected plugin approval changed"):
        store.run(active.id, "instructions", {}, expected_generation=active.generation)


@WINDOWS
def test_worker_denies_external_file_network_and_wasi(tmp_path, monkeypatch):
    monkeypatch.setenv("DODO_SANDBOX_TEST_TOKEN", "synthetic-not-a-credential")
    protected = tmp_path / "outside-runtime.txt"
    protected.write_text("fixture", encoding="utf-8")
    sandbox = Sandbox()
    try:
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            listener.listen(1)
            result = sandbox.execute(
                {"operation": "probe", "path": str(protected), "port": listener.getsockname()[1]}
            )
        assert not result["external_file_readable"]
        assert not result["loopback_connectable"]
        assert "DODO_SANDBOX_TEST_TOKEN" not in result["environment"]
        for wat in (
            '(module (import "wasi_snapshot_preview1" "fd_write" (func)))',
            (
                '(module (memory (export "memory") 1) (func (export "run")'
                " (param i32 i32) (result i64) (loop $again br $again) i64.const 0))"
            ),
        ):
            with pytest.raises(SandboxError, match="rejected"):
                sandbox.execute(
                    {
                        "operation": "run",
                        "context": {},
                        "module": base64.b64encode(wat2wasm(wat)).decode("ascii"),
                    }
                )
    finally:
        sandbox.close()
