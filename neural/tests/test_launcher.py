import asyncio
import json
import socket
import subprocess
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import httpx
import psutil
import pytest

from dodo_mem_neural import launcher
from dodo_mem_neural.artifacts import atomic_json


@pytest.mark.parametrize(
    "changed", [None, "binary", "argv0", "argv0_alias", "arguments", "cwd", "user", "created"]
)
def test_listener_identity_only_accepts_approved_interpreters_and_exact_arguments(
    tmp_path, monkeypatch, changed
):
    values = dict(
        line.split(" = ", 1)
        for line in (launcher.PYTHONW.parent.parent / "pyvenv.cfg").read_text().splitlines()
        if " = " in line
    )
    base = Path(values["home"]) / "pythonw.exe"
    executable = tmp_path / "foreign/pythonw.exe" if changed == "binary" else base
    argv0 = tmp_path / "foreign/pythonw.exe" if changed == "argv0" else base
    if changed == "argv0_alias":
        argv0 = tmp_path / "unlisted-alias.exe"
        try:
            argv0.symlink_to(base)
        except OSError as error:
            pytest.skip(f"File symlinks unavailable: {error}")
    arguments = (
        [*launcher.COMMAND[1:], "--foreign"] if changed == "arguments" else launcher.COMMAND[1:]
    )
    process = SimpleNamespace(
        exe=lambda: str(executable),
        cmdline=lambda: [str(argv0), *arguments],
        cwd=lambda: str(tmp_path if changed == "cwd" else launcher.REPO),
        username=lambda: "foreign" if changed == "user" else "synthetic-owner",
        create_time=lambda: 9 if changed == "created" else 10,
    )
    monkeypatch.setattr(
        launcher.psutil,
        "Process",
        lambda pid=None: (
            process if pid is not None else SimpleNamespace(username=lambda: "synthetic-owner")
        ),
    )
    if changed is None:
        assert launcher.service_process(123, created=10) is process
    else:
        with pytest.raises(launcher.LaunchError, match="nothing was stopped"):
            launcher.service_process(123, created=10)


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def isolated_launcher(tmp_path, monkeypatch):
    ports = {free_port(), free_port()}
    script = tmp_path / "service.py"
    script.write_text(
        "import http.server,json,threading,time\n"
        "class Handler(http.server.BaseHTTPRequestHandler):\n"
        " def do_GET(self):\n"
        "  self.send_response(200);self.end_headers();"
        "self.wfile.write(json.dumps({'ready':True,'client':'dodo','scope':'shared'}).encode())\n"
        " def log_message(self,*args):pass\n"
        f"for port in {sorted(ports)!r}:\n"
        " server=http.server.ThreadingHTTPServer(('127.0.0.1',port),Handler)\n"
        " threading.Thread(target=server.serve_forever,daemon=True).start()\n"
        "import asyncio\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "from dodo_mem_neural.launcher import watch_shutdown\n"
        "service=SimpleNamespace(stop_event=threading.Event(),worker_thread=None)\n"
        "asyncio.run(watch_shutdown(Path(__file__).parent/'state',service,[]))\n",
        encoding="utf-8",
    )
    home = tmp_path / "state"
    home.mkdir()
    (home / "selection.json").write_text('{"activated":true}')
    (home / "shared.token").write_text("synthetic-readiness-token-for-fixture")
    monkeypatch.setattr(launcher, "PORTS", ports)
    monkeypatch.setattr(launcher, "BROKER_URL", f"http://127.0.0.1:{min(ports)}", raising=False)
    monkeypatch.setattr(launcher, "COMMAND", [str(launcher.PYTHONW), str(script)])
    children = []
    popen = subprocess.Popen

    def own_child(*args, **kwargs):
        child = popen(*args, **kwargs)
        children.append((child, psutil.Process(child.pid).create_time()))
        return child

    monkeypatch.setattr(subprocess, "Popen", own_child)

    def scheduled_start(state, command, cwd):
        requested = time.time()
        ticket = {"requested_at": requested, "task_xml_sha256": "synthetic-task-definition"}
        atomic_json(state / "task-start-intent.json", ticket)
        subprocess.Popen(
            command,
            cwd=cwd,
            creationflags=subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS,
        )
        return ticket

    monkeypatch.setattr(launcher.task_runtime, "request_start", scheduled_start)
    monkeypatch.setattr(
        launcher.task_runtime,
        "verify_registered",
        lambda *args: {"xml_sha256": "synthetic-task-definition", "_state": 4},
    )
    try:
        yield home
    finally:
        for child, created in children:
            try:
                parent = psutil.Process(child.pid)
                if parent.create_time() != created:
                    continue
                for owned in reversed([parent, *parent.children(recursive=True)]):
                    if owned.is_running():
                        owned.terminate()
                child.wait(timeout=5)
            except psutil.NoSuchProcess:
                pass


@pytest.mark.skipif(not hasattr(subprocess, "CREATE_NO_WINDOW"), reason="native Windows launcher")
def test_hidden_start_adopts_only_its_redirected_listener(isolated_launcher):
    home = isolated_launcher
    assert launcher.start(home)["ready"] is True
    first = json.loads((home / "service-process.json").read_text())
    assert launcher.listening_service().pid == first["pid"]
    (home / "service-process.json").unlink()
    assert launcher.start(home)["ready"] is True
    assert json.loads((home / "service-process.json").read_text())["pid"] == first["pid"]
    launcher.stop(home)
    assert not psutil.pid_exists(first["pid"])
    assert not (home / "service-process.json").exists()
    assert json.loads((home / "shutdown-status.json").read_text())["outcome"] == "checkpointed"


def test_connect_timeout_during_readiness_is_retried(isolated_launcher, monkeypatch):
    probe = launcher.readiness
    calls = 0

    def temporarily_unready(home):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise httpx.ConnectTimeout("Synthetic early readiness timeout")
        return probe(home)

    monkeypatch.setattr(launcher, "readiness", temporarily_unready)
    assert launcher.start(isolated_launcher)["ready"] is True
    assert calls >= 2
    launcher.stop(isolated_launcher)


def test_reused_pid_cannot_be_stopped(isolated_launcher):
    home = isolated_launcher
    identity = {
        "pid": psutil.Process().pid,
        "created": psutil.Process().create_time() - 1,
        "executable": str(launcher.PYTHONW),
        "command": launcher.COMMAND,
    }
    (home / "service-process.json").write_text(json.dumps(identity))
    with pytest.raises(launcher.LaunchError, match="identity changed"):
        launcher.stop(home)
    assert psutil.Process(identity["pid"]).is_running()


def test_partial_listener_never_claims_or_stops_another_process(monkeypatch):
    ports = {free_port(), free_port()}
    monkeypatch.setattr(launcher, "PORTS", ports)
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", min(ports)))
        listener.listen()
        with pytest.raises(launcher.LaunchError, match="unexpected owners"):
            launcher.listening_service()
        assert listener.getsockname()[1] == min(ports)


def test_windows_bound_endpoints_still_require_one_exact_owned_process(monkeypatch):
    process = SimpleNamespace(pid=123)
    listeners = [
        SimpleNamespace(
            status=psutil.CONN_NONE,
            laddr=SimpleNamespace(ip="127.0.0.1", port=port),
            raddr=(),
            pid=123,
        )
        for port in launcher.PORTS
    ]
    monkeypatch.setattr(launcher.psutil, "net_connections", lambda **kwargs: listeners)
    monkeypatch.setattr(launcher, "service_process", lambda pid: process if pid == 123 else None)
    assert launcher.listening_service() is process
    listeners[0].pid = 456
    with pytest.raises(launcher.LaunchError, match="unexpected owners"):
        launcher.listening_service()


def test_interrupted_start_replaces_only_a_stale_owned_identity(isolated_launcher):
    home = isolated_launcher
    atomic_json(
        home / "service-process.json",
        {"pid": 0, "created": 0, "launch_owner": launcher.task_runtime.TASK_NAME},
    )
    launcher.task_runtime.request_start(home, launcher.COMMAND, launcher.REPO)
    deadline = time.monotonic() + 5
    while launcher.listening_service() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    assert launcher.start(home)["ready"] is True
    saved = json.loads((home / "service-process.json").read_text())
    assert saved["pid"] == launcher.listening_service().pid
    launcher.stop(home)


async def test_shutdown_checkpoints_before_stopping_and_ignores_old_process(tmp_path):
    signal = threading.Event()
    checkpoint = tmp_path / "checkpoint"

    def work():
        signal.wait(5)
        checkpoint.write_text("saved")

    worker = threading.Thread(target=work)
    worker.start()
    service = SimpleNamespace(stop_event=signal, worker_thread=worker)
    server = SimpleNamespace(should_exit=False)
    process = psutil.Process()
    nonce = uuid4().hex
    atomic_json(
        tmp_path / "shutdown.request.json",
        {"pid": process.pid, "created": process.create_time() - 1, "nonce": nonce},
    )
    watcher = asyncio.create_task(launcher.watch_shutdown(tmp_path, service, [server]))
    await asyncio.sleep(0.2)
    assert not server.should_exit and not signal.is_set()
    atomic_json(
        tmp_path / "shutdown.request.json",
        {"pid": process.pid, "created": process.create_time(), "nonce": nonce},
    )
    await asyncio.wait_for(watcher, timeout=3)
    assert checkpoint.read_text() == "saved"
    assert server.should_exit
    assert json.loads((tmp_path / "shutdown-status.json").read_text())["outcome"] == "checkpointed"


def test_readiness_rejects_a_malformed_authenticated_response(tmp_path, monkeypatch):
    (tmp_path / "shared.token").write_text("synthetic-readiness-token-for-fixture")
    client = httpx.Client
    transport = httpx.MockTransport(lambda request: httpx.Response(200, json=[]))
    monkeypatch.setattr(httpx, "Client", lambda **kwargs: client(transport=transport, **kwargs))
    with pytest.raises(launcher.LaunchError, match="invalid"):
        launcher.readiness(tmp_path)
