"""Own one hidden service process without claiming unrelated listeners or reused PIDs."""

import argparse
import asyncio
import contextlib
import ctypes
import json
import os
import subprocess
import time
from pathlib import Path
from uuid import uuid4

import httpx
import psutil

from . import task_runtime
from .artifacts import atomic_json

REPO = Path(__file__).resolve().parents[2]
from dodo_memory_desktop.configuration import runtime_path

STATE = runtime_path("neural")
PYTHONW = REPO / ".venv/Scripts/pythonw.exe"
COMMAND = [str(PYTHONW), "-m", "dodo_mem_neural.cli"]
PORTS = {8770, 8771}
BROKER_URL = "http://127.0.0.1:8770"


class LaunchError(RuntimeError):
    pass


@contextlib.contextmanager
def launch_lock(home):
    import msvcrt

    home.mkdir(parents=True, exist_ok=True)
    with (home / "launcher.lock").open("a+b") as stream:
        stream.seek(0, 2)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        deadline = time.monotonic() + 15
        while True:
            stream.seek(0)
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError:
                if time.monotonic() >= deadline:
                    raise LaunchError("Another neural launcher is still working.") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)


def service_process(pid, *, created=None):
    try:
        process = psutil.Process(pid)
        if created is not None and process.create_time() != created:
            raise LaunchError("Saved neural process identity changed; nothing was stopped.")
        configuration = PYTHONW.parent.parent / "pyvenv.cfg"
        values = dict(
            line.split(" = ", 1) for line in configuration.read_text().splitlines() if " = " in line
        )
        base = Path(values["home"]) / "pythonw.exe"
        approved = {PYTHONW.resolve(), base.resolve()}
        argv0 = {os.path.normcase(os.path.abspath(path)) for path in (PYTHONW, base)}
        command = process.cmdline()
        if (
            Path(process.exe()).resolve() not in approved
            or not command
            or os.path.normcase(os.path.abspath(command[0])) not in argv0
            or command[1:] != COMMAND[1:]
            or Path(process.cwd()).resolve() != REPO.resolve()
            or process.username() != psutil.Process().username()
        ):
            raise LaunchError("Listener belongs to another process; nothing was stopped.")
    except psutil.NoSuchProcess:
        return None
    return process


def listening_service():
    listeners = [
        item
        for item in psutil.net_connections(kind="tcp")
        if item.laddr.port in PORTS
        and (
            item.status == psutil.CONN_LISTEN
            # Windows can expose accepting loopback sockets as BOUND (psutil NONE).
            or (os.name == "nt" and item.status == psutil.CONN_NONE and not item.raddr)
        )
    ]
    if not listeners:
        return None
    if (
        len(listeners) != 2
        or {item.laddr.port for item in listeners} != PORTS
        or any(item.laddr.ip != "127.0.0.1" for item in listeners)
        or len({item.pid for item in listeners}) != 1
        or listeners[0].pid is None
    ):
        raise LaunchError("Neural ports have unexpected owners; nothing was stopped.")
    process = service_process(listeners[0].pid)
    if process is None:
        raise LaunchError("Neural process exited during inspection.")
    return process


def readiness(home=STATE):
    token_path = home / "shared.token"
    if token_path.resolve(strict=True) != token_path.absolute():
        raise LaunchError("Neural readiness credential contains a redirect.")
    with token_path.open("rb") as stream:
        raw_token = stream.read(4097)
    token = raw_token.decode("ascii").strip()
    if not 32 <= len(token) <= 4096 or any(not 33 <= ord(char) <= 126 for char in token):
        raise LaunchError("Neural readiness credential is invalid.")
    with (
        httpx.Client(trust_env=False, follow_redirects=False, timeout=2) as client,
        client.stream(
            "GET", BROKER_URL + "/status", headers={"Authorization": "Bearer " + token}
        ) as response,
    ):
        if response.status_code != 200:
            raise LaunchError("Neural shared status authentication failed.")
        body = bytearray()
        for part in response.iter_bytes():
            body.extend(part)
            if len(body) > 16384:
                raise LaunchError("Neural readiness response is invalid or oversized.")
    status = json.loads(body)
    if not isinstance(status, dict):
        raise LaunchError("Neural readiness response is invalid.")
    if status.get("scope") != "shared" or status.get("client") != "dodo":
        raise LaunchError("Neural readiness returned an unexpected credential scope.")
    if status.get("ready") is not True:
        raise LaunchError("Neural service has not reported readiness.")
    return {key: status.get(key) for key in ("ready", "state", "retrieval", "device", "model")}


def record_process(process, home, *, ticket=None):
    record = {
        "pid": process.pid,
        "created": process.create_time(),
        "executable": str(PYTHONW),
        "command": COMMAND,
    }
    if ticket is not None:
        record.update(launch_owner=task_runtime.TASK_NAME, **ticket)
    atomic_json(home / "service-process.json", record)


def start(home=STATE):
    selection = json.loads((home / "selection.json").read_text(encoding="utf-8"))
    if selection.get("activated") is not True:
        raise LaunchError("Neural startup is staged until acceptance is recorded.")
    with launch_lock(home):
        process = listening_service()
        if process is not None:
            saved = home / "service-process.json"
            identity = json.loads(saved.read_text(encoding="utf-8")) if saved.exists() else None
            if identity is not None and identity.get("launch_owner") != task_runtime.TASK_NAME:
                raise LaunchError("Existing neural service has another launch owner.")
            if identity is None or (identity.get("pid"), identity.get("created")) != (
                process.pid,
                process.create_time(),
            ):
                binding = task_runtime.verify_registered(home, COMMAND, REPO)
                pending = task_runtime._read(home / "task-start-intent.json")
                if (
                    binding["_state"] != 4
                    or pending.get("task_xml_sha256") != binding["xml_sha256"]
                    or process.create_time() < pending["requested_at"] - 2
                ):
                    raise LaunchError("Existing neural service has no matching owned task intent.")
                record_process(process, home, ticket=pending)
            identity = json.loads(saved.read_text(encoding="utf-8"))
            if identity.get("launch_owner") != task_runtime.TASK_NAME or (
                identity.get("pid"),
                identity.get("created"),
            ) != (process.pid, process.create_time()):
                raise LaunchError("Existing neural service has another launch owner.")
            binding = task_runtime.verify_registered(home, COMMAND, REPO)
            if identity.get("task_xml_sha256") != binding["xml_sha256"]:
                raise LaunchError("Neural service task identity changed.")
            ticket = {
                key: identity[key]
                for key in (
                    "requested_at",
                    "task_xml_sha256",
                    "task_instance_guid",
                    "task_engine_pid",
                )
                if key in identity
            }
        else:
            # The OS owns this task; a child spawned here would inherit Codex's Windows job.
            ticket = task_runtime.request_start(home, COMMAND, REPO)
        deadline = time.monotonic() + 90
        while time.monotonic() < deadline:
            try:
                status = readiness(home)
                owner = listening_service()
                if owner is None or owner.create_time() < ticket["requested_at"] - 2:
                    raise LaunchError("Neural listeners changed during readiness.")
                if process is not None and owner.pid != process.pid:
                    raise LaunchError("Neural listener owner changed during reuse.")
                task_runtime.verify_registered(home, COMMAND, REPO)
                record_process(owner, home, ticket=ticket)
                atomic_json(home / "launcher-status.json", {"outcome": "ready", **status})
                return status
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.ReadTimeout):
                time.sleep(0.25)
        raise LaunchError("Neural readiness timed out; inspect the service status before retrying.")


def stop(home=STATE):
    with launch_lock(home):
        saved = home / "service-process.json"
        if not saved.exists():
            raise LaunchError("No launcher-owned process identity exists; nothing was stopped.")
        identity = json.loads(saved.read_text(encoding="utf-8"))
        if identity.get("executable") != str(PYTHONW) or identity.get("command") != COMMAND:
            raise LaunchError("Saved neural launch command changed; nothing was stopped.")
        process = service_process(identity["pid"], created=identity["created"])
        if process is not None:
            request = home / "shutdown.request.json"
            value = read_shutdown_request(request) if request.exists() else None
            if value is None or (value["pid"], value["created"]) != (
                identity["pid"],
                identity["created"],
            ):
                value = {
                    "pid": identity["pid"],
                    "created": identity["created"],
                    "nonce": uuid4().hex,
                }
                atomic_json(request, value)
            try:
                process.wait(timeout=30)
            except psutil.TimeoutExpired as error:
                raise LaunchError(
                    "Neural checkpoint is still pending; process remains running."
                ) from error
            status = read_shutdown_request(home / "shutdown-status.json", status=True)
            if (
                status is None
                or any(status.get(key) != value[key] for key in value)
                or (status.get("outcome") != "checkpointed")
            ):
                raise LaunchError("Neural process exited without a checkpoint acknowledgement.")
        saved.unlink()
        atomic_json(home / "launcher-status.json", {"outcome": "stopped"})


def read_shutdown_request(path, *, status=False):
    if not path.exists() or path.resolve(strict=True) != path.absolute():
        return None
    with path.open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        return None
    try:
        value = json.loads(raw)
    except (ValueError, UnicodeDecodeError):
        return None
    required = {"pid", "created", "nonce"}
    if not isinstance(value, dict) or not required <= value.keys():
        return None
    if not status and value.keys() != required:
        return None
    if (
        type(value["pid"]) is not int
        or type(value["created"]) not in {int, float}
        or not isinstance(value["nonce"], str)
        or len(value["nonce"]) != 32
        or any(char not in "0123456789abcdef" for char in value["nonce"])
    ):
        return None
    return value


async def watch_shutdown(home, service, servers):
    process = psutil.Process()
    while True:
        request = read_shutdown_request(home / "shutdown.request.json")
        if request and (request["pid"], request["created"]) == (process.pid, process.create_time()):
            service.stop_event.set()
            worker = service.worker_thread
            if worker is not None:
                await asyncio.to_thread(worker.join, 20)
                if worker.is_alive():
                    atomic_json(
                        home / "shutdown-status.json",
                        {
                            **request,
                            "outcome": "blocked",
                            "reason": "Waiting for the active index operation to checkpoint.",
                        },
                    )
                    while worker.is_alive():
                        await asyncio.sleep(0.2)
            atomic_json(home / "shutdown-status.json", {**request, "outcome": "checkpointed"})
            for server in servers:
                server.should_exit = True
            return
        await asyncio.sleep(0.1)


def open_desktop():
    desktop_repo = REPO.parent / "dodo-memory-desktop"
    executable = desktop_repo / ".venv/Scripts/dodo-memory-desktop.exe"
    gateway = Path(os.environ["APPDATA"]) / (
        "Microsoft/Windows/Start Menu/Programs/Startup/Dodo Mem Gateway.lnk"
    )
    if gateway.exists():
        os.startfile(str(gateway))
    for process in psutil.process_iter(["exe", "cmdline"]):
        if process.info["exe"] and Path(process.info["exe"]).resolve() == executable.resolve():
            user32 = ctypes.windll.user32
            callback_type = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)

            @callback_type
            def show_window(window, _, user32=user32, process_id=process.pid):
                owner = ctypes.c_ulong()
                user32.GetWindowThreadProcessId(window, ctypes.byref(owner))
                if owner.value == process_id and user32.IsWindowVisible(window):
                    user32.ShowWindow(window, 9)
                    user32.SetForegroundWindow(window)
                    return False
                return True

            user32.EnumWindows(show_window, 0)
            return
    subprocess.Popen(
        [str(executable), "--whole-library"],
        cwd=desktop_repo,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--desktop", action="store_true")
    args = parser.parse_args()
    try:
        if args.stop:
            stop()
        elif args.check:
            process = listening_service()
            if process is None:
                raise LaunchError("Neural service is not listening.")
            atomic_json(STATE / "launcher-status.json", {"outcome": "ready", **readiness()})
        else:
            start()
    except (
        RuntimeError,
        OSError,
        psutil.Error,
        httpx.HTTPError,
        ValueError,
        KeyError,
        TypeError,
    ) as error:
        # Status contains no request bodies, credential values, or source excerpts.
        message = str(error) if isinstance(error, LaunchError) else type(error).__name__
        atomic_json(STATE / "launcher-status.json", {"outcome": "failed", "reason": message})
        return 1
    finally:
        if args.desktop:
            open_desktop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
