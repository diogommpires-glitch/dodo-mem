"""Contain a known WSL child separately from its owned Windows launcher."""

import json
import os
import secrets
import subprocess
import sys
import threading
from contextlib import contextmanager, suppress
from dataclasses import dataclass
from pathlib import Path

from .parsing import _owned_process

GUARDIAN = r"""
import hashlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

cfg = json.loads(sys.argv[1])
lease, heartbeat, cancel = (Path(cfg[name]) for name in ('lease', 'heartbeat', 'cancel'))
environment = dict(os.environ)
environment['DODO_OWNED_INDEX_NONCE'] = cfg['nonce']
process = record = None
code = 1
interrupted = False

def publish_lease():
    descriptor, temporary = tempfile.mkstemp(prefix='lease.', dir=lease.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            json.dump(record, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, lease)
    finally:
        Path(temporary).unlink(missing_ok=True)

def interrupt(*args):
    # Raising here can interrupt Popen's return before ownership metadata exists,
    # or interrupt the cleanup itself. Observe cancellation at safe checkpoints.
    global interrupted
    interrupted = True

def members():
    start_ticks = record['start_ticks'] if record is not None else 0
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) != process.pid or fields[0] == 'Z':
                continue
            marker = b'DODO_OWNED_INDEX_NONCE=' + cfg['nonce'].encode()
            try:
                owned = marker in (entry / 'environ').read_bytes().split(b'\x00')
            except PermissionError:
                owned = False
            if not owned:
                # Exiting members can become zombies after the live stat read.
                current = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
                if current[0] == 'Z':
                    continue
                raise ValueError('Unknown process in the owned Linux group')
            if int(fields[19]) < start_ticks:
                raise ValueError('Linux process creation predates ownership')
            found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found

for event in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(event, interrupt)
try:
    if interrupted or cancel.exists():
        raise SystemExit(0)
    process = subprocess.Popen(cfg['argv'], env=environment, start_new_session=True)
    # A new session gives the created child its own group. The nonce proves
    # membership even when metadata I/O fails before the lease is published.
    record = {
        'pid': process.pid, 'pgid': process.pid, 'start_ticks': 0,
        'nonce_sha256': hashlib.sha256(cfg['nonce'].encode()).hexdigest(),
        'job': cfg['job'],
    }
    if interrupted or cancel.exists():
        raise TimeoutError('Owner cancelled the Linux process')
    fields = Path(f'/proc/{process.pid}/stat').read_text().rsplit(') ', 1)[1].split()
    if int(fields[2]) != process.pid:
        raise ValueError('Created Linux child is not its own group leader')
    record['start_ticks'] = int(fields[19])
    publish_lease()
    lease.chmod(0o600)
    last, changed = heartbeat.stat().st_mtime_ns, time.monotonic()
    while process.poll() is None:
        if interrupted or cancel.exists():
            raise TimeoutError('Owner cancelled the Linux process')
        current = heartbeat.stat().st_mtime_ns
        if current != last:
            last, changed = current, time.monotonic()
        if time.monotonic() - changed > 10:
            raise TimeoutError('Owner heartbeat expired')
        time.sleep(.1)
    code = process.returncode
finally:
    if process is not None:
        if members():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            deadline = time.monotonic() + 5
            while members() and time.monotonic() < deadline:
                time.sleep(.05)
            if members():
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
        process.wait(timeout=5)
        if members():
            raise RuntimeError('Owned Linux group did not exit')
        # A failed diagnostic write must happen after containment, never before.
        if record is not None:
            record.update(exit_code=process.returncode, guardian_finished=True)
            if lease.parent.is_dir():
                publish_lease()
raise SystemExit(code)
"""

STOP_GROUP = r"""
import hashlib
import json
import os
import signal
import sys
import time
from pathlib import Path

cfg = json.loads(sys.argv[1])
lease = Path(cfg['lease'])
if not lease.exists():
    print(json.dumps({'owned_group_exists': False, 'cleanup': 'no lease'}))
    raise SystemExit(0)
record = json.loads(lease.read_text(encoding='utf-8'))
if (record['nonce_sha256'] != hashlib.sha256(cfg['nonce'].encode()).hexdigest()
        or record['job'] != cfg['job']):
    raise ValueError('Linux ownership lease differs')

def members():
    found = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
            if int(fields[2]) != record['pgid'] or fields[0] == 'Z':
                continue
            marker = b'DODO_OWNED_INDEX_NONCE=' + cfg['nonce'].encode()
            try:
                owned = marker in (entry / 'environ').read_bytes().split(b'\x00')
            except PermissionError:
                owned = False
            if not owned:
                current = (entry / 'stat').read_text().rsplit(') ', 1)[1].split()
                if current[0] == 'Z':
                    continue
                raise ValueError('Unknown process in the owned Linux group')
            if int(fields[19]) < record['start_ticks']:
                raise ValueError('Linux process creation predates ownership')
            found.append(int(entry.name))
        except (FileNotFoundError, ProcessLookupError):
            continue
    return found

before = members()
if before:
    try:
        os.killpg(record['pgid'], signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + 5
    while members() and time.monotonic() < deadline:
        time.sleep(.05)
    if members():
        try:
            os.killpg(record['pgid'], signal.SIGKILL)
        except ProcessLookupError:
            pass
        deadline = time.monotonic() + 1
        while members() and time.monotonic() < deadline:
            time.sleep(.05)
if members():
    raise RuntimeError('Owned Linux group did not exit')
print(json.dumps({
    'owned_group_exists': False, 'verified_owned_members': len(before),
    'cleanup': 'stopped' if before else 'already exited',
}))
"""

HOST_LAUNCH = r"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

gate = Path(os.environ['DODO_DOCLING_START_GATE'])
deadline = time.monotonic() + 10
while not gate.exists():
    if time.monotonic() > deadline:
        raise TimeoutError('Windows ownership start gate did not open')
    time.sleep(.01)
mode = sys.argv[2]
if mode not in {'inherit', 'null'}:
    raise ValueError('Unknown owned WSL standard-input mode')
raise SystemExit(subprocess.call(json.loads(sys.argv[1]),
    stdin=subprocess.DEVNULL if mode == 'null' else None))
"""


@dataclass
class OwnedWslProcess:
    process: subprocess.Popen
    runtime: Path
    cleanup: dict | None = None


def native_environment(environment):
    result = dict(environment)
    if os.name == "nt":
        for key, value in {
            "ProgramFiles": "C:/Program Files",
            "ProgramW6432": "C:/Program Files",
            "SystemRoot": "C:/Windows",
            "WINDIR": "C:/Windows",
            "OS": "Windows_NT",
            "PROCESSOR_ARCHITECTURE": "AMD64",
            "COMSPEC": "C:/Windows/System32/cmd.exe",
            "PATHEXT": ".COM;.EXE;.BAT;.CMD",
        }.items():
            result.setdefault(key, value)
    return result


def linux_path(path):
    value = Path(path).absolute().as_posix()
    if len(value) < 3 or value[1:3] != ":/":
        raise ValueError("WSL process files require an absolute Windows drive path")
    return "/mnt/" + value[0].lower() + value[2:]


def _write(path, data):
    with path.open("xb") as stream:
        stream.write(data)
    path.chmod(0o600)


def _stop(distribution, cfg, environment):
    result = subprocess.run(
        ["wsl.exe", "-d", distribution, "--exec", "python3", "-c", STOP_GROUP, json.dumps(cfg)],
        env=environment,
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)


@contextmanager
def owned_wsl_process(argv, *, distribution, directory, log, environment=None, stdin=None):
    """Own both kernels' children; never terminate a PID without its Linux lease."""
    if os.name != "nt":
        raise RuntimeError(
            "This bridge launcher requires Windows with an existing WSL distribution"
        )
    if stdin is not None and (type(stdin) is not int or stdin != subprocess.DEVNULL):
        raise ValueError("Owned WSL standard input must be inherited or subprocess.DEVNULL")
    if (
        not isinstance(distribution, str)
        or not distribution
        or any(character in distribution for character in "\0\r\n")
    ):
        raise ValueError("An explicit existing WSL distribution is required")
    if (
        not isinstance(argv, (list, tuple))
        or not argv
        or any(not isinstance(value, str) or "\0" in value for value in argv)
    ):
        raise ValueError("A structured non-empty native argument vector is required")
    directory = Path(directory).absolute()
    if directory.resolve() != directory or any(
        path.is_symlink() or getattr(path, "is_junction", lambda: False)()
        for path in (directory, *directory.parents)
    ):
        raise PermissionError("Owned process files cannot follow directory links")
    runtime = directory / ("owned-wsl-" + secrets.token_hex(16))
    runtime.mkdir(mode=0o700)
    guardian, lease = runtime / "guardian.py", runtime / "lease.json"
    heartbeat, cancel, gate = (runtime / name for name in ("heartbeat", "cancel", "windows-gate"))
    _write(guardian, GUARDIAN.encode("utf-8"))
    _write(heartbeat, b"owner active")
    cfg = {
        "argv": list(argv),
        "nonce": secrets.token_hex(32),
        "job": runtime.name,
        "lease": linux_path(lease),
        "heartbeat": linux_path(heartbeat),
        "cancel": linux_path(cancel),
    }
    _write(runtime / "control.json", json.dumps(cfg).encode("utf-8"))
    environment = native_environment(os.environ if environment is None else environment)
    command = [
        "wsl.exe",
        "-d",
        distribution,
        "--exec",
        "python3",
        linux_path(guardian),
        json.dumps(cfg),
    ]
    stop = threading.Event()
    heartbeat_errors = []

    def maintain_heartbeat():
        while not stop.wait(0.1):
            try:
                heartbeat.touch()
            except OSError as error:
                heartbeat_errors.append(error)
                return

    thread = threading.Thread(target=maintain_heartbeat, daemon=True)
    thread.start()
    owned = None
    try:
        with _owned_process(
            [
                sys._base_executable,
                "-I",
                "-S",
                "-c",
                HOST_LAUNCH,
                json.dumps(command),
                "null" if stdin == subprocess.DEVNULL else "inherit",
            ],
            environment,
            log,
            gate,
        ) as process:
            owned = OwnedWslProcess(process, runtime)
            try:
                yield owned
            finally:
                # Cancellation prevents a slow WSL startup from creating a new
                # Linux child after the host has already given up on its lease.
                _write(cancel, b"owner cancelled")
                owned.cleanup = _stop(distribution, cfg, environment)
                # Closing the owned Windows job is the final host fence.
                with suppress(subprocess.TimeoutExpired):
                    process.wait(timeout=3)
    finally:
        stop.set()
        thread.join(timeout=2)
        gate.unlink(missing_ok=True)
        if owned is not None:
            owned.cleanup = _stop(distribution, cfg, environment)
            _write(runtime / "cleanup.json", json.dumps(owned.cleanup).encode("utf-8"))
        if heartbeat_errors:
            raise RuntimeError("Owned Linux process heartbeat failed") from heartbeat_errors[0]
