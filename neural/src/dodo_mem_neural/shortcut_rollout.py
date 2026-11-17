"""Stage two shortcuts and preserve conflicting edits during install or rollback."""

import argparse
import ctypes
import hashlib
import json
import os
import subprocess
from contextlib import ExitStack, contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .artifacts import atomic_json, sha256
from .launcher import REPO, STATE

SHORTCUT_TARGETS = (
    Path.home() / "Desktop/Dodo Mem.lnk",
    Path(os.environ["APPDATA"])
    / "Microsoft/Windows/Start Menu/Programs/Startup/Dodo Mem Neural.lnk",
)


def _canonical(path):
    path = Path(path).absolute()
    parent = path if path.exists() else path.parent
    if parent.resolve(strict=True) != parent:
        raise ValueError(f"Shortcut path contains a redirect: {path.name}")
    return path


@contextmanager
def _exclusive(target, *, create=False):
    import msvcrt

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = [
        ctypes.c_wchar_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_ulong,
        ctypes.c_void_p,
    ]
    kernel.CreateFileW.restype = ctypes.c_void_p
    handle = kernel.CreateFileW(
        str(target), 0xC0010000, 0, None, 1 if create else 3, 0x00200000, None
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    descriptor = msvcrt.open_osfhandle(handle, os.O_RDWR | os.O_BINARY)
    with os.fdopen(descriptor, "r+b", buffering=0) as stream:
        yield stream


def _entries(path):
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = document["entries"]
    targets = set()
    for entry in entries:
        target = _canonical(entry["target"])
        if target not in SHORTCUT_TARGETS or target in targets:
            raise ValueError("Shortcut manifest contains an unexpected or repeated target.")
        targets.add(target)
        for name in ("staged", "backup"):
            if entry.get(name):
                source = _canonical(entry[name])
                if not source.is_relative_to(Path(path).absolute().parent):
                    raise ValueError("Shortcut payload escapes its staged directory.")
    return document, entries


def _rewrite(stream, payload):
    stream.seek(0)
    stream.write(payload)
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def install(path):
    document, entries = _entries(path)
    with ExitStack() as stack:
        changes = []
        for entry in entries:
            if sha256(entry["staged"]) != entry["after"]:
                raise ValueError("Staged shortcut payload changed.")
            payload = Path(entry["staged"]).read_bytes()
            target = Path(entry["target"])
            stream = stack.enter_context(_exclusive(target, create=entry["before"] is None))
            current = stream.read(1024 * 1024 + 1)
            if (
                entry["before"] is not None
                and hashlib.sha256(current).hexdigest() != entry["before"]
            ):
                raise ValueError(
                    "Existing shortcut changed after staging; nothing was overwritten."
                )
            changes.append((entry, stream, payload))
        atomic_json(
            Path(path).with_name("shortcut-journal.json"),
            {"state": "installing", "entries": entries},
        )
        for entry, stream, payload in changes:
            _rewrite(stream, payload)
            stream.seek(0)
            if hashlib.sha256(stream.read()).hexdigest() != entry["after"]:
                raise OSError("Installed shortcut checksum mismatch; retain its backup.")
    atomic_json(path, {**document, "state": "installed"})
    atomic_json(Path(path).with_name("shortcut-journal.json"), {"state": "installed"})


def rollback(path):
    document, entries = _entries(path)
    with ExitStack() as stack:
        changes = []
        for entry in entries:
            stream = stack.enter_context(_exclusive(Path(entry["target"])))
            current = hashlib.sha256(stream.read(1024 * 1024 + 1)).hexdigest()
            if current == entry["before"]:
                continue
            if current != entry["after"]:
                raise ValueError(
                    "Installed shortcut changed; rollback preserves the unexpected content."
                )
            payload = Path(entry["backup"]).read_bytes() if entry["backup"] else None
            if payload is not None and hashlib.sha256(payload).hexdigest() != entry["before"]:
                raise ValueError("Shortcut rollback backup changed.")
            changes.append((stream, payload))
        for stream, payload in changes:
            if payload is not None:
                _rewrite(stream, payload)
            else:
                import msvcrt

                deleted = ctypes.c_int(1)
                if not ctypes.windll.kernel32.SetFileInformationByHandle(
                    ctypes.c_void_p(msvcrt.get_osfhandle(stream.fileno())),
                    4,
                    ctypes.byref(deleted),
                    ctypes.sizeof(deleted),
                ):
                    raise ctypes.WinError()
    atomic_json(path, {**document, "state": "rolled_back"})


def stage():
    directory = STATE / "rollout" / datetime.now(UTC).strftime("shortcuts-%Y%m%dT%H%M%SZ")
    directory.mkdir(parents=True, exist_ok=False)
    entries = []
    quote = lambda value: "'" + str(value).replace("'", "''") + "'"
    for index, target in enumerate(SHORTCUT_TARGETS):
        _canonical(target)
        backup = directory / f"original-{index}.lnk" if target.exists() else None
        if backup:
            backup.write_bytes(target.read_bytes())
        candidate = directory / f"candidate-{index}.lnk"
        arguments = '"' + str(REPO / "scripts/Start-Dodo-Mem-Neural-Hidden.vbs") + '"'
        if index == 0:
            arguments += " --desktop"
        script = (
            "$shell=New-Object -ComObject WScript.Shell;"
            f"$s=$shell.CreateShortcut({quote(candidate)});"
            f"$s.TargetPath={quote(Path(os.environ['SYSTEMROOT']) / 'System32/wscript.exe')};"
            f"$s.Arguments={quote(arguments)};$s.WorkingDirectory={quote(REPO)};"
            "$s.WindowStyle=7;"
            + (
                f"$old=$shell.CreateShortcut({quote(target)});$s.IconLocation=$old.IconLocation;"
                if target.exists()
                else ""
            )
            + "$s.Save()"
        )
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            check=True,
            creationflags=subprocess.CREATE_NO_WINDOW,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        entries.append(
            {
                "target": str(target),
                "staged": str(candidate),
                "backup": str(backup) if backup else None,
                "before": sha256(target) if target.exists() else None,
                "after": sha256(candidate),
            }
        )
    path = directory / "manifest.json"
    atomic_json(path, {"schema": 1, "state": "staged", "entries": entries})
    atomic_json(STATE / "shortcut-stage.json", {"manifest": str(path), "sha256": sha256(path)})
    return path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("stage", "activate", "rollback"))
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    if args.action == "stage":
        print(stage())
        return
    if args.manifest is None or not args.manifest.absolute().is_relative_to(STATE / "rollout"):
        raise ValueError("Use the exact staged manifest under the local rollout directory.")
    if args.action == "activate":
        if json.loads((STATE / "selection.json").read_text())["activated"] is not True:
            raise ValueError("Neural release acceptance is not activated.")
        install(args.manifest)
    else:
        rollback(args.manifest)


if __name__ == "__main__":
    main()
