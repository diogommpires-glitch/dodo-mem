"""Install qualified native profiles through held handles and preserve conflicting edits."""

import argparse
import hashlib
import json
import os
import re
from contextlib import ExitStack
from pathlib import Path

from .artifacts import atomic_json, sha256
from .shortcut_rollout import _canonical, _exclusive, _rewrite

CLIENTS = ("cline", "opencode", "jan")


def qualify_reports(reports):
    if set(reports) != set(CLIENTS):
        raise ValueError("Actual Cline, OpenCode and Jan qualification reports are required.")
    for client, path in reports.items():
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        if report.get("client") != client or report.get("passed") is not True:
            raise ValueError(f"Actual {client} qualification has not passed.")


def _entries(path, home, appdata):
    path = _canonical(path)
    state = home / "local-ai/dodo-mem-neural"
    if not path.is_relative_to(state / "staged"):
        raise ValueError("Use a profile manifest inside the local staged directory.")
    models = json.loads((home / "local-ai/models.json").read_text(encoding="utf-8"))
    if any(not isinstance(key, str) or not re.fullmatch(r"[a-z0-9-]+", key) for key in models):
        raise ValueError(
            "Registered model keys must contain only lowercase letters, digits and hyphens."
        )
    profiles = [home / "local-ai" / model / "cline-data" for model in models]
    profiles.extend((home / "local-ai/cline-plan-act/cline-data", home / ".cline/data"))
    allowed = {
        root / name
        for root in profiles
        for name in (
            "globalState.json",
            "secrets.json",
            "settings/models.json",
            "settings/providers.json",
        )
    }
    allowed.add(appdata / "Jan/data/settings.json")
    allowed.update(
        home / "dodo-shared-memory" / name
        for name in (
            "profiles/opencode-memory.json",
            "profiles/opencode-coder-memory.json",
            "scripts/start_opencode_memory.py",
        )
    )
    entries = json.loads(path.read_text(encoding="utf-8"))["files"]
    seen = set()
    for entry in entries:
        target = _canonical(entry["target"])
        if target not in allowed or target in seen or not entry.get("before_sha256"):
            raise ValueError(
                "Profile target is unexpected, repeated or was not present when staged."
            )
        seen.add(target)
        for field, expected in (("staged", "after_sha256"), ("backup", "before_sha256")):
            source = _canonical(entry[field])
            if not source.is_relative_to(path.parent) or sha256(source) != entry[expected]:
                raise ValueError("Staged profile or backup checksum changed.")
            if source.stat().st_size > 4 * 1024**2:
                raise ValueError("Native profile exceeds the 4 MiB installation bound.")
    return entries, state


def _install_or_restore(path, *, home, appdata, restore):
    entries, state = _entries(path, home, appdata)
    with ExitStack() as stack:
        changes = []
        for entry in entries:
            stream = stack.enter_context(_exclusive(Path(entry["target"])))
            current_bytes = stream.read(4 * 1024**2 + 1)
            if len(current_bytes) > 4 * 1024**2:
                raise ValueError("Native profile exceeds the 4 MiB installation bound.")
            current = hashlib.sha256(current_bytes).hexdigest()
            desired = entry["before_sha256"] if restore else entry["after_sha256"]
            if current == desired:
                continue
            expected = entry["after_sha256"] if restore else entry["before_sha256"]
            if current != expected:
                raise ValueError("Native profile changed; the current content was preserved.")
            payload = Path(entry["backup"] if restore else entry["staged"]).read_bytes()
            changes.append((entry, stream, payload, desired))
        journal = state / "profile-rollout.json"
        atomic_json(
            journal,
            {
                "state": "restoring" if restore else "installing",
                "manifest": str(path),
                "manifest_sha256": sha256(path),
                "entries": entries,
            },
        )
        for entry, stream, payload, desired in changes:
            _rewrite(stream, payload)
            stream.seek(0)
            if hashlib.sha256(stream.read()).hexdigest() != desired:
                raise OSError(
                    "Native profile write failed verification; retain its recovery journal."
                )
    atomic_json(
        state / "clients.active.json",
        {"enabled": not restore, "manifest": str(path), "manifest_sha256": sha256(path)},
    )
    atomic_json(
        journal,
        {
            "state": "restored" if restore else "installed",
            "manifest": str(path),
            "manifest_sha256": sha256(path),
        },
    )


def install(path, *, reports, home=None, appdata=None):
    qualify_reports(reports)
    _install_or_restore(
        path,
        home=Path(home or Path.home()),
        appdata=Path(appdata or os.environ["APPDATA"]),
        restore=False,
    )


def rollback(path, *, home=None, appdata=None):
    _install_or_restore(
        path,
        home=Path(home or Path.home()),
        appdata=Path(appdata or os.environ["APPDATA"]),
        restore=True,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("activate", "rollback"))
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reports", type=Path)
    args = parser.parse_args()
    if args.action == "activate":
        if args.reports is None:
            raise ValueError("Provide the actual client qualification report mapping.")
        install(args.manifest, reports=json.loads(args.reports.read_text(encoding="utf-8")))
    else:
        rollback(args.manifest)
    print(
        f"Native profile {args.action} completed; model files and running servers were unchanged."
    )


if __name__ == "__main__":
    main()
