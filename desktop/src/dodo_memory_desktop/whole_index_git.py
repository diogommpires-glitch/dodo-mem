"""Read the checked-out Git metadata without fetching or using external diff drivers."""

import hashlib
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .whole_index import SECRET_BYTES, path_key, read_bytes, safe_path

HASH = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
MAX_GIT_OUTPUT = 16 * 1024 * 1024
MAX_COMMITS = 1000


def repository(path: Path) -> bool:
    try:
        git = path / ".git"
        safe_path(git)
        if not git.is_dir() or any(
            (git / name).exists() for name in ("commondir", "objects/info/alternates")
        ):
            return False
        config = read_bytes(git / "config", 1024 * 1024, internal=True)
    except (OSError, ValueError):
        return False
    return not re.search(rb"(?im)^\s*\[\s*include(?:if)?(?:\s|\])", config)


def git_output(path: Path, args: list[str]) -> bytes:
    executable = shutil.which("git")
    if not executable or not repository(path):
        raise ValueError("Local Git history unavailable")
    environment = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    environment.update(
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_NO_LAZY_FETCH="1",
        GIT_OPTIONAL_LOCKS="0",
        GIT_TERMINAL_PROMPT="0",
        GIT_PAGER="",
    )
    with tempfile.TemporaryFile() as output:
        subprocess.run(
            [
                executable,
                "--no-pager",
                "-c",
                "core.fsmonitor=false",
                "-c",
                "protocol.allow=never",
                "-C",
                str(path),
                *args,
            ],
            stdout=output,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            env=environment,
            timeout=12,
            check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        output.seek(0)
        data = output.read(MAX_GIT_OUTPUT + 1)
        if len(data) > MAX_GIT_OUTPUT:
            raise ValueError("Git history output exceeds its budget")
        return data


def revision_id(repo_id: str, commit: str) -> str:
    return hashlib.blake2b(f"{repo_id}:{commit}".encode(), digest_size=16).hexdigest()


def history(path: Path):
    data = git_output(
        path,
        [
            "log",
            "--all",
            f"--max-count={MAX_COMMITS}",
            "--no-ext-diff",
            "--no-textconv",
            "--no-show-signature",
            "--no-renames",
            "--name-only",
            "--decorate=short",
            "-m",
            "-z",
            "--format=%x1e%H%x00%P%x00%ct%x00%s%x00%D%x00",
        ],
    )
    rows = {}
    for record in data.split(b"\x1e")[1:]:
        fields = record.decode("utf-8", errors="replace").split("\x00")
        if len(fields) < 5:
            continue
        commit, parents, timestamp, subject, refs, *files = fields
        if not HASH.fullmatch(commit) or not timestamp.isdecimal():
            continue
        if SECRET_BYTES.search(subject.encode("utf-8", errors="replace")):
            subject = "Credential-like commit subject excluded"
        row = rows.setdefault(
            commit,
            {
                "commit": commit,
                "parents": [p for p in parents.split() if HASH.fullmatch(p)],
                "time_ns": int(timestamp) * 1_000_000_000,
                "subject": subject[:500],
                "files": set(),
                "refs": [ref.strip()[:250] for ref in refs.split(",") if ref.strip()][:80],
            },
        )
        for name in files:
            relative = Path(name.strip("\r\n"))
            if relative.parts and not relative.is_absolute() and ".." not in relative.parts:
                row["files"].add(path_key(path / relative))
    return list(rows.values())


def verify_commit(locator: str, commit: str) -> bool:
    if not HASH.fullmatch(commit) or not locator.endswith(f"#{commit}"):
        return False
    try:
        path = Path(locator.removesuffix(f"#{commit}"))
        return (
            git_output(path, ["rev-parse", "--verify", f"{commit}^{{commit}}"]).strip()
            == commit.encode()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False
