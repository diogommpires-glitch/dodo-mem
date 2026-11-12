"""Capture bounded working-tree code from explicitly approved Git roots."""

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from .lifecycle import _git
from .store import now

CODE_SUFFIXES = frozenset(
    {
        ".py",
        ".pyi",
        ".js",
        ".jsx",
        ".ts",
        ".tsx",
        ".c",
        ".h",
        ".cc",
        ".cpp",
        ".cxx",
        ".hh",
        ".hpp",
        ".cs",
        ".java",
        ".go",
        ".rs",
        ".rb",
        ".php",
        ".swift",
        ".kt",
        ".kts",
        ".scala",
        ".sh",
        ".bash",
        ".zsh",
        ".ps1",
        ".psm1",
        ".r",
        ".m",
        ".jl",
        ".lua",
        ".sql",
        ".html",
        ".css",
        ".scss",
        ".sass",
        ".vue",
        ".svelte",
    }
)
EXCLUDED_DIRECTORIES = frozenset(
    {
        "node_modules",
        "venv",
        "env",
        "site-packages",
        "vendor",
        "third_party",
        "third-party",
        "build",
        "dist",
        "out",
        "target",
        "generated",
        "__pycache__",
        "coverage",
        "htmlcov",
        "outputs",
        "checkpoints",
        "artifacts",
        "logs",
    }
)
SECRET_NAME = re.compile(
    r"(^|[_.-])(secrets?|credentials?|passwords?|tokens?|api[_-]?keys?|private[_-]?keys?)([_.-]|$)",
    re.IGNORECASE,
)
SECRET_LITERAL = re.compile(
    r"""(?:api[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret|password|passwd)"""
    r"""(?:["']\s*\]?)?\s*[:=]\s*["'][^"'\r\n]+["']""",
    re.IGNORECASE,
)
GENERATED_COMMENT = re.compile(
    r"""^\s*(?:#|//|/\*|\*|<!--|["']{3}).*(?:automatically generated|auto-generated|do not edit|@generated)""",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class _CapturedCode:
    path: Path
    content: bytes
    provenance: dict
    signature: tuple


class _Cancelled(RuntimeError):
    pass


def _linked(path):
    return any(
        part.is_symlink() or getattr(part, "is_junction", lambda: False)()
        for part in (path, *path.parents)
    )


def _relative(value):
    if not isinstance(value, str):
        raise TypeError("Selected repository paths must be relative strings")
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or "\\" in value
        or any(ord(character) < 32 for character in value)
        or path.as_posix() != value
    ):
        raise ValueError("Selected repository paths must be canonical relative paths")
    return path


def _repository(repository, allowed):
    root = Path(repository)
    if not root.is_absolute() or _linked(root):
        raise PermissionError("Repository root must be absolute and contain no links")
    root = root.resolve(strict=True)
    if root not in allowed:
        raise PermissionError("Repository root is not an exact explicitly approved root")
    actual = _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(actual).resolve() != root:
        raise PermissionError("Repository intake requires the exact Git root")
    if _git(root, "rev-parse", "--show-superproject-working-tree").strip():
        raise PermissionError("Repository intake excludes submodule roots")
    return root


def _snapshot(root):
    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD", optional=True)
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", optional=True)
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    index = _git(root, "ls-files", "--stage", "-z")
    return {
        "repository": str(root),
        "head": head.decode("ascii").strip() if head else None,
        "branch": branch.decode("utf-8").strip() if branch else None,
        "dirty": bool(status),
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
    }


def _path_exclusion(root, relative, submodules):
    name = relative.as_posix()
    parts = [part.casefold() for part in relative.parts]
    if any(name == submodule or name.startswith(submodule + "/") for submodule in submodules):
        return "submodule"
    if any(part.startswith(".") or part in EXCLUDED_DIRECTORIES for part in parts[:-1]):
        return "directory"
    if relative.suffix.casefold() not in CODE_SUFFIXES:
        return "extension"
    if SECRET_NAME.search(relative.name) or relative.name.casefold().startswith("local_settings."):
        return "credential_name"
    if any(marker in relative.name.casefold() for marker in (".min.", ".bundle.", ".generated.")):
        return "generated"
    local = root / relative
    if _linked(local) or local.resolve() != local.absolute():
        return "link"
    if not local.is_file():
        return "missing_or_non_regular"
    if any(
        (parent / ".git").exists()
        for parent in local.parents
        if parent != root and root in parent.parents
    ):
        return "nested_repository"
    return None


def decode_code(content, suffix):
    """Reject binary and structured-export data even under a language extension."""
    if suffix.casefold() not in CODE_SUFFIXES:
        raise ValueError("Code parser requires an explicitly supported language extension")
    if not content or b"\0" in content:
        raise ValueError("Code input must be non-empty text without NUL bytes")
    text = content.decode("utf-8", errors="strict")
    classification = text.lstrip("\ufeff \t\r\n")
    if not classification.strip():
        raise ValueError("Code input must contain non-empty source text")
    try:
        json.loads(classification)
    except json.JSONDecodeError:
        try:
            for line in classification.splitlines():
                record = line.lstrip("\ufeff \t\r")
                if record.strip():
                    json.loads(record)
        except json.JSONDecodeError:
            pass
        else:
            raise ValueError("JSONL exports cannot use the code parser")
    else:
        raise ValueError("JSON exports cannot use the code parser")
    return text


def _signature(info):
    return (info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns)


def _read_code(path, remaining, maximum, cancelled):
    before = path.stat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        return None, "non_regular_or_hardlink", 0
    if not 0 < before.st_size <= maximum:
        return None, "empty_or_oversized", 0
    if before.st_size > remaining:
        raise ValueError("Repository byte budget exceeded before queueing")
    with os.fdopen(os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)), "rb") as stream:
        descriptor = os.fstat(stream.fileno())
        # Windows path and descriptor st_ctime can describe different timestamps.
        if _linked(path) or _signature(descriptor)[:4] != _signature(before)[:4]:
            raise ValueError("Repository file changed before bounded capture")
        chunks, consumed = [], 0
        while True:
            if cancelled():
                raise _Cancelled()
            block = stream.read(min(65536, remaining - consumed + 1))
            if not block:
                break
            consumed += len(block)
            if consumed > remaining or consumed > maximum:
                raise ValueError("Repository byte budget exceeded during capture")
            chunks.append(block)
        if (
            _linked(path)
            or _signature(os.fstat(stream.fileno())) != _signature(descriptor)
            or _signature(path.stat()) != _signature(before)
        ):
            raise ValueError("Repository file changed during bounded capture")
    content = b"".join(chunks)
    try:
        text = decode_code(content, path.suffix)
    except (UnicodeError, ValueError):
        return None, "non_code_text", consumed
    if GENERATED_COMMENT.search("\n".join(text.splitlines()[:15])):
        return None, "generated", consumed
    if SECRET_LITERAL.search(text) or ("-----BEGIN " in text and "PRIVATE KEY-----" in text):
        return None, "credential_literal", consumed
    return (content, _signature(before)), None, consumed


def validate_capture(source, capture, allowed):
    """Check the internal captured-byte boundary without broadening generic intake."""
    if not isinstance(capture, _CapturedCode):
        raise PermissionError("Code intake requires an approved repository capture")
    provenance = capture.provenance
    root = Path(provenance["repository"])
    relative = _relative(provenance["relative_path"])
    path = Path(source).absolute()
    if (
        root not in allowed
        or path != root / relative
        or path != capture.path
        or _path_exclusion(root, relative, ())
        or _signature(path.stat()) != capture.signature
        or hashlib.sha256(capture.content).hexdigest() != provenance["content_sha256"]
    ):
        raise PermissionError("Repository capture no longer matches its approved path and bytes")
    decode_code(capture.content, path.suffix)
    identity = os.path.normcase(str(root)) + "\0" + os.path.normcase(relative.as_posix())
    return path, capture.content, hashlib.sha256(identity.encode()).hexdigest()[:32], provenance


def capture_repository(
    registry,
    repository,
    *,
    paths=None,
    scope="private",
    collection="code",
    owner="local",
    max_files=200,
    max_bytes=16 * 1024**2,
    max_file_bytes=1024**2,
    cancelled=lambda: False,
):
    """Preflight the whole selection, then report every durable per-file queue result."""
    if scope not in {"private", "shared"}:
        raise ValueError("Repository capture scope must be private or shared")
    for name, value, maximum in [
        ("file", max_files, 2000),
        ("byte", max_bytes, 100 * 1024**2),
        ("per-file byte", max_file_bytes, 10 * 1024**2),
    ]:
        if type(value) is not int or not 0 < value <= maximum:
            raise ValueError(f"Repository {name} limit must be between 1 and {maximum}")
    registry.vault.assert_retention_ready()
    root = _repository(repository, registry.repository_roots)
    snapshot = _snapshot(root)
    result = {
        "status": "queued",
        "repository": snapshot,
        "project": root.as_posix(),
        "jobs": [],
        "excluded": {},
        "bytes": 0,
        "scan_complete": False,
        "planned_files": None,
        "queued_files": 0,
        "remaining_files": None,
    }
    if cancelled():
        return {**result, "status": "cancelled"}
    candidates = set(
        _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
        .decode("utf-8")
        .split("\0")
    ) - {""}
    ignored = set(
        _git(root, "ls-files", "--cached", "--ignored", "--exclude-standard", "-z")
        .decode("utf-8")
        .split("\0")
    ) - {""}
    stages = _git(root, "ls-files", "--stage", "-z").decode("utf-8").split("\0")
    submodules = {entry.split("\t", 1)[1] for entry in stages if entry.startswith("160000 ")}
    excluded = Counter({"git_ignored": len(candidates & ignored)})
    candidates -= ignored
    if paths is not None:
        if not isinstance(paths, (list, tuple)) or not paths:
            raise ValueError("Selected repository paths must be a non-empty list")
        selected = {_relative(value).as_posix() for value in paths}
        if not selected <= candidates:
            raise ValueError("A selected repository path is unavailable or Git-ignored")
        candidates = selected
    captures, examined, consumed = [], 0, 0
    try:
        for name in sorted(candidates):
            if cancelled():
                raise _Cancelled()
            if name.endswith("/"):
                excluded["nested_repository"] += 1
                continue
            relative = _relative(name)
            reason = _path_exclusion(root, relative, submodules)
            if reason:
                excluded[reason] += 1
                continue
            examined += 1
            if examined > max_files:
                raise ValueError("Repository file limit exceeded before queueing")
            captured, reason, read = _read_code(
                root / relative, max_bytes - consumed, max_file_bytes, cancelled
            )
            consumed += read
            if reason:
                excluded[reason] += 1
                continue
            content, signature = captured
            provenance = {
                **snapshot,
                "relative_path": name,
                "captured_at": now(),
                "content_sha256": hashlib.sha256(content).hexdigest(),
                "representation": "working-tree-bytes",
            }
            captures.append(_CapturedCode(root / relative, content, provenance, signature))
    except _Cancelled:
        return {**result, "status": "cancelled", "excluded": dict(excluded)}
    if _snapshot(root) != snapshot:
        raise ValueError("Repository Git state changed during capture; nothing was queued")
    result["excluded"] = dict(excluded)
    result.update(scan_complete=True, planned_files=len(captures))
    for capture in captures:
        if cancelled():
            result["status"] = "cancelled"
            break
        try:
            job = registry.submit(
                capture.path,
                kind="code",
                scope=scope,
                collection=collection,
                project=root.as_posix(),
                owner=owner,
                parser="code-text-v1",
                capture_method="repository_worktree",
                repository_capture=capture,
            )
        except (OSError, ValueError, RuntimeError, sqlite3.DatabaseError) as error:
            result.update(
                status="partial_failure" if result["jobs"] else "failed",
                error_type=type(error).__name__,
            )
            break
        result["jobs"].append(job)
        result["bytes"] += len(capture.content)
    result["queued_files"] = len(result["jobs"])
    result["remaining_files"] = len(captures) - len(result["jobs"])
    if not captures:
        result["status"] = "empty"
    return result
