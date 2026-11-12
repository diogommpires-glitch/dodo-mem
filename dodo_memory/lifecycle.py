"""Observe repository drift and retain typed, historical coding checkpoints."""

import argparse
import asyncio
import hashlib
import json
import os
import re
import subprocess
import sys
from pathlib import Path, PurePosixPath

from .bridge import GATEWAY, _upstream
from .capture import capture_checkpoint
from .source_registry import atomic_bytes

MAX_CHANGED_BYTES = 16 * 1024**2
MAX_CHANGED_FILES = 500
HOOK_PHASES = {"SessionStart": "start", "PreCompact": "compact", "Stop": "stop"}


def _hook_path(value):
    path = Path(value)
    text = str(path)
    if os.name == "nt" and text.startswith(("\\\\?\\", "\\\\.\\")):
        if re.match(r"^\\\\\?\\[A-Za-z]:\\", text) is None:
            raise PermissionError("Native hooks require a local filesystem path")
        original = path.resolve(strict=True)
        canonical = Path(str(original).removeprefix("\\\\?\\")).resolve(strict=True)
        if not canonical.samefile(original):
            raise PermissionError("Native hook path aliases must identify the same directory")
        return canonical
    return path.resolve(strict=True)


def _git(repository, *arguments, optional=False, timeout=5):
    result = subprocess.run(
        [
            "git",
            "--no-optional-locks",
            "-c",
            "core.fsmonitor=false",
            "-C",
            str(repository),
            *arguments,
        ],
        capture_output=True,
        timeout=timeout,
        env={
            **{
                name: value
                for name, value in os.environ.items()
                if name
                not in {
                    "GIT_DIR",
                    "GIT_WORK_TREE",
                    "GIT_INDEX_FILE",
                    "GIT_COMMON_DIR",
                    "GIT_OBJECT_DIRECTORY",
                    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
                }
            },
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_PAGER": "cat",
        },
        check=False,
    )
    if optional and result.returncode == 1:
        return None
    if result.returncode:
        raise ValueError(f"Read-only Git inspection failed: {arguments[0]}")
    if len(result.stdout) > 1024**2:
        raise ValueError("Repository metadata exceeds the 1 MiB orientation limit")
    return result.stdout


def repository_snapshot(repository):
    """Hash bounded changed file contents without exporting source text or running Git hooks."""
    root = Path(repository).resolve(strict=True)
    actual = _git(root, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(actual).resolve() != root:
        raise ValueError("Repository orientation requires an explicitly named Git root")
    head = _git(root, "rev-parse", "--verify", "--quiet", "HEAD", optional=True)
    branch = _git(root, "symbolic-ref", "--quiet", "--short", "HEAD", optional=True)
    status = _git(root, "status", "--porcelain=v1", "-z", "--untracked-files=all")
    index = _git(root, "ls-files", "--stage", "-z")
    entries = status.decode("utf-8").split("\0")
    changed, consumed, position = [], 0, 0
    while position < len(entries) - 1:
        entry = entries[position]
        position += 1
        code, relative = entry[:2], entry[3:]
        names = [relative]
        if "R" in code or "C" in code:
            names.append(entries[position])
            position += 1
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or ":" in name or not path.parts:
                raise ValueError("Git returned a source path outside the repository")
            local = root / path
            if local.is_symlink():
                content = b"symlink\0" + os.readlink(local).encode("utf-8")
            elif not local.exists():
                content = b"missing"
            else:
                if local.resolve() != local or not local.is_file():
                    raise ValueError(
                        "Changed repository paths must be files without linked parents"
                    )
                remaining = MAX_CHANGED_BYTES - consumed
                if local.stat().st_size > remaining:
                    raise ValueError("Changed source bytes exceed the 16 MiB orientation limit")
                with local.open("rb") as stream:
                    content = stream.read(remaining + 1)
            consumed += len(content)
            if consumed > MAX_CHANGED_BYTES:
                raise ValueError("Changed source bytes exceed the 16 MiB orientation limit")
            changed.append(
                {"path": name, "status": code, "sha256": hashlib.sha256(content).hexdigest()}
            )
            if len(changed) > MAX_CHANGED_FILES:
                raise ValueError("Changed paths exceed the 500-file orientation limit")
    return {
        "repository": str(root),
        "branch": branch.decode("utf-8").strip() if branch is not None else None,
        "head": head.decode("ascii").strip() if head is not None else None,
        "status_sha256": hashlib.sha256(status).hexdigest(),
        "index_sha256": hashlib.sha256(index).hexdigest(),
        "worktree_sha256": hashlib.sha256(json.dumps(changed, sort_keys=True).encode()).hexdigest(),
        "changed_files": changed,
    }


def repository_drift(previous, current):
    """Report changed observations without modifying the historical checkpoint."""
    if previous is None:
        return ["No previous repository snapshot supplied; continuity has not been verified."]
    required = {"repository", "branch", "head", "status_sha256", "index_sha256", "worktree_sha256"}
    if not isinstance(previous, dict) or not required <= previous.keys():
        raise ValueError("Previous repository snapshot is missing its identity or signatures")
    warnings = []
    for field, message in [
        ("repository", "Repository differs from the historical checkpoint."),
        ("branch", "Branch differs from the historical checkpoint."),
        ("head", "HEAD differs from the historical checkpoint; inspect the live commit."),
        ("status_sha256", "Changed file paths/status differ from the historical checkpoint."),
        ("index_sha256", "Git index content differs from the historical checkpoint."),
        (
            "worktree_sha256",
            "Working file content differs from the historical checkpoint; re-read touched files.",
        ),
    ]:
        if previous[field] != current[field]:
            warnings.append(message)
    return warnings


def checkpoint_text(
    snapshot, phase, objective, latest_intent, decisions, failures, next_action, checks
):
    if phase not in {"start", "compact", "stop", "decision", "failure"}:
        raise ValueError("Checkpoint phase must be start, compact, stop, decision or failure")
    for name, value in [
        ("objective", objective),
        ("latest_intent", latest_intent),
        ("next_action", next_action),
    ]:
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError(f"Checkpoint {name} requires 1 to 4000 non-blank characters")
    for name, items in [("decisions", decisions), ("failures", failures)]:
        if (
            not isinstance(items, list)
            or len(items) > 32
            or any(
                not isinstance(item, str) or not item.strip() or len(item) > 2000 for item in items
            )
        ):
            raise ValueError(
                f"Checkpoint {name} requires at most 32 non-blank texts of up to 2000 characters"
            )
    if not isinstance(checks, list) or len(checks) > 32:
        raise ValueError("Checkpoint checks requires at most 32 reported checks")
    for check in checks:
        if (
            not isinstance(check, dict)
            or set(check) != {"command", "result", "platform"}
            or not isinstance(check["result"], str)
            or check["result"] not in {"passed", "failed", "not_run"}
            or any(
                not isinstance(check[name], str) or not 1 <= len(check[name]) <= 2000
                for name in ["command", "platform"]
            )
        ):
            raise ValueError(
                "A reported check requires command, platform and result passed/failed/not_run"
            )
    payload = {
        "schema_version": 1,
        "kind": "coding_checkpoint",
        "verification": "historical_agent_report",
        "phase": phase,
        "objective": objective,
        "latest_intent": latest_intent,
        "decisions": decisions,
        "failures": failures,
        "next_action": next_action,
        "checks": checks,
        "repository_snapshot": snapshot,
    }
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)
    if len(text) > 80_000:
        raise ValueError("Typed checkpoint exceeds the 80000-character limit")
    return text


def capture_lifecycle_checkpoint(
    vault,
    session_id,
    event_id,
    writer,
    repository,
    phase,
    objective,
    latest_intent,
    decisions,
    failures,
    next_action,
    checks,
):
    """Caller holds the gateway writer gate; preserve the existing immutable capture authority."""
    snapshot = repository_snapshot(repository)
    text = checkpoint_text(
        snapshot, phase, objective, latest_intent, decisions, failures, next_action, checks
    )
    return capture_checkpoint(
        vault, session_id, event_id, text, writer, snapshot["repository"], snapshot["head"]
    )


async def handle_hook(home, client, repository, event, gateway=GATEWAY):
    """Save bounded operational events through MCP; never infer decisions from raw transcripts."""
    if client not in {"codex", "claude"} or not isinstance(event, dict):
        raise ValueError("Lifecycle hooks require a Codex/Claude event object")
    name, session = event.get("hook_event_name"), event.get("session_id")
    if name not in HOOK_PHASES or not isinstance(session, str) or not session:
        raise ValueError("Lifecycle hooks require a supported event and non-empty session ID")
    cwd = event.get("cwd")
    if not isinstance(cwd, str) or not Path(cwd).is_absolute():
        return {}
    root = _hook_path(repository)
    try:
        working_directory = _hook_path(cwd)
    except (OSError, RuntimeError):
        return {}
    if not working_directory.is_dir() or not working_directory.is_relative_to(root):
        return {}
    actual_root = _git(working_directory, "rev-parse", "--show-toplevel").decode("utf-8").strip()
    if Path(actual_root).resolve() != root:
        return {}
    snapshot = repository_snapshot(root)
    identity = hashlib.sha256(snapshot["repository"].encode()).hexdigest()
    cache = Path(home).resolve() / ".state/lifecycle" / f"{client}-{identity}.json"
    previous = None
    if cache.exists():
        if (
            cache.is_symlink()
            or cache.resolve() != cache.absolute()
            or cache.stat().st_size > 1024**2
        ):
            raise ValueError("Lifecycle cache must be a bounded local file without links")
        previous = json.loads(cache.read_text(encoding="utf-8"))["snapshot"]
    warnings = repository_drift(previous, snapshot)
    text = checkpoint_text(
        snapshot,
        HOOK_PHASES[name],
        "Observe the coding lifecycle.",
        "No task intent or decisions were supplied by this automatic hook.",
        [],
        [],
        "Read live repository state and explicitly checkpoint significant task decisions.",
        [],
    )
    event_id = (
        "hook-"
        + hashlib.sha256(
            json.dumps(
                [client, name, session, event.get("turn_id"), event.get("source"), text],
                sort_keys=True,
            ).encode()
        ).hexdigest()[:48]
    )
    session_id = "native-" + hashlib.sha256(session.encode()).hexdigest()[:48]
    async with _upstream(home, client, gateway) as (memory, _initial):
        core = await memory.call_tool("get_core", {})
        if core.isError or not isinstance(core.structuredContent, dict):
            raise RuntimeError("Memory gateway did not return the reviewed core")
        saved = await memory.call_tool(
            "save_checkpoint",
            {
                "session_id": session_id,
                "event_id": event_id,
                "native_session_id": session,
                "text": text,
                "repository": snapshot["repository"],
                "commit": snapshot["head"],
            },
        )
        if (
            saved.isError
            or not isinstance(saved.structuredContent, dict)
            or not isinstance(saved.structuredContent.get("path"), str)
        ):
            raise RuntimeError("Memory gateway did not acknowledge a durable lifecycle checkpoint")
    atomic_bytes(
        cache,
        json.dumps({"snapshot": snapshot, "checkpoint": saved.structuredContent["path"]}).encode(),
    )
    context = json.dumps(
        {
            "repository": snapshot,
            "drift_warnings": warnings,
            "checkpoint": saved.structuredContent["path"],
        },
        ensure_ascii=False,
    )
    if name == "SessionStart":
        context += "\nReviewed core (core/CORE.md):\n" + core.structuredContent["text"]
        if len(context) > 24_000:
            raise ValueError("Startup context exceeds the 24000-character limit")
        return {"hookSpecificOutput": {"hookEventName": name, "additionalContext": context}}
    return {
        "systemMessage": "Dodo Mem lifecycle checkpoint saved; task claims require an explicit typed checkpoint."
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--home", required=True, type=Path)
    parser.add_argument("--client", required=True, choices=["codex", "claude"])
    parser.add_argument("--repository", required=True, type=Path)
    args = parser.parse_args()
    content = sys.stdin.buffer.read(64 * 1024 + 1)
    if len(content) > 64 * 1024:
        raise ValueError("Hook input exceeds the 64 KiB limit")
    result = asyncio.run(handle_hook(args.home, args.client, args.repository, json.loads(content)))
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
