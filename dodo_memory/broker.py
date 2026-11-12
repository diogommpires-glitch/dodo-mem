"""Server-internal, private native capture requests; no client-visible capture authority."""

import hashlib
import io
import json
import os
import re
import sqlite3
import subprocess
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from uuid import UUID, uuid4

from .lifecycle import _git
from .source_registry import MAX_BYTES, SourceRegistry, atomic_bytes, single_line
from .store import defer_busy_claim, now
from .transcripts import MAX_LINE_BYTES, MAX_RECORDS

CLIENTS = {"codex", "claude"}
MAX_REQUESTS = 1024
MAX_DISCOVERY_ENTRIES = 20_000
MAX_CAPTURE_SECONDS = 15
LEASE_SECONDS = 120
RETRY_SECONDS = 2
MAX_ATTEMPTS = 5


class _CaptureError(ValueError):
    def __init__(self, reason, *, retryable=False):
        super().__init__(reason)
        self.reason, self.retryable = reason, retryable


def _canonical_uuid(value):
    if not isinstance(value, str):
        return False
    try:
        return str(UUID(value)) == value
    except ValueError:
        return False


def _path(path, *, exists=True):
    """Reject links before resolving; owner roots are local absolute directories."""
    path = Path(path)
    if not path.is_absolute() or ".." in path.parts or path.drive.startswith("\\"):
        raise _CaptureError("unsafe_path")
    if any(
        part.is_symlink() or getattr(part, "is_junction", lambda: False)()
        for part in (path, *path.parents)
    ):
        raise _CaptureError("linked_path")
    if path.resolve(strict=exists) != path.absolute():
        raise _CaptureError("redirected_path")
    return path.absolute()


def _identity(path):
    info = path.stat()
    return info.st_dev, info.st_ino


def _signature(info):
    # Windows Python 3.14 path stat and handle stat disagree on ctime semantics.
    # Compare ctime only within the same API; these fields agree across APIs.
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns


class NativeCaptureBroker:
    """The server maps authenticated native tokens to names and never returns this state."""

    def __init__(self, vault, config):
        if not isinstance(config, dict) or set(config) - {
            "enabled",
            "allowed_repositories",
            "client_roots",
        }:
            raise TypeError("Native capture requires the owner-local broker configuration")
        self.vault = vault
        self.enabled = config.get("enabled", False)
        if type(self.enabled) is not bool:
            raise TypeError("Native capture enabled must be a boolean")
        repositories = config.get("allowed_repositories", [])
        roots = config.get("client_roots", {})
        if not isinstance(repositories, list) or len(repositories) > 16:
            raise TypeError("Native capture permits at most 16 explicit repository roots")
        if not isinstance(roots, dict) or set(roots) - CLIENTS:
            raise TypeError("Native capture roots must name only codex or claude")
        self.repositories, self.roots, self.anchors = {}, {}, {}
        for repository in repositories:
            root = _path(repository)
            if (
                not root.is_dir()
                or Path(_git(root, "rev-parse", "--show-toplevel").decode().strip()).resolve()
                != root
            ):
                raise ValueError("Approved repositories must be explicit Git roots")
            self.repositories[str(root)] = root
            self.anchors[root] = _identity(root)
        for client, locations in roots.items():
            if not isinstance(locations, list) or len(locations) > 8:
                raise TypeError("Each native client permits at most eight explicit local roots")
            self.roots[client] = tuple(_path(location) for location in locations)
            for root in self.roots[client]:
                if not root.is_dir():
                    raise ValueError("Native capture roots must be directories")
                self.anchors[root] = _identity(root)
        if self.enabled and not self.repositories:
            raise ValueError("Enabled native capture needs an approved repository")
        self.staging = _path(vault.home / ".state/native-capture", exists=False)
        self.staging.mkdir(parents=True, exist_ok=True, mode=0o700)
        _path(self.staging)
        self.anchors[self.staging] = _identity(self.staging)
        self.registry = SourceRegistry(vault, allowed_roots=(self.staging,))
        with vault.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS native_capture_requests (
                    client_name TEXT NOT NULL, session_id TEXT NOT NULL, repository TEXT NOT NULL,
                    generation INTEGER NOT NULL, claimed_generation INTEGER,
                    last_event_id TEXT NOT NULL, status TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
                    ready_at TEXT NOT NULL, claim_token TEXT, lease_until TEXT,
                    recoveries INTEGER NOT NULL DEFAULT 0, last_source_id TEXT, last_job_id TEXT,
                    captured_generation INTEGER,
                    provenance_json TEXT, error TEXT, recorded_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    PRIMARY KEY(client_name,session_id)
                );
                CREATE TABLE IF NOT EXISTS native_capture_control (
                    id INTEGER PRIMARY KEY CHECK(id=1), dropped_requests INTEGER NOT NULL DEFAULT 0
                );
                INSERT OR IGNORE INTO native_capture_control VALUES (1,0);
            """)

    def request(self, client_name, session_id, event_id, repository):
        """Always return: private capture acknowledgement is never a native response."""
        if (
            not self.enabled
            or not isinstance(client_name, str)
            or client_name not in CLIENTS
            or not self.roots.get(client_name)
        ):
            return
        if not _canonical_uuid(session_id) or not isinstance(repository, (str, Path)):
            return
        candidate = Path(repository)
        approved = next((root for root in self.repositories.values() if candidate == root), None)
        if approved is None or ".." in candidate.parts:
            return
        try:
            single_line(event_id, "native event ID", 160)
        except ValueError:
            return
        timestamp = now()
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            previous = db.execute(
                "SELECT repository FROM native_capture_requests WHERE client_name=? AND session_id=?",
                (client_name, session_id),
            ).fetchone()
            if previous is not None and previous["repository"] != str(approved):
                return
            if (
                previous is None
                and db.execute("SELECT count(*) FROM native_capture_requests").fetchone()[0]
                >= MAX_REQUESTS
            ):
                db.execute(
                    "UPDATE native_capture_control SET dropped_requests=dropped_requests+1 WHERE id=1"
                )
                return
            db.execute(
                """INSERT INTO native_capture_requests
                (client_name,session_id,repository,generation,last_event_id,status,ready_at,recorded_at,updated_at)
                VALUES (?,?,?,1,?,'queued',?,?,?)
                ON CONFLICT(client_name,session_id) DO UPDATE SET
                    generation=generation+1,last_event_id=excluded.last_event_id,
                    status=CASE WHEN status='running' THEN status ELSE 'queued' END,
                    attempts=CASE WHEN status='running' THEN attempts ELSE 0 END,
                    ready_at=excluded.ready_at,error=NULL,updated_at=excluded.updated_at""",
                (client_name, session_id, str(approved), event_id, timestamp, timestamp, timestamp),
            )
        return

    def recover(self):
        """Only expired claims are reclaimable; constructing another broker cannot steal work."""
        with self.vault.connect() as db:
            return db.execute(
                """UPDATE native_capture_requests SET status='queued',claim_token=NULL,lease_until=NULL,
                recoveries=recoveries+1,ready_at=?,updated_at=?,
                attempts=CASE WHEN generation!=claimed_generation THEN 0 ELSE attempts END
                WHERE status='running' AND lease_until<=?""",
                (now(), now(), now()),
            ).rowcount

    def status(self, client_name, session_id, *, actor_role):
        """Owner-only local API; it must never be registered as a native MCP tool."""
        if actor_role != "owner":
            raise PermissionError(
                "Only the authenticated local owner can inspect private capture state"
            )
        with self.vault.connect() as db:
            row = db.execute(
                "SELECT * FROM native_capture_requests WHERE client_name=? AND session_id=?",
                (client_name, session_id),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["provenance"] = json.loads(result.pop("provenance_json") or "null")
        result["last_job"] = None
        if result["last_job_id"] is not None:
            try:
                result["last_job"] = self.registry.job(result["last_job_id"], ("private",))
            except PermissionError:
                result["job_uncertainty"] = "Previously acknowledged private job is unavailable."
        return result

    def _check_anchor(self, root):
        if _path(root) != root or _identity(root) != self.anchors[root]:
            raise _CaptureError("configured_root_changed")

    @staticmethod
    def _deadline(deadline):
        if time.monotonic() > deadline:
            raise _CaptureError("capture_budget_exhausted", retryable=True)

    def _discover(self, client, session, deadline):
        matches, visited = [], 0
        pending = [(root, root, 0) for root in self.roots[client]]
        while pending:
            directory, root, depth = pending.pop()
            self._check_anchor(root)
            _path(directory)
            with os.scandir(directory) as entries:
                for entry in entries:
                    visited += 1
                    self._deadline(deadline)
                    if visited > MAX_DISCOVERY_ENTRIES:
                        raise _CaptureError("discovery_entry_limit", retryable=True)
                    matching = (
                        entry.name == session + ".jsonl"
                        if client == "claude"
                        else (
                            entry.name.startswith("rollout-")
                            and entry.name.endswith("-" + session + ".jsonl")
                        )
                    )
                    if matching:
                        matches.append(_path(entry.path))
                    elif depth < (3 if client == "codex" else 1) and entry.is_dir(
                        follow_symlinks=False
                    ):
                        if client == "codex" and not re.fullmatch(r"[0-9]{2,4}", entry.name):
                            continue
                        child = Path(entry.path)
                        if child.is_symlink() or getattr(child, "is_junction", lambda: False)():
                            continue
                        pending.append((child, root, depth + 1))
        if not matches:
            raise _CaptureError("native_source_not_available", retryable=True)
        if len(set(matches)) != 1:
            raise _CaptureError("ambiguous_native_source")
        return matches[0]

    def _snapshot(self, origin, deadline):
        roots = [
            root
            for locations in self.roots.values()
            for root in locations
            if origin.is_relative_to(root)
        ]
        for root in roots:
            self._check_anchor(root)
        _path(origin)
        before = origin.stat()
        if not origin.is_file() or not 0 < before.st_size <= MAX_BYTES:
            raise _CaptureError("native_source_size_limit")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        with os.fdopen(os.open(origin, flags), "rb") as stream:
            opened_before = os.fstat(stream.fileno())
            if _signature(opened_before) != _signature(before):
                raise _CaptureError("native_source_changed", retryable=True)
            data = stream.read(MAX_BYTES + 1)
            opened_after = os.fstat(stream.fileno())
        self._deadline(deadline)
        for root in roots:
            self._check_anchor(root)
        _path(origin)
        after = origin.stat()
        if (
            len(data) > MAX_BYTES
            or _signature(before) != _signature(opened_after)
            or _signature(before) != _signature(after)
            or before.st_ctime_ns != after.st_ctime_ns
            or opened_before.st_ctime_ns != opened_after.st_ctime_ns
        ):
            raise _CaptureError("native_source_changed", retryable=True)
        return data

    def _validate(self, data, client, session, repository, deadline):
        if not data.endswith(b"\n"):
            raise _CaptureError("partial_last_line", retryable=True)
        sessions, cwds = set(), set()
        for number, line in enumerate(io.BytesIO(data), 1):
            self._deadline(deadline)
            if number > MAX_RECORDS or len(line) > MAX_LINE_BYTES:
                raise _CaptureError("native_record_limit")
            try:
                record = json.loads(line)
            except (ValueError, RecursionError) as error:
                raise _CaptureError("invalid_native_jsonl") from error
            if not isinstance(record, dict) or not isinstance(record.get("type"), str):
                raise _CaptureError("invalid_native_record")
            payload = record.get("payload")
            if client == "codex":
                if record["type"] == "session_meta":
                    if not isinstance(payload, dict):
                        raise _CaptureError("invalid_native_session")
                    sessions.add(payload.get("session_id", payload.get("id")))
                    cwds.add(payload.get("cwd"))
                elif (
                    record["type"] == "turn_context"
                    and isinstance(payload, dict)
                    and "cwd" in payload
                ):
                    cwds.add(payload["cwd"])
            else:
                if "sessionId" in record:
                    sessions.add(record["sessionId"])
                if "cwd" in record:
                    cwds.add(record["cwd"])
                if record["type"] in {"user", "assistant"} and (
                    "sessionId" not in record or "cwd" not in record
                ):
                    raise _CaptureError("native_identity_missing", retryable=True)
        if sessions != {session} or not cwds or None in cwds or len(cwds) > 32:
            raise _CaptureError("native_identity_mismatch")
        for cwd in cwds:
            self._deadline(deadline)
            if not isinstance(cwd, str):
                raise _CaptureError("invalid_native_cwd")
            working = Path(cwd)
            if not working.is_absolute() or not working.is_relative_to(repository):
                raise _CaptureError("native_repository_mismatch")
            _path(working)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise _CaptureError("capture_budget_exhausted", retryable=True)
            try:
                actual = (
                    _git(working, "rev-parse", "--show-toplevel", timeout=min(5, remaining))
                    .decode()
                    .strip()
                )
            except subprocess.TimeoutExpired as error:
                raise _CaptureError("capture_budget_exhausted", retryable=True) from error
            self._deadline(deadline)
            if Path(actual).resolve() != repository:
                raise _CaptureError("native_repository_mismatch")
        return sorted(cwds)

    def _complete(self, claim, *, status, error=None, provenance=None, job=None):
        timestamp = now()
        ready = (datetime.now(UTC) + timedelta(seconds=RETRY_SECONDS)).isoformat()
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute(
                """UPDATE native_capture_requests SET
                status=CASE WHEN generation!=claimed_generation THEN 'queued' ELSE ? END,
                attempts=CASE WHEN generation!=claimed_generation THEN 0 ELSE attempts END,
                error=?,provenance_json=COALESCE(?,provenance_json),
                last_source_id=COALESCE(?,last_source_id),last_job_id=COALESCE(?,last_job_id),
                captured_generation=COALESCE(?,captured_generation),
                ready_at=CASE WHEN generation!=claimed_generation THEN ? ELSE ? END,
                claim_token=NULL,lease_until=NULL,updated_at=?
                WHERE client_name=? AND session_id=? AND claim_token=?""",
                (
                    status,
                    error,
                    json.dumps(provenance) if provenance is not None else None,
                    job["source_id"] if job else None,
                    job["id"] if job else None,
                    claim["claimed_generation"] if job else None,
                    timestamp,
                    ready,
                    timestamp,
                    claim["client_name"],
                    claim["session_id"],
                    claim["claim_token"],
                ),
            )
        return self.status(claim["client_name"], claim["session_id"], actor_role="owner")

    def _claim(self):
        with defer_busy_claim():
            self.recover()
            timestamp = now()
            token = uuid4().hex
            lease = (datetime.now(UTC) + timedelta(seconds=LEASE_SECONDS)).isoformat()
            with self.vault.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                row = db.execute(
                    "SELECT * FROM native_capture_requests WHERE status IN ('queued','deferred') AND ready_at<=? ORDER BY updated_at,client_name,session_id LIMIT 1",
                    (timestamp,),
                ).fetchone()
                if row is None:
                    return None
                db.execute(
                    """UPDATE native_capture_requests SET status='running',claimed_generation=generation,
                    claim_token=?,lease_until=?,attempts=attempts+1,updated_at=?
                    WHERE client_name=? AND session_id=?""",
                    (token, lease, timestamp, row["client_name"], row["session_id"]),
                )
                return dict(
                    db.execute(
                        "SELECT * FROM native_capture_requests WHERE client_name=? AND session_id=?",
                        (row["client_name"], row["session_id"]),
                    ).fetchone()
                )
        return None

    def run_next(self):
        """Execute one private capture request, never a public/private read on behalf of a client."""
        if not self.enabled or (claim := self._claim()) is None:
            return None
        token = claim["claim_token"]
        try:
            client, session = claim["client_name"], claim["session_id"]
            repository = self.repositories.get(claim["repository"])
            if repository is None or client not in self.roots:
                raise _CaptureError("capture_no_longer_authorised")
            self._check_anchor(repository)
            deadline = time.monotonic() + MAX_CAPTURE_SECONDS
            origin = self._discover(client, session, deadline)
            data = self._snapshot(origin, deadline)
            digest = hashlib.sha256(data).hexdigest()
            identity = hashlib.sha256((client + ":" + session).encode()).hexdigest()
            staged = self.staging / identity / (digest + ".jsonl")
            self._check_anchor(self.staging)
            _path(staged, exists=False)
            if staged.exists():
                if (
                    staged.stat().st_size > MAX_BYTES
                    or hashlib.sha256(staged.read_bytes()).hexdigest() != digest
                ):
                    raise _CaptureError("staged_original_integrity_failed")
            else:
                from .resources import admit_storage

                with self.vault.connect() as db:
                    db.execute("BEGIN IMMEDIATE")
                    admit_storage(
                        self.vault.home,
                        len(data) + 4 * 1024**2,
                        additional_entries=6,
                        temporary_bytes=len(data),
                        temporary_entries=4,
                    )
                    atomic_bytes(staged, data)
            provenance = {
                "client_name": client,
                "session_id": session,
                "repository": str(repository),
                "source_path": str(origin),
                "source_sha256": digest,
                "source_bytes": len(data),
                "staging_path": str(staged),
                "cwd": [],
                "captured_at": now(),
            }
            with self.vault.connect() as db:
                db.execute(
                    "UPDATE native_capture_requests SET provenance_json=? WHERE client_name=? AND session_id=? AND claim_token=?",
                    (json.dumps(provenance), client, session, token),
                )
            provenance["cwd"] = self._validate(data, client, session, repository, deadline)
            self._check_anchor(repository)
            owner = "native-" + identity
            with self.vault.connect() as db:
                live = db.execute(
                    "SELECT claim_token,lease_until FROM native_capture_requests WHERE client_name=? AND session_id=?",
                    (client, session),
                ).fetchone()
                if live["claim_token"] != token or live["lease_until"] <= now():
                    return self.status(client, session, actor_role="owner")
                sources = db.execute(
                    "SELECT id FROM sources WHERE owner=? AND scope='private' AND kind='conversation'",
                    (owner,),
                ).fetchall()
            if len(sources) > 1:
                raise _CaptureError("ambiguous_native_source_identity")
            source_id = sources[0]["id"] if sources else None
            self._deadline(deadline)
            job = self.registry.submit(
                staged,
                kind="conversation",
                scope="private",
                collection="native-" + client,
                project=hashlib.sha256(str(repository).encode()).hexdigest(),
                source_id=source_id,
                parser=client + "-jsonl-v1",
                owner=owner,
                capture_method="native_snapshot",
                assertion_type="historical_transcript",
                native_capture_claim=(client, session, token),
            )
            provenance["registry_original"] = job["original"]
            provenance["source_id"], provenance["job_id"] = job["source_id"], job["id"]
            return self._complete(claim, status="captured", provenance=provenance, job=job)
        except _CaptureError as error:
            retry = error.retryable and claim["attempts"] < MAX_ATTEMPTS
            return self._complete(
                claim, status="deferred" if retry else "failed", error=error.reason
            )
        except OSError:
            return self._complete(
                claim,
                status="deferred" if claim["attempts"] < MAX_ATTEMPTS else "failed",
                error="local_source_unavailable",
            )
        except (ValueError, TypeError, sqlite3.DatabaseError):
            return self._complete(claim, status="failed", error="capture_boundary_failed")
