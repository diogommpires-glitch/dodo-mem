"""Owner-reviewed, provenance-scoped purge with an external revocation authority."""

import hashlib
import json
import os
import re
import sqlite3
import subprocess
from contextlib import ExitStack, closing, contextmanager
from datetime import UTC, datetime
from pathlib import Path, PurePosixPath
from uuid import UUID, uuid4

from .source_registry import atomic_bytes, single_line
from .store import metadata, now, writer_lock


class PurgeBlocked(ValueError):
    """The reviewed purge still has unresolved provenance or retention dependencies."""


def _hash(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode()
    ).hexdigest()


def _file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _json_descriptor(path, limit=4 * 1024**2):
    with path.open("rb") as stream:
        content = stream.read(limit + 1)
    if len(content) > limit:
        raise PurgeBlocked("Provenance descriptor exceeds its bounded read budget")
    return json.loads(content)


def _authority_bytes(ledger):
    ledger["sha256"] = _hash({key: value for key, value in ledger.items() if key != "sha256"})
    return json.dumps(ledger, indent=2, sort_keys=True).encode("utf-8")


def _owner(actor_role, scopes):
    if actor_role != "owner" or set(scopes) != {"shared", "private"}:
        raise PermissionError("Purge administration requires authenticated owner and private scope")


def _has_table(db, name):
    return (
        db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (name,)).fetchone()
        is not None
    )


def _canonical_paths(record):
    return sorted(
        set(record.get("canonical_paths", []))
        | {row["path"] for row in record.get("canonical_views", [])}
        | {
            row["path"].removeprefix("vault/")
            for row in record["files"]
            if row["path"].startswith("vault/")
        }
    )


def _vector_inventory(db, paths):
    digest, count = hashlib.sha256(), 0
    if _has_table(db, "embedding_vectors"):
        for path in sorted(set(paths)):
            for row in db.execute(
                """SELECT space_id,path,document_sha256,chunk_sha256,
                window_index,window_sha256,vector FROM embedding_vectors WHERE path=?
                ORDER BY space_id,document_sha256,chunk_sha256,window_index""",
                (path,),
            ):
                value = [*row[:-1], hashlib.sha256(row[-1]).hexdigest()]
                digest.update(json.dumps(value, separators=(",", ":")).encode())
                count += 1
    return {"rows": count, "manifest_sha256": digest.hexdigest()}


def native_session_identity(client_name, session_id):
    if client_name not in {"codex", "claude"} or str(UUID(session_id)) != session_id:
        raise ValueError("Native revocation requires a canonical supported client/session identity")
    return hashlib.sha256(f"{client_name}:{session_id}".encode()).hexdigest()


def _metadata_file(path):
    with path.open(encoding="utf-8") as stream:
        if stream.readline(4097) != "---\n":
            return {}
        lines, count = [], 0
        while line := stream.readline(128 * 1024 - count + 1):
            if line == "---\n":
                return metadata("---\n" + "".join(lines) + "---\n")[0]
            count += len(line)
            if count > 128 * 1024:
                raise ValueError("Owned record frontmatter exceeds 128 KiB")
            lines.append(line)
    raise ValueError("Owned record frontmatter is truncated")


def _time(value):
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        raise ValueError("Retention expiry must include an explicit timezone")
    return parsed.astimezone(UTC).isoformat()


def load_authority(path):
    path = Path(path).absolute()
    if path.resolve(strict=True) != path or not path.is_file():
        raise ValueError("Revocation authority must be a real local file without redirects")
    ledger = json.loads(path.read_text(encoding="utf-8"))
    if (
        ledger.get("format") != 1
        or not isinstance(ledger.get("memory_id"), str)
        or not re.fullmatch(r"[a-f0-9]{32}", ledger.get("memory_id", ""))
        or not isinstance(ledger.get("events"), list)
        or ledger.get("sha256")
        != _hash({key: value for key, value in ledger.items() if key != "sha256"})
    ):
        raise ValueError("Invalid revocation authority format or memory identity")
    previous = None
    for sequence, event in enumerate(ledger["events"], 1):
        payload = {key: value for key, value in event.items() if key != "sha256"}
        if (
            event["sequence"] != sequence
            or event["previous"] != previous
            or _hash(payload) != event["sha256"]
        ):
            raise ValueError("Revocation authority hash chain is inconsistent")
        previous = event["sha256"]
    return ledger


def assert_evaluation_job_allowed(home, destination):
    """Offline workers call before reading inputs and again inside their root job lock."""
    home, destination = Path(home).absolute(), Path(destination).absolute()
    base = home / ".state/private-evaluation"
    if (
        destination == base
        or not destination.is_relative_to(base)
        or destination.resolve() != destination
    ):
        raise PurgeBlocked("Evaluation job is outside its canonical owner namespace")
    relative = destination.relative_to(home).as_posix()
    database = home / ".state/memory.sqlite"
    if database.resolve(strict=True) != database:
        raise PurgeBlocked("Evaluation authority database is redirected")
    revoked = set()
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True)) as db:
        if _has_table(db, "purge_denials"):
            revoked.update(
                row[0]
                for row in db.execute("SELECT value FROM purge_denials WHERE kind='evaluation_job'")
            )
        if _has_table(db, "retention_meta"):
            identity, authority, initialized = db.execute(
                "SELECT memory_id,authority,initialized FROM retention_meta WHERE id=1"
            ).fetchone()
            if initialized:
                ledger = load_authority(authority)
                if ledger["memory_id"] != identity:
                    raise PurgeBlocked("Evaluation revocation authority belongs to another memory")
                if _has_table(db, "purge_events"):
                    registered = {
                        row[0]
                        for row in db.execute(
                            "SELECT plan_hash FROM purge_events WHERE state IN ('revoked','complete')"
                        )
                    }
                    if not registered <= {event["plan_hash"] for event in ledger["events"]}:
                        raise PurgeBlocked(
                            "Evaluation revocation authority lost a registered event"
                        )
                revoked.update(
                    path
                    for event in ledger["events"]
                    for path in event["revocation"].get("evaluation_jobs", [])
                )
    if any(relative == value or relative.startswith(value + "/") for value in revoked):
        raise PurgeBlocked("Private evaluation job has been revoked")


def _retained_backup_paths(db, digest, allowed_sources):
    """Only exact registered aliases prove the surviving provenance exception."""
    if not allowed_sources or not _has_table(db, "source_versions"):
        return set()
    placeholders = ",".join("?" for _ in allowed_sources)
    retained = set()
    for original, owner, scope, kind, method in db.execute(
        f"SELECT v.original,s.owner,s.scope,s.kind,s.capture_method FROM source_versions v JOIN sources s ON s.id=v.source_id WHERE v.sha256=? AND v.source_id IN ({placeholders})",
        (digest, *allowed_sources),
    ):
        retained.add(original)
        if (
            scope == "private"
            and kind == "conversation"
            and method == "native_snapshot"
            and re.fullmatch(r"native-[a-f0-9]{64}", owner)
        ):
            retained.add(f".state/native-capture/{owner.removeprefix('native-')}/{digest}.jsonl")
    if _has_table(db, "ingest_jobs"):
        retained.update(
            "vault/" + row[0]
            for row in db.execute(
                f"SELECT path FROM ingest_jobs WHERE source_id IN ({placeholders}) AND view_sha256=? AND status='indexed' AND path IS NOT NULL",
                (*allowed_sources, digest),
            )
        )
    if _has_table(db, "retention_copies"):
        retained.update(
            row[0]
            for row in db.execute(
                f"SELECT c.path FROM retention_copies c JOIN source_versions v ON v.source_id=c.source_id AND v.sha256=c.source_version WHERE c.source_id IN ({placeholders}) AND c.sha256=?",
                (*allowed_sources, digest),
            )
        )
    if _has_table(db, "source_artifacts"):
        retained.update(
            row[0]
            for row in db.execute(
                f"SELECT a.path FROM source_artifacts a JOIN source_versions v ON v.source_id=a.source_id AND v.sha256=a.source_version JOIN ingest_jobs j ON j.id=a.job_id WHERE a.source_id IN ({placeholders}) AND a.sha256=? AND j.status='indexed'",
                (*allowed_sources, digest),
            )
        )
    return retained


def _forbidden_backup_aliases(db, manifest, hashes):
    retained = {
        digest: _retained_backup_paths(db, digest, allowed) for digest, allowed in hashes.items()
    }
    return sorted(
        name
        for name, record in manifest["files"].items()
        if record["sha256"] in hashes and name not in retained[record["sha256"]]
    )


def assert_restore_allowed(backup, authority):
    """Fail closed unless a snapshot contains the current completed revocation generation."""
    ledger = load_authority(authority)
    backup = Path(backup).absolute()
    manifest_path = backup / "manifest.json"
    database = backup / ".state/memory.sqlite"
    if (
        backup.resolve(strict=True) != backup
        or manifest_path.resolve(strict=True) != manifest_path
        or database.resolve(strict=True) != database
    ):
        raise PurgeBlocked("Restore backup or database contains a redirect")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = manifest.get("files", {}).get(".state/memory.sqlite", {})
    if _file_hash(database) != expected.get("sha256"):
        raise PurgeBlocked("Restore database does not match its backup inventory")
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)) as db:
        if not _has_table(db, "retention_meta"):
            raise PurgeBlocked("Legacy backup lacks verified retention identity; quarantine only")
        row = db.execute("SELECT memory_id FROM retention_meta WHERE id=1").fetchone()
        if row is None or row[0] != ledger["memory_id"]:
            raise PurgeBlocked("Backup belongs to a different or unverified memory identity")
        completed = dict(db.execute("SELECT plan_hash,state FROM purge_events"))
        if any(state != "complete" for state in completed.values()):
            raise PurgeBlocked("Backup contains an unfinished purge; restore only into quarantine")
        for event in ledger["events"]:
            if completed.get(event["plan_hash"]) != "complete":
                raise PurgeBlocked(
                    "Backup predates a completed revocation; stale restore is forbidden"
                )
            revoked = event["revocation"]
            if _has_table(db, "sources") and any(
                db.execute("SELECT 1 FROM sources WHERE id=?", (identity,)).fetchone()
                for identity in revoked["source_ids"]
            ):
                raise PurgeBlocked("Backup still contains a revoked source identity")
            for version in revoked["versions"]:
                if (
                    _has_table(db, "source_versions")
                    and db.execute(
                        "SELECT 1 FROM source_versions WHERE source_id=? AND sha256=?",
                        (version["source_id"], version["sha256"]),
                    ).fetchone()
                ):
                    raise PurgeBlocked("Backup still contains a revoked source version")
            deleted = {(record["path"], record["sha256"]) for record in revoked["files"]}
            canonical_paths = _canonical_paths(revoked)
            for table, label in (
                ("indexed_documents", "routing metadata"),
                ("chunks", "cached plaintext"),
                ("embedding_vectors", "vector data"),
                ("ordinary_note_revisions", "ordinary note revisions"),
                ("ordinary_note_pending", "pending note saves"),
            ):
                if _has_table(db, table) and any(
                    db.execute(f"SELECT 1 FROM {table} WHERE path=?", (path,)).fetchone()
                    for path in canonical_paths
                ):
                    raise PurgeBlocked("Backup still contains revoked " + label)
            if any("vault/" + path in manifest["files"] for path in canonical_paths):
                raise PurgeBlocked("Backup inventory still retains a revoked canonical path")
            if any(
                (name, record["sha256"]) in deleted for name, record in manifest["files"].items()
            ):
                raise PurgeBlocked("Backup inventory still retains purged canonical content")
            if _forbidden_backup_aliases(db, manifest, revoked["hashes"]):
                raise PurgeBlocked("Backup inventory retains purged bytes under an unproved alias")
    return {
        "allowed": True,
        "memory_id": ledger["memory_id"],
        "revocation_generation": len(ledger["events"]),
    }


class Retention:
    """Gateway authenticates the owner and holds the vault's single mutation gate."""

    def __init__(self, vault, *, authority, backup_roots):
        self.vault = vault
        self.authority = Path(authority).absolute()
        self.backup_roots = tuple(Path(path).absolute() for path in backup_roots)
        if (
            self.authority.resolve() != self.authority
            or self.authority.is_relative_to(vault.home)
            or any(
                self.authority.is_relative_to(path)
                or path.is_relative_to(vault.home)
                or path.resolve() != path
                for path in self.backup_roots
            )
        ):
            raise ValueError(
                "Authority must stay outside memory home/backups; backup roots must be explicit and external"
            )
        with vault.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS retention_meta (id INTEGER PRIMARY KEY CHECK(id=1), memory_id TEXT NOT NULL, authority TEXT NOT NULL, initialized INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS purge_events (plan_hash TEXT PRIMARY KEY, state TEXT NOT NULL, plan_json TEXT NOT NULL, reviewed_by TEXT NOT NULL, recorded_at TEXT NOT NULL, report_json TEXT);
                CREATE TABLE IF NOT EXISTS purge_denials (kind TEXT NOT NULL,value TEXT NOT NULL,plan_hash TEXT NOT NULL,PRIMARY KEY(kind,value,plan_hash));
                CREATE TABLE IF NOT EXISTS retention_copies (path TEXT NOT NULL,source_id TEXT NOT NULL,source_version TEXT NOT NULL,sha256 TEXT NOT NULL,kind TEXT NOT NULL,PRIMARY KEY(path,source_id,source_version));
            """)
            if "initialized" not in {
                row[1] for row in db.execute("PRAGMA table_info(retention_meta)")
            }:
                db.execute(
                    "ALTER TABLE retention_meta ADD COLUMN initialized INTEGER NOT NULL DEFAULT 1"
                )
            db.execute(
                "INSERT OR IGNORE INTO retention_meta VALUES (1,?,?,0)",
                (uuid4().hex, str(self.authority)),
            )
            row = db.execute("SELECT * FROM retention_meta WHERE id=1").fetchone()
            if row["authority"] != str(self.authority):
                raise ValueError("Memory already names a different revocation authority")
            self.memory_id = row["memory_id"]
        if self.authority.exists():
            self.synchronize_authority()
        vault.retention = self

    def initialize_authority(self, *, actor_role, scopes):
        _owner(actor_role, scopes)
        if self.authority.exists():
            ledger = self._ledger()
        else:
            with self.vault.connect() as db:
                initialized = db.execute(
                    "SELECT initialized FROM retention_meta WHERE id=1"
                ).fetchone()[0]
                if initialized or db.execute("SELECT 1 FROM purge_events LIMIT 1").fetchone():
                    raise PurgeBlocked(
                        "Missing authority cannot be reset after a purge; recover the original ledger"
                    )
            ledger = {"format": 1, "memory_id": self.memory_id, "events": []}
            atomic_bytes(self.authority, _authority_bytes(ledger))
        with self.vault.connect() as db:
            db.execute("UPDATE retention_meta SET initialized=1 WHERE id=1")
        return {"memory_id": self.memory_id, "generation": len(ledger["events"])}

    def _ledger(self):
        ledger = load_authority(self.authority)
        if ledger["memory_id"] != self.memory_id:
            raise PurgeBlocked("Revocation authority belongs to another memory identity")
        with self.vault.connect() as db:
            registered = {
                row[0]
                for row in db.execute(
                    "SELECT plan_hash FROM purge_events WHERE state IN ('revoked','complete')"
                )
            }
        if not registered <= {event["plan_hash"] for event in ledger["events"]}:
            raise PurgeBlocked("Revocation authority was rolled back or lost a registered event")
        return ledger

    def synchronize_authority(self):
        """Rebuild revocation metadata without deleting bytes or inventing completed jobs."""
        ledger = self._ledger()
        with self.vault.connect() as db:
            for event in ledger["events"]:
                revoked = event["revocation"]
                denials = [("source", value) for value in revoked["source_ids"]]
                denials += [
                    ("version", row["source_id"] + ":" + row["sha256"])
                    for row in revoked["versions"]
                ]
                denials += [("knowledge", value) for value in revoked["knowledge_keys"]]
                denials += [
                    ("native_session", value) for value in revoked.get("native_sessions", [])
                ]
                denials += [
                    ("evaluation_job", value) for value in revoked.get("evaluation_jobs", [])
                ]
                denials += [("path", path) for path in _canonical_paths(revoked)]
                for kind, value in denials:
                    db.execute(
                        "INSERT OR IGNORE INTO purge_denials VALUES (?,?,?)",
                        (kind, value, event["plan_hash"]),
                    )
        return {"memory_id": self.memory_id, "generation": len(ledger["events"])}

    def _owned(self, relative):
        value = PurePosixPath(relative)
        if (
            value.is_absolute()
            or value.as_posix() != relative
            or ".." in value.parts
            or "\\" in relative
            or ":" in relative
        ):
            raise ValueError("Purge files require canonical memory-relative paths")
        path = self.vault.home / relative
        if path.resolve() != path or not path.is_relative_to(self.vault.home):
            raise ValueError("Purge target escapes memory home or contains a redirect")
        return path

    def register_derived_copy(self, relative, *, source_versions, kind):
        if not source_versions or len(source_versions) > 100:
            raise ValueError("Copy registration requires 1 to 100 source-version references")
        if kind not in {"cache", "evaluation"} or not relative.startswith(
            (".state/cache/", ".state/evaluation/", ".state/private-evaluation/")
        ):
            raise ValueError(
                "Registered copies must be cache/evaluation files in their owned .state namespace"
            )
        self.assert_path_allowed(relative)
        path = self._owned(relative)
        digest = _file_hash(path)
        with self.vault.connect() as db:
            for source_id, version in source_versions:
                if not db.execute(
                    "SELECT 1 FROM source_versions WHERE source_id=? AND sha256=?",
                    (source_id, version),
                ).fetchone():
                    raise ValueError("Copy provenance must name a retained source version")
                self.assert_import_allowed(source_id=source_id, origin=None, sha256=version)
                db.execute(
                    "INSERT OR REPLACE INTO retention_copies VALUES (?,?,?,?,?)",
                    (relative, source_id, version, digest, kind),
                )
        return {"path": relative, "sha256": digest}

    def assert_import_allowed(self, *, source_id, origin, sha256):
        ledger = self._ledger()
        revocations = [event["revocation"] for event in ledger["events"]]
        with self.vault.connect() as db:
            for row in db.execute("SELECT plan_json FROM purge_events WHERE state='planned'"):
                plan = json.loads(row[0])
                revocations.append(
                    {
                        "source_ids": plan["source_ids"],
                        "locations": [row["path"] for row in plan["locations"]],
                        "hashes": plan["hash_exceptions"],
                    }
                )
        for revoked in revocations:
            if (
                source_id in revoked["source_ids"]
                or origin is not None
                and os.path.normcase(str(Path(origin).absolute()))
                in {os.path.normcase(value) for value in revoked["locations"]}
            ):
                raise PurgeBlocked("Source identity or intake location has been revoked")
            if sha256 in revoked["hashes"] and source_id not in revoked["hashes"][sha256]:
                raise PurgeBlocked("Purged bytes cannot be introduced as new provenance")
        with self.vault.connect() as db:
            if (
                source_id
                and db.execute(
                    "SELECT 1 FROM purge_denials WHERE kind='source' AND value=?", (source_id,)
                ).fetchone()
            ):
                raise PurgeBlocked("Source is quarantined by a durable purge job")

    def assert_path_allowed(self, relative):
        if any(
            record["path"] in {relative, "vault/" + relative}
            for event in self._ledger()["events"]
            for record in event["revocation"]["files"]
        ):
            raise PurgeBlocked("Canonical path has been revoked by the external authority")
        with self.vault.connect() as db:
            if db.execute(
                "SELECT 1 FROM purge_denials WHERE kind='path' AND value=?", (relative,)
            ).fetchone():
                raise PurgeBlocked("Canonical path has been revoked")

    def assert_knowledge_allowed(self, entry_key):
        if any(
            entry_key in event["revocation"]["knowledge_keys"] for event in self._ledger()["events"]
        ):
            raise PurgeBlocked("Working knowledge identity has been withdrawn")
        with self.vault.connect() as db:
            if db.execute(
                "SELECT 1 FROM purge_denials WHERE kind='knowledge' AND value=?", (entry_key,)
            ).fetchone():
                raise PurgeBlocked("Working knowledge identity has been withdrawn")

    def assert_native_session_allowed(self, client_name, session_id, *, db=None):
        identity = native_session_identity(client_name, session_id)
        if any(
            identity in event["revocation"].get("native_sessions", [])
            for event in self._ledger()["events"]
        ):
            raise PurgeBlocked("Native capture session has been revoked")
        if db is None:
            with self.vault.connect() as connection:
                self.assert_native_session_allowed(client_name, session_id, db=connection)
                return
        if db.execute(
            "SELECT 1 FROM purge_denials WHERE kind='native_session' AND value=?", (identity,)
        ).fetchone():
            raise PurgeBlocked("Native capture session is quarantined by a durable purge job")

    def _evaluation_dependencies(self, selected, add, hard):
        root = self._owned(".state/private-evaluation")
        if not root.is_dir():
            return [], [], set(), set()
        paths = []
        for path in root.rglob("*"):
            self._owned(path.relative_to(self.vault.home).as_posix())
            paths.append(path)
            if len(paths) > 20000:
                raise PurgeBlocked("Private evaluation inventory exceeds 20000 entries")
        files = [path for path in paths if path.is_file()]
        descriptors = {
            path.parent
            for path in files
            if path.name == "building.json"
            or path.name == "receipt.json"
            and (path.parent / "manifest.json").is_file()
        }
        recognized = {}
        for directory in sorted(descriptors):
            building, manifest = directory / "building.json", directory / "manifest.json"
            receipt = directory / "receipt.json"
            descriptor = None
            if building.is_file():
                candidate = _json_descriptor(building)
                if candidate.get("kind") == "owner-private-evaluation-building":
                    if candidate.get("sha256") != _hash(
                        {key: value for key, value in candidate.items() if key != "sha256"}
                    ):
                        hard.append("private_evaluation_building_integrity_mismatch")
                        continue
                    descriptor = candidate
            if manifest.is_file() and receipt.is_file():
                candidate, sealed = _json_descriptor(manifest), _json_descriptor(receipt)
                if candidate.get("kind") == "owner-private-evaluation":
                    if (
                        _hash(candidate) != sealed.get("manifest_sha256")
                        or sealed.get("job_id") != directory.name
                    ):
                        hard.append("private_evaluation_manifest_integrity_mismatch")
                        continue
                    if (
                        descriptor is not None
                        and descriptor["source_versions"] != candidate["source_versions"]
                    ):
                        hard.append("private_evaluation_descriptor_provenance_mismatch")
                        continue
                    descriptor = candidate
            if descriptor is None:
                continue
            edges = descriptor.get("source_versions", [])
            if (
                descriptor.get("scope") != "private"
                or descriptor.get("job_id") != directory.name
                or descriptor.get("retention_unit") != "whole-owner-job"
                or not isinstance(edges, list)
                or not 1 <= len(edges) <= 60
                or any(
                    not isinstance(edge, dict)
                    or not re.fullmatch(r"[a-f0-9]{32}", edge.get("source_id", ""))
                    or not re.fullmatch(r"[a-f0-9]{64}", edge.get("source_version", ""))
                    for edge in edges
                )
            ):
                hard.append("private_evaluation_provenance_unverified")
                continue
            recognized[directory] = {(edge["source_id"], edge["source_version"]) for edge in edges}
        if any(
            other != directory and other.is_relative_to(directory)
            for directory in recognized
            for other in recognized
        ):
            hard.append("overlapping_private_evaluation_owner_jobs")
        jobs, locks, known = [], [], set()
        for directory, edges in recognized.items():
            members = [path for path in files if path.is_relative_to(directory)]
            known.update(path.relative_to(self.vault.home).as_posix() for path in members)
            if not edges & selected:
                continue
            relative = directory.relative_to(self.vault.home).as_posix()
            jobs.append(relative)
            root_lock = directory / ".state/writer.lock"
            if not root_lock.is_file():
                hard.append("private_evaluation_root_lock_missing")
            for path in members:
                name = path.relative_to(self.vault.home).as_posix()
                if path.name == "writer.lock" and path.parent.name == ".state":
                    if path.stat().st_size:
                        hard.append("private_evaluation_coordination_lock_has_content")
                    locks.append(name)
                else:
                    add(name, "dependent_private_evaluation_owner_job")
        return (
            sorted(jobs),
            sorted(set(locks)),
            known,
            {path.relative_to(self.vault.home).as_posix() for path in files},
        )

    @contextmanager
    def _evaluation_locks(self, plan):
        with ExitStack() as stack:
            for relative in plan.get("evaluation_locks", []):
                path = self._owned(relative)
                if (
                    not relative.startswith(".state/private-evaluation/")
                    or path.name != "writer.lock"
                    or path.parent.name != ".state"
                    or not path.is_file()
                    or path.stat().st_size
                ):
                    raise PurgeBlocked("Reviewed evaluation coordination lock changed")
                try:
                    stack.enter_context(writer_lock(path.parent.parent))
                except RuntimeError as error:
                    raise PurgeBlocked(
                        "A private evaluation worker is active; stop it before purge"
                    ) from error
            yield

    def _git_copies(self, files):
        result = subprocess.run(
            ["git", "-C", str(self.vault.home), "rev-parse", "--git-dir"],
            capture_output=True,
            check=False,
            timeout=10,
        )
        if result.returncode == 128:
            if any(
                (path / ".git").exists() for path in (self.vault.home, *self.vault.home.parents)
            ):
                raise PurgeBlocked("Git exists but its object retention could not be verified")
            return 0
        if result.returncode:
            raise PurgeBlocked("Unable to verify Git retention")
        paths = "".join(record["path"] + "\n" for record in files)
        hashed = subprocess.run(
            ["git", "-C", str(self.vault.home), "hash-object", "--no-filters", "--stdin-paths"],
            input=paths,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        objects = subprocess.run(
            ["git", "-C", str(self.vault.home), "cat-file", "--batch-check=%(objecttype)"],
            input=hashed.stdout,
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
        return sum(line == "blob" for line in objects.stdout.splitlines())

    def preview(self, source_id, *, actor_role, scopes, mode="source", backup_retention=None):
        _owner(actor_role, scopes)
        ledger = self._ledger()
        if mode not in {"source", "source_hash_all"}:
            raise ValueError("Purge mode must be source or explicit source_hash_all")
        backup_retention = backup_retention or {}
        files, retained, hard = {}, [], []
        canonical_refs = {}
        with self.vault.connect() as db:
            source = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            if source is None or source["scope"] not in scopes:
                raise ValueError("Source is unavailable for these owner scope grants")
            initial = db.execute(
                "SELECT * FROM source_versions WHERE source_id=?", (source_id,)
            ).fetchall()
            hashes = {row["sha256"] for row in initial}
            versions = [
                dict(row)
                for row in db.execute("SELECT * FROM source_versions ORDER BY source_id,sha256")
                if row["source_id"] == source_id
                or mode == "source_hash_all"
                and row["sha256"] in hashes
            ]
            selected = {(row["source_id"], row["sha256"]) for row in versions}
            all_versions = [dict(row) for row in db.execute("SELECT * FROM source_versions")]
            removed_sources = sorted(
                {
                    row["source_id"]
                    for row in versions
                    if not any(
                        other["source_id"] == row["source_id"]
                        and (other["source_id"], other["sha256"]) not in selected
                        for other in all_versions
                    )
                }
            )
            locations = [
                dict(row)
                for row in db.execute("SELECT * FROM source_locations ORDER BY path")
                if row["source_id"] in removed_sources
            ]
            jobs = [
                dict(row)
                for row in db.execute("SELECT * FROM ingest_jobs ORDER BY id")
                if (row["source_id"], row["sha256"]) in selected
            ]
            source_scopes = {
                row["id"]: row["scope"] for row in db.execute("SELECT id,scope FROM sources")
            }
            job_ids = {row["id"] for row in jobs}
            requests = [
                dict(row)
                for row in db.execute("SELECT * FROM import_requests ORDER BY owner,request_id")
                if row["job_id"] in job_ids
            ]
            artifacts = (
                [
                    dict(row)
                    for row in db.execute("SELECT * FROM source_artifacts ORDER BY path")
                    if row["job_id"] in job_ids
                ]
                if _has_table(db, "source_artifacts")
                else []
            )
            if any(row["status"] in {"running", "parsing"} for row in jobs):
                hard.append("source_worker_running")

            def add(relative, reason, expected=None):
                path = self._owned(relative)
                if (
                    relative.startswith(("vault/core/", "generated/"))
                    or relative == ".state/memory.sqlite"
                ):
                    hard.append("protected_core_or_database_copy")
                    return
                allowed = (
                    "vault/documents/",
                    "vault/private/",
                    "vault/knowledge/",
                    "vault/inbox/",
                    "vault/episodes/",
                    "originals/",
                    "derived/",
                    ".state/cache/",
                    ".state/evaluation/",
                    ".state/private-evaluation/",
                    ".state/native-capture/",
                    ".state/ingest-logs/",
                    ".state/results/",
                )
                if not relative.startswith(allowed):
                    raise ValueError("Purge target is outside owned source/derived namespaces")
                if relative.startswith("vault/"):
                    canonical = relative.removeprefix("vault/")
                    prior = canonical_refs.get(canonical)
                    if prior and expected and prior != expected:
                        hard.append("canonical_view_identity_conflict")
                    canonical_refs[canonical] = expected or prior
                if path.is_file():
                    if len(files) >= 10000:
                        raise PurgeBlocked("Purge preview exceeds its 10000-file bounded inventory")
                    actual = _file_hash(path)
                    if relative.startswith("vault/") and canonical_refs[canonical] is None:
                        canonical_refs[canonical] = actual
                    if expected and actual != expected:
                        hard.append("registered_file_integrity_mismatch")
                    files[relative] = {
                        "path": relative,
                        "sha256": actual,
                        "bytes": path.stat().st_size,
                        "reason": reason,
                    }

            native_sessions = {
                row["owner"].removeprefix("native-")
                for row in db.execute("SELECT * FROM sources")
                if row["id"] in removed_sources
                and row["scope"] == "private"
                and row["kind"] == "conversation"
                and row["capture_method"] == "native_snapshot"
                and re.fullmatch(r"native-[a-f0-9]{64}", row["owner"])
            }
            if any(
                row["id"] not in removed_sources and row["owner"] == "native-" + identity
                for identity in native_sessions
                for row in db.execute("SELECT id,owner FROM sources")
            ):
                hard.append("native_session_has_retained_source_provenance")
            native_requests = []
            if _has_table(db, "native_capture_requests"):
                for row in db.execute(
                    "SELECT * FROM native_capture_requests ORDER BY client_name,session_id"
                ):
                    identity = native_session_identity(row["client_name"], row["session_id"])
                    if row["last_source_id"] in removed_sources or identity in native_sessions:
                        native_requests.append(dict(row))
                        native_sessions.add(identity)
                        if row["status"] == "running":
                            hard.append("native_capture_worker_running")
            for identity in native_sessions:
                folder = self._owned(f".state/native-capture/{identity}")
                if folder.is_dir():
                    for path in folder.iterdir():
                        relative = path.relative_to(self.vault.home).as_posix()
                        self._owned(relative)
                        if (
                            not re.fullmatch(r"[a-f0-9]{64}\.jsonl", path.name)
                            or not path.is_file()
                        ):
                            hard.append("unrecognized_owned_native_staging_entry")
                            continue
                        add(relative, "revoked_native_session_snapshot", path.stem)
            for version in versions:
                origin = Path(version["origin"]).absolute()
                if origin.is_relative_to(self.vault.home / ".state/native-capture"):
                    add(
                        origin.relative_to(self.vault.home).as_posix(),
                        "source_native_snapshot",
                        version["sha256"],
                    )

            for artifact in artifacts:
                add(artifact["path"], "registered_parser_artifact", artifact["sha256"])
            for job in jobs:
                if job["status"] == "indexed" and (
                    not job["path"]
                    or not isinstance(job["view_sha256"], str)
                    or not re.fullmatch(r"[0-9a-f]{64}", job["view_sha256"])
                ):
                    hard.append("indexed_source_view_identity_unknown")
                prefix = "private" if source_scopes[job["source_id"]] == "private" else "documents"
                canonical = f"{prefix}/{job['source_id']}/{job['sha256']}/{job['parser']}.md"
                if job["path"] and job["path"] != canonical:
                    add("vault/" + job["path"], "source_view", job["view_sha256"])
                candidate = self._owned("vault/" + canonical)
                expected_fields = {
                    "source_id": job["source_id"],
                    "source_version": job["sha256"],
                    "parser": job["parser"],
                    "ingest_job_id": job["id"],
                    "source": "sha256:" + job["sha256"],
                }
                fields = _metadata_file(candidate) if candidate.is_file() else {}
                if candidate.exists() and (
                    not candidate.is_file()
                    or any(fields.get(name) != value for name, value in expected_fields.items())
                ):
                    hard.append("canonical_source_view_binding_mismatch")
                else:
                    # A rolled-back publication can leave this view without a stored path or SHA.
                    # Its observed hash identifies a deletion target, never historical parser truth.
                    expected = job["view_sha256"] if job["path"] == canonical else None
                    add("vault/" + canonical, "source_view", expected)
                add(f".state/ingest-logs/{job['id']}.log", "source_parse_log")
                folder = self._owned(f"derived/{job['source_id']}/{job['sha256']}/{job['parser']}")
                if folder.is_dir():
                    for path in folder.rglob("*"):
                        self._owned(path.relative_to(self.vault.home).as_posix())
                        if path.is_file():
                            add(path.relative_to(self.vault.home).as_posix(), "parser_artifact")
            legacy = []
            for path in self.vault.root.rglob("*.md"):
                relative = path.relative_to(self.vault.home).as_posix()
                self._owned(relative)
                fields = _metadata_file(path)
                digest = fields.get("source", "").removeprefix("sha256:")
                if digest in hashes and not fields.get("source_id"):
                    if mode == "source_hash_all":
                        add(relative, "legacy_source_view")
                    else:
                        legacy.append({"path": relative, "sha256": digest})
            for version in versions:
                others = [
                    row["source_id"]
                    for row in all_versions
                    if row["sha256"] == version["sha256"]
                    and (row["source_id"], row["sha256"]) not in selected
                ]
                aliases = [row["path"] for row in legacy if row["sha256"] == version["sha256"]]
                if others or aliases:
                    retained.append(
                        {
                            "path": version["original"],
                            "sha256": version["sha256"],
                            "source_ids": sorted(set(others)),
                            "legacy_views": sorted(aliases),
                        }
                    )
                else:
                    add(version["original"], "unreferenced_original", version["sha256"])
            knowledge_keys, proposal_ids = set(), set()
            knowledge_records = []
            for table in ["accepted_knowledge_versions", "accepted_knowledge_proposals"]:
                if _has_table(db, table):
                    for row in db.execute(f"SELECT * FROM {table}"):
                        path = self.vault.resolve(row["path"])
                        if _file_hash(path) != row["sha256"]:
                            hard.append("knowledge_integrity_mismatch")
                        fields = _metadata_file(path)
                        info = json.loads(fields["review"])
                        knowledge_records.append((table, dict(row), info))
            changed = True
            while changed:
                changed = False
                evidence_records = {
                    (path, digest) for path, digest in canonical_refs.items() if digest is not None
                }
                for table, row, info in knowledge_records:
                    if row["entry_key"] in knowledge_keys or any(
                        (ref["path"], ref["sha256"]) in evidence_records for ref in info["evidence"]
                    ):
                        if row["entry_key"] not in knowledge_keys:
                            knowledge_keys.add(row["entry_key"])
                            changed = True
                        relative = "vault/" + row["path"]
                        if row["path"] not in canonical_refs:
                            add(relative, "dependent_working_knowledge", row["sha256"])
                            changed = True
                        if table == "accepted_knowledge_proposals":
                            proposal_ids.add(row["id"])
            copy_paths = []
            registered_copies = set()
            for row in db.execute("SELECT * FROM retention_copies ORDER BY path"):
                registered_copies.add(row["path"])
                if (row["source_id"], row["source_version"]) in selected:
                    add(row["path"], "registered_derived_copy", row["sha256"])
                    copy_paths.append(row["path"])
            evaluation_jobs, evaluation_locks, known_evaluation, all_evaluation = (
                self._evaluation_dependencies(selected, add, hard)
            )
            unregistered_evaluation = len(all_evaluation - known_evaluation - registered_copies)
            copy_paths.extend(
                name for name in known_evaluation if name in registered_copies and name in files
            )
            unregistered_native = 0
            native_root = self._owned(".state/native-capture")
            if native_root.is_dir():
                for path in native_root.rglob("*"):
                    relative = path.relative_to(self.vault.home).as_posix()
                    self._owned(relative)
                    if path.is_file() and relative not in files:
                        unregistered_native += 1
                        if unregistered_native > 10000:
                            raise PurgeBlocked("Unregistered native inventory exceeds 10000 files")
            checkpoints = []
            episode_root = self.vault.root / "episodes"
            episode_count = 0
            if episode_root.exists():
                for path in episode_root.glob("*.md"):
                    episode_count += 1
                    self._owned(path.relative_to(self.vault.home).as_posix())
                    fields = _metadata_file(path)
                    if (
                        fields.get("source_id") in removed_sources
                        or fields.get("source_version") in hashes
                        or fields.get("source_sha256") in hashes
                    ):
                        relative = path.relative_to(self.vault.home).as_posix()
                        add(relative, "provably_source_linked_checkpoint")
                        checkpoints.append(relative)
            peer_jobs = []
            if _has_table(db, "jobs"):
                source_paths = set(canonical_refs)
                if source_paths:
                    placeholders = ",".join("?" for _ in source_paths)
                    rows = db.execute(
                        f"SELECT DISTINCT j.id,j.status FROM jobs j,json_each(j.transcript) turn,json_each(turn.value,'$.memory_sources') src WHERE src.value IN ({placeholders})",
                        tuple(source_paths),
                    )
                    for row in rows:
                        peer_jobs.append(row["id"])
                        if row["status"] in {"running", "queued"}:
                            hard.append("source_peer_job_running")
                        add(f".state/results/{row['id']}.json", "source_linked_peer_result")
        files_list = sorted(files.values(), key=lambda row: row["path"])
        if mode == "source_hash_all":
            target_hashes = hashes | {record["sha256"] for record in files_list}
            protected = [self.vault.root / "core/CORE.md"]
            generated = self.vault.home / "generated"
            if generated.exists():
                protected.extend(path for path in generated.iterdir() if path.is_file())
            if any(path.is_file() and _file_hash(path) in target_hashes for path in protected):
                hard.append("protected_core_copy_requires_separate_review")
        if self._git_copies(files_list):
            hard.append("known_content_retained_in_git_objects")
        remaining_hashes = {
            digest: sorted(
                {
                    row["source_id"]
                    for row in all_versions
                    if row["sha256"] == digest and (row["source_id"], row["sha256"]) not in selected
                }
            )
            for digest in hashes
        }
        with self.vault.connect() as db:
            known_digests = {record["sha256"] for record in files_list} | {
                digest for digest in canonical_refs.values() if digest is not None
            }
            for digest in sorted(known_digests):
                if digest not in remaining_hashes:
                    remaining_hashes[digest] = sorted(
                        {
                            row[0]
                            for row in db.execute(
                                "SELECT source_id FROM source_versions WHERE sha256=?", (digest,)
                            )
                            if (row[0], digest) not in selected
                        }
                    )
                if _has_table(db, "source_artifacts"):
                    remaining_hashes[digest] = sorted(
                        set(remaining_hashes[digest])
                        | {
                            row[0]
                            for row in db.execute(
                                "SELECT a.source_id,a.source_version FROM source_artifacts a JOIN ingest_jobs j ON j.id=a.job_id WHERE a.sha256=? AND j.status='indexed'",
                                (digest,),
                            )
                            if (row[0], row[1]) not in selected
                        }
                    )
        backups = []
        for root in self.backup_roots:
            if not root.is_dir() or root.resolve() != root:
                hard.append("backup_root_unavailable_or_redirected")
                continue
            for manifest_path in sorted(root.glob("*/manifest.json")):
                if manifest_path.resolve() != manifest_path:
                    hard.append("backup_manifest_redirected")
                    continue
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if manifest.get("format") != 1 or not isinstance(manifest.get("files"), dict):
                    hard.append("invalid_known_backup_manifest")
                    continue
                matches = {
                    record["path"]
                    for record in files_list
                    if manifest["files"].get(record["path"], {}).get("sha256") == record["sha256"]
                }
                matches.update(
                    "vault/" + path
                    for path, digest in canonical_refs.items()
                    if manifest["files"].get("vault/" + path, {}).get("sha256") == digest
                )
                if any(
                    record["sha256"] in remaining_hashes for record in manifest["files"].values()
                ):
                    database = manifest_path.parent / ".state/memory.sqlite"
                    if (
                        not database.is_file()
                        or database.resolve() != database
                        or _file_hash(database)
                        != manifest["files"].get(".state/memory.sqlite", {}).get("sha256")
                    ):
                        hard.append("known_backup_provenance_unverified")
                        continue
                    with closing(
                        sqlite3.connect(database.as_uri() + "?mode=ro&immutable=1", uri=True)
                    ) as backup_db:
                        matches.update(
                            _forbidden_backup_aliases(backup_db, manifest, remaining_hashes)
                        )
                if matches:
                    identity = _file_hash(manifest_path)
                    expiry = (
                        _time(backup_retention[identity]) if identity in backup_retention else None
                    )
                    if expiry is None:
                        hard.append("backup_retention_disposition_required")
                    backups.append(
                        {
                            "directory": str(manifest_path.parent),
                            "manifest_sha256": identity,
                            "matching_files": sorted(matches),
                            "retained_until": expiry,
                        }
                    )
        plan = {
            "schema_version": 1,
            "memory_id": self.memory_id,
            "source_id": source_id,
            "mode": mode,
            "authority_sha256": _file_hash(self.authority),
            "authority_generation": len(ledger["events"]),
            "versions": versions,
            "source_ids": removed_sources,
            "locations": locations,
            "ingest_jobs": jobs,
            "import_requests": requests,
            "source_artifacts": artifacts,
            "native_capture_requests": native_requests,
            "native_sessions": sorted(native_sessions),
            "evaluation_jobs": evaluation_jobs,
            "evaluation_locks": evaluation_locks,
            "files": files_list,
            "canonical_views": [
                {"path": path, "sha256": digest, "file_present": "vault/" + path in files}
                for path, digest in sorted(canonical_refs.items())
            ],
            "retained_blobs": retained,
            "knowledge_keys": sorted(knowledge_keys),
            "proposal_ids": sorted(proposal_ids),
            "copy_paths": sorted(set(copy_paths)),
            "checkpoints": sorted(checkpoints),
            "peer_jobs": sorted(peer_jobs),
            "hash_exceptions": remaining_hashes,
            "backup_roots": [str(path) for path in self.backup_roots],
            "backups": backups,
            "backup_retention": backup_retention,
            "unresolved": [
                row
                for row in [
                    {
                        "code": "external_origin_locations_retained",
                        "count": sum(
                            not Path(row["path"]).absolute().is_relative_to(self.vault.home)
                            for row in locations
                        ),
                    },
                    {
                        "code": "unregistered_private_evaluation_copies",
                        "count": unregistered_evaluation,
                    },
                    {
                        "code": "other_native_staging_snapshots_unresolved",
                        "count": unregistered_native,
                    },
                    {
                        "code": "ultimate_native_client_history_retained",
                        "count": len(native_sessions),
                    },
                    {
                        "code": "external_clients_provider_history_and_unregistered_caches",
                        "count": None,
                    },
                    {
                        "code": "unlinked_historical_checkpoints",
                        "count": episode_count - len(checkpoints),
                    },
                ]
                if row["count"] is None or row["count"] > 0
            ],
            "hard_blockers": sorted(set(hard)),
        }
        with self.vault.connect() as db:
            plan["vector_index"] = _vector_inventory(db, _canonical_paths(plan))
        return {**plan, "plan_hash": _hash(plan)}

    def execute(
        self, plan, *, expected_plan_hash, actor_role, scopes, reviewed_by, acknowledged_unresolved
    ):
        _owner(actor_role, scopes)
        single_line(reviewed_by, "purge owner reviewer", 160)
        payload = {key: value for key, value in plan.items() if key != "plan_hash"}
        if plan.get("plan_hash") != expected_plan_hash or _hash(payload) != expected_plan_hash:
            raise PurgeBlocked("Purge plan differs from the exact owner-reviewed hash")
        if plan["memory_id"] != self.memory_id:
            raise PurgeBlocked("Purge plan belongs to another memory")
        with self._evaluation_locks(plan):
            with self.vault.connect() as db:
                db.execute("BEGIN IMMEDIATE")
                previous = db.execute(
                    "SELECT * FROM purge_events WHERE plan_hash=?", (expected_plan_hash,)
                ).fetchone()
                if previous is not None:
                    if previous["state"] == "complete":
                        return json.loads(previous["report_json"])
                else:
                    self._stage_review(
                        db,
                        plan,
                        expected_plan_hash,
                        actor_role,
                        scopes,
                        reviewed_by,
                        acknowledged_unresolved,
                    )
            return self._resume_locked(expected_plan_hash, actor_role=actor_role, scopes=scopes)

    def _stage_review(
        self, db, plan, expected_plan_hash, actor_role, scopes, reviewed_by, acknowledged_unresolved
    ):
        fresh = self.preview(
            plan["source_id"],
            actor_role=actor_role,
            scopes=scopes,
            mode=plan["mode"],
            backup_retention=plan["backup_retention"],
        )
        if fresh["plan_hash"] != expected_plan_hash:
            raise PurgeBlocked("Source, dependent copy, backup or authority changed after review")
        if plan["hard_blockers"]:
            raise PurgeBlocked(
                "Purge has unresolved hard blockers: " + ",".join(plan["hard_blockers"])
            )
        if set(acknowledged_unresolved) != {row["code"] for row in plan["unresolved"]}:
            raise PurgeBlocked(
                "Owner must explicitly acknowledge the reported limits on unknown copies"
            )
        approval = {
            "reviewed_by": reviewed_by,
            "acknowledged_unresolved": sorted(acknowledged_unresolved),
        }
        db.execute("PRAGMA secure_delete=ON")
        db.execute("INSERT INTO chunks(chunks,rank) VALUES('secure-delete',1)")
        db.execute(
            "INSERT INTO purge_events VALUES (?,'planned',?,?,?,?)",
            (
                expected_plan_hash,
                json.dumps(plan, sort_keys=True),
                reviewed_by,
                now(),
                json.dumps({"owner_review": approval}),
            ),
        )
        denials = [("source", identity) for identity in plan["source_ids"]]
        denials += [("version", row["source_id"] + ":" + row["sha256"]) for row in plan["versions"]]
        denials += [("path", path) for path in _canonical_paths(plan)]
        denials += [("knowledge", key) for key in plan["knowledge_keys"]]
        denials += [("native_session", identity) for identity in plan["native_sessions"]]
        denials += [("evaluation_job", path) for path in plan["evaluation_jobs"]]
        for kind, value in denials:
            db.execute(
                "INSERT INTO purge_denials VALUES (?,?,?)", (kind, value, expected_plan_hash)
            )
        for relative in _canonical_paths(plan):
            db.execute("DELETE FROM chunks WHERE path=?", (relative,))
            if _has_table(db, "indexed_documents"):
                db.execute("DELETE FROM indexed_documents WHERE path=?", (relative,))
            if _has_table(db, "embedding_vectors"):
                db.execute("DELETE FROM embedding_vectors WHERE path=?", (relative,))
            if _has_table(db, "ordinary_note_revisions"):
                db.execute("DELETE FROM ordinary_note_revisions WHERE path=?", (relative,))
            if _has_table(db, "ordinary_note_pending"):
                db.execute("DELETE FROM ordinary_note_pending WHERE path=?", (relative,))
        for job in plan["ingest_jobs"]:
            db.execute("UPDATE ingest_jobs SET status='purging' WHERE id=?", (job["id"],))
        for row in plan["native_capture_requests"]:
            db.execute(
                "UPDATE native_capture_requests SET status='purging',claim_token=NULL,lease_until=NULL WHERE client_name=? AND session_id=?",
                (row["client_name"], row["session_id"]),
            )

    def resume(self, plan_hash, *, actor_role, scopes):
        """Resume only durable owner intent, while excluding private evaluation workers."""
        _owner(actor_role, scopes)
        with self.vault.connect() as db:
            row = db.execute(
                "SELECT plan_json FROM purge_events WHERE plan_hash=?", (plan_hash,)
            ).fetchone()
        if row is None:
            raise ValueError("No durable owner-reviewed purge exists")
        plan = json.loads(row["plan_json"])
        if _hash({key: value for key, value in plan.items() if key != "plan_hash"}) != plan_hash:
            raise PurgeBlocked("Durable purge plan failed its integrity check")
        with self._evaluation_locks(plan):
            return self._resume_locked(plan_hash, actor_role=actor_role, scopes=scopes)

    def _resume_locked(self, plan_hash, *, actor_role, scopes):
        """Resume only a durably owner-reviewed job; never create new deletion intent."""
        _owner(actor_role, scopes)
        with self.vault.connect() as db:
            row = db.execute(
                "SELECT * FROM purge_events WHERE plan_hash=?", (plan_hash,)
            ).fetchone()
        if row is None:
            raise ValueError("No durable owner-reviewed purge exists")
        if row["state"] == "complete":
            return json.loads(row["report_json"])
        plan = json.loads(row["plan_json"])
        if _hash({key: value for key, value in plan.items() if key != "plan_hash"}) != plan_hash:
            raise PurgeBlocked("Durable purge plan failed its integrity check")
        ledger = self._ledger()
        approval = json.loads(row["report_json"])["owner_review"]
        if plan_hash not in {event["plan_hash"] for event in ledger["events"]}:
            revocation = {
                "source_ids": plan["source_ids"],
                "versions": [
                    {"source_id": row["source_id"], "sha256": row["sha256"]}
                    for row in plan["versions"]
                ],
                "locations": [row["path"] for row in plan["locations"]],
                "hashes": plan["hash_exceptions"],
                "files": [{"path": row["path"], "sha256": row["sha256"]} for row in plan["files"]],
                "canonical_paths": _canonical_paths(plan),
                "knowledge_keys": plan["knowledge_keys"],
                "native_sessions": plan["native_sessions"],
                "evaluation_jobs": plan["evaluation_jobs"],
                "backups": plan["backups"],
            }
            event = {
                "sequence": len(ledger["events"]) + 1,
                "previous": ledger["events"][-1]["sha256"] if ledger["events"] else None,
                "plan_hash": plan_hash,
                "revoked_at": now(),
                "revocation": revocation,
                "owner_review": approval,
            }
            event["sha256"] = _hash(event)
            ledger["events"].append(event)
            atomic_bytes(self.authority, _authority_bytes(ledger))
        with self.vault.connect() as db:
            db.execute("PRAGMA secure_delete=ON")
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE purge_events SET state='revoked' WHERE plan_hash=?", (plan_hash,))
            for request in plan["native_capture_requests"]:
                db.execute(
                    "DELETE FROM native_capture_requests WHERE client_name=? AND session_id=?",
                    (request["client_name"], request["session_id"]),
                )
            for request in plan["import_requests"]:
                db.execute(
                    "DELETE FROM import_requests WHERE owner=? AND request_id=?",
                    (request["owner"], request["request_id"]),
                )
            for job in plan["ingest_jobs"]:
                if _has_table(db, "source_artifacts"):
                    db.execute("DELETE FROM source_artifacts WHERE job_id=?", (job["id"],))
                db.execute("DELETE FROM ingest_jobs WHERE id=?", (job["id"],))
            for version in plan["versions"]:
                db.execute(
                    "DELETE FROM source_versions WHERE source_id=? AND sha256=?",
                    (version["source_id"], version["sha256"]),
                )
            for identity in plan["source_ids"]:
                db.execute("DELETE FROM source_locations WHERE source_id=?", (identity,))
                db.execute("DELETE FROM sources WHERE id=?", (identity,))
            if _has_table(db, "accepted_knowledge_versions"):
                for key in plan["knowledge_keys"]:
                    db.execute("DELETE FROM accepted_knowledge_heads WHERE entry_key=?", (key,))
                    db.execute("DELETE FROM accepted_knowledge_versions WHERE entry_key=?", (key,))
                    db.execute("DELETE FROM accepted_knowledge_proposals WHERE entry_key=?", (key,))
            for path in plan["copy_paths"]:
                db.execute("DELETE FROM retention_copies WHERE path=?", (path,))
            for job_id in plan["peer_jobs"]:
                db.execute("DELETE FROM jobs WHERE id=?", (job_id,))
        removed = 0
        for record in plan["files"]:
            path = self._owned(record["path"])
            if path.exists():
                if not path.is_file() or _file_hash(path) != record["sha256"]:
                    raise PurgeBlocked(
                        "Reviewed purge file changed; revoked data stays quarantined"
                    )
                path.unlink()
                removed += 1
        with self.vault.connect() as db:
            checkpoint = db.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
            if checkpoint[0] != 0:
                raise PurgeBlocked(
                    "Readers retain the old WAL; keep purge revoked and retry after reader release"
                )
        report = {
            "plan_hash": plan_hash,
            "status": "complete",
            "files_in_plan": len(plan["files"]),
            "removed_this_attempt": removed,
            "retained_blobs": len(plan["retained_blobs"]),
            "retained_backups": len(plan["backups"]),
            "unresolved": plan["unresolved"],
            "semantic_erasure_claimed": False,
            "revocation_generation": len(ledger["events"]),
            "owner_review": approval,
            "sqlite_secure_delete": True,
            "wal_truncated": True,
        }
        with self.vault.connect() as db:
            db.execute(
                "UPDATE purge_events SET state='complete',report_json=? WHERE plan_hash=?",
                (json.dumps(report), plan_hash),
            )
        return report

    def backup_expiry_preview(self, *, actor_role, scopes, at=None):
        """Return reviewed snapshots due for expiry, without deleting any backup path."""
        _owner(actor_role, scopes)
        at = _time(at) if at else now()
        due = {}
        for event in self._ledger()["events"]:
            for backup in event["revocation"]["backups"]:
                if backup["retained_until"] <= at:
                    directory = Path(backup["directory"]).absolute()
                    if not any(directory.parent == root for root in self.backup_roots):
                        raise PurgeBlocked("Backup is outside registered snapshot roots")
                    manifest = directory / "manifest.json"
                    if (
                        manifest.resolve(strict=True) != manifest
                        or _file_hash(manifest) != backup["manifest_sha256"]
                    ):
                        raise PurgeBlocked("Known backup manifest changed after retention review")
                    due[backup["manifest_sha256"]] = {**backup, "deletion_authorized": False}
        return {
            "as_of": at,
            "snapshots_due": list(due.values()),
            "policy": "Root operations must separately verify provenance/inventory and owner approval before scoped expiry",
        }
