"""Durable intake identities, immutable originals and restartable parsing jobs."""

import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from uuid import uuid4

from .parsing import DOCLING_PARSER
from .store import defer_busy_claim, now

SCOPES = ("shared", "private")
NATIVE_PARSERS = {"codex-jsonl-v1", "claude-jsonl-v1"}
MAX_BYTES = 100 * 1024 * 1024


def single_line(value, name, limit=2048):
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= limit
        or value.splitlines() != [value]
        or any(ord(char) < 32 for char in value)
    ):
        raise ValueError(f"{name} must be a single line of 1 to {limit} characters")
    return value


def atomic_bytes(path, content, *, replace=True):
    """Publish a complete file, optionally requiring an absent destination."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_symlink() or path.resolve() != path.absolute():
        raise ValueError("Derived and original paths cannot contain links")
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


class SourceRegistry:
    def __init__(self, vault, allowed_roots=(), *, repository_roots=()):
        self.vault = vault
        self.allowed_roots = tuple(Path(path).resolve(strict=True) for path in allowed_roots)
        roots = tuple(Path(path) for path in repository_roots)
        if any(
            not root.is_absolute()
            or any(
                part.is_symlink() or getattr(part, "is_junction", lambda: False)()
                for part in (root, *root.parents)
            )
            for root in roots
        ):
            raise PermissionError("Approved repository roots must be absolute and contain no links")
        self.repository_roots = tuple(root.resolve(strict=True) for root in roots)
        with vault.connect() as db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS sources (
                    id TEXT PRIMARY KEY, kind TEXT NOT NULL, scope TEXT NOT NULL,
                    collection TEXT NOT NULL, project TEXT, owner TEXT NOT NULL,
                    capture_method TEXT NOT NULL, assertion_type TEXT NOT NULL,
                    recorded_at TEXT NOT NULL, observed_at TEXT, schema_version INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_locations (
                    path TEXT PRIMARY KEY, source_id TEXT NOT NULL REFERENCES sources(id)
                );
                CREATE TABLE IF NOT EXISTS source_versions (
                    source_id TEXT NOT NULL REFERENCES sources(id), sha256 TEXT NOT NULL,
                    original TEXT NOT NULL, bytes INTEGER NOT NULL, suffix TEXT NOT NULL,
                    origin TEXT NOT NULL, recorded_at TEXT NOT NULL, observed_at TEXT,
                    repository_provenance TEXT,
                    PRIMARY KEY(source_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS ingest_jobs (
                    id TEXT PRIMARY KEY, source_id TEXT NOT NULL, sha256 TEXT NOT NULL,
                    parser TEXT NOT NULL, status TEXT NOT NULL, path TEXT, error TEXT, view_sha256 TEXT,
                    recoveries INTEGER NOT NULL DEFAULT 0, cancel_requested INTEGER NOT NULL DEFAULT 0,
                    recorded_at TEXT NOT NULL, updated_at TEXT NOT NULL,
                    UNIQUE(source_id, sha256, parser),
                    FOREIGN KEY(source_id, sha256) REFERENCES source_versions(source_id, sha256)
                );
                CREATE TABLE IF NOT EXISTS import_requests (
                    owner TEXT NOT NULL, request_id TEXT NOT NULL, fingerprint TEXT NOT NULL,
                    job_id TEXT NOT NULL REFERENCES ingest_jobs(id), PRIMARY KEY(owner, request_id)
                );
                CREATE TABLE IF NOT EXISTS ingest_control (
                    id INTEGER PRIMARY KEY CHECK(id=1), paused INTEGER NOT NULL
                );
                CREATE TABLE IF NOT EXISTS source_artifacts (
                    path TEXT PRIMARY KEY, source_id TEXT NOT NULL, source_version TEXT NOT NULL,
                    parser TEXT NOT NULL, sha256 TEXT NOT NULL, bytes INTEGER NOT NULL,
                    job_id TEXT NOT NULL REFERENCES ingest_jobs(id)
                );
                INSERT OR IGNORE INTO ingest_control VALUES (1,0);
            """)

            if "view_sha256" not in {
                row[1] for row in db.execute("PRAGMA table_info(ingest_jobs)")
            }:
                db.execute("ALTER TABLE ingest_jobs ADD COLUMN view_sha256 TEXT")
            if "repository_provenance" not in {
                row[1] for row in db.execute("PRAGMA table_info(source_versions)")
            }:
                db.execute("ALTER TABLE source_versions ADD COLUMN repository_provenance TEXT")

    def _input(self, source):
        path = Path(source).absolute()
        if any(
            part.is_symlink() or getattr(part, "is_junction", lambda: False)()
            for part in (path, *path.parents)
        ):
            raise PermissionError("Intake links are not allowed")
        path = path.resolve(strict=True)
        if not any(path.is_relative_to(root) for root in self.allowed_roots):
            raise PermissionError("Intake path is outside the explicitly allowed roots")
        if not path.is_file() or path.suffix.lower() not in {
            ".md",
            ".txt",
            ".pdf",
            ".json",
            ".jsonl",
        }:
            raise ValueError("Intake requires a Markdown, text, PDF or conversation JSON file")
        single_line(str(path), "source path")
        with path.open("rb") as stream:
            content = stream.read(MAX_BYTES + 1)
        if not 0 < len(content) <= MAX_BYTES:
            raise ValueError("Intake files must contain 1 byte to 100 MiB")
        return path, content

    def submit(
        self,
        source,
        *,
        kind="note",
        scope="shared",
        collection="references",
        project=None,
        source_id=None,
        parser=None,
        request_id=None,
        owner="local",
        observed_at=None,
        capture_method="explicit_import",
        assertion_type="source_claim",
        native_capture_claim=None,
        repository_capture=None,
    ):
        self.vault.assert_retention_ready()
        if kind not in {"note", "paper", "conversation", "code"} or scope not in SCOPES:
            raise ValueError("Intake kind or scope is unsupported")
        for name, value in [
            ("collection", collection),
            ("owner", owner),
            ("capture method", capture_method),
            ("assertion type", assertion_type),
        ]:
            single_line(value, name, 160)
        for name, value in [
            ("project", project),
            ("observed_at", observed_at),
            ("request_id", request_id),
        ]:
            if value is not None:
                single_line(value, name)
        if source_id is not None and not re.fullmatch(r"[a-f0-9]{32}", source_id):
            raise ValueError("source_id must be an existing 32-character lowercase hex ID")
        explicit_identity = source_id is not None
        repository_provenance = None
        if parser == "code-text-v1":
            if kind != "code" or native_capture_claim is not None:
                raise PermissionError("Code parsing requires a separate repository code capture")
            from .repositories import validate_capture

            path, content, code_source_id, repository_provenance = validate_capture(
                source, repository_capture, self.repository_roots
            )
            if source_id is not None and source_id != code_source_id:
                raise PermissionError(
                    "Repository code identity is bound to its root and relative path"
                )
            source_id = code_source_id
            explicit_identity = False
        else:
            if repository_capture is not None:
                raise PermissionError("Repository captures require the code-text-v1 parser")
            path, content = self._input(source)
        suffix = path.suffix.lower()
        parser = parser or {".pdf": DOCLING_PARSER, ".json": "conversation-json-v1"}.get(
            suffix, "text-v1"
        )
        if suffix == ".jsonl" and parser not in NATIVE_PARSERS:
            raise ValueError(
                "JSONL capture requires an explicit supported native transcript parser"
            )
        if parser in NATIVE_PARSERS | {"conversation-json-v1"} and (
            kind != "conversation" or scope != "private"
        ):
            raise PermissionError(
                "Raw conversation capture requires kind conversation and private scope"
            )
        if kind == "conversation" and scope != "private":
            raise PermissionError("Raw conversation captures must remain private")
        if not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", parser):
            raise ValueError(
                "parser must be a revision ID of 1 to 80 letters, digits, dots or dashes"
            )
        if native_capture_claim is not None:
            if (
                not isinstance(native_capture_claim, tuple)
                or len(native_capture_claim) != 3
                or not all(isinstance(value, str) for value in native_capture_claim)
            ):
                raise PermissionError(
                    "Native capture requires an internal client/session/claim tuple"
                )
            client, session, claim_token = native_capture_claim
            identity = hashlib.sha256((client + ":" + session).encode()).hexdigest()
            if (
                client not in {"codex", "claude"}
                or not re.fullmatch(
                    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", session
                )
                or owner != "native-" + identity
                or capture_method != "native_snapshot"
                or parser != client + "-jsonl-v1"
                or scope != "private"
                or kind != "conversation"
            ):
                raise PermissionError(
                    "Native capture claim does not match its private source namespace"
                )
        digest = hashlib.sha256(content).hexdigest()
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    str(path),
                    digest,
                    kind,
                    scope,
                    collection,
                    project,
                    owner if native_capture_claim is not None else source_id,
                    parser,
                    observed_at,
                    capture_method,
                    assertion_type,
                ]
                + ([repository_provenance] if repository_provenance is not None else []),
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if native_capture_claim is not None:
                live = db.execute(
                    "SELECT status,claim_token,lease_until FROM native_capture_requests WHERE client_name=? AND session_id=?",
                    (client, session),
                ).fetchone()
                if (
                    live is None
                    or live["status"] != "running"
                    or live["claim_token"] != claim_token
                    or live["lease_until"] is None
                    or live["lease_until"] <= now()
                ):
                    raise PermissionError("Native capture publication lease is no longer valid")
                if (
                    db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone()
                    and db.execute(
                        "SELECT 1 FROM purge_denials WHERE kind='native_session' AND value=?",
                        (identity,),
                    ).fetchone()
                ):
                    raise PermissionError("Native capture session has been revoked")
                native_sources = db.execute(
                    "SELECT id FROM sources WHERE owner=? AND scope='private' AND kind='conversation'",
                    (owner,),
                ).fetchall()
                if len(native_sources) > 1:
                    raise ValueError("Native capture namespace has ambiguous logical sources")
                if native_sources:
                    if source_id is not None and source_id != native_sources[0]["id"]:
                        raise PermissionError(
                            "Native capture logical source changed before publication"
                        )
                    source_id = native_sources[0]["id"]
            location = db.execute(
                "SELECT source_id FROM source_locations WHERE path=?", (str(path),)
            ).fetchone()
            if source_id and location and location["source_id"] != source_id:
                raise ValueError("Intake location already belongs to another source")
            if source_id is None:
                source_id = location["source_id"] if location else uuid4().hex
            if self.vault.retention is not None:
                self.vault.retention.assert_import_allowed(
                    source_id=source_id, origin=path, sha256=digest
                )
            if request_id is not None:
                previous = db.execute(
                    "SELECT * FROM import_requests WHERE owner=? AND request_id=?",
                    (owner, request_id),
                ).fetchone()
                if previous:
                    if previous["fingerprint"] != fingerprint:
                        raise ValueError("Import request ID was reused with a different payload")
                    job = self._job(db, previous["job_id"], (scope,))
                    retained = self.vault.home / job["original"]
                    if (
                        retained.is_symlink()
                        or retained.resolve() != retained.absolute()
                        or hashlib.sha256(retained.read_bytes()).hexdigest() != job["sha256"]
                    ):
                        raise ValueError(
                            "Retained original has been modified; refusing acknowledgement"
                        )
                    return job
            existing = db.execute("SELECT * FROM sources WHERE id=?", (source_id,)).fetchone()
            if existing:
                if any(
                    existing[key] != value
                    for key, value in [
                        ("scope", scope),
                        ("owner", owner),
                        ("kind", kind),
                        ("collection", collection),
                        ("project", project),
                    ]
                ):
                    raise PermissionError(
                        "Source identity, ownership and collection cannot change on intake"
                    )
            elif explicit_identity:
                raise ValueError("Unknown source identity")
            original = self.vault.home / "originals" / (digest + suffix)
            known_job = db.execute(
                "SELECT id FROM ingest_jobs WHERE source_id=? AND sha256=? AND parser=?",
                (source_id, digest, parser),
            ).fetchone()
            if (
                not original.exists()
                or not existing
                or not location
                or not known_job
                or request_id is not None
            ):
                from .resources import admit_storage

                admit_storage(
                    self.vault.home,
                    (0 if original.exists() else len(content)) + 4 * 1024**2,
                    additional_entries=8,
                    temporary_bytes=len(content),
                    temporary_entries=4,
                )
            timestamp = now()
            if not existing:
                db.execute(
                    "INSERT INTO sources VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (
                        source_id,
                        kind,
                        scope,
                        collection,
                        project,
                        owner,
                        capture_method,
                        assertion_type,
                        timestamp,
                        observed_at,
                        1,
                    ),
                )
            db.execute(
                "INSERT OR IGNORE INTO source_locations VALUES (?,?)", (str(path), source_id)
            )
            original = self.vault.home / "originals" / (digest + suffix)
            if original.exists():
                if original.is_symlink() or original.resolve() != original.absolute():
                    raise ValueError("Retained originals cannot contain links")
                if hashlib.sha256(original.read_bytes()).hexdigest() != digest:
                    raise ValueError("Retained original has been modified; refusing overwrite")
            else:
                atomic_bytes(original, content)
            db.execute(
                """INSERT OR IGNORE INTO source_versions
                (source_id,sha256,original,bytes,suffix,origin,recorded_at,observed_at,repository_provenance)
                VALUES (?,?,?,?,?,?,?,?,?)""",
                (
                    source_id,
                    digest,
                    original.relative_to(self.vault.home).as_posix(),
                    len(content),
                    suffix,
                    str(path),
                    timestamp,
                    observed_at,
                    json.dumps(repository_provenance, sort_keys=True)
                    if repository_provenance is not None
                    else None,
                ),
            )
            previous = db.execute(
                "SELECT id FROM ingest_jobs WHERE source_id=? AND sha256=? AND parser=?",
                (source_id, digest, parser),
            ).fetchone()
            job_id = previous["id"] if previous else uuid4().hex
            if not previous:
                db.execute(
                    """INSERT INTO ingest_jobs
                    (id,source_id,sha256,parser,status,recorded_at,updated_at)
                    VALUES (?,?,?,?,'queued',?,?)""",
                    (job_id, source_id, digest, parser, timestamp, timestamp),
                )
            if request_id is not None:
                db.execute(
                    "INSERT INTO import_requests VALUES (?,?,?,?)",
                    (owner, request_id, fingerprint, job_id),
                )
            return self._job(db, job_id, (scope,))

    def _job(self, db, job_id, scopes):
        placeholders = ",".join("?" for _ in scopes)
        revoked = self.vault.source_revocation_sql(db, "s.id", "j.sha256")
        row = db.execute(
            f"""SELECT j.*,v.original,v.bytes,v.suffix,v.origin,
            s.scope,s.kind,s.collection,s.project,s.capture_method,s.assertion_type,
            v.observed_at,v.repository_provenance,s.schema_version FROM ingest_jobs j
            JOIN sources s ON s.id=j.source_id
            JOIN source_versions v ON v.source_id=j.source_id AND v.sha256=j.sha256
            WHERE j.id=? AND s.scope IN ({placeholders}) AND j.status!='purging'{revoked}""",
            (job_id, *scopes),
        ).fetchone()
        if row is None:
            raise PermissionError("Intake job is unavailable in the granted scopes")
        job = dict(row)
        if job["repository_provenance"] is not None:
            job["repository_provenance"] = json.loads(job["repository_provenance"])
        return job

    def job(self, job_id, scopes=("shared",)):
        self.vault.assert_retention_ready()
        with self.vault.connect() as db:
            return self._job(db, job_id, scopes)

    def request(self, request_id, owner="local", scopes=("shared",)):
        self.vault.assert_retention_ready()
        with self.vault.connect() as db:
            row = db.execute(
                "SELECT job_id FROM import_requests WHERE owner=? AND request_id=?",
                (owner, request_id),
            ).fetchone()
            if row is None:
                raise PermissionError("Import request is unavailable")
            return self._job(db, row["job_id"], scopes)

    def source(self, source_id, scopes=("shared",)):
        self.vault.assert_retention_ready()
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self.vault.source_revocation_sql(db, "sources.id")
            row = db.execute(
                f"SELECT * FROM sources WHERE id=? AND scope IN ({placeholders}){revoked}",
                (source_id, *scopes),
            ).fetchone()
            if row is None:
                raise PermissionError("Source is unavailable in the granted scopes")
            return {
                **dict(row),
                "versions": [
                    dict(version)
                    for version in db.execute(
                        f"SELECT * FROM source_versions WHERE source_id=?{self.vault.source_revocation_sql(db, 'source_versions.source_id', 'source_versions.sha256')} ORDER BY recorded_at,sha256",
                        (source_id,),
                    )
                ],
                "locations": [
                    item[0]
                    for item in db.execute(
                        "SELECT path FROM source_locations WHERE source_id=? ORDER BY path",
                        (source_id,),
                    )
                ],
            }

    def sources(self, scopes=("shared",)):
        self.vault.assert_retention_ready()
        placeholders = ",".join("?" for _ in scopes)
        with self.vault.connect() as db:
            revoked = self.vault.source_revocation_sql(db, "sources.id")
            return [
                dict(row)
                for row in db.execute(
                    f"SELECT * FROM sources WHERE scope IN ({placeholders}){revoked} ORDER BY recorded_at,id",
                    scopes,
                )
            ]

    def pause(self, paused):
        if type(paused) is not bool:
            raise ValueError("paused must be a boolean")
        with self.vault.connect() as db:
            db.execute("UPDATE ingest_control SET paused=? WHERE id=1", (int(paused),))

    def cancel(self, job_id, scopes=("shared",)):
        with self.vault.connect() as db:
            self._job(db, job_id, scopes)
            db.execute(
                """UPDATE ingest_jobs SET cancel_requested=1, updated_at=?,
                status=CASE WHEN status='queued' THEN 'cancelled' ELSE status END WHERE id=?""",
                (now(), job_id),
            )

    def reparse(
        self,
        source_id,
        version,
        *,
        parser,
        actor_role,
        scopes=("shared",),
        owner="local",
        request_id=None,
        force=False,
    ):
        """Queue an owner-reviewed parser job from verified retained bytes only."""
        if actor_role != "owner":
            raise PermissionError("Only an authenticated owner may reparse a retained source")
        self.vault.assert_retention_ready()
        if (
            not isinstance(source_id, str)
            or not re.fullmatch(r"[0-9a-f]{32}", source_id)
            or not isinstance(version, str)
            or not re.fullmatch(r"[0-9a-f]{64}", version)
        ):
            raise ValueError("Reparse requires an exact source ID and SHA-256 version")
        parsers = {
            "text-v1",
            "text-v2",
            "pypdf-v1",
            DOCLING_PARSER,
            "conversation-json-v1",
            "code-text-v1",
            "pdf-annotations-v1",
            *NATIVE_PARSERS,
        }
        if not isinstance(parser, str) or parser not in parsers:
            raise ValueError("Reparse requires an installed explicit parser revision")
        if type(force) is not bool:
            raise TypeError("Reparse force must be a boolean")
        if force and request_id is None:
            raise ValueError("Forced reparse requires an idempotent request ID")
        single_line(owner, "owner", 160)
        if request_id is not None:
            single_line(request_id, "request_id")
        grants = tuple(scope for scope in scopes if scope in SCOPES)
        if not grants:
            raise PermissionError("No source scope is granted for reparse")
        fingerprint = hashlib.sha256(
            json.dumps(
                ["reparse", source_id, version, parser, force], separators=(",", ":")
            ).encode()
        ).hexdigest()
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            placeholders = ",".join("?" for _ in grants)
            revoked = self.vault.source_revocation_sql(db, "s.id", "v.sha256")
            source = db.execute(
                f"""SELECT v.*,s.kind,s.scope,s.capture_method,s.owner AS source_owner
                FROM source_versions v
                JOIN sources s ON s.id=v.source_id
                WHERE v.source_id=? AND v.sha256=? AND s.scope IN ({placeholders}){revoked}""",
                (source_id, version, *grants),
            ).fetchone()
            if source is None:
                raise PermissionError(
                    "Retained source version is unavailable in the granted scopes"
                )
            if source["capture_method"] == "native_snapshot":
                identity = source["source_owner"].removeprefix("native-")
                if (
                    not source["source_owner"].startswith("native-")
                    or not re.fullmatch(r"[0-9a-f]{64}", identity)
                    or (
                        db.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='purge_denials'"
                        ).fetchone()
                        and db.execute(
                            "SELECT 1 FROM purge_denials WHERE kind='native_session' AND value=?",
                            (identity,),
                        ).fetchone()
                    )
                ):
                    raise PermissionError("Native capture namespace is unavailable or revoked")
            suffix = source["suffix"]
            if (
                (suffix == ".jsonl" and parser not in NATIVE_PARSERS)
                or (
                    parser in NATIVE_PARSERS | {"conversation-json-v1"}
                    and (source["kind"] != "conversation" or source["scope"] != "private")
                )
                or (parser in NATIVE_PARSERS and suffix != ".jsonl")
                or (parser == "conversation-json-v1" and suffix != ".json")
                or (parser in {"text-v1", "text-v2"} and suffix not in {".md", ".txt"})
                or (
                    parser in {"pypdf-v1", DOCLING_PARSER, "pdf-annotations-v1"}
                    and suffix != ".pdf"
                )
                or (
                    parser == "code-text-v1"
                    and (source["kind"] != "code" or source["repository_provenance"] is None)
                )
            ):
                raise PermissionError("Parser does not match the retained source type and privacy")
            if (
                source["capture_method"] == "native_snapshot"
                and not db.execute(
                    "SELECT 1 FROM ingest_jobs WHERE source_id=? AND sha256=? AND parser=?",
                    (source_id, version, parser),
                ).fetchone()
            ):
                raise PermissionError(
                    "Native snapshots retain their original native parser namespace"
                )
            if self.vault.retention is not None:
                self.vault.retention.assert_import_allowed(
                    source_id=source_id, origin=source["origin"], sha256=version
                )
            if source["original"] != f"originals/{version}{suffix}":
                raise ValueError(
                    "Retained original differs from its canonical source-version binding"
                )
            original = self.vault.home / source["original"]
            if (
                original.resolve() != original.absolute()
                or not original.is_relative_to(self.vault.home / "originals")
                or any(
                    part.is_symlink() or getattr(part, "is_junction", lambda: False)()
                    for part in (original, *original.parents)
                )
            ):
                raise ValueError(
                    "Retained original must stay inside its unlinked original directory"
                )
            before = original.stat()
            with original.open("rb") as stream:
                descriptor = os.fstat(stream.fileno())
                content = stream.read(MAX_BYTES + 1)
                after = os.fstat(stream.fileno())
            path_after = original.stat()
            stable_fields = ("st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns")
            if (
                len(content) != source["bytes"]
                or not 0 < len(content) <= MAX_BYTES
                or (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
                != (
                    descriptor.st_dev,
                    descriptor.st_ino,
                    descriptor.st_size,
                    descriptor.st_mtime_ns,
                )
                or tuple(getattr(descriptor, name) for name in stable_fields)
                != tuple(getattr(after, name) for name in stable_fields)
                or tuple(getattr(path_after, name) for name in stable_fields)
                != tuple(getattr(before, name) for name in stable_fields)
                or hashlib.sha256(content).hexdigest() != version
            ):
                raise ValueError("Retained original failed its bounded hash integrity check")
            previous = (
                db.execute(
                    "SELECT * FROM import_requests WHERE owner=? AND request_id=?",
                    (owner, request_id),
                ).fetchone()
                if request_id is not None
                else None
            )
            if previous is not None:
                if previous["fingerprint"] != fingerprint:
                    raise ValueError("Reparse request ID was reused with a different payload")
                return self._job(db, previous["job_id"], grants)
            existing = db.execute(
                "SELECT * FROM ingest_jobs WHERE source_id=? AND sha256=? AND parser=?",
                (source_id, version, parser),
            ).fetchone()
            timestamp = now()
            if existing is None:
                job_id = uuid4().hex
                db.execute(
                    """INSERT INTO ingest_jobs
                    (id,source_id,sha256,parser,status,recorded_at,updated_at)
                    VALUES (?,?,?,?,'queued',?,?)""",
                    (job_id, source_id, version, parser, timestamp, timestamp),
                )
            else:
                job_id = existing["id"]
                if force:
                    if existing["status"] in {"running", "purging"}:
                        raise ValueError("Cannot force a running or purging parser job")
                    if existing["path"] is not None:
                        db.execute("DELETE FROM chunks WHERE path=?", (existing["path"],))
                        if db.execute(
                            "SELECT 1 FROM sqlite_master WHERE name='indexed_documents'"
                        ).fetchone():
                            db.execute(
                                "DELETE FROM indexed_documents WHERE path=?", (existing["path"],)
                            )
                    db.execute(
                        """UPDATE ingest_jobs SET status='queued',cancel_requested=0,error=NULL,
                        recoveries=recoveries+1,updated_at=? WHERE id=?""",
                        (timestamp, job_id),
                    )
            if request_id is not None:
                db.execute(
                    "INSERT INTO import_requests VALUES (?,?,?,?)",
                    (owner, request_id, fingerprint, job_id),
                )
            return self._job(db, job_id, grants)

    def retry(self, job_id, scopes=("shared",)):
        with self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = self._job(db, job_id, scopes)
            if job["status"] not in {"failed", "cancelled"}:
                raise ValueError("Only a failed or cancelled import can be explicitly retried")
            db.execute(
                """UPDATE ingest_jobs SET status='queued',cancel_requested=0,error=NULL,
                recoveries=recoveries+1,updated_at=? WHERE id=?""",
                (now(), job_id),
            )
            return self._job(db, job_id, scopes)

    def recover(self):
        with self.vault.connect() as db:
            db.execute(
                """UPDATE ingest_jobs SET
                status=CASE WHEN cancel_requested=1 THEN 'cancelled' ELSE 'queued' END,
                recoveries=recoveries+1,updated_at=? WHERE status='running'""",
                (now(),),
            )

    def claim(self, job_id=None):
        self.vault.assert_retention_ready()
        with defer_busy_claim(), self.vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if db.execute("SELECT paused FROM ingest_control WHERE id=1").fetchone()[0]:
                return None
            condition = " AND id=?" if job_id is not None else ""
            revoked = self.vault.source_revocation_sql(
                db, "ingest_jobs.source_id", "ingest_jobs.sha256"
            )
            row = db.execute(
                f"SELECT id FROM ingest_jobs WHERE status='queued'{condition}{revoked} ORDER BY rowid LIMIT 1",
                (job_id,) if job_id is not None else (),
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE ingest_jobs SET status='running',updated_at=? WHERE id=?", (now(), row[0])
            )
            return self._job(db, row[0], SCOPES)
        return None

    def register_artifacts(self, job, artifacts, *, db):
        """Publish byte provenance inside the same transaction as the managed view."""
        current = self._job(db, job["id"], SCOPES)
        if (
            any(current[key] != job[key] for key in ("source_id", "sha256", "parser"))
            or current["status"] != "running"
        ):
            raise ValueError("Artifact registration requires the matching claimed job transaction")
        if len(artifacts) > 10000:
            raise ValueError("Artifact inventory exceeds 10000 files")
        rows = []
        for name, content in artifacts.items():
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", name) or not isinstance(
                content, bytes
            ):
                raise ValueError("Artifact registration requires plain filenames and bytes")
            relative = f"derived/{job['source_id']}/{job['sha256']}/{job['parser']}/{name}"
            target = self.vault.home / relative
            checksum = hashlib.sha256(content).hexdigest()
            if (
                target.resolve() != target
                or not target.is_file()
                or target.stat().st_size != len(content)
            ):
                raise ValueError("Published artifact bytes differ from their registration")
            with target.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != checksum:
                    raise ValueError("Published artifact bytes differ from their registration")
            rows.append(
                (
                    relative,
                    job["source_id"],
                    job["sha256"],
                    job["parser"],
                    checksum,
                    len(content),
                    job["id"],
                )
            )
        db.execute("DELETE FROM source_artifacts WHERE job_id=?", (job["id"],))
        db.executemany("INSERT INTO source_artifacts VALUES (?,?,?,?,?,?,?)", rows)
        return {"artifacts": len(rows)}

    def finish(self, job_id, *, path=None, error=None):
        with self.vault.connect() as db:
            row = self._job(db, job_id, SCOPES)
            if row["status"] != "running":
                raise ValueError("Only a claimed intake job can finish")
            status = "cancelled" if row["cancel_requested"] else "failed" if error else "indexed"
            db.execute(
                "UPDATE ingest_jobs SET status=?,path=?,error=?,updated_at=? WHERE id=?",
                (status, path, error, now(), job_id),
            )
