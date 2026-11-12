import errno
import hashlib
import json
import os
import re
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

if os.name == "nt":
    import msvcrt
else:
    import fcntl


def now():
    return datetime.now(UTC).isoformat()


@contextmanager
def defer_busy_claim():
    """Let the next poll claim work after a contended pre-work transaction rolls back."""
    try:
        yield
    except sqlite3.OperationalError as error:
        code = getattr(error, "sqlite_errorcode", 0) & 0xFF
        if code not in {sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED}:
            raise


@contextmanager
def writer_lock(home):
    state = home / ".state"
    state.mkdir(parents=True, exist_ok=True)
    with (state / "writer.lock").open("a+b") as lock:
        try:
            if os.name == "nt":
                lock.seek(0)
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as err:
            if err.errno not in {errno.EACCES, errno.EAGAIN}:
                raise
            raise RuntimeError("Another writer owns this vault; use its MCP tools") from err
        yield


def metadata(text):
    fields = {}
    if text.startswith("---\n"):
        head, text = text[4:].split("\n---\n", 1)
        fields = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    return fields, text


def scope_for(path):
    return path.parts[0] if path.parts[0] in {"private", "archive", "inbox"} else "shared"


def read_retention_config(home):
    home = Path(home).resolve()
    path = home / "retention.local.json"
    if not path.exists():
        return None
    if path.resolve() != path or path.is_symlink():
        raise ValueError("Retention configuration cannot contain redirects")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or set(config) != {"authority", "backup_roots"}:
        raise ValueError("retention.local.json requires authority and backup_roots only")
    if not isinstance(config["backup_roots"], list):
        raise TypeError("Retention backup_roots must be an explicit list")
    paths = [config["authority"], *config["backup_roots"]]
    if any(not isinstance(value, str) or not Path(value).is_absolute() for value in paths):
        raise ValueError("Retention paths must be explicit absolute local paths")
    if any(Path(value).drive.startswith("\\\\") for value in paths):
        raise ValueError("Retention authority and backups must be local paths")
    return config


def _search_label(value, name, limit):
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or not 1 <= len(value) <= limit
        or value != value.strip()
        or value.splitlines() != [value]
        or any(ord(character) < 32 for character in value)
    ):
        raise ValueError(
            f"{name} must be a nonblank single line of 1 to {limit} characters without edge whitespace"
        )
    return value


class Vault:
    def __init__(self, home, *, _retention_setup=False):
        self.home = Path(home).resolve()
        config = read_retention_config(self.home)
        self._retention_config = config
        if config is not None and not Path(config["authority"]).is_file() and not _retention_setup:
            from .retention import PurgeBlocked

            raise PurgeBlocked(
                "Configured current revocation authority is missing; owner maintenance must initialize or recover it"
            )
        self._retrieval_config = read_retrieval_config(self.home)
        self.default_search_mode = (self._retrieval_config or {}).get("default_mode", "keyword")
        self._embedding_check = None
        self.root = self.home / "vault"
        self.state = self.home / ".state"
        self.state.mkdir(parents=True, exist_ok=True)
        self.db_path = self.state / "memory.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.execute("""CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
                path, title, heading, body, scope UNINDEXED, source UNINDEXED,
                tokenize='unicode61 remove_diacritics 2')""")
            db.execute("""CREATE TABLE IF NOT EXISTS indexed_documents (
                path TEXT PRIMARY KEY, project TEXT, collection TEXT,
                accepted INTEGER NOT NULL CHECK(accepted IN (0,1)), sha256 TEXT NOT NULL
            )""")
            retained_identity = db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='retention_meta'"
            ).fetchone()
        self.retention = None
        if config is None and retained_identity:
            from .retention import PurgeBlocked

            raise PurgeBlocked(
                "This memory requires its current retention.local.json and external authority"
            )
        if config is not None:
            from .retention import Retention

            self.retention = Retention(
                self, authority=config["authority"], backup_roots=config["backup_roots"]
            )
            if not _retention_setup:
                self.assert_retention_ready()

    def assert_retention_ready(self):
        if self.retention is not None:
            from .retention import PurgeBlocked

            try:
                if read_retention_config(self.home) != self._retention_config:
                    raise PurgeBlocked(
                        "Retention configuration changed; restart with the owner-reviewed policy"
                    )
                self.retention._ledger()
            except (OSError, ValueError) as error:
                raise PurgeBlocked(
                    "Configured current revocation authority is unavailable or invalid"
                ) from error

    def source_revocation_sql(self, db, source_expression, version_expression=None):
        if not db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone():
            return ""
        clause = f" AND NOT EXISTS (SELECT 1 FROM purge_denials d WHERE d.kind='source' AND d.value={source_expression})"
        if version_expression is not None:
            clause += f" AND NOT EXISTS (SELECT 1 FROM purge_denials d WHERE d.kind='version' AND d.value={source_expression}||':'||{version_expression})"
        return clause

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.db_path, timeout=15)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def resolve(self, relative):
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root) or path.suffix != ".md":
            raise ValueError(f"Expected a Markdown path within the vault, got {relative!r}")
        return path

    def get(self, relative, scopes=("shared",), *, expected_sha256=None):
        path = self.resolve(relative)
        relative = path.relative_to(self.root)
        if scope_for(relative) not in scopes:
            raise PermissionError(f"Scope {scope_for(relative)!r} is not granted")
        self.assert_retention_ready()
        if self.retention is not None:
            self.retention.assert_path_allowed(relative.as_posix())
        content = path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        if expected_sha256 is not None and content_hash != expected_sha256:
            raise ValueError(
                "Search view changed after indexing; explicitly reindex before searching"
            )
        fields, body = metadata(content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
        if not self.document_published(relative.as_posix(), fields, content_hash=content_hash):
            raise ValueError("Managed source view has not been committed by its ingestion job")
        return {
            "path": relative.as_posix(),
            "sha256": content_hash,
            "source": fields.get("source", relative.as_posix()),
            "status": fields.get("status", "curated"),
            "recorded": fields.get("recorded", fields.get("reviewed")),
            "text": body,
            **{
                key: json.loads(fields[key])
                for key in ("extraction_uncertainty", "figure_caption_coverage")
                if key in fields
            },
            **{
                key: fields[key]
                for key in (
                    "record_id",
                    "source_id",
                    "source_version",
                    "content_hash",
                    "schema_version",
                    "parser",
                    "kind",
                    "text_origin",
                    "citation_label",
                    "collection",
                    "project",
                    "observed_at",
                    "assertion_type",
                    "knowledge_key",
                    "knowledge_revision",
                )
                if key in fields
            },
        }

    def core(self):
        return self.get("core/CORE.md")

    @contextmanager
    def embedding(self):
        if read_retrieval_config(self.home) != self._retrieval_config:
            raise ValueError("Retrieval configuration changed; restart with the reviewed policy")
        if self._retrieval_config is None:
            yield None
            return
        from .embeddings import LocalEmbedder, make_runtime_check

        if self._embedding_check is None:
            self._embedding_check = make_runtime_check(self.home)
        with LocalEmbedder(
            self.home / "profiles/qwen3-embedding-cpu.json",
            self.home / "secrets/embedding-api.key",
            runtime_check=self._embedding_check,
        ) as embedder:
            yield embedder

    def _sync_vectors(self, db, chunks, *, paths=None, reset=False):
        from .retrieval import synchronize_vectors

        return synchronize_vectors(self, db, chunks, paths=paths, reset=reset)

    def index(self):
        self.assert_retention_ready()
        chunks = []
        with self.connect() as db:
            db.execute("DELETE FROM chunks")
            db.execute("DELETE FROM indexed_documents")
            for path in sorted(self.root.rglob("*.md")):
                chunks.extend(self.document_chunks(path, db=db))
            db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
            self._sync_vectors(db, chunks, reset=True)
        return {"chunks": len(chunks), "documents": len({row[0] for row in chunks})}

    def document_published(self, relative, fields, db=None, content_hash=None):
        job_id, key = fields.get("ingest_job_id"), fields.get("knowledge_key")
        if db is None:
            with self.connect() as connection:
                return self.document_published(
                    relative, fields, db=connection, content_hash=content_hash
                )
        if (
            db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone()
            and db.execute(
                "SELECT 1 FROM purge_denials WHERE (kind='path' AND value=?) OR (kind='source' AND value=?) OR (kind='knowledge' AND value=?) OR (kind='version' AND value=?)",
                (
                    relative,
                    fields.get("source_id"),
                    key,
                    fields.get("source_id", "") + ":" + fields.get("source_version", ""),
                ),
            ).fetchone()
        ):
            return False
        if job_id is None and key is None:
            return True
        if content_hash is None:
            content_hash = hashlib.sha256(self.resolve(relative).read_bytes()).hexdigest()
        if job_id is not None:
            if not db.execute("SELECT 1 FROM sqlite_master WHERE name='ingest_jobs'").fetchone():
                return False
            row = db.execute(
                "SELECT status,path,view_sha256 FROM ingest_jobs WHERE id=?", (job_id,)
            ).fetchone()
            if (
                row is None
                or row["status"] != "indexed"
                or row["path"] != relative
                or row["view_sha256"] != content_hash
            ):
                return False
        if key is not None:
            revision = fields.get("knowledge_revision", "")
            if (
                not revision.isdecimal()
                or not db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='accepted_knowledge_versions'"
                ).fetchone()
            ):
                return False
            row = db.execute(
                "SELECT path,sha256 FROM accepted_knowledge_versions WHERE entry_key=? AND revision=?",
                (key, int(revision)),
            ).fetchone()
            if row is None or row["path"] != relative or row["sha256"] != content_hash:
                return False
        return True

    def document_chunks(self, path, db=None):
        relative = path.relative_to(self.root)
        if path.is_symlink() or path.resolve() != path:
            raise ValueError(f"Symlinks are not indexed: {relative}")
        content = path.read_bytes()
        fields, body = metadata(content.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n"))
        if not self.document_published(
            relative.as_posix(), fields, db=db, content_hash=hashlib.sha256(content).hexdigest()
        ):
            return
        if fields.get("kind") == "knowledge_proposal":
            return
        if db is not None:
            # The publication guard above verified the registered accepted version, not just its label.
            accepted = (
                fields.get("kind") == "accepted_knowledge"
                and fields.get("knowledge_key") is not None
            )
            db.execute(
                "INSERT OR REPLACE INTO indexed_documents VALUES (?,?,?,?,?)",
                (
                    relative.as_posix(),
                    _search_label(fields.get("project"), "project", 2048),
                    _search_label(fields.get("collection"), "collection", 160),
                    int(accepted),
                    hashlib.sha256(content).hexdigest(),
                ),
            )
        title = next((s.lstrip("# ") for s in body.splitlines() if s.startswith("# ")), path.stem)
        from .chunks import section_chunks

        for heading, chunk in section_chunks(body, title):
            yield (
                relative.as_posix(),
                title,
                heading,
                chunk,
                scope_for(relative),
                fields.get("source", relative.as_posix()),
            )

    def upsert_document(self, relative, db=None):
        path = self.resolve(relative)
        relative = path.relative_to(self.root).as_posix()
        if db is None:
            with self.connect() as connection:
                return self.upsert_document(relative, db=connection)
        db.execute("DELETE FROM indexed_documents WHERE path=?", (relative,))
        chunks = list(self.document_chunks(path, db=db))
        db.execute("DELETE FROM chunks WHERE path=?", (relative,))
        db.executemany("INSERT INTO chunks VALUES (?,?,?,?,?,?)", chunks)
        self._sync_vectors(db, chunks, paths=[relative])
        return {"path": relative, "chunks": len(chunks)}

    def search(
        self,
        query,
        scopes=("shared",),
        limit=8,
        *,
        project=None,
        collection=None,
        broad=False,
        mode=None,
    ):
        if type(limit) is not int or not 1 <= limit <= 20:
            raise ValueError(f"limit must be between 1 and 20, got {limit}")
        project = _search_label(project, "project", 2048)
        collection = _search_label(collection, "collection", 160)
        if type(broad) is not bool:
            raise ValueError("broad must be an explicit boolean")
        if broad and project is not None:
            raise ValueError(
                "broad cannot be combined with project; choose a deliberate discovery mode"
            )
        if not isinstance(scopes, (tuple, list, set, frozenset)) or any(
            not isinstance(scope, str) or scope not in {"shared", "private", "archive", "inbox"}
            for scope in scopes
        ):
            raise ValueError("scopes must be an explicit collection of known scope names")
        scopes = tuple(scopes)
        if not scopes:
            return []
        self.assert_retention_ready()
        if not isinstance(query, str) or len(query) > 4000:
            raise ValueError("Search query must be a string of at most 4000 characters")
        default_requested = mode is None
        mode = self.default_search_mode if default_requested else mode
        if mode not in {"keyword", "exact", "dense", "hybrid"}:
            raise ValueError("Search mode must be keyword, exact, dense or hybrid")
        if mode != "keyword":
            from .embeddings import EmbeddingError
            from .retrieval import routed_search

            try:
                return routed_search(
                    self,
                    query,
                    scopes,
                    limit,
                    project=project,
                    collection=collection,
                    broad=broad,
                    mode=mode,
                )
            except EmbeddingError as error:
                if not default_requested:
                    raise
                result = self.search(
                    query,
                    scopes,
                    limit,
                    project=project,
                    collection=collection,
                    broad=broad,
                    mode="keyword",
                )
                return [
                    {**row, "retrieval_mode": "keyword", "retrieval_warning": str(error)}
                    for row in result
                ]
        terms = re.findall(r"[^\W_]+", query, re.UNICODE)[:32]
        if not terms:
            return []
        match = " OR ".join('"' + word + '"' for word in terms)
        placeholders = ",".join("?" for _ in scopes)
        with self.connect() as db:
            revoked = ""
            if db.execute("SELECT 1 FROM sqlite_master WHERE name='purge_denials'").fetchone():
                revoked = " AND NOT EXISTS (SELECT 1 FROM purge_denials d WHERE d.kind='path' AND d.value=chunks.path)"
                if db.execute("SELECT 1 FROM sqlite_master WHERE name='ingest_jobs'").fetchone():
                    revoked += " AND NOT EXISTS (SELECT 1 FROM ingest_jobs j JOIN purge_denials d ON (d.kind='source' AND d.value=j.source_id) OR (d.kind='version' AND d.value=j.source_id||':'||j.sha256) WHERE j.path=chunks.path)"
                if db.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='accepted_knowledge_versions'"
                ).fetchone():
                    revoked += " AND NOT EXISTS (SELECT 1 FROM accepted_knowledge_versions v JOIN purge_denials d ON d.kind='knowledge' AND d.value=v.entry_key WHERE v.path=chunks.path)"
            temporal, times = "", ()
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE name='accepted_knowledge_versions'"
            ).fetchone():
                at = now()
                temporal = """ AND NOT EXISTS (
                    SELECT 1 FROM accepted_knowledge_versions v WHERE v.path=chunks.path AND (
                        v.revision != (SELECT MAX(r.revision) FROM accepted_knowledge_versions r
                            WHERE r.entry_key=v.entry_key AND (r.valid_from IS NULL OR r.valid_from<=?))
                        OR (v.valid_from IS NOT NULL AND v.valid_from>?)
                        OR (v.valid_to IS NOT NULL AND v.valid_to<=?)
                    ))"""
                times = (at, at, at)
            eligible = f"chunks MATCH ? AND scope IN ({placeholders}){revoked}{temporal}"
            parameters = (match, *scopes, *times)
            if db.execute(
                f"""SELECT 1 FROM chunks WHERE {eligible} AND NOT EXISTS (
                    SELECT 1 FROM indexed_documents d WHERE d.path=chunks.path) LIMIT 1""",
                parameters,
            ).fetchone():
                raise ValueError(
                    "Search context metadata is missing; explicitly reindex before searching"
                )
            context, filters = "", ()
            if not broad:
                if project is None:
                    context = " AND d.project IS NULL"
                else:
                    context = " AND (d.project=? OR d.project IS NULL)"
                    filters = (project,)
            if collection is not None:
                context += " AND d.collection=?"
                filters += (collection,)
            rows = db.execute(
                f"""SELECT DISTINCT chunks.path,title,heading,body,source,bm25(chunks) AS rank,
                    d.sha256 AS indexed_sha256 FROM chunks
                    JOIN indexed_documents d ON d.path=chunks.path
                    WHERE {eligible}{context}
                    ORDER BY CASE WHEN d.project=? THEN 0
                        WHEN d.project IS NULL AND d.accepted=1 THEN 1 ELSE 2 END,
                        rank,chunks.path,chunks.rowid LIMIT ?""",
                (*parameters, *filters, project, limit),
            ).fetchall()
        results = []
        for row in rows:
            entry = dict(row)
            note = self.get(entry["path"], scopes, expected_sha256=entry.pop("indexed_sha256"))
            entry.update(
                {key: value for key, value in note.items() if key not in {"path", "source", "text"}}
            )
            results.append(entry)
        return results

    def propose(self, title, text, writer):
        if not title.strip() or len(title) > 160 or "\n" in title:
            raise ValueError("title must be a single non-empty line under 161 characters")
        if not 1 <= len(text) <= 100_000:
            raise ValueError("proposal text must contain 1 to 100000 characters")
        relative = f"inbox/{uuid4().hex}.md"
        path = self.resolve(relative)
        from .resources import admit_storage
        from .source_registry import atomic_bytes

        content = f"---\nkind: proposal\nsource: {writer}\nrecorded: {now()}\n---\n# {title}\n\n{text}\n".encode()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            admit_storage(
                self.home,
                len(content) + 4 * 1024**2,
                additional_entries=4,
                temporary_bytes=len(content),
                temporary_entries=2,
            )
            atomic_bytes(path, content)
        return {"path": relative, "status": "pending_review"}

    def import_file(self, source, kind="note", scope="shared"):
        """Synchronously import through durable intake; caller owns the single-writer gate."""
        from .ingest import Ingestor
        from .source_registry import SourceRegistry

        source = Path(source).absolute()
        registry = SourceRegistry(self, [source.parent])
        parser = "pypdf-v1" if source.suffix.lower() == ".pdf" else None
        job = registry.submit(source, kind=kind, scope=scope, parser=parser)
        if job["status"] == "queued":
            Ingestor(self, registry).run_next(job["id"])
        job = registry.job(job["id"], (scope,))
        if job["status"] != "indexed":
            raise ValueError(
                f"Import {job['id']} is {job['status']}: {job['error']}. "
                "Inspect the retained original and parser diagnostics; PDFs may require OCR/Docling."
            )
        return job


def pin_core(vault):
    text = vault.core()["text"]
    root = vault.home / "generated"
    root.mkdir(exist_ok=True)
    for name in ["AGENTS.md", "CLAUDE.md", "SYSTEM_PROMPT.md"]:
        (root / name).write_text(text, encoding="utf-8")
    (root / "core.mdc").write_text("---\nalwaysApply: true\n---\n" + text, encoding="utf-8")
    return {"directory": str(root), "sha256": hashlib.sha256(text.encode()).hexdigest()}


def read_config(home):
    path = Path(home) / "agents.local.json"
    if not path.exists():
        return {}
    cfg = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(cfg, dict):
        raise TypeError("agents.local.json must be an object keyed by agent name")
    enabled = {}
    for name, agent in cfg.items():
        if not re.fullmatch(r"[a-z0-9_-]{1,32}", name):
            raise ValueError(f"Invalid registered agent name: {name!r}")
        if not isinstance(agent, dict):
            raise TypeError(f"Agent {name!r} must be an object")
        kind = agent.get("kind")
        if kind not in ["codex", "claude", "local"]:
            raise ValueError(f"Agent {name!r} must have kind codex, claude or local")
        if not isinstance(agent.get("enabled", False), bool):
            raise TypeError(f"enabled for agent {name!r} must be a boolean")
        if not agent.get("enabled", False):
            continue
        for field, minimum, maximum in [
            ("max_tokens", 1, 32768),
            ("thinking_budget", 0, 24576),
        ]:
            if field in agent and (
                type(agent[field]) is not int or not minimum <= agent[field] <= maximum
            ):
                raise ValueError(
                    f"{field} for agent {name!r} must be an integer in [{minimum}, {maximum}]"
                )
        if (
            "max_tokens" in agent
            and "thinking_budget" in agent
            and agent["thinking_budget"] >= agent["max_tokens"]
        ):
            raise ValueError(f"thinking_budget for agent {name!r} must be less than max_tokens")
        timeout = agent.get("timeout_seconds", 180)
        if type(timeout) not in {int, float} or not 1 <= timeout <= 1200:
            raise ValueError(f"timeout_seconds for agent {name!r} must be finite and in [1, 1200]")
        if "reasoning_effort" in agent and agent["reasoning_effort"] not in [
            "none",
            "minimal",
            "low",
            "medium",
            "high",
            "xhigh",
        ]:
            raise ValueError(f"Unsupported reasoning_effort for agent {name!r}")
        if kind == "local":
            model = agent.get("model")
            if not isinstance(model, str) or not model.strip():
                raise ValueError(f"model for agent {name!r} must be a non-empty string")
            base = agent.get("base_url")
            if not isinstance(base, str):
                raise ValueError(f"base_url for agent {name!r} must be an HTTP loopback URL")
            url = urlsplit(base)
            if (
                url.scheme != "http"
                or url.hostname not in {"localhost", "127.0.0.1", "::1"}
                or url.username
                or url.password
                or url.query
                or url.fragment
            ):
                raise ValueError(f"base_url for agent {name!r} must be an HTTP loopback URL")
        else:
            command = agent.get("executable", kind)
            command = [command] if isinstance(command, str) else command
            if (
                not isinstance(command, list)
                or not command
                or any(not isinstance(part, str) or not part.strip() for part in command)
            ):
                raise ValueError(f"executable for agent {name!r} must name a non-empty command")
        enabled[name] = agent
    return enabled


def read_intake_config(home):
    path = Path(home) / "intake.local.json"
    config = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    names = {"allowed_roots", "repository_roots"}
    if not isinstance(config, dict) or set(config) - names:
        raise ValueError("intake.local.json supports allowed_roots and repository_roots only")
    for name in names:
        values = config.get(name, [])
        if not isinstance(values, list) or any(
            not isinstance(value, str) or not value.strip() or any(ord(c) < 32 for c in value)
            for value in values
        ):
            raise ValueError(f"Intake {name} must contain explicit local path strings")
    return {name: config.get(name, []) for name in names}


def read_retrieval_config(home):
    path = Path(home) / "retrieval.local.json"
    if not path.exists():
        return None
    if path.resolve() != path.absolute():
        raise ValueError("Retrieval configuration cannot contain redirects")
    config = json.loads(path.read_text(encoding="utf-8"))
    if (
        not isinstance(config, dict)
        or set(config) != {"default_mode"}
        or config["default_mode"] not in {"keyword", "exact", "dense", "hybrid"}
    ):
        raise ValueError("retrieval.local.json requires an explicit supported default_mode only")
    return config


def read_milestone_config(home):
    path = Path(home) / "milestones.local.json"
    if not path.is_file() or path.resolve() != path.absolute():
        raise ValueError("Owner milestone checks require an explicit unlinked local configuration")
    config = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict) or not {"checks"} <= set(config) <= {"checks", "timeout"}:
        raise ValueError("milestones.local.json supports explicit checks and optional timeout only")
    return config
