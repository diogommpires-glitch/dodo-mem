"""Owner-local catalogue reads and verified source access."""

import hashlib
import json
import os
import re
import sqlite3
import stat
from collections import Counter, OrderedDict
from contextlib import closing, contextmanager
from functools import lru_cache
from itertools import zip_longest
from pathlib import Path
from random import Random
from threading import Lock

from .configuration import runtime_path, source_roots

MAX_SOURCE_BYTES = 16 * 1024 * 1024
METADATA_ONLY_ROOTS = (runtime_path("neural").parent,)
EVIDENCE_POLICY_VERSION = 1
REPARSE = 0x400
PLACEHOLDER = 0x1000 | 0x40000 | 0x400000
PROTECTED_NAMES = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".git",
        ".aws",
        ".azure",
        ".kube",
        ".config",
        ".codex",
        ".claude",
        "secrets",
        "credentials",
        "credential",
        "keychains",
        "keyrings",
        ".npmrc",
        ".netrc",
        ".pypirc",
        "_netrc",
        "id_rsa",
        "id_ed25519",
        ".docker",
        ".password-store",
        ".git-credentials",
        "auth.json",
        "tokens.json",
    }
)
PROTECTED_SUFFIXES = frozenset({".pfx", ".p12", ".key", ".pem", ".kdbx", ".keystore"})
SECRET_BYTES = re.compile(
    rb"(?i:\bBearer\s+[A-Za-z0-9._~+/-]{24,}={0,2})|"
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{25,})\b|"
    rb"(?im:\b[A-Za-z_]*(?:auth_?token|api_?key|access_?token|password)[\"']?\s*[:=]\s*[\"'][^\r\n\"']{8,}[\"'])"
)
NODE_FIELDS = (
    "id,parent_id,root_id,project_id,path,name,kind,category,state,size,mtime_ns,digest,descendants"
)
GRAPH_BLOCKED = frozenset({"offline", "inaccessible", "protected", "redirected", "placeholder"})
GRAPH_BLOCKED_SQL = ",".join(f"'{state}'" for state in sorted(GRAPH_BLOCKED))
GRAPH_RELATIONS = ("reference", "citation", "duplicate", "topic")
GRAPH_RELATIONS_SQL = ",".join(f"'{kind}'" for kind in GRAPH_RELATIONS)
GRAPH_NODE_LIMIT = 900
GRAPH_EDGE_LIMIT = 3000
EXTRACTION_SAMPLE_BLOCKS = 128
EXTRACTION_SAMPLE_ROWS = 64
SCHEMA = """
CREATE TABLE IF NOT EXISTS catalogue (
    id INTEGER PRIMARY KEY CHECK(id=1), generation INTEGER NOT NULL,
    scanning INTEGER NOT NULL, message TEXT NOT NULL
);
INSERT OR IGNORE INTO catalogue VALUES (1,0,0,'Waiting for the initial inventory.');
CREATE TABLE IF NOT EXISTS nodes (
    id TEXT PRIMARY KEY, parent_id TEXT, root_id TEXT NOT NULL, project_id TEXT,
    path TEXT NOT NULL, path_key TEXT NOT NULL UNIQUE, name TEXT NOT NULL,
    kind TEXT NOT NULL, category TEXT NOT NULL, state TEXT NOT NULL, size INTEGER NOT NULL,
    mtime_ns INTEGER NOT NULL, file_id TEXT NOT NULL, digest TEXT, descendants INTEGER NOT NULL DEFAULT 0,
    seen INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS node_parent ON nodes(parent_id,name COLLATE NOCASE,id);
CREATE INDEX IF NOT EXISTS node_digest ON nodes(digest) WHERE digest IS NOT NULL;
CREATE INDEX IF NOT EXISTS node_project ON nodes(project_id,kind,mtime_ns);
CREATE INDEX IF NOT EXISTS node_file_name ON nodes(name COLLATE NOCASE,id) WHERE kind='file';
DROP INDEX IF EXISTS node_work;
CREATE INDEX IF NOT EXISTS node_pending_recent ON nodes(seen DESC,size) WHERE state='pending';
CREATE TABLE IF NOT EXISTS filename_index_progress (
    id INTEGER PRIMARY KEY CHECK(id=1), through_row INTEGER NOT NULL, ready INTEGER NOT NULL
);
INSERT OR IGNORE INTO filename_index_progress VALUES (1,0,0);
CREATE VIRTUAL TABLE IF NOT EXISTS filename_search USING fts5(
    name, content='nodes', content_rowid='rowid', tokenize='trigram', detail=none
);
CREATE TRIGGER IF NOT EXISTS filename_add AFTER INSERT ON nodes
WHEN (SELECT ready OR new.rowid<=through_row FROM filename_index_progress)
BEGIN
    INSERT INTO filename_search(rowid,name) VALUES(new.rowid,new.name);
END;
CREATE TRIGGER IF NOT EXISTS filename_remove AFTER DELETE ON nodes
WHEN (SELECT ready OR old.rowid<=through_row FROM filename_index_progress)
BEGIN
    INSERT INTO filename_search(filename_search,rowid,name) VALUES('delete',old.rowid,old.name);
END;
CREATE TRIGGER IF NOT EXISTS filename_rename AFTER UPDATE OF name ON nodes
WHEN old.name<>new.name
    AND (SELECT ready OR new.rowid<=through_row FROM filename_index_progress)
BEGIN
    INSERT INTO filename_search(filename_search,rowid,name) VALUES('delete',old.rowid,old.name);
    INSERT INTO filename_search(rowid,name) VALUES(new.rowid,new.name);
END;
CREATE TABLE IF NOT EXISTS passages (
    id INTEGER PRIMARY KEY, node_id TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, start INTEGER NOT NULL, end INTEGER NOT NULL, text TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS passage_node ON passages(node_id,start);
CREATE VIRTUAL TABLE IF NOT EXISTS passage_search USING fts5(
    text, content='passages', content_rowid='id'
);
CREATE TRIGGER IF NOT EXISTS passage_add AFTER INSERT ON passages BEGIN
    INSERT INTO passage_search(rowid,text) VALUES(new.id,new.text);
END;
CREATE TRIGGER IF NOT EXISTS passage_remove AFTER DELETE ON passages BEGIN
    INSERT INTO passage_search(passage_search,rowid,text) VALUES('delete',old.id,old.text);
END;
CREATE TABLE IF NOT EXISTS edges (
    source TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    target TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    kind TEXT NOT NULL, evidence TEXT NOT NULL, PRIMARY KEY(source,target,kind)
);
CREATE INDEX IF NOT EXISTS edge_target ON edges(target,kind,source);
CREATE TABLE IF NOT EXISTS commit_metadata (
    node_id TEXT PRIMARY KEY REFERENCES nodes(id) ON DELETE CASCADE,
    refs TEXT NOT NULL, files TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS declared_links (
    source TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    destination TEXT NOT NULL, line INTEGER NOT NULL, PRIMARY KEY(source,destination)
);
CREATE TABLE IF NOT EXISTS note_titles (
    source TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    title TEXT NOT NULL, PRIMARY KEY(source,title)
);
CREATE INDEX IF NOT EXISTS note_title_lookup ON note_titles(title,source);
CREATE TABLE IF NOT EXISTS topics (
    source TEXT NOT NULL REFERENCES nodes(id) ON DELETE CASCADE,
    topic TEXT NOT NULL, line INTEGER NOT NULL, PRIMARY KEY(source,topic)
);
CREATE INDEX IF NOT EXISTS topics_by_name ON topics(topic,source);
CREATE TABLE IF NOT EXISTS inventory_roots (
    id TEXT PRIMARY KEY, phase TEXT NOT NULL CHECK(phase IN ('pending','complete','offline')),
    visited INTEGER NOT NULL DEFAULT 0
);
"""
GRAPH_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS graph_revision (
    id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER NOT NULL
);
INSERT OR IGNORE INTO graph_revision VALUES (1,0);
CREATE TRIGGER IF NOT EXISTS graph_node_add AFTER INSERT ON nodes BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
CREATE TRIGGER IF NOT EXISTS graph_node_remove AFTER DELETE ON nodes BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
CREATE TRIGGER IF NOT EXISTS graph_node_change AFTER UPDATE ON nodes
WHEN old.id IS NOT new.id OR old.parent_id IS NOT new.parent_id
    OR old.root_id IS NOT new.root_id OR old.project_id IS NOT new.project_id
    OR old.path IS NOT new.path OR old.name IS NOT new.name OR old.kind IS NOT new.kind
    OR old.category IS NOT new.category OR old.descendants IS NOT new.descendants
    OR (old.state<>new.state AND (
        new.kind='root' OR old.state IN ({GRAPH_BLOCKED_SQL}) OR new.state IN ({GRAPH_BLOCKED_SQL})
    ))
BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
CREATE TRIGGER IF NOT EXISTS graph_edge_add AFTER INSERT ON edges
WHEN new.kind IN ({GRAPH_RELATIONS_SQL}) BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
CREATE TRIGGER IF NOT EXISTS graph_edge_remove AFTER DELETE ON edges
WHEN old.kind IN ({GRAPH_RELATIONS_SQL}) BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
CREATE TRIGGER IF NOT EXISTS graph_edge_change AFTER UPDATE ON edges
WHEN old.kind IN ({GRAPH_RELATIONS_SQL}) OR new.kind IN ({GRAPH_RELATIONS_SQL}) BEGIN
    UPDATE graph_revision SET revision=revision+1 WHERE id=1;
END;
"""


def default_database() -> Path:
    return runtime_path("desktop") / "catalogue.sqlite3"


def path_key(path: str | Path) -> str:
    return os.path.normcase(os.path.abspath(path))


def path_id(path: str | Path) -> str:
    return hashlib.blake2b(path_key(path).encode("utf-8"), digest_size=16).hexdigest()


def history_root_paths() -> tuple[Path, ...]:
    return source_roots("DODO_MEM_CHAT_ROOTS")


def history_source(path: Path) -> str | None:
    for root in history_root_paths():
        if not path.is_relative_to(root):
            continue
        if path.suffix.casefold() == ".jsonl":
            if root.parent.name == ".codex" and path.name.startswith("rollout-"):
                return "codex"
            if root.parent.name == ".claude" and (
                re.fullmatch(r"[0-9a-f-]{36}", path.stem, re.IGNORECASE)
                or path.name.startswith("agent-")
            ):
                return "claude"
        if (
            root.parent.name == ".claude"
            and path.suffix.casefold() == ".md"
            and "memory" in path.parts
        ):
            return "plain"
    return None


def metadata_only_runtime(path: Path) -> bool:
    return any(path.is_relative_to(root) for root in METADATA_ONLY_ROOTS)


def protected(path: Path) -> bool:
    if metadata_only_runtime(path):
        return True
    if history_source(path):
        return False
    return path.suffix.casefold() in PROTECTED_SUFFIXES or any(
        (part := name.casefold()) in PROTECTED_NAMES
        or part.startswith((".env", "credentials.", "secrets.", "secret."))
        for name in path.parts
    )


def entry_state(info: os.stat_result) -> str | None:
    attrs = getattr(info, "st_file_attributes", 0)
    if attrs & PLACEHOLDER:
        return "placeholder"
    if attrs & REPARSE or stat.S_ISLNK(info.st_mode):
        return "redirected"
    return None


def safe_path(path: Path) -> os.stat_result:
    if not path.is_absolute() or not path.drive and os.name == "nt":
        raise ValueError("A local absolute path is required")
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        if state := entry_state(info):
            raise OSError(f"Unavailable path: {state}")
    return info


def read_bytes(path: Path, cap: int = MAX_SOURCE_BYTES, *, internal: bool = False) -> bytes:
    if metadata_only_runtime(path) or protected(path) and not internal:
        raise PermissionError("Protected content")
    info = safe_path(path)
    if not stat.S_ISREG(info.st_mode) or info.st_size > cap:
        raise ValueError("Not a regular file within the extraction budget")
    if os.name == "nt":
        from .whole_index_windows import open_verified

        stream = open_verified(path)
    else:
        stream = os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW), "rb")
    with stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode) or opened.st_size > cap:
            raise ValueError("Source size or type changed")
        data = stream.read(cap + 1)
        final = os.fstat(stream.fileno())
        if len(data) > cap or (opened.st_mtime_ns, opened.st_size) != (
            final.st_mtime_ns,
            final.st_size,
        ):
            raise OSError("Source changed during read")
    if SECRET_BYTES.search(data) and not internal:
        raise PermissionError("Protected content")
    return data


def bounded(limit: int, maximum: int = 400) -> int:
    return max(1, min(limit, maximum))


def writable_database(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_path(path.parent)
    for candidate in (path, path.with_name(path.name + "-wal"), path.with_name(path.name + "-shm")):
        if os.path.lexists(candidate):
            safe_path(candidate)
    connection = sqlite3.connect(path, timeout=5)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA secure_delete=ON")
    connection.executescript(f"BEGIN IMMEDIATE;\n{SCHEMA}\n{GRAPH_SCHEMA}\nCOMMIT;")
    connection.execute("INSERT INTO passage_search(passage_search,rank) VALUES('secure-delete',1)")
    connection.execute(
        "INSERT INTO filename_search(filename_search,rank) VALUES('secure-delete',1)"
    )
    version = connection.execute("PRAGMA user_version").fetchone()[0]
    if version < EVIDENCE_POLICY_VERSION:
        # Never reuse passages extracted before decoded documents received credential checks.
        connection.executescript("""
            BEGIN IMMEDIATE;
            DELETE FROM passages WHERE node_id IN (SELECT id FROM nodes WHERE kind='file');
            DELETE FROM declared_links;
            DELETE FROM topics;
            DELETE FROM edges WHERE kind IN ('reference','duplicate','topic');
            UPDATE nodes SET state='pending',digest=NULL WHERE kind='file' AND digest IS NOT NULL;
            UPDATE catalogue SET generation=generation+1,scanning=1;
            PRAGMA user_version=1;
            COMMIT;
        """)
        connection.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if version < 2:
        # Older indexes stored explicit paths and wikilinks in the same format.
        connection.executescript("""
            BEGIN IMMEDIATE;
            CREATE TEMP TABLE wiki_upgrade AS
                SELECT id FROM nodes WHERE kind='file' AND state='indexed'
                AND (lower(name) LIKE '%.md' OR lower(name) LIKE '%.markdown');
            DELETE FROM passages WHERE node_id IN (SELECT id FROM wiki_upgrade);
            DELETE FROM declared_links WHERE source IN (SELECT id FROM wiki_upgrade);
            DELETE FROM topics WHERE source IN (SELECT id FROM wiki_upgrade);
            DELETE FROM edges WHERE kind IN ('reference','duplicate','topic')
                AND (source IN (SELECT id FROM wiki_upgrade) OR target IN (SELECT id FROM wiki_upgrade));
            UPDATE nodes SET state='pending',digest=NULL WHERE id IN (SELECT id FROM wiki_upgrade);
            UPDATE catalogue SET generation=generation+1,scanning=1
                WHERE EXISTS (SELECT 1 FROM wiki_upgrade);
            DROP TABLE wiki_upgrade;
            PRAGMA user_version=2;
            COMMIT;
        """)
    connection.commit()
    return connection


def advance_name_index(db: sqlite3.Connection, limit: int = 8192) -> bool:
    """Backfill one committed range of legacy filenames using the existing writer."""
    through, ready = db.execute("SELECT through_row,ready FROM filename_index_progress").fetchone()
    if ready:
        return True
    last = db.execute("SELECT coalesce(max(rowid),0) FROM nodes").fetchone()[0]
    end = db.execute(
        "SELECT max(rowid) FROM (SELECT rowid FROM nodes WHERE rowid>? ORDER BY rowid LIMIT ?)",
        (through, limit),
    ).fetchone()[0]
    with db:
        if end is not None:
            db.execute(
                "INSERT INTO filename_search(rowid,name) SELECT rowid,name FROM nodes "
                "WHERE rowid>? AND rowid<=?",
                (through, end),
            )
        ready = end is None or end == last
        db.execute(
            "UPDATE filename_index_progress SET through_row=?,ready=?", (end or through, ready)
        )
    return ready


def _graph_revision(db):
    # The current reader can be opened before the worker has applied the schema.
    if db.execute("SELECT 1 FROM sqlite_master WHERE name='graph_revision'").fetchone():
        return db.execute("SELECT revision FROM graph_revision").fetchone()[0]
    return None


class Catalogue:
    def __init__(self, database: Path | None = None):
        self.database = Path(database or default_database()).absolute()
        self._bundle_cache = OrderedDict()
        self._bundle_lock = Lock()

    @contextmanager
    def _read(self):
        if not self.database.is_file():
            yield None
            return
        safe_path(self.database)
        with closing(
            sqlite3.connect(self.database.as_uri() + "?mode=ro", uri=True, timeout=3)
        ) as db:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA query_only=ON")
            db.execute("BEGIN")
            available = db.execute("SELECT 1 FROM sqlite_master WHERE name='catalogue'").fetchone()
            yield db if available else None

    def revision(self) -> int:
        with self._read() as db:
            return 0 if db is None else db.execute("SELECT generation FROM catalogue").fetchone()[0]

    def graph_status(self) -> dict:
        return self.status(summary=True)

    def extraction_counts(self) -> dict | None:
        """Summarise terminal file states without scanning a large catalogue."""
        with self._read() as db:
            if db is None:
                return None
            low, high = db.execute("""
                SELECT (SELECT min(rowid) FROM nodes), (SELECT max(rowid) FROM nodes)
            """).fetchone()
            if low is None:
                return {"extracted": 0, "skipped": 0, "failed": 0, "estimated": False}
            span = high - low + 1
            estimated = span > EXTRACTION_SAMPLE_BLOCKS * EXTRACTION_SAMPLE_ROWS
            ranges = [(low, high)]
            if estimated:
                rng = Random(0)
                ranges = []
                for block in range(EXTRACTION_SAMPLE_BLOCKS):
                    start = low + span * block // EXTRACTION_SAMPLE_BLOCKS
                    end = low + span * (block + 1) // EXTRACTION_SAMPLE_BLOCKS
                    sample = rng.randrange(start, end - EXTRACTION_SAMPLE_ROWS + 1)
                    ranges.append((sample, sample + EXTRACTION_SAMPLE_ROWS - 1))

            intervals = ",".join("(?,?)" for _ in ranges)
            rows, extracted, skipped, failed = db.execute(
                f"""
                WITH sample(low, high) AS (VALUES {intervals})
                SELECT count(*),
                    count(*) FILTER (WHERE kind='file' AND state='indexed'),
                    count(*) FILTER (WHERE kind='file' AND state NOT IN (
                        'pending','indexed','failed','inaccessible'
                    )),
                    count(*) FILTER (WHERE kind='file' AND state IN ('failed','inaccessible'))
                FROM sample CROSS JOIN nodes WHERE rowid BETWEEN sample.low AND sample.high
                    AND kind<>'commit'
                """,
                [bound for interval in ranges for bound in interval],
            ).fetchone()
            if not estimated:
                return {
                    "extracted": extracted,
                    "skipped": skipped,
                    "failed": failed,
                    "estimated": False,
                }
            if not rows:
                return None
            total = db.execute("""
                SELECT coalesce(sum(1+descendants),0) FROM nodes
                WHERE parent_id IS NULL AND kind='root'
            """).fetchone()[0]
            return {
                "extracted": round(total * extracted / rows),
                "skipped": round(total * skipped / rows),
                "failed": round(total * failed / rows),
                "estimated": True,
            }

    def status(self, *, summary=False) -> dict:
        status = {
            "generation": 0,
            "nodes": 0,
            "inventory_pending": 0,
            "inventory_finished": 0,
            "inventory_offline": 0,
            "inventory_visited": 0,
            "content_pending": 0,
            "scanning": False,
            "message": "Waiting for the local index worker.",
        }
        if not summary:
            status.update(files=0, folders=0, indexed=0, edges=0)
        with self._read() as db:
            if db is None:
                return status
            status.update(
                db.execute("SELECT generation,scanning,message FROM catalogue").fetchone()
            )
            if summary:
                status["graph_revision"] = _graph_revision(db)
                # Folder counters already track the physical inventory at each commit.
                status.update(
                    db.execute("""
                    SELECT coalesce(sum(1+descendants),0) nodes,
                        CASE WHEN count(*) FILTER (WHERE state<>'online') = 0
                            THEN (SELECT count(*) FROM nodes WHERE state='pending')
                            ELSE (SELECT count(*) FROM nodes n JOIN nodes r ON r.id=n.root_id
                                WHERE n.state='pending' AND r.state='online')
                        END content_pending
                    FROM nodes WHERE parent_id IS NULL AND kind='root'
                """).fetchone()
                )
            else:
                status.update(
                    db.execute("""
                SELECT count(*) nodes, count(*) FILTER (WHERE kind='file') files,
                    count(*) FILTER (WHERE kind IN ('root','folder','project')) folders,
                    count(*) FILTER (WHERE state='indexed') AS "indexed",
                    (SELECT count(*) FROM edges) + count(parent_id) edges,
                    count(*) FILTER (WHERE state='pending' AND root_id IN (
                        SELECT id FROM nodes WHERE parent_id IS NULL AND kind='root' AND state='online'
                    )) content_pending
                FROM nodes
                """).fetchone()
                )
            if db.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='inventory_roots'"
            ).fetchone():
                status.update(
                    db.execute("""
                    SELECT count(*) FILTER (WHERE phase='pending') inventory_pending,
                        count(*) FILTER (WHERE phase='complete') inventory_finished,
                        count(*) FILTER (WHERE phase='offline') inventory_offline,
                        coalesce(sum(visited) FILTER (WHERE phase='pending'),0) inventory_visited
                    FROM inventory_roots
                """).fetchone()
                )
        status["scanning"] = bool(status["scanning"])
        return status

    @staticmethod
    def _effective(db, rows, inactive=None):
        result = []
        if inactive is None:
            inactive = {
                row[0]: row[1]
                for row in db.execute(
                    """SELECT id,state FROM nodes WHERE parent_id IS NULL AND kind='root'
                    AND state IN ('offline','inaccessible')"""
                )
            }
        for row in rows:
            item = dict(row)
            runtime = metadata_only_runtime(Path(item["path"]))
            if runtime and item["kind"] == "file":
                if "match_start" in item:
                    continue
                item["state"], item["digest"] = "protected", None
            if item["root_id"] in inactive:
                item["state"] = inactive[item["root_id"]]
            item["reason"] = {
                "offline": "Source drive or folder is offline.",
                "inaccessible": "Windows did not allow access to this location.",
                "protected": "Credential or restricted content was excluded.",
                "redirected": "Link or junction was recorded without following it.",
                "placeholder": "Cloud placeholder was recorded without downloading it.",
                "metadata": "No supported searchable content was available.",
                "encrypted": "Document is encrypted.",
                "oversized": "File exceeds the 16 MiB extraction limit.",
                "failed": "Local parser failed or reached an extraction limit.",
            }.get(item["state"], "")
            if runtime:
                item["reason"] = "Dodo Mem runtime credentials or derived cache; metadata only."
            result.append(item)
        return result

    @classmethod
    def _get(cls, db, field, value):
        row = db.execute(f"SELECT {NODE_FIELDS} FROM nodes WHERE {field}=?", (value,)).fetchone()
        if row is None:
            return None
        root = db.execute("SELECT state FROM nodes WHERE id=?", (row["root_id"],)).fetchone()
        inactive = {row["root_id"]: root[0]} if root and root[0] in GRAPH_BLOCKED else {}
        rows = cls._effective(db, (row,), inactive)
        return rows[0] if rows else None

    def node(self, node_id: str) -> dict | None:
        with self._read() as db:
            return self._get(db, "id", node_id) if db is not None else None

    def resolve_path(self, path: str | Path) -> dict | None:
        with self._read() as db:
            return self._get(db, "path_key", path_key(path)) if db is not None else None

    def roots(self) -> list[dict]:
        with self._read() as db:
            if db is None:
                return []
            return self._effective(
                db,
                db.execute(f"""SELECT {NODE_FIELDS} FROM nodes
                    WHERE parent_id IS NULL AND kind='root' ORDER BY path_key"""),
            )

    def addin_packages(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._read() as db:
            if db is None:
                return []
            return self._effective(
                db,
                db.execute(
                    f"""
                    SELECT {NODE_FIELDS} FROM nodes
                    WHERE kind='file' AND category='add-in' AND lower(name) LIKE '%.dodo-pack.zip'
                    AND state NOT IN ('offline','inaccessible','protected','redirected','placeholder')
                    AND root_id IN (SELECT id FROM nodes WHERE kind='root' AND state='online')
                    ORDER BY name COLLATE NOCASE,id LIMIT ? OFFSET ?
                    """,
                    (bounded(limit), max(0, offset)),
                ),
            )

    @classmethod
    def _children(cls, db, node_id, limit, offset):
        return cls._effective(
            db,
            db.execute(
                f"""
            SELECT {NODE_FIELDS} FROM nodes WHERE parent_id=? AND kind<>'commit'
            ORDER BY CASE WHEN kind IN ('folder','project') THEN 0 ELSE 1 END,name COLLATE NOCASE,id
            LIMIT ? OFFSET ?
        """,
                (node_id, bounded(limit), max(0, offset)),
            ),
        )

    def children(self, node_id: str, limit: int = 100, offset: int = 0) -> list[dict]:
        with self._read() as db:
            return self._children(db, node_id, limit, offset) if db is not None else []

    def search(self, query: str, limit: int = 20, scope: str | None = None) -> list[dict]:
        terms = re.findall(r"[^\W_]+", query[:500].casefold())[:12]
        if not terms:
            return []
        maximum = bounded(limit, 100)
        with self._read() as db:
            if db is None:
                return []
            predicate, args = "", []
            if scope is not None:
                selected = db.execute(
                    "SELECT path_key,project_id FROM nodes WHERE id=?", (scope,)
                ).fetchone()
                if not selected:
                    return []
                prefix = selected["path_key"].rstrip(os.sep) + os.sep
                predicate = "AND (n.id=? OR n.project_id=? OR n.path_key BETWEEN ? AND ?)"
                args.extend((scope, scope, prefix, prefix + "\U0010ffff"))
            match = " AND ".join(f'"{term}"*' for term in terms)
            fields = ",".join(f"n.{field}" for field in NODE_FIELDS.split(","))
            filename = query.strip()
            if "." in filename and os.path.basename(filename) == filename:
                exact = self._effective(
                    db,
                    db.execute(
                        f"""
                        SELECT {fields} FROM nodes n CROSS JOIN nodes r ON r.id=n.root_id
                        WHERE n.kind='file' AND n.name=? COLLATE NOCASE
                        AND r.state NOT IN ('offline','inaccessible') {predicate}
                        ORDER BY n.name COLLATE NOCASE,n.id LIMIT ?
                        """,
                        (filename, *args, maximum),
                    ),
                )
                if exact:
                    return exact
            records = self._effective(
                db,
                db.execute(
                    f"""
                SELECT {fields},min(hits.start) AS match_start FROM (
                    SELECT p.node_id,p.start,rank AS score FROM passage_search
                    JOIN passages p ON p.id=passage_search.rowid
                    JOIN nodes n ON n.id=p.node_id JOIN nodes r ON r.id=n.root_id
                    WHERE passage_search MATCH ? AND n.state='indexed'
                    AND r.state NOT IN ('offline','inaccessible') {predicate}
                    ORDER BY rank LIMIT 1000
                ) hits JOIN nodes n ON n.id=hits.node_id
                GROUP BY n.id ORDER BY min(hits.score) LIMIT ?
            """,
                    (match, *args, maximum),
                ),
            )
            if len(records) < maximum:
                patterns = [f"%{term}%" for term in terms]
                anchor = max(terms, key=len)
                source = "nodes n"
                indexed_name = ""
                if (
                    len(anchor) >= 3
                    and db.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='filename_index_progress'"
                    ).fetchone()
                    and db.execute("SELECT ready FROM filename_index_progress").fetchone()[0]
                ):
                    source = "filename_search CROSS JOIN nodes n ON n.rowid=filename_search.rowid"
                    indexed_name = "filename_search.name LIKE ? AND "
                    patterns.insert(0, f"%{anchor}%")
                names = indexed_name + " AND ".join("lower(n.name) LIKE ?" for _ in terms)
                known = {row["id"] for row in records}
                for row in self._effective(
                    db,
                    db.execute(
                        f"""
                    SELECT {fields} FROM {source} JOIN nodes r ON r.id=n.root_id
                    WHERE {names} AND r.state NOT IN ('offline','inaccessible') {predicate}
                    ORDER BY n.name COLLATE NOCASE LIMIT ?
                """,
                        (*patterns, *args, maximum),
                    ),
                ):
                    if row["id"] not in known:
                        records.append(row)
            return records[:maximum]

    def neighbours(
        self, node_id: str, limit: int = 40, include_inferred: bool = False
    ) -> list[dict]:
        maximum = bounded(limit)
        with self._read() as db:
            if db is None or (node := self._get(db, "id", node_id)) is None:
                return []
            if node["state"] in {
                "offline",
                "inaccessible",
                "protected",
                "redirected",
                "placeholder",
            }:
                return []
            rows = db.execute(
                """
                SELECT target AS other,source,target,kind,evidence
                FROM edges WHERE source=? AND (? OR kind<>'topic')
                UNION ALL
                SELECT source AS other,source,target,kind,evidence
                FROM edges WHERE target=? AND (? OR kind<>'topic')
                ORDER BY kind,other LIMIT ?
            """,
                (node_id, include_inferred, node_id, include_inferred, maximum),
            )
            result = []
            for row in rows:
                other = self._get(db, "id", row["other"])
                if other and other["state"] not in {
                    "offline",
                    "inaccessible",
                    "protected",
                    "redirected",
                    "placeholder",
                }:
                    result.append(
                        {
                            "node": other,
                            "source": row["source"],
                            "target": row["target"],
                            "kind": row["kind"],
                            "evidence": row["evidence"],
                        }
                    )
            if len(result) < maximum and node["parent_id"]:
                parent = self._get(db, "id", node["parent_id"])
                if parent:
                    result.append(
                        {
                            "node": parent,
                            "source": parent["id"],
                            "target": node_id,
                            "kind": "contains",
                            "evidence": "Parent folder",
                        }
                    )
            result.extend(
                {
                    "node": child,
                    "source": node_id,
                    "target": child["id"],
                    "kind": "contains",
                    "evidence": "Child entry",
                }
                for child in self._children(db, node_id, maximum, 0)[
                    : max(0, maximum - len(result))
                ]
            )
            return result

    def read_evidence(self, node_id: str, limit: int = 1000, start_line: int = 1) -> dict | None:
        with self._read() as db:
            if db is None or (node := self._get(db, "id", node_id)) is None:
                return None
            if node["state"] != "indexed" or not node["digest"]:
                return None
            passage = db.execute(
                """
                SELECT kind,start,end,text FROM passages WHERE node_id=?
                ORDER BY CASE WHEN end>=? THEN 0 ELSE 1 END,start LIMIT 1
            """,
                (node_id, max(1, start_line)),
            ).fetchone()
        if passage is None or SECRET_BYTES.search(
            passage["text"].encode("utf-8", errors="replace")
        ):
            return None
        if node["kind"] == "commit":
            from .whole_index_git import verify_commit

            valid = verify_commit(node["path"], node["digest"])
        else:
            try:
                path = Path(node["path"])
                valid = (
                    hashlib.sha256(
                        read_bytes(path, internal=history_source(path) in {"codex", "claude"})
                    ).hexdigest()
                    == node["digest"]
                )
            except (OSError, ValueError):
                return None
        if not valid:
            return None
        latest = self.node(node_id)
        if latest is None or latest["state"] != "indexed" or latest["digest"] != node["digest"]:
            return None
        text = passage["text"]
        kind, first, end = passage["kind"], passage["start"], passage["end"]
        maximum = bounded(limit, 12000)
        if kind == "lines":
            offset = min(max(start_line - first, 0), max(end - first, 0))
            text = "\n".join(text.splitlines()[offset:])[:maximum]
            first += offset
            locator = {"kind": "lines", "start": first, "end": first + text.count("\n")}
        elif kind == "page":
            locator = {"kind": "page", "page": first}
        elif kind == "commit":
            locator = {"kind": "commit", "commit": node["digest"]}
        else:
            locator = {"kind": kind, "start": first, "end": end}
        return {
            "node_id": node_id,
            "path": node["path"],
            "digest": node["digest"],
            "locator": locator,
            "text": text[:maximum],
        }

    @staticmethod
    def _visible_edges(db, nodes, kind=None):
        ids = {
            node["id"]
            for node in nodes
            if node["state"]
            not in {"offline", "inaccessible", "protected", "redirected", "placeholder"}
        }
        if not ids:
            return []
        bind = ",".join("?" for _ in ids)
        predicate = "AND kind=?" if kind else ""
        args = (*ids, *ids, kind) if kind else (*ids, *ids)
        return [
            dict(row)
            for row in db.execute(
                f"""
            SELECT source,target,kind,evidence FROM edges
            WHERE source IN ({bind}) AND target IN ({bind}) {predicate}
            ORDER BY source,target,kind LIMIT 12000
        """,
                args,
            )
        ]

    @staticmethod
    def _cluster_edges(db, nodes):
        scopes = []
        for node in nodes:
            if node["state"] in {
                "offline",
                "inaccessible",
                "protected",
                "redirected",
                "placeholder",
            }:
                continue
            low = path_key(node["path"]).rstrip(os.sep) + os.sep
            high = low + "\U0010ffff" if node["kind"] in {"root", "folder", "project"} else low
            scopes.append((node["id"], low, high))
        if not scopes:
            return []
        values = ",".join("(?,?,?)" for _ in scopes)
        args = [item for scope in scopes for item in scope]
        return [
            dict(row)
            for row in db.execute(
                f"""
            WITH buckets(id,low,high) AS (VALUES {values}),
            members(node,bucket) AS MATERIALIZED (
                SELECT id,id FROM buckets
                UNION ALL
                SELECT n.id,b.id FROM buckets b JOIN nodes n ON n.path_key BETWEEN b.low AND b.high
                WHERE n.kind<>'commit' AND n.state NOT IN (
                    'offline','inaccessible','protected','redirected','placeholder'
                )
            )
            SELECT m.bucket source,t.bucket target,e.kind,count(*) count,
                CASE WHEN count(*)=1 THEN min(e.evidence) ELSE count(*)||' indexed connections' END evidence
            FROM members m JOIN edges e ON e.source=m.node JOIN members t ON t.node=e.target
            WHERE m.bucket<>t.bucket AND e.kind IN ('reference','duplicate','topic')
            GROUP BY m.bucket,t.bucket,e.kind ORDER BY source,target,kind LIMIT 12000
        """,
                args,
            )
        ]

    def projection(self, scope: str | None = None, offset: int = 0, limit: int = 120) -> dict:
        with self._read() as db:
            if db is None:
                return {"generation": 0, "scope": None, "nodes": [], "edges": [], "total": 0}
            generation = db.execute("SELECT generation FROM catalogue").fetchone()[0]
            selected = self._get(db, "id", scope) if scope is not None else None
            if scope is None:
                nodes = self._effective(
                    db,
                    db.execute(
                        f"""
                    SELECT {NODE_FIELDS} FROM nodes
                    WHERE parent_id IN (SELECT id FROM nodes WHERE kind='root')
                    OR (kind='root' AND descendants=0)
                    ORDER BY name COLLATE NOCASE,id LIMIT ? OFFSET ?
                """,
                        (bounded(limit), max(0, offset)),
                    ),
                )
                total = db.execute("""
                    SELECT count(*) FROM nodes WHERE parent_id IN (
                        SELECT id FROM nodes WHERE kind='root'
                    ) OR (kind='root' AND descendants=0)
                """).fetchone()[0]
            else:
                nodes = self._children(db, scope, limit, offset)
                total = db.execute(
                    "SELECT count(*) FROM nodes WHERE parent_id=? AND kind<>'commit'", (scope,)
                ).fetchone()[0]
            return {
                "generation": generation,
                "scope": selected,
                "nodes": nodes,
                "edges": self._cluster_edges(db, nodes),
                "total": total,
            }

    def _graph_bundles(self, db, revision, hubs, kinds, maximum):
        key = revision, tuple(node["id"] for node in hubs), kinds
        with self._bundle_lock:
            cached = self._bundle_cache.pop(key, None)
            if cached is not None:
                self._bundle_cache[key] = cached
        if cached is None:
            groups = Counter()
            representatives = {}
            active = [node for node in hubs if node["state"] not in GRAPH_BLOCKED]
            if active:
                marks = ",".join("?" for _ in kinds)
                folders_by_root = {}
                active_ids = {node["id"] for node in active}
                for hub in active:
                    if hub["kind"] in {"root", "folder", "project"}:
                        prefix = path_key(hub["path"]).rstrip(os.sep) + os.sep
                        folders_by_root.setdefault(hub["root_id"], {})[prefix] = hub["id"]

                @lru_cache(maxsize=8192)
                def cluster_ids(node, root, path, kind):
                    ids = [node] if node in active_ids else []
                    folders = folders_by_root.get(root)
                    if folders and kind != "commit":
                        end = path.find(os.sep)
                        while end != -1:
                            if hub := folders.get(path[: end + 1]):
                                ids.append(hub)
                            end = path.find(os.sep, end + 1)
                    return ids

                # Start with relationships. Walking every file under each cluster made
                # a mostly unlinked catalogue with millions of entries too slow to open.
                db.execute("PRAGMA cache_size=-32768")  # Retain node pages for both joins.
                rows = db.execute(
                    f"""
                    SELECT e.source,e.target,e.kind,a.root_id,a.path_key,a.kind,
                           b.root_id,b.path_key,b.kind
                    FROM edges e CROSS JOIN nodes a ON a.id=e.source
                    CROSS JOIN nodes b ON b.id=e.target
                    WHERE e.kind IN ({marks})
                    AND a.state NOT IN ({GRAPH_BLOCKED_SQL})
                    AND b.state NOT IN ({GRAPH_BLOCKED_SQL})
                    """,
                    kinds,
                )
                for a, b, kind, a_root, a_path, a_kind, b_root, b_path, b_kind in rows:
                    for source in cluster_ids(a, a_root, a_path, a_kind):
                        for target in cluster_ids(b, b_root, b_path, b_kind):
                            if source != target:
                                first, second = sorted((source, target))
                                groups[first, second, kind] += 1
                            if a_kind == b_kind == "file":
                                pair = (kind == "topic", *sorted((a, b)))
                                for hub in {source, target}:
                                    options = representatives.setdefault(hub, set())
                                    options.add(pair)
                                    if len(options) > 2:
                                        options.remove(max(options))
            ranked = sorted(groups.items(), key=lambda pair: (-pair[1], pair[0]))[:GRAPH_EDGE_LIMIT]
            pairs = sorted(set().union(*representatives.values()))[:8]
            pins = tuple(dict.fromkeys(identity for _, a, b in pairs for identity in (a, b)))
            cached = ranked, sum(groups.values()), pins
            with self._bundle_lock:
                self._bundle_cache[key] = cached
                if len(self._bundle_cache) > 4:
                    self._bundle_cache.popitem(last=False)
        ranked, linked, pins = cached
        return (
            [
                {
                    "source": source,
                    "target": target,
                    "kind": kind,
                    "count": count,
                    "evidence": "",
                    "bundle": True,
                }
                for (source, target, kind), count in ranked[:maximum]
            ],
            linked,
            pins,
        )

    @classmethod
    def _graph_members(cls, db, hubs, expanded, inactive, selected, pins=()):
        nodes = {node["id"]: {**node, "cluster_id": node["id"]} for node in hubs}
        if selected:
            # Scoped diagrams keep their real parent and immediate-child lines.
            nodes.setdefault(selected["id"], {**selected, "cluster_id": selected["id"]})
        # Keep expanded members and paths to a few real links before sampling other files.
        for identity in dict.fromkeys((*expanded, *pins)):
            lineage, ancestor = [], identity
            while ancestor and ancestor not in nodes and len(lineage) < 64:
                rows = cls._effective(
                    db,
                    db.execute(f"SELECT {NODE_FIELDS} FROM nodes WHERE id=?", (ancestor,)),
                    inactive,
                )
                if not rows or rows[0]["state"] in GRAPH_BLOCKED or rows[0]["kind"] == "commit":
                    break
                lineage.append(rows[0])
                ancestor = rows[0]["parent_id"]
            if ancestor in nodes and len(nodes) + len(lineage) <= GRAPH_NODE_LIMIT:
                cluster = nodes[ancestor]["cluster_id"]
                for node in reversed(lineage):
                    nodes[node["id"]] = {**node, "cluster_id": cluster}

        def sample(branch, maximum):
            if branch["state"] in GRAPH_BLOCKED or not branch["descendants"]:
                return
            remaining = min(maximum, GRAPH_NODE_LIMIT - len(nodes))
            if remaining <= 0:
                return
            frontier = [branch]
            visible = Counter(node["parent_id"] for node in nodes.values())
            deferred = []

            def add(node):
                nodes[node["id"]] = {**node, "cluster_id": branch["cluster_id"]}

            for depth in range(3):
                if not frontier or not remaining:
                    break
                breadth = max(4, 2 * ((remaining + len(frontier) - 1) // len(frontier)))
                groups = [
                    cls._effective(
                        db,
                        db.execute(
                            f"""
                            SELECT {NODE_FIELDS} FROM nodes WHERE parent_id=?
                            AND kind<>'commit' AND state NOT IN ({GRAPH_BLOCKED_SQL})
                            ORDER BY (descendants=0),name COLLATE NOCASE,id LIMIT ?
                            """,
                            (parent["id"], breadth + visible[parent["id"]]),
                        ),
                        inactive,
                    )
                    for parent in frontier
                ]
                candidates = [
                    node for batch in zip_longest(*groups) for node in batch if node is not None
                ]
                quota = remaining
                if depth < 2 and any(node["descendants"] for node in candidates):
                    # Keep part of the cluster budget for children of these folders.
                    quota = max(1, (remaining * 2 + 2) // 3)
                frontier = []
                for node in candidates:
                    if node["id"] not in nodes:
                        if quota <= 0:
                            deferred.append(node)
                            continue
                        add(node)
                        quota -= 1
                        remaining -= 1
                    if node["descendants"]:
                        frontier.append(node)
            for node in deferred[:remaining]:
                add(node)

        overview_slots = GRAPH_NODE_LIMIT - 120 - len(hubs) - bool(selected)
        sample_size = min(48, max(1, overview_slots // max(1, len(hubs))))
        for node in hubs:
            sample(nodes[node["id"]], sample_size)
        for identity in expanded:
            if identity in nodes:
                sample(nodes[identity], 48)
        return nodes

    @staticmethod
    def _graph_link_members(db, nodes, kinds):
        remaining = min(40, GRAPH_NODE_LIMIT - len(nodes))
        if remaining <= 0:
            return
        ids = tuple(
            identity for identity, node in nodes.items() if node["state"] not in GRAPH_BLOCKED
        )
        if not ids:
            return
        endpoints = ",".join("?" for _ in ids)
        marks = ",".join("?" for _ in kinds)
        rows = db.execute(
            f"""
            WITH outgoing AS (
                SELECT target AS id FROM edges
                WHERE source IN ({endpoints}) AND kind IN ({marks}) LIMIT 1600
            ), incoming AS (
                SELECT source AS id FROM edges
                WHERE target IN ({endpoints}) AND kind IN ({marks}) LIMIT 1600
            ), related AS (
                SELECT id FROM outgoing UNION SELECT id FROM incoming
            )
            SELECT {NODE_FIELDS} FROM nodes
            WHERE id IN (SELECT id FROM related) AND id NOT IN ({endpoints})
              AND parent_id IN ({endpoints}) AND kind='file'
              AND state NOT IN ({GRAPH_BLOCKED_SQL})
            ORDER BY id LIMIT ?
            """,
            (*ids, *kinds, *ids, *kinds, *ids, *ids, remaining),
        )
        for row in rows:
            parent = nodes[row["parent_id"]]
            if row["root_id"] == parent["root_id"]:
                nodes[row["id"]] = {**dict(row), "cluster_id": parent["cluster_id"]}

    @staticmethod
    def _graph_direct(db, nodes, kinds, maximum):
        ids = [node["id"] for node in nodes.values() if node["state"] not in GRAPH_BLOCKED]
        if len(ids) < 2 or maximum <= 0:
            return []
        endpoints = ",".join("?" for _ in ids)
        marks = ",".join("?" for _ in kinds)
        return [
            dict(row)
            for row in db.execute(
                f"""
                SELECT source,target,kind,evidence,1 count FROM edges
                WHERE source IN ({endpoints}) AND target IN ({endpoints}) AND kind IN ({marks})
                ORDER BY source,target,kind LIMIT ?
                """,
                (*ids, *ids, *kinds, maximum),
            )
            if nodes[row["source"]]["cluster_id"] != row["source"]
            or nodes[row["target"]]["cluster_id"] != row["target"]
        ]

    def graph_projection(
        self, scope=None, offset=0, limit=120, *, expanded=(), include_inferred=False
    ):
        """Sample real nodes for rendering while counting the underlying inventory and bundles."""
        empty = {
            "generation": 0,
            "graph_revision": None,
            "scope": None,
            "nodes": [],
            "hubs": [],
            "edges": [],
            "total": 0,
            "counts": {"nodes": 0, "containment": 0, "bundled": 0},
        }
        with self._read() as db:
            if db is None:
                return empty
            generation = db.execute("SELECT generation FROM catalogue").fetchone()[0]
            graph_revision = _graph_revision(db)
            roots = db.execute(
                f"SELECT {NODE_FIELDS} FROM nodes WHERE parent_id IS NULL AND kind='root'"
            ).fetchall()
            inactive = {row["id"]: row["state"] for row in roots if row["state"] in GRAPH_BLOCKED}
            selected = None
            if scope is None:
                condition = """
                    (parent_id IN (SELECT id FROM nodes WHERE parent_id IS NULL AND kind='root')
                    OR (parent_id IS NULL AND kind='root' AND descendants=0))
                """
                args = ()
                all_items = sum(1 + root["descendants"] for root in roots)
                hierarchy = all_items - len(roots)
            else:
                rows = self._effective(
                    db,
                    db.execute(f"SELECT {NODE_FIELDS} FROM nodes WHERE id=?", (scope,)),
                    inactive,
                )
                if not rows:
                    return {**empty, "generation": generation, "graph_revision": graph_revision}
                selected = rows[0]
                condition, args = (
                    ("parent_id=? AND kind<>'commit'", (scope,))
                    if selected["descendants"]
                    else ("id=?", (scope,))
                )
                hierarchy = selected["descendants"]
                all_items = 1 + hierarchy
            hubs = self._effective(
                db,
                db.execute(
                    f"""
                    SELECT {NODE_FIELDS} FROM nodes WHERE {condition}
                    ORDER BY name COLLATE NOCASE,id LIMIT ? OFFSET ?
                    """,
                    (*args, bounded(limit, 120), max(0, offset)),
                ),
                inactive,
            )
            total = db.execute(f"SELECT count(*) FROM nodes WHERE {condition}", args).fetchone()[0]
            expanded = tuple(sorted(set(expanded)))[:16]
            kinds = GRAPH_RELATIONS if include_inferred else GRAPH_RELATIONS[:-1]
            bundles, linked, pins = self._graph_bundles(
                db,
                ("catalogue", generation) if graph_revision is None else ("graph", graph_revision),
                hubs,
                kinds,
                GRAPH_EDGE_LIMIT,
            )
            nodes = self._graph_members(db, hubs, expanded, inactive, selected, pins)
            has_links = db.execute("SELECT 1 FROM edges LIMIT 1").fetchone() is not None
            if has_links:
                self._graph_link_members(db, nodes, kinds)
            edges = [
                {
                    "source": node["parent_id"],
                    "target": node["id"],
                    "kind": "contains",
                    "evidence": "Parent folder",
                    "count": 1,
                }
                for node in nodes.values()
                if node["parent_id"] in nodes
                and node["state"] not in GRAPH_BLOCKED
                and nodes[node["parent_id"]]["state"] not in GRAPH_BLOCKED
            ]
            edges.extend(bundles[: (GRAPH_EDGE_LIMIT - len(edges)) * 3 // 4])
            if has_links:
                edges.extend(self._graph_direct(db, nodes, kinds, GRAPH_EDGE_LIMIT - len(edges)))
            return {
                "generation": generation,
                "graph_revision": graph_revision,
                "scope": selected,
                "nodes": list(nodes.values()),
                "hubs": [node["id"] for node in hubs],
                "edges": edges,
                "total": total,
                "counts": {"nodes": all_items, "containment": hierarchy, "bundled": linked},
            }

    def graph_evidence(self, edge, *, limit=20):
        """Resolve displayed links to accessible file pairs and recheck their source bytes."""
        source, target, kind = edge.get("source"), edge.get("target"), edge.get("kind")
        if not source or not target or (kind != "contains" and kind not in GRAPH_RELATIONS):
            return []
        with self._read() as db:
            if db is None:
                return []
            endpoints = {key: self._get(db, "id", key) for key in (source, target)}
            if any(not row or row["state"] in GRAPH_BLOCKED for row in endpoints.values()):
                return []
            if kind == "contains":
                if endpoints[target]["parent_id"] != source:
                    return []
                return [
                    {
                        "node": endpoints[source],
                        "target": endpoints[target],
                        "source": None,
                        "evidence": "Parent folder",
                        "kind": kind,
                    }
                ]
            if edge.get("bundle"):
                bounds = []
                for node in endpoints.values():
                    low = path_key(node["path"]).rstrip(os.sep) + os.sep
                    high = (
                        low + "\U0010ffff" if node["kind"] in {"root", "folder", "project"} else low
                    )
                    bounds.extend((node["id"], node["root_id"], low, high))
                links = db.execute(
                    f"""
                    WITH buckets(id,root,low,high) AS (VALUES (?,?,?,?),(?,?,?,?))
                    SELECT e.source,e.target,e.evidence FROM edges e
                    CROSS JOIN nodes n ON n.id=e.source CROSS JOIN nodes m ON m.id=e.target
                    CROSS JOIN buckets a CROSS JOIN buckets b
                    WHERE a.id<>b.id AND e.kind=?
                    AND (n.id=a.id OR (n.root_id=a.root AND n.kind<>'commit'
                        AND n.path_key BETWEEN a.low AND a.high))
                    AND (m.id=b.id OR (m.root_id=b.root AND m.kind<>'commit'
                        AND m.path_key BETWEEN b.low AND b.high))
                    AND n.state NOT IN ({GRAPH_BLOCKED_SQL})
                    AND m.state NOT IN ({GRAPH_BLOCKED_SQL})
                    ORDER BY e.source,e.target LIMIT ?
                    """,
                    (*bounds, kind, bounded(limit, 20)),
                ).fetchall()
            else:
                links = db.execute(
                    "SELECT source,target,evidence FROM edges WHERE source=? AND target=? AND kind=?",
                    (source, target, kind),
                ).fetchall()
        return [
            item for link in links if (item := self._verify_graph_relation(link, kind)) is not None
        ]

    def _verify_graph_relation(self, link, kind):
        origin, destination = self.node(link["source"]), self.node(link["target"])
        if any(not row or row["state"] in GRAPH_BLOCKED for row in (origin, destination)):
            return None
        line = re.search(r"\bline ([1-9][0-9]*)\b", link["evidence"])
        source_node = destination if kind == "topic" else origin
        start_line = int(line[1]) if line else 1
        source = self.read_evidence(source_node["id"], 1500, start_line)
        if source is None:
            return None
        if line and source["locator"].get("start") != start_line:
            return None
        if kind == "duplicate":
            other = self.read_evidence(destination["id"], 1)
            if other is None or other["digest"] != source["digest"]:
                return None
        else:
            try:
                safe_path(Path(destination["path"]))
            except (OSError, ValueError):
                return None
        with self._read() as db:
            if (
                db is None
                or not db.execute(
                    f"""
                SELECT 1 FROM edges e JOIN nodes a ON a.id=e.source JOIN nodes b ON b.id=e.target
                JOIN nodes ra ON ra.id=a.root_id JOIN nodes rb ON rb.id=b.root_id
                WHERE e.source=? AND e.target=? AND e.kind=? AND e.evidence=?
                AND a.state NOT IN ({GRAPH_BLOCKED_SQL}) AND b.state NOT IN ({GRAPH_BLOCKED_SQL})
                AND ra.state NOT IN ({GRAPH_BLOCKED_SQL}) AND rb.state NOT IN ({GRAPH_BLOCKED_SQL})
                AND (CASE WHEN e.kind='topic' THEN b.digest ELSE a.digest END)=?
                """,
                    (origin["id"], destination["id"], kind, link["evidence"], source["digest"]),
                ).fetchone()
            ):
                return None
        return {
            "node": source_node,
            "target": origin if kind == "topic" else destination,
            "source": source,
            "evidence": link["evidence"],
            "kind": kind,
        }

    def git_history(self, repo_id: str, limit: int = 100, offset: int = 0) -> dict:
        with self._read() as db:
            if db is None:
                return {"generation": 0, "scope": None, "nodes": [], "edges": [], "total": 0}
            generation = db.execute("SELECT generation FROM catalogue").fetchone()[0]
            selected = self._get(db, "id", repo_id)
            nodes = self._effective(
                db,
                db.execute(
                    f"""
                SELECT {NODE_FIELDS} FROM nodes WHERE project_id=? AND kind='commit'
                ORDER BY mtime_ns DESC,id LIMIT ? OFFSET ?
            """,
                    (repo_id, bounded(limit), max(0, offset)),
                ),
            )
            total = db.execute(
                "SELECT count(*) FROM nodes WHERE project_id=? AND kind='commit'", (repo_id,)
            ).fetchone()[0]
            for node in nodes:
                metadata = db.execute(
                    "SELECT refs,files FROM commit_metadata WHERE node_id=?", (node["id"],)
                ).fetchone()
                if metadata:
                    node["refs"] = json.loads(metadata["refs"])
                    node["changed_files"] = json.loads(metadata["files"])
            return {
                "generation": generation,
                "scope": selected,
                "nodes": nodes,
                "edges": self._visible_edges(db, nodes, "commit_parent"),
                "total": total,
            }
