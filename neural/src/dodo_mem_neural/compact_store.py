"""Rebuildable neural metadata uses compact E: records and a C: cache binding."""

import hashlib
import json
import os
import re
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path

from .artifacts import atomic_json, replace_file, sha256
from .index_store import IndexStore
from .resources import check_resources

SCHEMA = """
CREATE TABLE IF NOT EXISTS neural_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(source_id TEXT PRIMARY KEY,digest TEXT NOT NULL,
    path TEXT NOT NULL,scope TEXT NOT NULL,version TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1,
    ready INTEGER NOT NULL DEFAULT 0,scan_epoch INTEGER NOT NULL DEFAULT 0,
    provider TEXT NOT NULL DEFAULT 'catalogue');
CREATE TABLE IF NOT EXISTS jobs(id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,
    version TEXT NOT NULL,status TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL,priority INTEGER NOT NULL DEFAULT 0,UNIQUE(source_id,version),
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE);
CREATE INDEX IF NOT EXISTS job_priority ON jobs(status,priority DESC,id);
CREATE TABLE IF NOT EXISTS embedding_spaces(id INTEGER PRIMARY KEY,encoder_id TEXT NOT NULL,
    chunker_id TEXT NOT NULL,UNIQUE(encoder_id,chunker_id));
CREATE TABLE IF NOT EXISTS chunk_records(id INTEGER PRIMARY KEY,identity BLOB NOT NULL UNIQUE,
    text_hash BLOB NOT NULL,text TEXT,space_id INTEGER NOT NULL REFERENCES embedding_spaces(id));
CREATE VIEW IF NOT EXISTS chunks AS SELECT c.id,c.identity,c.text_hash,c.text,
    e.encoder_id,e.chunker_id FROM chunk_records c JOIN embedding_spaces e ON e.id=c.space_id;
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(text,
    content='chunk_records',content_rowid='id');
CREATE TRIGGER IF NOT EXISTS compact_chunk_add AFTER INSERT ON chunk_records
WHEN new.text IS NOT NULL BEGIN
    INSERT INTO chunk_search(rowid,text) VALUES(new.id,new.text);
END;
CREATE TRIGGER IF NOT EXISTS compact_chunk_fill AFTER UPDATE OF text ON chunk_records
WHEN old.text IS NULL AND new.text IS NOT NULL BEGIN
    INSERT INTO chunk_search(rowid,text) VALUES(new.id,new.text);
END;
CREATE TABLE IF NOT EXISTS occurrence_records(chunk_id INTEGER NOT NULL REFERENCES chunk_records(id),
    job_id INTEGER NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,locator TEXT NOT NULL,
    start INTEGER NOT NULL,end INTEGER NOT NULL,role TEXT,timestamp TEXT,historical INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(job_id,locator)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS occurrence_key ON occurrence_records(chunk_id,job_id);
CREATE VIEW IF NOT EXISTS occurrences AS SELECT o.chunk_id,j.source_id,j.version,o.locator,
    o.start,o.end,o.role,o.timestamp,o.historical FROM occurrence_records o JOIN jobs j ON j.id=o.job_id;
CREATE TRIGGER IF NOT EXISTS compact_occurrence_add INSTEAD OF INSERT ON occurrences BEGIN
    INSERT INTO occurrence_records VALUES(new.chunk_id,
        (SELECT id FROM jobs WHERE source_id=new.source_id AND version=new.version),
        new.locator,new.start,new.end,new.role,new.timestamp,new.historical);
END;
CREATE TRIGGER IF NOT EXISTS compact_occurrence_remove INSTEAD OF DELETE ON occurrences BEGIN
    DELETE FROM occurrence_records WHERE job_id=(SELECT id FROM jobs
        WHERE source_id=old.source_id AND version=old.version) AND locator=old.locator;
END;
CREATE TABLE IF NOT EXISTS centroid_keys(id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,
    version TEXT NOT NULL,UNIQUE(source_id,version));
CREATE TABLE IF NOT EXISTS centroids(source_id TEXT PRIMARY KEY,version TEXT NOT NULL,
    vector_key INTEGER NOT NULL,vector BLOB NOT NULL,weight REAL NOT NULL);
CREATE UNIQUE INDEX IF NOT EXISTS centroid_vector ON centroids(vector_key);
CREATE TABLE IF NOT EXISTS neighbour_records(source_key INTEGER NOT NULL REFERENCES centroid_keys(id),
    target_key INTEGER NOT NULL REFERENCES centroid_keys(id),weight REAL NOT NULL,
    PRIMARY KEY(source_key,target_key)) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS neighbour_target ON neighbour_records(target_key,source_key);
CREATE VIEW IF NOT EXISTS semantic_neighbours AS SELECT s.source_id AS source,t.source_id AS target,
    s.version AS source_version,t.version AS target_version,n.weight FROM neighbour_records n
    JOIN centroid_keys s ON s.id=n.source_key JOIN centroid_keys t ON t.id=n.target_key;
CREATE TRIGGER IF NOT EXISTS compact_neighbour_add INSTEAD OF INSERT ON semantic_neighbours BEGIN
    INSERT INTO neighbour_records VALUES((SELECT id FROM centroid_keys
        WHERE source_id=new.source AND version=new.source_version),
        (SELECT id FROM centroid_keys WHERE source_id=new.target AND version=new.target_version),new.weight);
END;
CREATE TRIGGER IF NOT EXISTS compact_neighbour_remove INSTEAD OF DELETE ON semantic_neighbours BEGIN
    DELETE FROM neighbour_records WHERE source_key=(SELECT id FROM centroid_keys
        WHERE source_id=old.source AND version=old.source_version) AND target_key=(SELECT id
        FROM centroid_keys WHERE source_id=old.target AND version=old.target_version);
END;
CREATE TABLE IF NOT EXISTS support_records(source_key INTEGER NOT NULL,target_key INTEGER NOT NULL,
    source_chunk INTEGER NOT NULL,target_chunk INTEGER NOT NULL,source_similarity REAL NOT NULL,
    target_similarity REAL NOT NULL,source_sampled INTEGER NOT NULL,target_sampled INTEGER NOT NULL,
    source_total INTEGER NOT NULL,target_total INTEGER NOT NULL,
    PRIMARY KEY(source_key,target_key),CHECK(source_key<target_key)) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS communities(source_id TEXT PRIMARY KEY,source_version TEXT NOT NULL,
    community_id TEXT NOT NULL,generation TEXT NOT NULL,partial INTEGER NOT NULL);
"""


def compact_locator(value):
    try:
        record = json.loads(value)
    except (ValueError, TypeError):
        return value
    if isinstance(record, list) and len(record) == 4:
        # The exact chunk hash is stored once in chunk_records, not repeated in every locator.
        return json.dumps([record[0], None, record[2], record[3]], separators=(",", ":"))
    return value


class CompactIndexStore(IndexStore):
    def __init__(
        self,
        path,
        *,
        encoder_id="default",
        chunker_id="tokens224-overlap32-retokenise-v3-compact-locator",
        derived_dir=None,
        resource_guard=None,
    ):
        self.legacy_path = Path(path).absolute()
        self.derived_dir = Path(derived_dir).absolute() if derived_dir else None
        self.path = self.derived_dir / "index.sqlite3" if self.derived_dir else self.legacy_path
        self.binding = self.legacy_path.with_suffix(".cache.json") if self.derived_dir else None
        self.encoder_id, self.chunker_id = encoder_id, chunker_id
        self.resource_guard = resource_guard or (check_resources if self.derived_dir else None)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.resolve() != self.path or self.path.is_symlink():
            raise ValueError("Neural metadata storage cannot contain redirects.")
        if self.path.is_file():
            probe = self._connect(self.path, readonly=True)
            try:
                old_chunks = probe.execute(
                    "SELECT type FROM sqlite_master WHERE name='chunks'"
                ).fetchone()
                if old_chunks and old_chunks[0] == "table":
                    raise ValueError(
                        "Derived neural metadata has the legacy layout; migrate it from C: first."
                    )
            finally:
                probe.close()
        migrated = False
        if self.derived_dir and not self.path.exists() and self.legacy_path.is_file():
            guard = resource_guard or check_resources
            guard(background=True, additional_disk=self.legacy_path.stat().st_size * 3)
            temporary = self.path.with_suffix(f".migrating-{uuid.uuid4().hex}.sqlite3")
            self.db = self._connect(temporary)
            try:
                self.db.executescript(SCHEMA)
                self._migrate(self.legacy_path)
                self.db.close()
                with temporary.open("r+b") as stream:
                    os.fsync(stream.fileno())
                replace_file(temporary, self.path)
                migrated = True
            finally:
                self.db.close()
                if not migrated:
                    for suffix in ("", "-journal", "-wal", "-shm"):
                        temporary.with_name(temporary.name + suffix).unlink(missing_ok=True)
        self.db = self._connect(self.path)
        try:
            self.db.executescript(SCHEMA)
            self.cache_uuid = self.get_meta("cache_uuid")
            if self.cache_uuid is None:
                self.cache_uuid = uuid.uuid4().hex
                with self.db:
                    self.set_meta("cache_uuid", self.cache_uuid)
            if not re.fullmatch(r"[a-f0-9]{32}", self.cache_uuid):
                self.db.close()
                raise ValueError(
                    "Derived neural cache UUID must be exactly 32 hexadecimal characters."
                )
            create_binding = False
            if self.binding and self.binding.exists():
                bound = json.loads(self.binding.read_text(encoding="utf-8"))
                if bound["cache_uuid"] != self.cache_uuid or Path(bound["path"]) != self.path:
                    self.db.close()
                    raise ValueError(
                        "Neural metadata cache UUID differs from its C: publication binding."
                    )
            elif self.binding:
                create_binding = True
            self.space_id = self._space(encoder_id, chunker_id)
            version = json.dumps([encoder_id, chunker_id], separators=(",", ":"))
            previous = self.get_meta("index_version")
            with self.db:
                if self.get_meta("index_progress_epoch") is None:
                    self.reset_progress("Current derived publication counters initialised.")
                if migrated or previous is not None and previous != version:
                    self.db.execute("UPDATE sources SET ready=0")
                    self.db.execute("UPDATE jobs SET status='pending' WHERE status<>'obsolete'")
                    self.reset_progress(
                        "Derived cache migrated or embedding/chunking version changed."
                    )
                if self.get_meta("compact_rebuild_priority") is None:
                    # Inherited versions need revalidation; newly scanned work must not wait
                    # behind every old high-priority job before producing usable results.
                    self.db.execute("UPDATE jobs SET priority=-10 WHERE status='pending'")
                    self.set_meta("compact_rebuild_priority", "1")
                self.set_meta("index_version", version)
                self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'")
            if create_binding:
                atomic_json(
                    self.binding,
                    {
                        "cache_uuid": self.cache_uuid,
                        "path": str(self.path),
                        "migration_sha256": sha256(self.path),
                        "legacy_preserved": self.legacy_path.is_file(),
                    },
                )
        except BaseException:
            self.db.close()
            raise

    @staticmethod
    def _connect(path, *, readonly=False, timeout=10):
        connection = sqlite3.connect(
            Path(path).absolute().as_uri() + ("?mode=ro" if readonly else "?mode=rwc"),
            uri=True,
            timeout=timeout,
            check_same_thread=False,
        )
        try:
            connection.row_factory = sqlite3.Row
            if readonly:
                connection.execute("PRAGMA query_only=ON")
            else:
                connection.execute("PRAGMA journal_mode=DELETE")
                connection.execute("PRAGMA synchronous=FULL")
                connection.execute("PRAGMA foreign_keys=ON")
            return connection
        except BaseException:
            connection.close()
            raise

    def _space(self, encoder, chunker):
        self.db.execute(
            "INSERT OR IGNORE INTO embedding_spaces(encoder_id,chunker_id) VALUES (?,?)",
            (encoder, chunker),
        )
        return self.db.execute(
            "SELECT id FROM embedding_spaces WHERE encoder_id=? AND chunker_id=?",
            (encoder, chunker),
        ).fetchone()[0]

    @contextmanager
    def read_snapshot(self, *, timeout=10):
        snapshot = object.__new__(CompactIndexStore)
        for name in (
            "path",
            "encoder_id",
            "chunker_id",
            "space_id",
            "cache_uuid",
            "legacy_path",
            "binding",
            "derived_dir",
            "resource_guard",
        ):
            setattr(snapshot, name, getattr(self, name))
        snapshot.db = self._connect(self.path, readonly=True, timeout=timeout)
        try:
            if snapshot.get_meta("cache_uuid") != self.cache_uuid:
                raise OSError(
                    "Derived neural cache identity changed; its old vectors are ineligible."
                )
            snapshot.db.execute("BEGIN")
            yield snapshot
        finally:
            snapshot.db.close()

    def _available(self):
        if not self.path.is_file():
            raise OSError("Derived neural metadata is unavailable; use the source catalogue.")
        if (
            self.binding
            and json.loads(self.binding.read_text(encoding="utf-8"))["cache_uuid"]
            != self.cache_uuid
        ):
            raise OSError("Derived neural cache binding changed; its old vectors are ineligible.")

    def visible_occurrences(self, chunk_ids, scopes, *, limit=1000):
        self._available()
        return super().visible_occurrences(chunk_ids, scopes, limit=limit)

    def lexical_search(self, query, scopes, limit=40):
        self._available()
        return super().lexical_search(query, scopes, limit=limit)

    def _migrate(self, source_path):
        source = sqlite3.connect(source_path.as_uri() + "?mode=ro", uri=True, timeout=10)
        source.row_factory = sqlite3.Row
        source.execute("BEGIN")
        try:
            for table in (
                "neural_meta",
                "sources",
                "jobs",
                "centroid_keys",
                "centroids",
                "communities",
            ):
                fields = [row[1] for row in source.execute(f"PRAGMA table_info({table})")]
                if not fields:
                    continue
                cursor = source.execute(f"SELECT {','.join(fields)} FROM {table}")
                while rows := cursor.fetchmany(1024):
                    self._reserve_migration(rows)
                    self.db.executemany(
                        f"INSERT INTO {table}({','.join(fields)}) VALUES ({','.join('?' for _ in fields)})",
                        (tuple(row) for row in rows),
                    )
            cursor = source.execute("""SELECT c.*,EXISTS(SELECT 1 FROM occurrences o JOIN sources s
                ON s.source_id=o.source_id WHERE o.chunk_id=c.id AND s.provider<>'catalogue') keep_text
                FROM chunks c ORDER BY id""")
            while rows := cursor.fetchmany(512):
                self._reserve_migration(rows)
                for row in rows:
                    self.db.execute(
                        "INSERT INTO chunk_records VALUES (?,?,?,?,?)",
                        (
                            row["id"],
                            bytes.fromhex(row["identity"]),
                            hashlib.sha256(row["text"].encode()).digest(),
                            row["text"] if row["keep_text"] else None,
                            self._space(row["encoder_id"], row["chunker_id"]),
                        ),
                    )
            cursor = source.execute("SELECT * FROM occurrences")
            while rows := cursor.fetchmany(1024):
                self._reserve_migration(rows)
                self.db.executemany(
                    "INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        (
                            row["chunk_id"],
                            row["source_id"],
                            row["version"],
                            compact_locator(row["locator"]),
                            row["start"],
                            row["end"],
                            row["role"],
                            row["timestamp"],
                            row["historical"],
                        )
                        for row in rows
                    ),
                )
            cursor = source.execute("SELECT * FROM semantic_neighbours")
            while rows := cursor.fetchmany(1024):
                self._reserve_migration(rows)
                self.db.executemany(
                    "INSERT INTO semantic_neighbours VALUES (?,?,?,?,?)",
                    (tuple(row) for row in rows),
                )
            self.db.commit()
            if self.db.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Migrated neural metadata failed integrity verification.")
        finally:
            source.close()

    def _reserve_migration(self, rows):
        if self.resource_guard:
            size = sum(
                len(value.encode())
                if isinstance(value, str)
                else len(value)
                if isinstance(value, bytes)
                else 8
                for row in rows
                for value in row
            )
            self.resource_guard(background=True, additional_disk=3 * size + 4096)

    def stage_chunks(self, source_id, version, chunks):
        source = self.db.execute(
            "SELECT provider FROM sources WHERE source_id=? AND version=? AND active=1",
            (source_id, version),
        ).fetchone()
        if source is None:
            return []
        keep_text = source["provider"] != "catalogue"
        result = []
        with self.db:
            self.db.execute(
                "DELETE FROM occurrences WHERE source_id=? AND version=?", (source_id, version)
            )
            for number, chunk in enumerate(chunks):
                if self.resource_guard and number % 128 == 0:
                    self.resource_guard(background=True, additional_disk=2 * 1024**2)
                identity = hashlib.sha256(
                    json.dumps(
                        [self.encoder_id, self.chunker_id, chunk["text"]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                ).digest()
                self.db.execute(
                    "INSERT OR IGNORE INTO chunk_records(identity,text_hash,text,space_id) VALUES (?,?,?,?)",
                    (
                        identity,
                        hashlib.sha256(chunk["text"].encode()).digest(),
                        chunk["text"] if keep_text else None,
                        self.space_id,
                    ),
                )
                key = self.db.execute(
                    "SELECT id FROM chunk_records WHERE identity=?", (identity,)
                ).fetchone()[0]
                if keep_text:
                    self.db.execute(
                        "UPDATE chunk_records SET text=? WHERE id=? AND text IS NULL",
                        (chunk["text"], key),
                    )
                locator = compact_locator(chunk.get("locator", str(number)))
                self.db.execute(
                    "INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)",
                    (
                        key,
                        source_id,
                        version,
                        locator,
                        chunk["start"],
                        chunk["end"],
                        chunk.get("role"),
                        chunk.get("timestamp"),
                        int(chunk.get("historical", False)),
                    ),
                )
                result.append({**chunk, "locator": locator, "chunk_id": key})
        return result

    def replace_semantic_support(self, source_id, records):
        with self.db:
            self.db.execute(
                """DELETE FROM support_records WHERE source_key IN
                (SELECT id FROM centroid_keys WHERE source_id=?) OR target_key IN
                (SELECT id FROM centroid_keys WHERE source_id=?)""",
                (source_id, source_id),
            )
            for source, target, version, target_version, encoded in records:
                evidence = json.loads(encoded)
                a, b = evidence["source_passage"], evidence["target_passage"]
                if "chunk_id" not in a or "chunk_id" not in b:
                    continue
                source_key = self.db.execute(
                    "SELECT id FROM centroid_keys WHERE source_id=? AND version=?",
                    (source, version),
                ).fetchone()[0]
                target_key = self.db.execute(
                    "SELECT id FROM centroid_keys WHERE source_id=? AND version=?",
                    (target, target_version),
                ).fetchone()[0]
                if source_key > target_key:
                    source_key, target_key, a, b = target_key, source_key, b, a
                self.db.execute(
                    "INSERT OR REPLACE INTO support_records VALUES (?,?,?,?,?,?,?,?,?,?)",
                    (
                        source_key,
                        target_key,
                        a["chunk_id"],
                        b["chunk_id"],
                        a["similarity_to_other_centroid"],
                        b["similarity_to_other_centroid"],
                        a["sampled_unique_passages"],
                        b["sampled_unique_passages"],
                        a["total_unique_passages"],
                        b["total_unique_passages"],
                    ),
                )

    def representative_passages(self, source_id, version):
        return self.db.execute(
            """WITH unique_passages AS (SELECT chunk_id,min(locator) locator
            FROM occurrences WHERE source_id=? AND version=? GROUP BY chunk_id)
            SELECT o.chunk_id,o.locator,o.start,o.end,c.text_hash,count(*) OVER() total,
            CASE WHEN json_valid(o.locator) THEN coalesce(json_extract(o.locator,'$[3]')-
                json_extract(o.locator,'$[2]'),0) ELSE 0 END token_weight
            FROM unique_passages u JOIN occurrences o ON o.chunk_id=u.chunk_id AND o.locator=u.locator
            JOIN chunks c ON c.id=o.chunk_id WHERE o.source_id=? AND o.version=?
            ORDER BY token_weight DESC,o.chunk_id LIMIT 8""",
            (source_id, version, source_id, version),
        ).fetchall()

    def _support(self, source, target, version, target_version):
        keys = self.db.execute(
            "SELECT id,source_id FROM centroid_keys WHERE (source_id=? AND version=?) OR (source_id=? AND version=?)",
            (source, version, target, target_version),
        ).fetchall()
        keymap = {row["source_id"]: row["id"] for row in keys}
        if len(keymap) != 2:
            return {"unavailable": "passage provenance pending"}
        a, b = keymap[source], keymap[target]
        row = self.db.execute(
            "SELECT * FROM support_records WHERE source_key=? AND target_key=?",
            (min(a, b), max(a, b)),
        ).fetchone()
        if row is None:
            return {"basis": "file centroids", "unavailable": "passage provenance pending"}
        values = dict(row)

        def passage(node_id, node_version, prefix):
            chunk = values[prefix + "_chunk"]
            occurrence = self.db.execute(
                "SELECT o.locator,o.start,o.end,c.text_hash,s.digest FROM occurrences o JOIN chunks c ON c.id=o.chunk_id JOIN sources s ON s.source_id=o.source_id WHERE o.source_id=? AND o.version=? AND o.chunk_id=? ORDER BY o.locator LIMIT 1",
                (node_id, node_version, chunk),
            ).fetchone()
            if occurrence is None:
                return {"unavailable": "passage occurrence changed"}
            return {
                "source_id": node_id,
                "source_version": node_version,
                "source_digest": occurrence["digest"],
                "chunk_id": chunk,
                "locator": occurrence["locator"],
                "start_line": occurrence["start"],
                "end_line": occurrence["end"],
                "text_hash": bytes(occurrence["text_hash"]).hex(),
                "similarity_to_other_centroid": values[prefix + "_similarity"],
                "sampled_unique_passages": values[prefix + "_sampled"],
                "total_unique_passages": values[prefix + "_total"],
                "selection": "closest to other centroid among up to eight longest-token unique passages",
            }

        return {
            "basis": "mutual token-weighted file centroids",
            "embedding_space": self.encoder_id,
            "chunking_version": self.chunker_id,
            "source_passage": passage(source, version, "source" if a < b else "target"),
            "target_passage": passage(target, target_version, "target" if a < b else "source"),
        }

    def semantic_neighbours(self, source_id, scopes=("shared", "private"), limit=8):
        allowed = sorted(set(scopes) & {"private", "shared"})
        if not allowed:
            return []
        if not 1 <= limit <= 8:
            raise ValueError("Semantic neighbour count must be in [1,8]")
        marks = ",".join("?" for _ in allowed)
        rows = self.db.execute(
            f"""SELECT f.source,f.target,min(f.weight,r.weight) weight,
            s.version source_version,t.version target_version,t.path,t.scope FROM semantic_neighbours f
            JOIN semantic_neighbours r ON r.source=f.target AND r.target=f.source
            JOIN sources s ON s.source_id=f.source AND s.version=f.source_version AND s.version=r.target_version
            JOIN sources t ON t.source_id=f.target AND t.version=f.target_version AND t.version=r.source_version
            WHERE f.source=? AND s.active=1 AND s.ready=1 AND t.active=1 AND t.ready=1 AND min(f.weight,r.weight)>=?
            AND s.scope IN ({marks}) AND t.scope IN ({marks}) ORDER BY weight DESC,f.target LIMIT ?""",
            (
                source_id,
                float(self.get_meta("semantic_threshold", "0.82")),
                *allowed,
                *allowed,
                limit,
            ),
        )
        return [
            {
                **dict(row),
                "kind": "semantic",
                "inferred": True,
                "evidence": self._support(
                    row["source"], row["target"], row["source_version"], row["target_version"]
                ),
            }
            for row in rows
        ]
