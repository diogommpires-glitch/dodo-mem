"""Versioned local jobs, deduplicated chunks, and independently authorised occurrences."""

import hashlib
import json
import re
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS neural_meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS sources(
    source_id TEXT PRIMARY KEY,digest TEXT NOT NULL,path TEXT NOT NULL,scope TEXT NOT NULL,
    version TEXT NOT NULL,active INTEGER NOT NULL DEFAULT 1,ready INTEGER NOT NULL DEFAULT 0,
    scan_epoch INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS jobs(
    id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,version TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'pending',attempts INTEGER NOT NULL DEFAULT 0,
    updated REAL NOT NULL,UNIQUE(source_id,version),
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS job_status ON jobs(status,id);
CREATE TABLE IF NOT EXISTS chunks(
    id INTEGER PRIMARY KEY,identity TEXT NOT NULL UNIQUE,text TEXT NOT NULL,
    encoder_id TEXT NOT NULL,chunker_id TEXT NOT NULL
);
CREATE VIRTUAL TABLE IF NOT EXISTS chunk_search USING fts5(text,content='chunks',content_rowid='id');
CREATE TRIGGER IF NOT EXISTS chunk_add AFTER INSERT ON chunks BEGIN
    INSERT INTO chunk_search(rowid,text) VALUES(new.id,new.text);
END;
CREATE TABLE IF NOT EXISTS occurrences(
    chunk_id INTEGER NOT NULL,source_id TEXT NOT NULL,version TEXT NOT NULL,
    locator TEXT NOT NULL,start INTEGER NOT NULL,end INTEGER NOT NULL,
    role TEXT,timestamp TEXT,historical INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY(source_id,version,locator),
    FOREIGN KEY(chunk_id) REFERENCES chunks(id),
    FOREIGN KEY(source_id) REFERENCES sources(source_id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS occurrence_chunk ON occurrences(chunk_id,source_id,version);
CREATE TABLE IF NOT EXISTS centroid_keys(
    id INTEGER PRIMARY KEY,source_id TEXT NOT NULL,version TEXT NOT NULL,
    UNIQUE(source_id,version)
);
CREATE TABLE IF NOT EXISTS centroids(
    source_id TEXT PRIMARY KEY,version TEXT NOT NULL,vector_key INTEGER NOT NULL,
    vector BLOB NOT NULL,weight REAL NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS centroid_vector ON centroids(vector_key);
CREATE TABLE IF NOT EXISTS semantic_neighbours(
    source TEXT NOT NULL,target TEXT NOT NULL,source_version TEXT NOT NULL,
    target_version TEXT NOT NULL,weight REAL NOT NULL,PRIMARY KEY(source,target)
);
CREATE INDEX IF NOT EXISTS semantic_target ON semantic_neighbours(target,source);
CREATE TABLE IF NOT EXISTS semantic_support(
    source TEXT NOT NULL,target TEXT NOT NULL,source_version TEXT NOT NULL,
    target_version TEXT NOT NULL,evidence TEXT NOT NULL,PRIMARY KEY(source,target)
);
CREATE TABLE IF NOT EXISTS communities(
    source_id TEXT PRIMARY KEY,source_version TEXT NOT NULL,community_id TEXT NOT NULL,
    generation TEXT NOT NULL,partial INTEGER NOT NULL
);
"""


class IndexStore:
    def __init__(
        self, path: Path, *, encoder_id="default", chunker_id="tokens224-overlap32-retokenise-v2"
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.encoder_id = encoder_id
        self.chunker_id = chunker_id
        self.db = sqlite3.connect(self.path, timeout=10, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        columns = {row[1] for row in self.db.execute("PRAGMA table_info(sources)")}
        if "provider" not in columns:
            self.db.execute(
                "ALTER TABLE sources ADD COLUMN provider TEXT NOT NULL DEFAULT 'catalogue'"
            )
        if "priority" not in {row[1] for row in self.db.execute("PRAGMA table_info(jobs)")}:
            self.db.execute("ALTER TABLE jobs ADD COLUMN priority INTEGER NOT NULL DEFAULT 0")
        self.db.execute("CREATE INDEX IF NOT EXISTS job_priority ON jobs(status,priority DESC,id)")
        version = json.dumps([encoder_id, chunker_id], separators=(",", ":"))
        previous = self.get_meta("index_version")
        with self.db:
            if previous is not None and previous != version:
                self.db.execute("UPDATE sources SET ready=0")
                self.db.execute("""UPDATE jobs SET status='pending' WHERE EXISTS
                    (SELECT 1 FROM sources s WHERE s.source_id=jobs.source_id
                    AND s.version=jobs.version AND s.active=1)""")
            self.set_meta("index_version", version)
            self.db.execute("UPDATE jobs SET status='pending' WHERE status='running'")

    def close(self):
        self.db.close()

    @contextmanager
    def read_snapshot(self, *, timeout=3):
        snapshot = object.__new__(IndexStore)
        snapshot.path, snapshot.encoder_id, snapshot.chunker_id = (
            self.path,
            self.encoder_id,
            self.chunker_id,
        )
        snapshot.db = sqlite3.connect(
            self.path.absolute().as_uri() + "?mode=ro", uri=True, timeout=timeout
        )
        snapshot.db.row_factory = sqlite3.Row
        try:
            snapshot.db.execute("PRAGMA query_only=ON")
            snapshot.db.execute("BEGIN")
            yield snapshot
        finally:
            snapshot.db.close()

    def get_meta(self, key, default=None):
        row = self.db.execute("SELECT value FROM neural_meta WHERE key=?", (key,)).fetchone()
        return row[0] if row else default

    def replace_semantic_support(self, source_id, records):
        with self.db:
            self.db.execute("DELETE FROM semantic_support WHERE source=?", (source_id,))
            self.db.executemany("INSERT INTO semantic_support VALUES (?,?,?,?,?)", records)

    def representative_passages(self, source_id, version):
        rows = self.db.execute(
            """WITH unique_passages AS (SELECT chunk_id,min(locator) locator
            FROM occurrences WHERE source_id=? AND version=? GROUP BY chunk_id)
            SELECT o.chunk_id,o.locator,o.start,o.end,c.text,count(*) OVER() total,
                CASE WHEN json_valid(o.locator) THEN coalesce(json_extract(o.locator,'$[3]')-
                    json_extract(o.locator,'$[2]'),0) ELSE 0 END token_weight
            FROM unique_passages u JOIN occurrences o ON o.chunk_id=u.chunk_id AND o.locator=u.locator
            JOIN chunks c ON c.id=o.chunk_id WHERE o.source_id=? AND o.version=?
            ORDER BY token_weight DESC,o.chunk_id LIMIT 8""",
            (source_id, version, source_id, version),
        )
        return [
            {**dict(row), "text_hash": hashlib.sha256(row["text"].encode()).digest()}
            for row in rows
        ]

    def set_meta(self, key, value):
        self.db.execute(
            "INSERT INTO neural_meta VALUES (?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, str(value)),
        )

    def reset_progress(self, reason):
        previous = json.loads(self.get_meta("index_progress", "{}"))
        if previous:
            self.set_meta(
                "historical_index_progress",
                json.dumps({"reason": reason, "saved_at": time.time(), "progress": previous}),
            )
        self.set_meta("index_progress", "{}")
        self.set_meta("index_progress_reset_reason", reason)
        self.set_meta("index_progress_epoch", time.time_ns())

    def queue_source(
        self, source_id, digest, path, scope, version, *, scan_epoch=0, provider="catalogue"
    ):
        if scope not in {"private", "shared", "archive"}:
            raise ValueError(f"Unknown source scope: {scope}")
        with self.db:
            previous = self.db.execute(
                "SELECT version,scope,digest FROM sources WHERE source_id=?", (source_id,)
            ).fetchone()
            priority = 10 if previous and tuple(previous) != (version, scope, digest) else 0
            self.db.execute(
                """INSERT INTO sources(source_id,digest,path,scope,version,scan_epoch,provider)
                VALUES (?,?,?,?,?,?,?) ON CONFLICT(source_id) DO UPDATE SET
                digest=excluded.digest,path=excluded.path,scope=excluded.scope,
                ready=CASE WHEN sources.version=excluded.version AND sources.scope=excluded.scope
                    AND sources.active=1
                    THEN sources.ready ELSE 0 END,
                version=excluded.version,active=1,scan_epoch=excluded.scan_epoch,provider=excluded.provider""",
                (source_id, digest, str(path), scope, version, scan_epoch, provider),
            )
            self.db.execute(
                "UPDATE jobs SET status='obsolete' WHERE source_id=? AND version<>?",
                (source_id, version),
            )
            self.db.execute(
                """INSERT INTO jobs(source_id,version,updated,priority) VALUES (?,?,?,?)
                ON CONFLICT(source_id,version) DO UPDATE SET priority=max(jobs.priority,excluded.priority),status=CASE
                    WHEN jobs.status IN ('complete','obsolete') AND (SELECT ready FROM sources
                    WHERE source_id=excluded.source_id)=0 THEN 'pending' ELSE jobs.status END""",
                (source_id, version, time.time(), priority),
            )

    def claim_job(self):
        with self.db:
            row = self.db.execute(
                """SELECT j.id,j.source_id,j.version,s.digest,s.path,s.scope FROM jobs j
                JOIN sources s ON s.source_id=j.source_id AND s.version=j.version
                WHERE j.status='pending' AND s.active=1 ORDER BY j.priority DESC,j.id LIMIT 1"""
            ).fetchone()
            if row is None:
                return None
            self.db.execute(
                "UPDATE jobs SET status='running',attempts=attempts+1,updated=? WHERE id=?",
                (time.time(), row["id"]),
            )
            return dict(row)

    def retry_job(self, job_id):
        with self.db:
            self.db.execute(
                "UPDATE jobs SET status='pending',updated=? WHERE id=? AND status='running'",
                (time.time(), job_id),
            )

    def fail_job(self, job_id, reason):
        with self.db:
            self.db.execute("""CREATE TABLE IF NOT EXISTS job_failures(
                job_id INTEGER PRIMARY KEY REFERENCES jobs(id) ON DELETE CASCADE,
                reason TEXT NOT NULL,updated REAL NOT NULL)""")
            self.db.execute(
                "UPDATE sources SET ready=0 WHERE source_id=(SELECT source_id FROM jobs WHERE id=?)",
                (job_id,),
            )
            self.db.execute(
                "UPDATE jobs SET status='failed',updated=? WHERE id=? AND status='running'",
                (time.time(), job_id),
            )
            self.db.execute(
                "INSERT OR REPLACE INTO job_failures VALUES (?,?,?)",
                (job_id, reason, time.time()),
            )

    def pending_count(self):
        return self.db.execute(
            "SELECT count(*) FROM jobs WHERE status IN ('pending','running')"
        ).fetchone()[0]

    def stage_chunks(self, source_id, version, chunks):
        source = self.db.execute(
            "SELECT 1 FROM sources WHERE source_id=? AND version=? AND active=1",
            (source_id, version),
        ).fetchone()
        if source is None:
            return []
        rows = []
        with self.db:
            self.db.execute(
                "DELETE FROM occurrences WHERE source_id=? AND version=?", (source_id, version)
            )
            for number, chunk in enumerate(chunks):
                identity = hashlib.sha256(
                    json.dumps(
                        [self.encoder_id, self.chunker_id, chunk["text"]],
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode()
                ).hexdigest()
                self.db.execute(
                    """INSERT OR IGNORE INTO chunks(identity,text,encoder_id,chunker_id)
                    VALUES (?,?,?,?)""",
                    (identity, chunk["text"], self.encoder_id, self.chunker_id),
                )
                key = self.db.execute(
                    "SELECT id FROM chunks WHERE identity=?", (identity,)
                ).fetchone()[0]
                self.db.execute(
                    """INSERT INTO occurrences VALUES (?,?,?,?,?,?,?,?,?)""",
                    (
                        key,
                        source_id,
                        version,
                        chunk.get("locator", str(number)),
                        chunk["start"],
                        chunk["end"],
                        chunk.get("role"),
                        chunk.get("timestamp"),
                        int(chunk.get("historical", False)),
                    ),
                )
                rows.append({"chunk_id": key, **chunk})
        return rows

    def complete_source(self, source_id, version):
        with self.db:
            changed = self.db.execute(
                "UPDATE sources SET ready=1 WHERE source_id=? AND version=? AND active=1",
                (source_id, version),
            ).rowcount
            if changed:
                self.db.execute(
                    "UPDATE jobs SET status='complete',updated=? WHERE source_id=? AND version=?",
                    (time.time(), source_id, version),
                )
                self.db.execute(
                    "DELETE FROM occurrences WHERE source_id=? AND version<>?", (source_id, version)
                )
        return bool(changed)

    def invalidate_source(self, source_id):
        with self.db:
            self.db.execute("UPDATE sources SET active=0,ready=0 WHERE source_id=?", (source_id,))
            self.db.execute("UPDATE jobs SET status='obsolete' WHERE source_id=?", (source_id,))

    def finish_scan(self, epoch):
        with self.db:
            self.db.execute(
                "UPDATE sources SET active=0,ready=0 WHERE scan_epoch<>? AND provider='catalogue'",
                (epoch,),
            )
            self.db.execute(
                """UPDATE jobs SET status='obsolete' WHERE source_id IN
                (SELECT source_id FROM sources WHERE active=0)"""
            )
            self.set_meta("scan_cursor", "")

    def visible_occurrences(self, chunk_ids, scopes, *, limit=1000):
        if not 1 <= limit <= 1000:
            raise ValueError("Occurrence count must be in [1,1000]")
        allowed = sorted(set(scopes) & {"private", "shared"})
        if not chunk_ids or not allowed:
            return []
        rows = []
        for offset in range(0, len(chunk_ids), 500):
            keys = list(chunk_ids[offset : offset + 500])
            marks = ",".join("?" for _ in keys)
            permissions = ",".join("?" for _ in allowed)
            result = self.db.execute(
                f"""SELECT o.chunk_id,s.source_id,s.source_id AS node_id,s.digest,s.path,s.scope,
                s.version,o.start,o.end,o.start AS start_line,o.end AS end_line,
                o.locator,o.role,o.timestamp,o.historical,c.text
                FROM occurrences o JOIN sources s ON s.source_id=o.source_id AND s.version=o.version
                JOIN chunks c ON c.id=o.chunk_id
                WHERE o.chunk_id IN ({marks}) AND s.scope IN ({permissions})
                AND s.active=1 AND s.ready=1 AND c.encoder_id=? AND c.chunker_id=?
                ORDER BY o.chunk_id,s.source_id,o.locator LIMIT ?""",
                (*keys, *allowed, self.encoder_id, self.chunker_id, limit - len(rows)),
            )
            rows.extend(dict(row) for row in result)
            if len(rows) == limit:
                break
        return rows

    def lexical_search(self, query, scopes, limit=40):
        if not 1 <= limit <= 1000 or not isinstance(query, str) or len(query) > 2000:
            raise ValueError("Lexical search requires <=2000 characters and limit in [1,1000]")
        allowed = sorted(set(scopes) & {"private", "shared"})
        terms = re.findall(r"\w+", query, flags=re.UNICODE)[:32]
        if not allowed or not terms:
            return []
        expression = " OR ".join('"' + term + '"' for term in terms)
        permissions = ",".join("?" for _ in allowed)
        rows = self.db.execute(
            f"""SELECT c.id,bm25(chunk_search) AS rank FROM chunk_search
            JOIN chunks c ON c.id=chunk_search.rowid
            WHERE chunk_search MATCH ? AND c.encoder_id=? AND c.chunker_id=?
            AND EXISTS (SELECT 1 FROM occurrences o JOIN sources s ON s.source_id=o.source_id
                AND s.version=o.version WHERE o.chunk_id=c.id AND s.active=1 AND s.ready=1
                AND s.scope IN ({permissions})) ORDER BY rank,c.id LIMIT ?""",
            (expression, self.encoder_id, self.chunker_id, *allowed, limit),
        ).fetchall()
        rank = {row["id"]: row["rank"] for row in rows}
        occurrences = []
        for key in rank:
            occurrences.extend(
                self.visible_occurrences([key], allowed, limit=limit - len(occurrences))
            )
            if len(occurrences) == limit:
                break
        return [{**row, "score": -rank[row["chunk_id"]]} for row in occurrences[:limit]]

    def save_centroid(self, source_id, version, vector, weight):
        vector = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(vector)
        if vector.shape != (384,) or not np.isfinite(vector).all() or norm <= 0 or weight <= 0:
            raise ValueError("Centroid requires 384 finite dimensions and positive weight/norm")
        with self.db:
            self.db.execute(
                "INSERT OR IGNORE INTO centroid_keys(source_id,version) VALUES (?,?)",
                (source_id, version),
            )
            key = self.db.execute(
                "SELECT id FROM centroid_keys WHERE source_id=? AND version=?", (source_id, version)
            ).fetchone()[0]
            self.db.execute(
                """INSERT INTO centroids VALUES (?,?,?,?,?) ON CONFLICT(source_id)
                DO UPDATE SET version=excluded.version,vector_key=excluded.vector_key,
                vector=excluded.vector,weight=excluded.weight""",
                (source_id, version, key, (vector / norm).astype("<f4").tobytes(), float(weight)),
            )
        return key

    def centroid_rows(self, *, after="", limit=25_000):
        if not 1 <= limit <= 25_000:
            raise ValueError("Centroid batch must be in [1,25000]")
        rows = self.db.execute(
            """SELECT c.*,s.scope,s.path FROM centroids c JOIN sources s ON s.source_id=c.source_id
            AND s.version=c.version WHERE s.active=1 AND s.ready=1 AND c.source_id>?
            ORDER BY c.source_id LIMIT ?""",
            (after, limit),
        )
        return [
            {**dict(row), "vector": np.frombuffer(row["vector"], dtype="<f4").copy()}
            for row in rows
        ]

    def centroid_count(self):
        return self.db.execute(
            """SELECT count(*) FROM centroids c JOIN sources s ON s.source_id=c.source_id
            AND s.version=c.version WHERE s.active=1 AND s.ready=1"""
        ).fetchone()[0]

    def semantic_neighbours(self, source_id, scopes=("shared", "private"), limit=8):
        allowed = sorted(set(scopes) & {"private", "shared"})
        if not allowed:
            return []
        if not 1 <= limit <= 8:
            raise ValueError("Semantic neighbour count must be in [1,8]")
        marks = ",".join("?" for _ in allowed)
        rows = self.db.execute(
            f"""SELECT f.source,f.target,min(f.weight,r.weight) AS weight,
            s.version AS source_version,t.version AS target_version,t.path,t.scope,p.evidence AS support
            FROM semantic_neighbours f JOIN semantic_neighbours r
            ON r.source=f.target AND r.target=f.source
            JOIN sources s ON s.source_id=f.source AND s.version=f.source_version
            AND s.version=r.target_version
            JOIN sources t ON t.source_id=f.target AND t.version=f.target_version
            AND t.version=r.source_version
            LEFT JOIN semantic_support p ON p.source=f.source AND p.target=f.target
            AND p.source_version=f.source_version AND p.target_version=f.target_version
            WHERE f.source=? AND s.active=1 AND s.ready=1 AND t.active=1 AND t.ready=1
            AND min(f.weight,r.weight)>=?
            AND s.scope IN ({marks}) AND t.scope IN ({marks})
            ORDER BY weight DESC,f.target LIMIT ?""",
            (
                source_id,
                float(self.get_meta("semantic_threshold", "0.82")),
                *allowed,
                *allowed,
                limit,
            ),
        )
        result = []
        for row in rows:
            edge = dict(row)
            support = edge.pop("support")
            edge.update(
                kind="semantic",
                inferred=True,
                evidence=json.loads(support)
                if support
                else {
                    "basis": "file centroids",
                    "unavailable": "passage evidence backfill pending",
                },
            )
            result.append(edge)
        return result

    def community_rows(self, *, after="", limit=25_000):
        if not 1 <= limit <= 25_000:
            raise ValueError("Community batch must be in [1,25000]")
        return [
            dict(row)
            for row in self.db.execute(
                """SELECT c.* FROM communities c JOIN sources s ON s.source_id=c.source_id
            AND s.version=c.source_version WHERE s.active=1 AND s.ready=1
            AND c.source_id>? ORDER BY c.source_id LIMIT ?""",
                (after, limit),
            )
        ]

    def topics(self, scopes=("shared", "private"), limit=900):
        allowed = sorted(set(scopes) & {"private", "shared"})
        if not 1 <= limit <= 900:
            raise ValueError("Topic member count must be in [1,900]")
        if not allowed:
            return {"members": [], "edges": [], "coverage": {"covered": 0, "total": 0}}
        marks = ",".join("?" for _ in allowed)
        members = [
            dict(row)
            for row in self.db.execute(
                f"""SELECT c.source_id,c.community_id,c.generation,c.partial,s.scope,s.path,
            s.digest,s.version AS source_version
            FROM communities c JOIN sources s ON s.source_id=c.source_id
            AND s.version=c.source_version WHERE s.active=1 AND s.ready=1
            AND s.scope IN ({marks}) ORDER BY c.source_id LIMIT ?""",
                (*allowed, limit),
            )
        ]
        visible = {row["source_id"] for row in members}
        edges = [
            edge
            for source in sorted(visible)
            for edge in self.semantic_neighbours(source, allowed)
            if edge["target"] in visible and source < edge["target"]
        ]
        counts = self.db.execute(
            f"""SELECT count(*) AS total,count(c.source_id) AS covered FROM sources s
            LEFT JOIN communities c ON c.source_id=s.source_id AND c.source_version=s.version
            WHERE s.active=1 AND s.ready=1 AND s.scope IN ({marks})""",
            allowed,
        ).fetchone()
        return {
            "members": members,
            "edges": edges,
            "coverage": {
                **dict(counts),
                "partial": any(row["partial"] for row in members)
                or counts["covered"] < counts["total"],
                "method": self.get_meta("organisation_method", "not-run"),
            },
        }
