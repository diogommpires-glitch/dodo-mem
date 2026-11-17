"""Coalesced catalogue changes remain durable until neural jobs have been queued."""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

from .resources import ResourcePressure


@contextmanager
def defer_sqlite_busy():
    try:
        yield
    except sqlite3.OperationalError as error:
        if (getattr(error, "sqlite_errorcode", 0) & 0xFF) in {
            sqlite3.SQLITE_BUSY,
            sqlite3.SQLITE_LOCKED,
        }:
            raise ResourcePressure(
                "SQLite writer is busy; neural work remains queued for retry."
            ) from error
        raise


SCHEMA = """
CREATE TABLE IF NOT EXISTS neural_changes(
    node_id TEXT PRIMARY KEY,revision INTEGER NOT NULL,changed_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS neural_change_time ON neural_changes(changed_at,node_id);
CREATE TRIGGER IF NOT EXISTS neural_file_add AFTER INSERT ON nodes WHEN new.kind='file'
BEGIN
    INSERT INTO neural_changes VALUES(new.id,1,(julianday('now')-2440587.5)*86400)
    ON CONFLICT(node_id) DO UPDATE SET revision=revision+1,changed_at=excluded.changed_at;
END;
CREATE TRIGGER IF NOT EXISTS neural_file_remove AFTER DELETE ON nodes WHEN old.kind='file'
BEGIN
    INSERT INTO neural_changes VALUES(old.id,1,(julianday('now')-2440587.5)*86400)
    ON CONFLICT(node_id) DO UPDATE SET revision=revision+1,changed_at=excluded.changed_at;
END;
DROP TRIGGER IF EXISTS neural_file_change;
CREATE TRIGGER neural_file_change
AFTER UPDATE OF digest,state,size,mtime_ns,file_id,path ON nodes
WHEN (old.kind='file' OR new.kind='file') AND (
    old.digest IS NOT new.digest OR old.state IS NOT new.state OR old.size IS NOT new.size
    OR old.mtime_ns IS NOT new.mtime_ns OR old.file_id IS NOT new.file_id
    OR old.path IS NOT new.path)
BEGIN
    INSERT INTO neural_changes VALUES(new.id,1,(julianday('now')-2440587.5)*86400)
    ON CONFLICT(node_id) DO UPDATE SET revision=revision+1,changed_at=excluded.changed_at;
END;
"""


@defer_sqlite_busy()
def open_changes(path):
    connection = sqlite3.connect(Path(path).absolute().as_uri() + "?mode=rw", uri=True, timeout=3)
    fields = {row[1] for row in connection.execute("PRAGMA table_info(nodes)")}
    if (
        not {"id", "root_id", "path", "kind", "state", "digest", "size", "mtime_ns", "file_id"}
        <= fields
    ):
        connection.close()
        raise ValueError(
            "Neural change capture requires the expected whole-library catalogue schema."
        )
    try:
        connection.executescript("BEGIN IMMEDIATE;" + SCHEMA + "COMMIT;")
    except sqlite3.Error:
        connection.close()
        raise
    return connection
