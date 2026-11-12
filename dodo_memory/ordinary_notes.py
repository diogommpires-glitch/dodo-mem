"""Owner-authored shared Markdown and its durable revision history."""

import hashlib
import re
import stat
from pathlib import PurePosixPath

from .evidence import _relative_note, _signature
from .personal_profile import PROFILE_PATH
from .source_registry import atomic_bytes
from .store import metadata, now

MAX_BYTES = 256 * 1024
PROTECTED = {
    "archive",
    "core",
    "documents",
    "episodes",
    "history",
    "inbox",
    "knowledge",
    "preferences",
    "private",
    "profile",
}
MANAGED_FIELDS = {
    "content_hash",
    "ingest_job_id",
    "knowledge_key",
    "knowledge_revision",
    "parser",
    "record_id",
    "review",
    "source_id",
    "source_version",
}
WINDOWS_DEVICE = re.compile(
    r"(?:AUX|CON|CONIN\$|CONOUT\$|NUL|PRN|COM[1-9¹²³]|LPT[1-9¹²³])", re.IGNORECASE
)
FRONTMATTER = re.compile(r"\A---\r?\n.*?\r?\n---\r?\n", re.DOTALL)


def _sha256(content):
    return hashlib.sha256(content).hexdigest() if content is not None else None


def _path(vault, path):
    relative = _relative_note(path, ("shared",))
    parts = PurePosixPath(relative).parts
    if (
        parts[0].casefold() in PROTECTED
        or relative.casefold() == PROFILE_PATH
        or len(relative) > 1024
        or any(
            any(char in part for char in '<>"|?*')
            or WINDOWS_DEVICE.fullmatch(part.split(".", 1)[0].rstrip(" "))
            for part in parts
        )
    ):
        raise PermissionError("Only ordinary shared note paths can be edited")
    target = vault.root / relative
    if target.resolve().as_posix() != target.as_posix() or any(
        part.is_symlink() or getattr(part, "is_junction", lambda: False)()
        for part in (target, *target.parents)
    ):
        raise PermissionError("Ordinary notes require a canonical path without redirects")
    vault.assert_retention_ready()
    if vault.retention is not None:
        vault.retention.assert_path_allowed(relative)
    with vault.connect() as db:
        if not vault.document_published(relative, {}, db=db):
            raise PermissionError("Ordinary note is unavailable")
        names = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for table in ("ingest_jobs", "accepted_knowledge_proposals", "accepted_knowledge_versions"):
            if (
                table in names
                and db.execute(f"SELECT 1 FROM {table} WHERE path=?", (relative,)).fetchone()
            ):
                raise PermissionError("Managed, reviewed and historical documents are read-only")
    return relative, target


def _read(target):
    try:
        before = target.stat()
    except FileNotFoundError:
        return None
    if not stat.S_ISREG(before.st_mode) or before.st_size > MAX_BYTES:
        raise ValueError("Ordinary note exceeds the 256 KiB regular-file limit")
    with target.open("rb") as stream:
        content = stream.read(MAX_BYTES + 1)
    if len(content) > MAX_BYTES or _signature(target.stat()) != _signature(before):
        raise ValueError("Ordinary note changed while reading")
    return content


def _body(relative, content):
    decoded = content.decode("utf-8")
    fields, text = metadata(decoded.replace("\r\n", "\n").replace("\r", "\n"))
    if (
        fields.keys() & MANAGED_FIELDS
        or fields.get("kind", "note") != "note"
        or fields.get("status", "curated") != "curated"
        or fields.get("source", relative) != relative
    ):
        raise PermissionError("Managed, reviewed and historical documents are read-only")
    match = FRONTMATTER.match(decoded)
    return match[0] if match else "", text


def read(vault, path):
    relative, target = _path(vault, path)
    content = _read(target)
    if content is None:
        raise FileNotFoundError("Ordinary note is unavailable")
    _, text = _body(relative, content)
    digest = _sha256(content)
    return {"path": relative, "text": text, "sha256": digest}


def _record(db, path, content):
    digest = _sha256(content)
    head = db.execute(
        "SELECT revision,sha256 FROM ordinary_note_revisions WHERE path=? ORDER BY revision DESC LIMIT 1",
        (path,),
    ).fetchone()
    if head is not None and head["sha256"] == digest:
        return head["revision"]
    revision = head["revision"] + 1 if head is not None else 1
    db.execute(
        "INSERT INTO ordinary_note_revisions VALUES (?,?,?,?,?)",
        (path, revision, digest, now(), content),
    )
    return revision


def _index_save(vault, db, path, previous, content):
    if previous is not None:
        _record(db, path, previous)
    revision = _record(db, path, content)
    vault.upsert_document(path, db=db)
    db.execute("DELETE FROM ordinary_note_pending WHERE path=?", (path,))
    return revision


def prepare(vault):
    """Recover an interrupted file/index commit using the recorded digests."""
    with vault.connect() as db:
        db.executescript("""
            CREATE TABLE IF NOT EXISTS ordinary_note_revisions (
                path TEXT NOT NULL, revision INTEGER NOT NULL, sha256 TEXT NOT NULL,
                recorded_at TEXT NOT NULL, content BLOB NOT NULL, PRIMARY KEY(path,revision)
            );
            CREATE TABLE IF NOT EXISTS ordinary_note_pending (
                path TEXT PRIMARY KEY, previous BLOB, expected TEXT NOT NULL
            );
        """)
        pending = db.execute("SELECT * FROM ordinary_note_pending ORDER BY path").fetchall()
    for row in pending:
        relative, target = _path(vault, row["path"])
        content = _read(target)
        with vault.connect() as db:
            if content == row["previous"]:
                db.execute("DELETE FROM ordinary_note_pending WHERE path=?", (relative,))
            elif _sha256(content) == row["expected"]:
                _body(relative, content)
                db.execute("BEGIN IMMEDIATE")
                _index_save(vault, db, relative, row["previous"], content)


def save(vault, path, text, expected_sha256):
    relative, target = _path(vault, path)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    if not text.strip() or text.startswith("---\n") or "\0" in text:
        raise ValueError("Write nonblank Markdown body text without a frontmatter block or NUL")
    if expected_sha256 is not None and re.fullmatch("[0-9a-f]{64}", expected_sha256) is None:
        raise ValueError("Expected revision must be a lowercase SHA-256")
    previous = _read(target)
    if _sha256(previous) != expected_sha256:
        raise ValueError("Note revision changed. Reload the current note before saving.")
    if previous is None and not relative.startswith("notes/"):
        raise PermissionError("New ordinary notes must be created under notes/")
    header = _body(relative, previous)[0] if previous is not None else ""
    newline = "\r\n" if previous is not None and b"\r\n" in previous else "\n"
    content = (header + text.replace("\n", newline)).encode("utf-8")
    if len(content) > MAX_BYTES:
        raise ValueError("Ordinary note exceeds 256 KiB")
    digest = _sha256(content)
    with vault.connect() as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("SELECT 1 FROM ordinary_note_pending WHERE path=?", (relative,)).fetchone():
            raise ValueError("Ordinary note has an unresolved earlier save")
        db.execute("INSERT INTO ordinary_note_pending VALUES (?,?,?)", (relative, previous, digest))
    published = False
    try:
        with vault.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            _path(vault, relative)
            if _read(target) != previous:
                raise ValueError("Note revision changed. Reload the current note before saving.")
            atomic_bytes(target, content, replace=previous is not None)
            published = True
            revision = _index_save(vault, db, relative, previous, content)
    except BaseException:
        with vault.connect() as db:
            pending = db.execute(
                "SELECT 1 FROM ordinary_note_pending WHERE path=?", (relative,)
            ).fetchone()
        if pending:
            restored = not published
            if published and _read(target) == content:
                if previous is None:
                    target.unlink()
                else:
                    atomic_bytes(target, previous)
                restored = True
            if restored:
                with vault.connect() as db:
                    db.execute("DELETE FROM ordinary_note_pending WHERE path=?", (relative,))
        raise
    return {"path": relative, "text": text, "sha256": digest, "revision": revision}


def history(vault, path, limit):
    path = read(vault, path)["path"]
    if not 1 <= limit <= 100:
        raise ValueError("History limit must be from 1 to 100")
    with vault.connect() as db:
        rows = db.execute(
            """SELECT revision,sha256,recorded_at FROM ordinary_note_revisions
            WHERE path=? ORDER BY revision DESC LIMIT ?""",
            (path, limit + 1),
        ).fetchall()
    return {
        "path": path,
        "records": [dict(row) for row in rows[:limit]],
        "truncated": len(rows) > limit,
    }


def revision(vault, path, number):
    path = read(vault, path)["path"]
    with vault.connect() as db:
        row = db.execute(
            "SELECT * FROM ordinary_note_revisions WHERE path=? AND revision=?", (path, number)
        ).fetchone()
    if row is None or _sha256(row["content"]) != row["sha256"]:
        raise ValueError("Ordinary note revision is unavailable")
    _, text = _body(path, row["content"])
    return {
        "path": path,
        "revision": number,
        "sha256": row["sha256"],
        "recorded_at": row["recorded_at"],
        "text": text,
    }
