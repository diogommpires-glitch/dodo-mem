"""Read-only Cline session catalogues backed by bounded consistent online snapshots."""

import hashlib
import json
import re
import sqlite3
import stat
import threading
import time
from pathlib import Path

from .chat_adapters import ChatMessage, HistoricalSourceError, parse_json_chat, read_cline_snapshot

_SECRET = re.compile(
    rb"-----BEGIN [A-Z ]*PRIVATE KEY-----|\b(?:AKIA[A-Z0-9]{16}|gh[pousr]_[A-Za-z0-9]{20,}"
    rb"|sk-(?:proj-)?[A-Za-z0-9_-]{25,})\b|(?i:\bBearer\s+[A-Za-z0-9._~+/-]{24,}={0,2})|"
    rb"(?im:\b[A-Za-z_]*(?:auth_?token|api_?key|access_?token|password)[\"']?\s*[:=]\s*[\"'][^\r\n\"']{8,}[\"'])"
)


def discover_cline_databases(home=None):
    if home is None:
        from dodo_memory_desktop.configuration import source_roots

        return [path for path in source_roots("DODO_MEM_CLINE_DATABASES") if path.is_file()]
    home = Path(home)
    paths = [home / ".cline/data/db/sessions.db"]
    profiles = home / "local-ai"
    if profiles.is_dir():
        paths.extend(sorted(profiles.glob("*/cline-data/db/sessions.db"))[:16])
    return [path for path in paths if path.is_file() and not path.is_symlink()]


def _safe_file(path, *, max_bytes):
    path = Path(path).absolute()
    if ".." in path.parts:
        raise ValueError("Cline source paths must not contain parent traversal")
    for item in (*reversed(path.parents), path):
        info = item.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & (
            0x400 | 0x1000 | 0x40000 | 0x400000
        ):
            raise HistoricalSourceError("provider_source_unavailable")
    if not stat.S_ISREG(info.st_mode) or info.st_size > max_bytes:
        raise ValueError("Cline source exceeds the regular-file size bound")
    return info


class ClineHistory:
    def __init__(self, database, snapshot_dir, *, resource_guard=None):
        self.database = Path(database).absolute()
        self.session_root = self.database.parent.parent / "sessions"
        self.identity = hashlib.sha256(str(self.database).casefold().encode()).hexdigest()[:24]
        self.prefix = f"cline-sqlite:{self.database.as_posix()}:"
        self.snapshot = Path(snapshot_dir) / f"cline-{self.identity}.sqlite3"
        self.resource_guard = resource_guard
        self._signature = None
        self._lock = threading.RLock()

    def accepts(self, source_id):
        return source_id.startswith(self.prefix)

    def _refresh(self):
        try:
            info = _safe_file(self.database, max_bytes=512 * 1024**2)
            wal = self.database.with_name(self.database.name + "-wal")
            wal_info = _safe_file(wal, max_bytes=512 * 1024**2) if wal.exists() else None
        except (OSError, ValueError) as error:
            raise HistoricalSourceError("provider_source_unavailable") from error
        signature = (
            info.st_size,
            info.st_mtime_ns,
            wal_info.st_size if wal_info else 0,
            wal_info.st_mtime_ns if wal_info else 0,
        )
        if signature != self._signature or not self.snapshot.exists():
            list(
                read_cline_snapshot(
                    self.database, self.snapshot, resource_guard=self.resource_guard
                )
            )
            self._signature = signature

    def _row(self, session_id):
        self._refresh()
        db = sqlite3.connect(self.snapshot.absolute().as_uri() + "?mode=ro", uri=True)
        db.row_factory = sqlite3.Row
        try:
            columns = {row[1] for row in db.execute("PRAGMA table_info(sessions)")}
            required = {"session_id", "started_at", "updated_at", "prompt", "messages_path"}
            if not required <= columns:
                raise ValueError("Unsupported Cline sessions schema")
            row = db.execute(
                "SELECT session_id,started_at,updated_at,prompt,messages_path FROM sessions WHERE session_id=?",
                (session_id,),
            ).fetchone()
            return dict(row) if row else None
        finally:
            db.close()

    def _record(self, source_id):
        if not self.accepts(source_id):
            return None
        session_id = source_id[len(self.prefix) :]
        with self._lock:
            row = self._row(session_id)
            if row is None:
                return None
            raw = b""
            messages = []
            if row["messages_path"]:
                path = Path(row["messages_path"]).absolute()
                if ".." in path.parts or not path.is_relative_to(self.session_root):
                    raise ValueError("Cline message path escapes the session directory")
                before = _safe_file(path, max_bytes=16 * 1024**2)
                with path.open("rb") as stream:
                    raw = stream.read(16 * 1024**2 + 1)
                if len(raw) > 16 * 1024**2:
                    raise ValueError("Cline message source exceeds the regular-file 16 MiB bound")
                after = _safe_file(path, max_bytes=16 * 1024**2)
                if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
                    raise OSError("Cline messages changed while being read")
                messages = [
                    message
                    for message in parse_json_chat(raw, "cline")
                    if not _SECRET.search(message.text.encode())
                ]
            elif row["prompt"] and not _SECRET.search(row["prompt"].encode()):
                messages = [ChatMessage(row["prompt"], "user", row["started_at"], "prompt")]
            digest = hashlib.sha256(
                json.dumps(row, sort_keys=True).encode() + b"\0" + raw
            ).hexdigest()
            return {
                "id": source_id,
                "path": f"{self.database}#session={session_id}",
                "name": "Historical Cline session",
                "kind": "file",
                "state": "indexed",
                "digest": digest,
                "size": len(raw),
                "mtime_ns": 0,
                "file_id": self.identity + ":" + session_id,
                "historical": True,
            }, messages

    def node(self, source_id):
        try:
            record = self._record(source_id)
        except (OSError, ValueError, sqlite3.Error):
            return None
        return record[0] if record else None

    def authorized(self, source_id, scope):
        return scope == "full_private" and self.node(source_id) is not None

    def passages(self, source_id):
        record = self._record(source_id)
        if record is None:
            return
        line = 1
        for message in record[1]:
            prefix = f"[historical {message.role}; time={message.timestamp or 'unknown'}] "
            text = prefix + message.text
            end = line + text.count("\n")
            yield {
                "kind": "chat",
                "start": line,
                "end": end,
                "text": text,
                "role": message.role,
                "timestamp": message.timestamp,
                "historical": True,
                "locator": message.locator,
            }
            line = end + 1

    def read_evidence(self, source_id, *, limit=1000, start_line=1):
        if not 1 <= limit <= 16_000 or start_line < 1:
            raise ValueError("Cline evidence requires limit in [1,16000] and positive start line")
        try:
            record = self._record(source_id)
            if record is None:
                return None
            lines = []
            for message in record[1]:
                text = f"[historical {message.role}; time={message.timestamp or 'unknown'}] {message.text}"
                lines.extend(text.splitlines())
        except (OSError, ValueError, sqlite3.Error):
            return None
        text = "\n".join(lines[start_line - 1 :])[:limit]
        return {
            "node_id": source_id,
            "path": record[0]["path"],
            "digest": record[0]["digest"],
            "locator": {
                "kind": "historical_chat_lines",
                "start": start_line,
                "end": start_line + text.count("\n"),
            },
            "text": text,
            "historical": True,
        }

    def scan(self, store, version, *, limit=256):
        if not 1 <= limit <= 4096:
            raise ValueError("Cline scan batch must be in [1,4096]")
        key = "cline_scan:" + self.identity
        cursor = store.get_meta(key + ":cursor", "")
        epoch = store.get_meta(key + ":epoch") if cursor else str(time.time_ns())
        with self._lock:
            self._refresh()
            db = sqlite3.connect(self.snapshot.absolute().as_uri() + "?mode=ro", uri=True)
            try:
                sessions = [
                    row[0]
                    for row in db.execute(
                        "SELECT session_id FROM sessions WHERE session_id>? ORDER BY session_id LIMIT ?",
                        (cursor, limit),
                    )
                ]
            finally:
                db.close()
        queued = 0
        for session_id in sessions:
            source_id = self.prefix + session_id
            node = self.node(source_id)
            if node is None:
                store.invalidate_source(source_id)
                continue
            store.queue_source(
                source_id,
                node["digest"],
                node["path"],
                "private",
                version(node),
                scan_epoch=int(epoch),
                provider=self.prefix,
            )
            queued += 1
        complete = len(sessions) < limit
        with store.db:
            store.set_meta(key + ":cursor", "" if complete else sessions[-1])
            store.set_meta(key + ":epoch", epoch)
            if complete:
                store.db.execute(
                    "UPDATE sources SET active=0,ready=0 WHERE provider=? AND scan_epoch<>?",
                    (self.prefix, int(epoch)),
                )
        return {"queued": queued, "complete": complete}
