"""Local historical chat adapters; snapshots never modify the source database."""

import json
import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from .artifacts import replace_file

MAX_RECORD_BYTES = 16 * 1024 * 1024


class HistoricalSourceError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class ChatMessage:
    text: str
    role: str
    timestamp: str | None
    locator: str
    historical: bool = True


def _message_text(message):
    content = message.get("content", message.get("text", ""))
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, dict):
        content = content.get("parts", [])
    if not isinstance(content, list):
        return ""
    parts = []
    for part in content:
        if isinstance(part, str):
            parts.append(part)
        elif (
            isinstance(part, dict)
            and part.get("type") in {"text", "input_text", "output_text"}
            and isinstance(part.get("text"), str)
        ):
            parts.append(part["text"])
    return "\n".join(parts).strip()


def _message(record, locator, timestamp=None):
    author = record.get("author", {})
    role = record.get("role") or record.get("sender")
    if role is None and isinstance(author, dict):
        role = author.get("role")
    if role == "human":
        role = "user"
    if role not in {"user", "assistant"}:
        return None
    text = _message_text(record)
    if not text:
        return None
    date = record.get(
        "timestamp",
        record.get("create_time", record.get("created_at", record.get("ts", timestamp))),
    )
    return ChatMessage(text, role, None if date is None else str(date), locator)


def _records(value, prefix=""):
    if isinstance(value, list):
        for index, item in enumerate(value):
            yield from _records(item, f"{prefix}/{index}")
    elif isinstance(value, dict):
        if "mapping" in value and isinstance(value["mapping"], dict):
            mapping = value["mapping"]
            current = value.get("current_node")
            if current in mapping:
                lineage = []
                seen = set()
                while current in mapping and current not in seen:
                    seen.add(current)
                    lineage.append(current)
                    current = mapping[current].get("parent")
                keys = reversed(lineage)
            else:
                keys = sorted(mapping)
            for key in keys:
                node = mapping[key]
                if isinstance(node, dict) and isinstance(node.get("message"), dict):
                    yield node["message"], f"{prefix}/mapping/{key}"
        elif any(key in value for key in ("chat_messages", "messages", "apiConversationHistory")):
            for key in ("chat_messages", "messages", "apiConversationHistory"):
                if key in value:
                    yield from _records(value[key], f"{prefix}/{key}")
        else:
            yield value, prefix


def parse_json_chat(data: str | bytes, source: str):
    if len(data if isinstance(data, bytes) else data.encode()) > MAX_RECORD_BYTES:
        raise ValueError("Chat input exceeds the 16 MiB adapter bound")
    if isinstance(data, bytes):
        data = data.decode("utf-8-sig")
    if source in {"codex", "claude"}:
        for line_number, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue  # A running session may end in an incomplete JSONL record.
            if not isinstance(event, dict):
                continue
            if source == "codex":
                record = event.get("payload")
                if event.get("type") != "response_item" or not isinstance(record, dict):
                    continue
                if record.get("type") != "message":
                    continue
            else:
                record = event.get("message")
                if event.get("type") not in {"user", "assistant"} or not isinstance(record, dict):
                    continue
                record = {**record, "role": record.get("role", event["type"])}
            message = _message(record, f"line:{line_number}", event.get("timestamp"))
            if message:
                yield message
        return
    value = json.loads(data)
    for record, locator in _records(value):
        message = _message(record, locator)
        if message:
            yield message


def read_cline_snapshot(
    source: Path, snapshot: Path, *, resource_guard=None, max_snapshot_bytes=512 * 1024**2
):
    """Read committed WAL content through SQLite's online backup API."""
    source, snapshot = Path(source).absolute(), Path(snapshot).absolute()
    if source == snapshot:
        raise ValueError("Cline snapshot must differ from its source")
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    temporary = snapshot.with_name(snapshot.name + ".partial")
    try:
        origin = sqlite3.connect(source.as_uri() + "?mode=ro", uri=True, timeout=2)
    except sqlite3.OperationalError as error:
        if getattr(error, "sqlite_errorcode", 0) & 0xFF == sqlite3.SQLITE_CANTOPEN:
            raise HistoricalSourceError("provider_source_unavailable") from error
        raise
    destination = None
    try:
        page_size = origin.execute("PRAGMA page_size").fetchone()[0]
        snapshot_size = page_size * origin.execute("PRAGMA page_count").fetchone()[0]
        if not 0 < snapshot_size <= max_snapshot_bytes:
            raise ValueError("Cline snapshot exceeds the configured size bound")
        if resource_guard:
            resource_guard(background=True, additional_disk=snapshot_size)
        started = time.monotonic()

        def progress(status, remaining, total):
            if total * page_size > max_snapshot_bytes:
                raise ValueError("Cline snapshot grew beyond the configured size bound")
            if time.monotonic() - started > 30:
                raise TimeoutError("Cline snapshot exceeded its 30 second online backup budget")

        destination = sqlite3.connect(temporary)
        origin.backup(destination, pages=256, progress=progress)
    finally:
        if destination is not None:
            destination.close()
        origin.close()
    with temporary.open("rb+") as stream:
        os.fsync(stream.fileno())
    replace_file(temporary, snapshot)
    db = sqlite3.connect(snapshot.as_uri() + "?mode=ro", uri=True)
    try:
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "ItemTable" in tables:
            rows = db.execute(
                """SELECT key,value FROM ItemTable WHERE key IN
                ('taskHistory','apiConversationHistory','messages')
                OR key LIKE 'apiConversationHistory:%' OR key LIKE 'messages:%'"""
            )
            for key, value in rows:
                if isinstance(value, (str, bytes)) and len(value) <= MAX_RECORD_BYTES:
                    for message in parse_json_chat(value, "cline"):
                        yield ChatMessage(
                            message.text,
                            message.role,
                            message.timestamp,
                            f"sqlite:{key}{message.locator}",
                        )
        if "messages" in tables:
            columns = {row[1] for row in db.execute("PRAGMA table_info(messages)")}
            if {"role", "content"} <= columns:
                date = "timestamp" if "timestamp" in columns else "NULL"
                for rowid, role, content, timestamp in db.execute(
                    f"SELECT rowid,role,content,{date} FROM messages ORDER BY rowid"
                ):
                    message = _message(
                        {"role": role, "content": content}, f"sqlite:messages:{rowid}", timestamp
                    )
                    if message:
                        yield message
    finally:
        db.close()
