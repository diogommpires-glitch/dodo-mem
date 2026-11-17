import json
import sqlite3

import pytest

from dodo_mem_neural.chat_adapters import parse_json_chat, read_cline_snapshot


def test_codex_roles_time_and_tool_exclusion():
    data = "\n".join(
        json.dumps(row)
        for row in [
            {
                "type": "response_item",
                "timestamp": "2026-01-01T00:00:00Z",
                "payload": {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hello"}],
                },
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "tool", "content": "secret"},
            },
            {
                "type": "response_item",
                "payload": {"type": "message", "role": "assistant", "content": "reply"},
            },
        ]
    )
    messages = list(parse_json_chat(data, "codex"))
    assert [(m.role, m.text) for m in messages] == [("user", "hello"), ("assistant", "reply")]
    assert messages[0].timestamp == "2026-01-01T00:00:00Z"
    assert all(m.historical for m in messages)


def test_cline_online_backup_includes_wal_and_preserves_roles(tmp_path):
    source = tmp_path / "state.db"
    db = sqlite3.connect(source)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("CREATE TABLE ItemTable(key TEXT PRIMARY KEY,value TEXT)")
    db.execute(
        "INSERT INTO ItemTable VALUES (?,?)",
        (
            "taskHistory",
            json.dumps(
                [
                    {
                        "id": "t1",
                        "messages": [{"role": "user", "content": "wal message", "timestamp": 123}],
                    }
                ]
            ),
        ),
    )
    db.commit()
    messages = list(read_cline_snapshot(source, tmp_path / "snapshot.db"))
    assert [(m.role, m.text, m.timestamp) for m in messages] == [("user", "wal message", "123")]
    assert (tmp_path / "snapshot.db").exists()
    db.close()


def test_cline_snapshot_rejects_size_before_copy(tmp_path):
    source = tmp_path / "state.db"
    db = sqlite3.connect(source)
    db.execute("CREATE TABLE messages(role TEXT,content TEXT)")
    db.commit()
    with pytest.raises(ValueError, match="snapshot"):
        list(read_cline_snapshot(source, tmp_path / "snapshot.db", max_snapshot_bytes=1))
    assert not (tmp_path / "snapshot.db").exists()
    db.close()
