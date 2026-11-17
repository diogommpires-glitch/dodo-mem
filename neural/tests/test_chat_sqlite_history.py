import json
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from dodo_mem_neural import sqlite_history as history_module
from dodo_mem_neural.chat_adapters import HistoricalSourceError
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.sqlite_history import ClineHistory
from dodo_mem_neural.vector_store import VectorStore
from dodo_mem_neural.worker import NeuralWorker


def test_offline_cline_database_isolated_and_resource_pressure_still_propagates(tmp_path):
    data = tmp_path / "cline-data/db"
    data.mkdir(parents=True)
    database = data / "sessions.db"
    source = sqlite3.connect(database)
    source.execute(
        "CREATE TABLE sessions(session_id TEXT PRIMARY KEY,started_at TEXT,updated_at TEXT,prompt TEXT,messages_path TEXT)"
    )
    source.execute("INSERT INTO sessions VALUES ('one','a','b','word',NULL)")
    source.commit()
    source.close()
    provider = ClineHistory(database, tmp_path / "snapshots")
    node = provider.node(provider.prefix + "one")
    catalogue = tmp_path / "catalogue.db"
    db = sqlite3.connect(catalogue)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');
        INSERT INTO nodes VALUES ('valid','root','valid.md','file','indexed','fixture',4,1,'v');
        INSERT INTO passages VALUES (1,'valid','line',1,1,'word');
    """)
    db.close()
    store = IndexStore(tmp_path / "state.db")
    vectors = VectorStore(tmp_path / "vectors", tmp_path / "publication")
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    worker = NeuralWorker(
        catalogue,
        store,
        vectors,
        lambda texts, **_: np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1)),
        tokenizer,
        source_providers=[provider],
    )
    offline = database.with_suffix(".offline")
    database.rename(offline)
    with pytest.raises(HistoricalSourceError, match="provider_source_unavailable"):
        list(worker._chunks(node))
    scan = worker.scan()
    assert scan["unavailable_providers"] == 1 and scan["complete"] is False
    assert worker.run_pending()["completed"] == 1
    offline.rename(database)
    provider.snapshot.unlink()

    def pressure(**_):
        raise ResourcePressure("synthetic storage reserve")

    provider.resource_guard = pressure
    with pytest.raises(ResourcePressure, match="storage reserve"):
        worker.scan()
    worker.close()
    store.close()


def test_cline_source_removed_between_stat_and_sqlite_connect_is_typed(tmp_path, monkeypatch):
    database = tmp_path / "sessions.db"
    source = sqlite3.connect(database)
    source.execute("CREATE TABLE fixture(value TEXT)")
    source.close()
    provider = ClineHistory(database, tmp_path / "snapshots")
    original = history_module._safe_file

    def disappear(path, **kwargs):
        info = original(path, **kwargs)
        if path == database:
            database.unlink()
        return info

    monkeypatch.setattr(history_module, "_safe_file", disappear)
    with pytest.raises(HistoricalSourceError, match="provider_source_unavailable"):
        provider._refresh()


def test_cline_message_growth_does_not_read_beyond_16_mib(tmp_path, monkeypatch):
    data = tmp_path / "cline-data"
    (data / "db").mkdir(parents=True)
    (data / "sessions").mkdir()
    messages = data / "sessions/test.json"
    messages.write_text('{"messages":[{"role":"user","content":"word"}]}')
    database = data / "db/sessions.db"
    source = sqlite3.connect(database)
    source.execute(
        "CREATE TABLE sessions(session_id TEXT PRIMARY KEY,started_at TEXT,updated_at TEXT,prompt TEXT,messages_path TEXT)"
    )
    source.execute("INSERT INTO sessions VALUES ('one','a','b','word',?)", (str(messages),))
    source.commit()
    source.close()
    provider = ClineHistory(database, tmp_path / "snapshots")
    original_safe_file = history_module._safe_file
    original_open = Path.open
    actual_reads = []
    grown = False

    def grow(path, **kwargs):
        nonlocal grown
        info = original_safe_file(path, **kwargs)
        if path == messages and not grown:
            grown = True
            with original_open(messages, "ab") as stream:
                stream.truncate(17 * 1024**2)
        return info

    class MeasuredRead:
        def __init__(self, stream):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.stream.close()

        def read(self, size=-1):
            raw = self.stream.read(size)
            actual_reads.append(len(raw))
            return raw

    def open_file(path, mode="r", *args, **kwargs):
        stream = original_open(path, mode, *args, **kwargs)
        return MeasuredRead(stream) if path == messages and mode == "rb" else stream

    monkeypatch.setattr(history_module, "_safe_file", grow)
    monkeypatch.setattr(Path, "open", open_file)
    with pytest.raises(ValueError, match="bound"):
        provider._record(provider.prefix + "one")
    assert actual_reads and max(actual_reads) <= 16 * 1024**2 + 1


def test_cline_provider_registers_wal_sessions_and_rereads_current_evidence(tmp_path):
    data = tmp_path / "cline-data"
    (data / "db").mkdir(parents=True)
    (data / "sessions").mkdir()
    database = data / "db/sessions.db"
    messages = data / "sessions/test.messages.json"
    messages.write_text(
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "historical request", "ts": 100},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "historical answer"}],
                        "ts": 101,
                    },
                    {"role": "tool", "content": "hidden tool"},
                    {"role": "user", "content": "Authorization: Bearer " + "synthetic_token_" * 3},
                ]
            }
        ),
        encoding="utf-8",
    )
    db = sqlite3.connect(database)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("""CREATE TABLE sessions(session_id TEXT PRIMARY KEY,started_at TEXT,
        updated_at TEXT,prompt TEXT,messages_path TEXT)""")
    db.execute(
        "INSERT INTO sessions VALUES ('one','2026-01-01','2026-01-02','prompt',?)", (str(messages),)
    )
    db.commit()
    provider = ClineHistory(database, tmp_path / "snapshots")
    store = IndexStore(tmp_path / "state.db", encoder_id="test")
    result = provider.scan(store, lambda node: node["digest"], limit=10)
    assert result["queued"] == 1 and result["complete"] is True
    job = store.claim_job()
    assert job["source_id"].startswith("cline-sqlite:") and job["scope"] == "private"
    node = provider.node(job["source_id"])
    evidence = provider.read_evidence(job["source_id"], limit=1000)
    assert evidence["digest"] == node["digest"]
    assert "historical request" in evidence["text"] and "historical answer" in evidence["text"]
    assert "hidden tool" not in evidence["text"]
    assert "Authorization" not in evidence["text"]
    assert provider.authorized(job["source_id"], "shared") is False
    assert provider.authorized(job["source_id"], "full_private") is True
    catalogue = tmp_path / "catalogue.db"
    catalogue_db = sqlite3.connect(catalogue)
    catalogue_db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,start INTEGER,
            end INTEGER,text TEXT);
    """)
    catalogue_db.close()
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    vectors = VectorStore(tmp_path / "vectors", tmp_path / "publication", encoder_id="test")

    def embed(texts, query=False):
        return np.tile(np.eye(1, 384, dtype=np.float32), (len(texts), 1))

    worker = NeuralWorker(catalogue, store, vectors, embed, tokenizer, source_providers=[provider])
    assert worker.scan(limit=10)["queued"] == 1
    assert worker.run_pending(limit=1)["completed"] == 1
    assert store.lexical_search("historical", ["shared"]) == []
    assert store.lexical_search("historical", ["private"])[0]["historical"] == 1
    worker.close()
    messages.write_text('{"messages":[{"role":"user","content":"changed"}]}', encoding="utf-8")
    assert provider.node(job["source_id"])["digest"] != node["digest"]
    store.finish_scan(999)
    assert (
        store.db.execute(
            "SELECT active FROM sources WHERE source_id=?", (job["source_id"],)
        ).fetchone()[0]
        == 1
    )
    store.close()
    db.close()


def test_cline_provider_refuses_message_paths_outside_session_root(tmp_path):
    data = tmp_path / "cline-data"
    (data / "db").mkdir(parents=True)
    database = data / "db/sessions.db"
    outside = tmp_path / "outside.json"
    outside.write_text('{"messages":[{"role":"user","content":"outside"}]}', encoding="utf-8")
    db = sqlite3.connect(database)
    db.execute(
        "CREATE TABLE sessions(session_id TEXT PRIMARY KEY,started_at TEXT,updated_at TEXT,prompt TEXT,messages_path TEXT)"
    )
    db.execute("INSERT INTO sessions VALUES ('one','a','b','',?)", (str(outside),))
    db.commit()
    provider = ClineHistory(database, tmp_path / "snapshots")
    assert provider.node(provider.prefix + "one") is None
    db.close()
