import hashlib
import sqlite3
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from dodo_mem_neural import worker as worker_module
from dodo_mem_neural.compact_store import CompactIndexStore
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.vector_store import VectorStore
from dodo_mem_neural.worker import NeuralWorker, chunk_text


def tokenizer():
    value = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    value.pre_tokenizer = pre_tokenizers.Whitespace()
    value.enable_truncation(8)
    value.enable_padding(length=8)
    return value


def test_chat_growth_after_metadata_check_is_still_bounded(tmp_path, monkeypatch):
    path = tmp_path / ".codex/rollout-growth.jsonl"
    path.parent.mkdir()
    path.write_bytes(b"{}\n")
    original = Path.open
    grown = False

    def grow_on_open(selected, mode="r", *args, **kwargs):
        nonlocal grown
        if selected == path and mode == "rb" and not grown:
            grown = True
            with original(selected, "ab") as stream:
                stream.truncate(16 * 1024**2 + 64)
        return original(selected, mode, *args, **kwargs)

    monkeypatch.setattr(Path, "open", grow_on_open)
    worker = object.__new__(NeuralWorker)
    with pytest.raises(ValueError, match="16 MiB bound"):
        worker._chat_messages({"path": str(path), "digest": "old-fixture"})


@pytest.mark.parametrize("store_type", [IndexStore, CompactIndexStore])
def test_missing_chat_fails_only_that_version_and_next_source_finishes(tmp_path, store_type):
    database = tmp_path / "catalogue.db"
    db = sqlite3.connect(database)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,
            start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');
    """)
    missing = tmp_path / ".codex/sessions/rollout-synthetic.jsonl"
    missing.parent.mkdir(parents=True)
    missing.write_text(
        '{"type":"response_item","payload":{"type":"message","role":"user","content":"word"}}\n'
    )
    valid = tmp_path / "valid.md"
    valid.write_text("word")
    for identity, path in (("a-missing", missing), ("b-valid", valid)):
        info = path.stat()
        db.execute(
            "INSERT INTO nodes VALUES (?,'root',?,'file','indexed',?,?,?,?)",
            (
                identity,
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                info.st_size,
                info.st_mtime_ns,
                f"{info.st_dev}:{info.st_ino}",
            ),
        )
        db.execute(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,'line',1,1,'word')",
            (identity,),
        )
    db.commit()
    store = store_type(tmp_path / "state.db")
    vectors = VectorStore(tmp_path / "vectors", tmp_path / "publication")

    def embed(texts, query=False):
        result = np.zeros((len(texts), 384), dtype=np.float32)
        result[:, 0] = 1
        return result

    worker = NeuralWorker(database, store, vectors, embed, tokenizer(), source_providers=())
    worker.scan()
    missing.unlink()
    batch = worker.run_pending()
    assert batch["completed"] == 1 and batch["failed"] == 1 and batch["pending"] == 0
    assert (
        store.db.execute("SELECT status FROM jobs WHERE source_id='a-missing'").fetchone()[0]
        == "failed"
    )
    assert (
        store.db.execute("SELECT ready FROM sources WHERE source_id='a-missing'").fetchone()[0] == 0
    )
    failure = store.db.execute("SELECT reason FROM job_failures").fetchone()[0]
    assert failure == "historical_source_unavailable" and str(tmp_path) not in failure
    worker.scan()
    assert worker.run_pending()["failed"] == 0
    assert (
        store.db.execute("SELECT ready FROM sources WHERE source_id='b-valid'").fetchone()[0] == 1
    )
    worker.close()
    store.close()
    db.close()


def test_catalogue_changes_jump_the_backlog_and_acknowledge_only_current_revision(tmp_path):
    database = tmp_path / "catalogue.db"
    db = sqlite3.connect(database)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,
            start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');
    """)
    path = tmp_path / "note.md"
    path.write_text("word")
    db.execute(
        "INSERT INTO nodes VALUES ('a','root',?,'file','indexed','old',4,1,'f')", (str(path),)
    )
    db.commit()
    store = IndexStore(tmp_path / "state.db")
    vectors = VectorStore(tmp_path / "cache", tmp_path / "publication")
    worker = NeuralWorker(
        database, store, vectors, lambda *args, **kwargs: None, tokenizer(), source_providers=()
    )
    with store.db:
        store.set_meta("scan_cursor", "z")
        store.set_meta("scan_epoch", 7)
    for number in range(20):
        store.queue_source(f"backlog-{number}", "digest", path, "private", "v", scan_epoch=7)
    db.execute("UPDATE nodes SET digest='new',mtime_ns=2 WHERE id='a'")
    db.commit()
    changed = worker.changes.execute(
        "SELECT revision FROM neural_changes WHERE node_id='a'"
    ).fetchone()[0]
    db.execute("UPDATE nodes SET digest='newer',mtime_ns=3 WHERE id='a'")
    db.commit()
    with worker.changes:
        worker.changes.execute(
            "DELETE FROM neural_changes WHERE node_id='a' AND revision=?", (changed,)
        )
    worker.changes.execute("PRAGMA busy_timeout=1")
    db.execute("BEGIN IMMEDIATE")
    with pytest.raises(ResourcePressure, match="writer is busy"):
        worker.consume_changes()
    assert worker.changes.execute("SELECT count(*) FROM neural_changes").fetchone()[0] == 1
    db.rollback()
    assert worker.consume_changes() == 1
    assert store.claim_job()["source_id"] == "a"
    assert worker.changes.execute("SELECT count(*) FROM neural_changes").fetchone()[0] == 0
    assert store.db.execute("SELECT scan_epoch FROM sources WHERE source_id='a'").fetchone()[0] == 7
    db.execute("DELETE FROM nodes WHERE id='a'")
    db.commit()
    assert worker.consume_changes() == 1
    assert store.db.execute("SELECT active FROM sources WHERE source_id='a'").fetchone()[0] == 0
    worker.close()
    store.close()
    db.close()


def test_old_ordinary_passages_with_bearer_credentials_are_not_embedded(tmp_path):
    database = tmp_path / "catalogue.db"
    db = sqlite3.connect(database)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,
            start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');
    """)
    path = tmp_path / "old-extraction.md"
    text = "An old cached extraction contains Bearer " + "Z" * 32
    path.write_text(text)
    info = path.stat()
    db.execute(
        "INSERT INTO nodes VALUES ('a','root',?,'file','indexed',?,?,?,?)",
        (
            str(path),
            hashlib.sha256(path.read_bytes()).hexdigest(),
            info.st_size,
            info.st_mtime_ns,
            f"{info.st_dev}:{info.st_ino}",
        ),
    )
    db.execute("INSERT INTO passages VALUES (1,'a','lines',1,1,?)", (text,))
    db.commit()
    store = IndexStore(tmp_path / "state.db")
    vectors = VectorStore(tmp_path / "cache", tmp_path / "publication")

    def unexpected_embedding(*args, **kwargs):
        raise AssertionError("credential-containing text reached inference")

    worker = NeuralWorker(
        database, store, vectors, unexpected_embedding, tokenizer(), source_providers=()
    )
    worker.scan()
    assert worker.run_pending()["embedded"] == 0
    assert store.lexical_search("Bearer", ("private",)) == []
    worker.close()
    store.close()
    db.close()


def test_chunking_disables_runtime_padding_and_truncation():
    chunks = list(chunk_text(" ".join(["word"] * 500), tokenizer()))
    assert [(c["token_start"], c["token_end"]) for c in chunks] == [
        (0, 224),
        (192, 416),
        (384, 500),
    ]
    assert all(len(c["text"].split()) <= 224 for c in chunks)


def test_failed_worker_startup_closes_its_connections(tmp_path, monkeypatch):
    database = tmp_path / "catalogue.db"
    db = sqlite3.connect(database)
    db.execute("""CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
        digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT)""")
    db.close()
    store = IndexStore(tmp_path / "state.db")
    vectors = VectorStore(tmp_path / "vectors", tmp_path / "publication")
    connections = []
    connect = sqlite3.connect

    def capture(*args, **kwargs):
        connection = connect(*args, **kwargs)
        connections.append(connection)
        return connection

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(worker_module.sqlite3, "connect", capture)
    monkeypatch.setattr(store, "get_meta", locked)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        NeuralWorker(database, store, vectors, None, tokenizer(), source_providers=())
    assert len(connections) == 2
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
    assert store.db.execute("SELECT 1").fetchone()[0] == 1
    store.close()


def test_chunk_boundary_retokenisation_respects_224_body_tokens():
    value = Tokenizer(
        models.WordPiece(
            {
                "[UNK]": 0,
                "a": 1,
                "##a": 2,
                "##bcd": 3,
                "b": 4,
                "##c": 5,
                "##d": 6,
            },
            unk_token="[UNK]",
        )
    )
    value.pre_tokenizer = pre_tokenizers.Whitespace()
    chunks = list(chunk_text(" ".join(["aaabcdaaa"] * 75), value))
    assert (
        max(len(value.encode(chunk["text"], add_special_tokens=False).ids) for chunk in chunks)
        <= 224
    )


def test_worker_restarts_deduplicates_and_rejects_stale_sources(tmp_path):
    catalogue = tmp_path / "catalogue.db"
    db = sqlite3.connect(catalogue)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,
            start INTEGER,end INTEGER,text TEXT);
    """)
    db.execute("INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r')")
    for name in ("a", "b"):
        path = tmp_path / f"{name}.md"
        path.write_text("word word", encoding="utf-8")
        info = path.stat()
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        db.execute(
            "INSERT INTO nodes VALUES (?,'root',?,'file','indexed',?,?,?,?)",
            (
                name,
                str(path),
                digest,
                info.st_size,
                info.st_mtime_ns,
                f"{info.st_dev}:{info.st_ino}",
            ),
        )
        db.execute(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,'line',1,1,?)",
            (name, "word word"),
        )
    db.commit()
    encoded = []

    def embed(texts, query=False):
        encoded.extend(texts)
        vectors = np.zeros((len(texts), 384), dtype=np.float32)
        vectors[:, 0] = 1
        return vectors

    state = IndexStore(tmp_path / "state.db", encoder_id="test")
    vectors = VectorStore(tmp_path / "cache", tmp_path / "publication", encoder_id="test")
    events = []
    worker = NeuralWorker(
        catalogue,
        state,
        vectors,
        embed,
        tokenizer(),
        source_providers=(),
        progress_callback=events.append,
    )
    assert worker.scan(limit=1)["complete"] is False
    assert worker.scan(limit=1)["complete"] is False
    assert worker.scan(limit=1)["complete"] is True
    assert worker.run_pending(limit=2)["completed"] == 2
    assert encoded == ["word word"]
    assert {event["stage"] for event in events} >= {
        "scanning",
        "reading",
        "embedding",
        "publishing",
    }
    assert max(event.get("batch_embedded", 0) for event in events) == 1
    assert all("text" not in event and "path" not in event for event in events)
    keys = [key for key, _ in vectors.search(np.eye(1, 384, dtype=np.float32)[0], 3)]
    assert {row["source_id"] for row in state.visible_occurrences(keys, ["private"])} == {"a", "b"}
    db.execute("UPDATE nodes SET digest='edited' WHERE id='a'")
    db.execute("UPDATE passages SET text='word changed' WHERE node_id='a'")
    db.commit()
    assert worker.refresh_changed(limit=128)["changed"] == 1
    assert worker.run_pending(limit=1)["completed"] == 1
    db.execute("DELETE FROM nodes WHERE id='b'")
    db.commit()
    worker.scan(limit=10)
    worker.scan(limit=10)
    keys = [key for key, _ in vectors.search(np.eye(1, 384, dtype=np.float32)[0], 3)]
    assert {row["source_id"] for row in state.visible_occurrences(keys, ["private"])} == {"a"}
    worker.close()
    state.close()
    db.close()


@pytest.mark.parametrize("store_class", [IndexStore, CompactIndexStore])
def test_file_centroid_uses_unique_passage_vectors_and_rebuilds_changed_policy(
    tmp_path, store_class
):
    database = tmp_path / "catalogue.db"
    db = sqlite3.connect(database)
    db.executescript("""
        CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,state TEXT,
            digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,
            start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');
    """)
    path = tmp_path / "repeated.md"
    path.write_text("word word\nword word\nword word\nother")
    db.execute(
        "INSERT INTO nodes VALUES ('a','root',?,'file','indexed','digest',33,1,'f')", (str(path),)
    )
    db.executemany(
        "INSERT INTO passages VALUES (?,'a','lines',?,?,?)",
        ((n, n, n, text) for n, text in enumerate(["word word"] * 3 + ["other"], 1)),
    )
    db.commit()
    store = store_class(tmp_path / "state.db", encoder_id="test")
    vectors = VectorStore(tmp_path / "vectors", tmp_path / "publication", encoder_id="test")

    def embed(texts, query=False):
        values = np.zeros((len(texts), 384), dtype=np.float32)
        for n, text in enumerate(texts):
            values[n, 0 if text == "word word" else 1] = 1
        return values

    worker = NeuralWorker(database, store, vectors, embed, tokenizer(), source_providers=())
    worker.scan()
    assert worker.run_pending()["embedded"] == 2
    centroid = store.centroid_rows()[0]
    np.testing.assert_allclose(centroid["vector"][:2], np.array([2, 1]) / np.sqrt(5), atol=1e-6)
    assert centroid["weight"] == 3
    worker.close()
    with store.db:
        store.set_meta("centroid_policy", "older-policy")
        store.set_meta("scan_cursor", "z")
        store.set_meta("index_progress", '{"embedded":1000,"completed_jobs":500}')
    worker = NeuralWorker(database, store, vectors, embed, tokenizer(), source_providers=())
    assert store.db.execute("SELECT ready FROM sources").fetchone()[0] == 0
    assert store.pending_count() == 1 and store.get_meta("scan_cursor") == ""
    assert store.get_meta("index_progress") == "{}"
    assert '"embedded": 1000' in store.get_meta("historical_index_progress")
    worker.scan()
    assert worker.run_pending()["completed"] == 1
    worker.close()
    store.close()
    db.close()
