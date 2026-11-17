import hashlib
import json
import sqlite3

import numpy as np
import pytest
from tokenizers import Tokenizer, models, pre_tokenizers

from dodo_mem_neural.compact_store import CompactIndexStore
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.organisation import refresh_neighbours
from dodo_mem_neural.vector_store import VectorStore
from dodo_mem_neural.worker import NeuralWorker


def test_compact_worker_publishes_deduplicated_vectors_then_revalidates_sources(tmp_path):
    database = tmp_path / "source-catalogue.sqlite3"
    db = sqlite3.connect(database)
    db.executescript("""CREATE TABLE nodes(id TEXT PRIMARY KEY,root_id TEXT,path TEXT,kind TEXT,
        state TEXT,digest TEXT,size INTEGER,mtime_ns INTEGER,file_id TEXT);
        CREATE TABLE passages(id INTEGER PRIMARY KEY,node_id TEXT,kind TEXT,start INTEGER,end INTEGER,text TEXT);
        INSERT INTO nodes VALUES ('root','root','/','root','online',NULL,0,0,'r');""")
    for name in ("a", "b"):
        path = tmp_path / (name + ".md")
        path.write_text("word word")
        info = path.stat()
        db.execute(
            "INSERT INTO nodes VALUES (?,'root',?,'file','indexed',?,?,?,?)",
            (
                name,
                str(path),
                hashlib.sha256(path.read_bytes()).hexdigest(),
                info.st_size,
                info.st_mtime_ns,
                f"{info.st_dev}:{info.st_ino}",
            ),
        )
        db.execute(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES (?,'lines',1,1,'word word')",
            (name,),
        )
    db.commit()
    store = CompactIndexStore(
        tmp_path / "c-state/index.sqlite3",
        encoder_id="test",
        derived_dir=tmp_path / "e-cache",
        resource_guard=lambda **kwargs: None,
    )
    vectors = VectorStore(
        tmp_path / "e-vectors",
        tmp_path / "c-publication",
        encoder_id="test",
        cache_uuid=store.cache_uuid,
    )
    tokenizer = Tokenizer(models.WordLevel({"[UNK]": 0, "word": 1}, unk_token="[UNK]"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    def encode(texts, query=False):
        values = np.zeros((len(texts), 384), dtype=np.float32)
        values[:, 0] = 1
        return values

    worker = NeuralWorker(database, store, vectors, encode, tokenizer, source_providers=())
    worker.scan()
    batch = worker.run_pending()
    assert batch["completed"] == 2 and batch["embedded"] == 1
    assert (
        store.db.execute("SELECT count(*) FROM chunk_records WHERE text IS NOT NULL").fetchone()[0]
        == 0
    )
    keys = [key for key, _ in vectors.search(encode(["word"])[0])]
    assert {row["source_id"] for row in store.visible_occurrences(keys, ("private",))} == {"a", "b"}
    db.execute("DELETE FROM nodes WHERE id='b'")
    db.commit()
    worker.consume_changes()
    assert {row["source_id"] for row in store.visible_occurrences(keys, ("private",))} == {"a"}
    worker.close()
    store.close()
    db.close()


def test_compact_semantic_support_stores_one_numeric_pair_and_recovers_evidence(tmp_path):
    store = CompactIndexStore(tmp_path / "state.sqlite3", encoder_id="test")
    vectors = VectorStore(
        tmp_path / "vectors", tmp_path / "pub", encoder_id="test", cache_uuid=store.cache_uuid
    )
    files = VectorStore(
        tmp_path / "files",
        tmp_path / "file-pub",
        encoder_id="test:files",
        cache_uuid=store.cache_uuid,
    )
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    for name in ("a", "b"):
        store.queue_source(name, "digest-" + name, name, "private", "v")
        row = store.stage_chunks(
            name,
            "v",
            [
                {
                    "text": "shared semantic content",
                    "start": 1,
                    "end": 1,
                    "locator": '["lines:1","' + "a" * 64 + '",0,4]',
                }
            ],
        )[0]
        vectors.add([row["chunk_id"]], vector[None, :])
        key = store.save_centroid(name, "v", vector, 4)
        files.add([key], vector[None, :])
        store.complete_source(name, "v")
    vectors.publish()
    files.publish()
    for name in ("a", "b"):
        refresh_neighbours(store, files, name, passage_vectors=vectors)
    assert store.db.execute("SELECT count(*) FROM support_records").fetchone()[0] == 1
    edge = store.semantic_neighbours("b")[0]
    assert edge["evidence"]["source_passage"]["source_digest"] == "digest-b"
    assert (
        edge["evidence"]["target_passage"]["text_hash"]
        == hashlib.sha256(b"shared semantic content").hexdigest()
    )
    assert store.db.execute("SELECT text FROM chunks").fetchone()[0] is None
    store.close()


def test_ordinary_text_is_not_duplicated_but_external_chat_text_remains_searchable(tmp_path):
    store = CompactIndexStore(tmp_path / "index.sqlite3", encoder_id="test")
    store.queue_source("ordinary", "a", "a.md", "private", "a")
    store.queue_source("cline", "b", "b.json", "private", "b", provider="cline-sqlite:fixture")
    text = "A planted contextual preference."
    first = store.stage_chunks(
        "ordinary",
        "a",
        [{"text": text, "start": 1, "end": 1, "locator": '["lines:1","' + "d" * 64 + '",0,5]'}],
    )[0]
    assert (
        store.db.execute("SELECT text FROM chunks WHERE id=?", (first["chunk_id"],)).fetchone()[0]
        is None
    )
    assert "d" * 64 not in first["locator"]
    second = store.stage_chunks("cline", "b", [{"text": text, "start": 1, "end": 1}])[0]
    assert first["chunk_id"] == second["chunk_id"]
    store.complete_source("ordinary", "a")
    store.complete_source("cline", "b")
    assert len(store.lexical_search("planted", ("private",))) == 2
    assert store.visible_occurrences([first["chunk_id"]], ("shared",)) == []
    with store.read_snapshot() as snapshot:
        assert snapshot.cache_uuid == store.cache_uuid
        assert snapshot.pending_count() == 0
    store.close()


def test_legacy_copy_migrates_to_derived_drive_and_preserves_source(tmp_path):
    legacy = tmp_path / "c-state/index.sqlite3"
    old = IndexStore(legacy, encoder_id="test")
    old.queue_source("file", "digest", "file.md", "private", "v")
    old.stage_chunks("file", "v", [{"text": "ordinary body", "start": 1, "end": 1}])
    old.complete_source("file", "v")
    old.close()
    before = hashlib.sha256(legacy.read_bytes()).hexdigest()
    store = CompactIndexStore(
        legacy,
        encoder_id="test",
        derived_dir=tmp_path / "e-cache",
        resource_guard=lambda **kwargs: None,
    )
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert store.path.parent == tmp_path / "e-cache"
    assert store.db.execute("PRAGMA journal_mode").fetchone()[0] == "delete"
    assert store.db.execute("SELECT text FROM chunks").fetchone()[0] is None
    assert store.db.execute("SELECT ready FROM sources").fetchone()[0] == 0
    assert store.pending_count() == 1
    binding = json.loads(store.binding.read_text())
    assert binding["cache_uuid"] == store.cache_uuid
    original_uuid = store.cache_uuid
    store.close()
    reopened = CompactIndexStore(
        legacy,
        encoder_id="test",
        derived_dir=tmp_path / "e-cache",
        resource_guard=lambda **kwargs: None,
    )
    assert reopened.cache_uuid == original_uuid
    reopened.close()
    binding["cache_uuid"] = "0" * 32
    legacy.with_suffix(".cache.json").write_text(json.dumps(binding))
    with pytest.raises(ValueError, match="UUID"):
        CompactIndexStore(
            legacy,
            encoder_id="test",
            derived_dir=tmp_path / "e-cache",
            resource_guard=lambda **kwargs: None,
        )


def test_rescanned_jobs_progress_before_inherited_rebuild_backlog(tmp_path):
    legacy = tmp_path / "c-state/index.sqlite3"
    old = IndexStore(legacy, encoder_id="test")
    old.queue_source("inherited", "digest", "inherited.md", "private", "old-version")
    with old.db:
        old.db.execute("UPDATE jobs SET priority=10")
    old.close()
    options = {
        "encoder_id": "test",
        "derived_dir": tmp_path / "e-cache",
        "resource_guard": lambda **kwargs: None,
    }
    store = CompactIndexStore(legacy, **options)
    store.queue_source("rescanned", "digest", "current.md", "private", "current-version")
    assert store.claim_job()["source_id"] == "rescanned"
    store.close()
    # The correction is one-time: a restart must not demote newly queued work.
    store = CompactIndexStore(legacy, **options)
    row = store.db.execute("SELECT priority FROM jobs WHERE source_id='rescanned'").fetchone()
    assert row["priority"] == 0
    assert store.claim_job()["source_id"] == "rescanned"
    store.close()


@pytest.mark.parametrize("failure_stage", ["pragma", "metadata"])
def test_failed_compact_constructor_closes_target_connection(tmp_path, monkeypatch, failure_stage):
    connect = sqlite3.connect
    connections = []

    class InterruptedConnection(sqlite3.Connection):
        def execute(self, query, *args, **kwargs):
            if failure_stage == "pragma" and query == "PRAGMA journal_mode=DELETE":
                raise sqlite3.OperationalError("database is locked")
            return super().execute(query, *args, **kwargs)

    def capture(*args, **kwargs):
        connection = connect(*args, factory=InterruptedConnection, **kwargs)
        connections.append(connection)
        return connection

    def locked(*args, **kwargs):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(sqlite3, "connect", capture)
    if failure_stage == "metadata":
        monkeypatch.setattr(CompactIndexStore, "_space", locked)
    with pytest.raises(sqlite3.OperationalError, match="locked"):
        CompactIndexStore(tmp_path / "index.sqlite3")
    assert connections
    for connection in connections:
        with pytest.raises(sqlite3.ProgrammingError, match="closed"):
            connection.execute("SELECT 1")
