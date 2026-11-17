"""Independent fault checks for the rebuildable metadata publication boundary."""

import hashlib
import json

import numpy as np
import pytest

from dodo_mem_neural import compact_store
from dodo_mem_neural.compact_store import CompactIndexStore
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.vector_store import VectorStore


def legacy_fixture(path):
    store = IndexStore(path, encoder_id="test")
    store.queue_source("source", "digest", "source.md", "private", "v")
    store.stage_chunks("source", "v", [{"text": "synthetic evidence", "start": 1, "end": 1}])
    store.complete_source("source", "v")
    store.close()
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_migration_cleans_owned_temporary_and_preserves_legacy(tmp_path, monkeypatch):
    legacy = tmp_path / "state/index.sqlite3"
    before = legacy_fixture(legacy)
    migrate = CompactIndexStore._migrate

    def fail_after_copy(self, source):
        migrate(self, source)
        raise ResourcePressure("Synthetic interruption after copying.")

    monkeypatch.setattr(CompactIndexStore, "_migrate", fail_after_copy)
    for _ in range(2):
        with pytest.raises(ResourcePressure):
            CompactIndexStore(
                legacy, derived_dir=tmp_path / "derived", resource_guard=lambda **kw: None
            )
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert not legacy.with_suffix(".cache.json").exists()
    assert not list((tmp_path / "derived").glob("*.migrating-*"))


def test_pointer_failure_recovers_only_committed_derived_database(tmp_path, monkeypatch):
    legacy = tmp_path / "state/index.sqlite3"
    before = legacy_fixture(legacy)
    publish = compact_store.atomic_json

    def fail_publication(*args, **kwargs):
        raise OSError("Synthetic C: pointer write failure.")

    monkeypatch.setattr(compact_store, "atomic_json", fail_publication)
    with pytest.raises(OSError):
        CompactIndexStore(
            legacy, derived_dir=tmp_path / "derived", resource_guard=lambda **kw: None
        )
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert not legacy.with_suffix(".cache.json").exists()
    monkeypatch.setattr(compact_store, "atomic_json", publish)
    store = CompactIndexStore(
        legacy, derived_dir=tmp_path / "derived", resource_guard=lambda **kw: None
    )
    assert store.db.execute("PRAGMA quick_check").fetchone()[0] == "ok"
    assert store.db.execute("SELECT ready FROM sources").fetchone()[0] == 0
    assert json.loads(store.binding.read_text())["cache_uuid"] == store.cache_uuid
    store.close()


def test_migration_honours_checkpoint_between_batches_and_can_retry(tmp_path):
    legacy = tmp_path / "state/index.sqlite3"
    before = legacy_fixture(legacy)
    calls = 0

    def checkpoint(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ResourcePressure("Synthetic service checkpoint request.")

    with pytest.raises(ResourcePressure, match="checkpoint"):
        CompactIndexStore(legacy, derived_dir=tmp_path / "derived", resource_guard=checkpoint)
    assert calls == 2
    assert hashlib.sha256(legacy.read_bytes()).hexdigest() == before
    assert not legacy.with_suffix(".cache.json").exists()
    assert not list((tmp_path / "derived").glob("*.migrating-*"))
    store = CompactIndexStore(
        legacy, derived_dir=tmp_path / "derived", resource_guard=lambda **kw: None
    )
    assert store.db.execute("SELECT count(*) FROM sources").fetchone()[0] == 1
    store.close()


def test_wrong_cache_uuid_does_not_reuse_colliding_vector_keys(tmp_path):
    first = VectorStore(tmp_path / "vectors", tmp_path / "pub", cache_uuid="a" * 32)
    vector = np.zeros((1, 384), dtype=np.float32)
    vector[0, 0] = 1
    first.add([1], vector)
    first.publish()
    second = VectorStore(tmp_path / "vectors", tmp_path / "pub", cache_uuid="b" * 32)
    assert second.contains([1]) == [False]
    assert second.search(vector[0]) == []
