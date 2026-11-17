import numpy as np
import pytest

from dodo_mem_neural.vector_store import VectorStore


def test_shards_publish_reopen_and_fall_back_from_corrupt_current(tmp_path):
    cache, state = tmp_path / "cache", tmp_path / "state"
    vectors = np.zeros((3, 384), dtype=np.float32)
    vectors[:, :3] = np.eye(3)
    store = VectorStore(cache, state, encoder_id="test", max_mutable=2, max_shard=2)
    store.add([1, 2], vectors[:2])
    store.publish()
    store.add([3], vectors[2:])
    manifest = store.publish()
    assert max(shard["count"] for shard in manifest["shards"]) <= 2
    reopened = VectorStore(cache, state, encoder_id="test", max_mutable=2, max_shard=2)
    assert reopened.search(vectors[2], 1)[0][0] == 3
    (state / "CURRENT").write_text("broken", encoding="utf-8")
    recovered = VectorStore(cache, state, encoder_id="test", max_mutable=2, max_shard=2)
    assert recovered.contains([1, 2, 3]) == [True, True, False]
    assert recovered.search(vectors[0], 1)[0][0] == 1


def test_small_publications_compact_tail_and_keep_only_two_generations(tmp_path):
    store = VectorStore(
        tmp_path / "cache", tmp_path / "state", encoder_id="test", max_mutable=2, max_shard=4
    )
    vectors = np.zeros((6, 384), dtype=np.float32)
    vectors[:, :6] = np.eye(6)
    for offset in range(0, 6, 2):
        store.add([offset + 1, offset + 2], vectors[offset : offset + 2])
        manifest = store.publish()
    assert [row["count"] for row in manifest["shards"]] == [4, 2]
    assert len(list((tmp_path / "cache").glob("shard-*.usearch"))) <= 3
    assert store.get([6])[0, 5] > 0.99
    reopened = VectorStore(
        tmp_path / "cache", tmp_path / "state", encoder_id="test", max_mutable=2, max_shard=4
    )
    assert reopened.contains([1, 2, 3, 4, 5, 6]) == [True] * 6


def test_missing_derived_directory_is_rejected_before_mapped_search(tmp_path):
    store = VectorStore(tmp_path / "cache", tmp_path / "state")
    vector = np.zeros((1, 384), dtype=np.float32)
    vector[0, 0] = 1
    store.add([1], vector)
    store.publish()
    store.cache_dir = tmp_path / "offline-drive"
    with pytest.raises(OSError, match="storage is unavailable"):
        store.search(vector[0])


def test_cache_uuid_change_cannot_reuse_prior_integer_vector_keys(tmp_path):
    cache, state = tmp_path / "cache", tmp_path / "state"
    vector = np.zeros((1, 384), dtype=np.float32)
    vector[0, 0] = 1
    original = VectorStore(cache, state, cache_uuid="a" * 32)
    original.add([1], vector)
    original.publish()
    changed = VectorStore(cache, state, cache_uuid="b" * 32)
    assert changed.search(vector[0]) == []
    assert changed.contains([1]) == [False]
    assert original.search(vector[0])[0][0] == 1
