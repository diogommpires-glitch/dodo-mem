import numpy as np
import pytest

from dodo_mem_neural.compact_store import CompactIndexStore
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.organisation import organise, publish_communities, refresh_neighbours
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.vector_store import VectorStore


def test_mutual_neighbours_and_communities_are_stable():
    vectors = np.zeros((6, 384), dtype=np.float32)
    vectors[:3, 0] = 1
    vectors[3:, 1] = 1
    identities = ["f", "e", "d", "c", "b", "a"]
    first = organise(identities, vectors, neighbours=2)
    second = organise(list(reversed(identities)), vectors[::-1], neighbours=2)
    assert first == second
    assert len(set(first["communities"].values())) == 2
    assert all(edge[2] > 0 for edge in first["edges"])
    assert max(sum(node in edge[:2] for edge in first["edges"]) for node in identities) <= 2


def test_persistent_mutual_graph_keeps_scope_and_reports_partial_coverage(tmp_path):
    store = IndexStore(tmp_path / "state.db", encoder_id="test")
    files = VectorStore(tmp_path / "cache", tmp_path / "publication", encoder_id="test:files")
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    for source, scope in [("a", "shared"), ("b", "private"), ("c", "shared")]:
        store.queue_source(source, source, source, scope, "v")
        key = store.save_centroid(source, "v", vector, 224)
        files.add([key], vector[None, :])
        store.complete_source(source, "v")
    files.publish()
    for source in ["a", "b", "c"]:
        refresh_neighbours(store, files, source)
    assert [row["target"] for row in store.semantic_neighbours("a", ["shared"])] == ["c"]
    result = publish_communities(store, limit=2)
    assert result["partial"] is True
    assert result["covered"] == 2 and result["total"] == 3
    first_ids = {row["source_id"]: row["community_id"] for row in store.community_rows()}
    publish_communities(store, limit=2)
    next_ids = {row["source_id"]: row["community_id"] for row in store.community_rows()}
    assert all(next_ids[source] == community for source, community in first_ids.items())
    assert all(row["scope"] == "shared" for row in store.topics(["shared"])["members"])
    store.invalidate_source("c")
    assert store.semantic_neighbours("a", ["shared"]) == []
    store.close()


def test_later_neighbour_refresh_returns_affected_old_sources(tmp_path):
    store = IndexStore(tmp_path / "state.db", encoder_id="test")
    files = VectorStore(tmp_path / "cache", tmp_path / "publication", encoder_id="test:files")
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    for source in ["a", "b"]:
        store.queue_source(source, source, source, "private", "v")
        key = store.save_centroid(source, "v", vector, 224)
        files.add([key], vector[None, :])
        store.complete_source(source, "v")
        files.publish()
        affected = refresh_neighbours(store, files, source)
        for old_source in affected:
            refresh_neighbours(store, files, old_source)
    assert [row["target"] for row in store.semantic_neighbours("a")] == ["b"]
    store.close()


def test_semantic_edge_keeps_sampled_passage_provenance(tmp_path):
    store = IndexStore(tmp_path / "state.db", encoder_id="test")
    files = VectorStore(tmp_path / "files", tmp_path / "file-state", encoder_id="test:files")
    passages = VectorStore(tmp_path / "passages", tmp_path / "passage-state", encoder_id="test")
    vector = np.zeros(384, dtype=np.float32)
    vector[0] = 1
    for source in ["a", "b"]:
        store.queue_source(source, "raw-digest-" + source, source, "private", "v")
        rows = store.stage_chunks(
            source,
            "v",
            [
                {
                    "text": "same semantic evidence",
                    "start": 1,
                    "end": 1,
                    "locator": '["lines:1","text-hash",0,12]',
                }
            ],
        )
        passages.add([rows[0]["chunk_id"]], vector[None, :])
        key = store.save_centroid(source, "v", vector, 224)
        files.add([key], vector[None, :])
        store.complete_source(source, "v")
    passages.publish()
    files.publish()
    for source in ["a", "b"]:
        refresh_neighbours(store, files, source, passage_vectors=passages)
    edge = store.semantic_neighbours("a")[0]
    evidence = edge["evidence"]
    assert evidence["basis"] == "mutual token-weighted file centroids"
    assert evidence["source_passage"]["source_digest"] == "raw-digest-a"
    assert len(evidence["source_passage"]["text_hash"]) == 64
    assert evidence["source_passage"]["locator"] == '["lines:1","text-hash",0,12]'
    assert evidence["source_passage"]["sampled_unique_passages"] == 1
    store.close()


@pytest.mark.parametrize("store_type", [IndexStore, CompactIndexStore])
def test_global_publication_includes_more_than_one_cohort_and_keeps_ids(tmp_path, store_type):
    store = store_type(tmp_path / "state.db", encoder_id="test")
    count = 25_002
    vector = np.zeros(384, dtype="<f4").tobytes()
    with store.db:
        store.db.executemany(
            """INSERT INTO sources(source_id,digest,path,scope,version,active,ready)
            VALUES (?,?,?,'private','v',1,1)""",
            ((f"source-{i:06}", "digest", "fixture") for i in range(count)),
        )
        store.db.executemany(
            "INSERT INTO centroid_keys VALUES (?,?,'v')",
            ((i + 1, f"source-{i:06}") for i in range(count)),
        )
        store.db.executemany(
            "INSERT INTO centroids VALUES (?,'v',?,?,1)",
            ((f"source-{i:06}", i + 1, vector) for i in range(count)),
        )
        store.db.executemany(
            "INSERT INTO semantic_neighbours VALUES (?,?,'v','v',0.91)",
            ((f"source-{i:06}", f"source-{i ^ 1:06}") for i in range(count)),
        )

    def fixture_policy(**kwargs):
        # The disposable fixture is bounded separately from the production E: library quota.
        assert kwargs.get("additional_disk", 0) <= 128 * 1024**2

    first = publish_communities(store, global_dir=tmp_path / "cold", resource_guard=fixture_policy)
    assert first["partial"] is False and first["covered"] == count
    assert first["cohort"] == count and first["global"] is True
    before = dict(store.db.execute("SELECT source_id,community_id FROM communities"))
    with store.db:
        store.set_meta("organisation_global_completed_at", 0)
    second = publish_communities(store, global_dir=tmp_path / "cold", resource_guard=fixture_policy)
    after = dict(store.db.execute("SELECT source_id,community_id FROM communities"))
    assert before == after
    assert second["generation"] != first["generation"]
    assert store.get_meta("organisation_global_lastgood") is not None
    previous = store.get_meta("organisation_global_current")
    from dodo_mem_neural.organisation_global import publish_global

    def interrupted(**kwargs):
        if store.get_meta("organisation_global_state") == "running":
            raise ResourcePressure("test resource pressure")

    with pytest.raises(ResourcePressure, match="test resource pressure"):
        publish_global(store, tmp_path / "cold", resource_guard=interrupted)
    assert store.get_meta("organisation_global_current") == previous
    assert dict(store.db.execute("SELECT source_id,community_id FROM communities")) == before
    store.close()
