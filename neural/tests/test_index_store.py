import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from dodo_mem_neural.index_store import IndexStore


def test_dedup_keeps_permissions_and_rejects_stale_occurrences(tmp_path):
    store = IndexStore(tmp_path / "state.db", encoder_id="test", chunker_id="224:32")
    store.queue_source("a", "digest-a", "a.md", "private", "version-a")
    store.queue_source("b", "digest-b", "b.md", "shared", "version-b")
    first = store.stage_chunks("a", "version-a", [{"text": "same text", "start": 1, "end": 1}])
    second = store.stage_chunks("b", "version-b", [{"text": "same text", "start": 2, "end": 2}])
    assert first[0]["chunk_id"] == second[0]["chunk_id"]
    store.complete_source("a", "version-a")
    store.complete_source("b", "version-b")
    key = first[0]["chunk_id"]
    assert [row["source_id"] for row in store.visible_occurrences([key], ["shared"])] == ["b"]
    store.queue_source("b", "new-digest", "b.md", "shared", "version-new")
    assert store.visible_occurrences([key], ["shared"]) == []
    assert [row["source_id"] for row in store.visible_occurrences([key], ["private"])] == ["a"]
    store.close()


def test_threaded_read_snapshots_are_consistent_read_only_and_do_not_reset_jobs(tmp_path):
    store = IndexStore(tmp_path / "state.db")
    store.queue_source("running", "digest", "r.md", "private", "v")
    store.claim_job()

    def read(_):
        with store.read_snapshot() as snapshot:
            assert (
                snapshot.db.execute("SELECT status FROM jobs WHERE source_id='running'").fetchone()[
                    0
                ]
                == "running"
            )
            first = snapshot.pending_count()
            for _ in range(20):
                assert snapshot.pending_count() == first
                snapshot.topics()
            with pytest.raises(sqlite3.OperationalError, match="readonly"):
                snapshot.set_meta("forbidden", "write")
        return first

    with ThreadPoolExecutor(max_workers=8) as threads:
        futures = [threads.submit(read, number) for number in range(32)]
        for number in range(20):
            store.queue_source(f"new-{number}", "digest", "n.md", "private", "v")
        assert all(1 <= future.result() <= 21 for future in futures)
    assert (
        store.db.execute("SELECT status FROM jobs WHERE source_id='running'").fetchone()[0]
        == "running"
    )
    store.close()


def test_policy_change_and_returning_source_requeue_completed_jobs(tmp_path):
    store = IndexStore(tmp_path / "state.db")
    store.queue_source("a", "digest", "a.md", "private", "same-version")
    store.complete_source("a", "same-version")
    store.queue_source("a", "digest", "a.md", "shared", "same-version")
    assert store.claim_job()["scope"] == "shared"
    store.complete_source("a", "same-version")
    store.invalidate_source("a")
    store.queue_source("a", "digest", "a.md", "shared", "same-version")
    assert store.claim_job()["source_id"] == "a"
    store.close()


def test_lexical_filters_permissions_before_candidate_limit(tmp_path):
    store = IndexStore(tmp_path / "state.db")
    store.queue_source("a", "d-a", "a.md", "private", "v-a")
    store.queue_source("b", "d-b", "b.md", "shared", "v-b")
    store.stage_chunks("a", "v-a", [{"text": "neural neural", "start": 1, "end": 1}])
    store.stage_chunks("b", "v-b", [{"text": "neural shared", "start": 2, "end": 2}])
    store.complete_source("a", "v-a")
    store.complete_source("b", "v-b")
    assert [
        row["source_id"] for row in store.lexical_search('neural OR "', ["shared"], limit=1)
    ] == ["b"]
    store.close()


def test_interrupted_jobs_resume_and_version_change_invalidates(tmp_path):
    path = tmp_path / "state.db"
    store = IndexStore(path, encoder_id="one", chunker_id="224:32")
    store.queue_source("a", "d", "a.md", "private", "v")
    assert store.claim_job()["source_id"] == "a"
    store.close()
    store = IndexStore(path, encoder_id="one", chunker_id="224:32")
    assert store.claim_job()["source_id"] == "a"
    store.close()
    store = IndexStore(path, encoder_id="two", chunker_id="224:32")
    assert store.pending_count() == 1
    assert store.visible_occurrences([1], ["private"]) == []
    store.close()
