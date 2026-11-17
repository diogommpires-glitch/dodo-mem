"""The background worker survives transient startup locks without blocking the broker."""

import json
import sqlite3
import threading
from types import SimpleNamespace

from dodo_mem_neural import service
from dodo_mem_neural.resources import ResourcePressure


def fixture(tmp_path, monkeypatch, startup):
    instance = service.OrganisationService.__new__(service.OrganisationService)
    instance.home = instance.space_home = tmp_path
    instance.space = "fixture"
    instance.runtime = SimpleNamespace(space_version="test", encode=None, tokenizer=None)
    instance.selection = {"semantic_threshold": 0.82}
    instance.background_paused = threading.Event()
    instance.stop_event = threading.Event()
    instance.progress = {}
    instance.activity = SimpleNamespace(active=False)
    instance.catalogue = SimpleNamespace(database=tmp_path / "catalogue.sqlite3")
    instance.authorised = SimpleNamespace(source_scope=None, providers=[])
    instance.status = lambda: dict(instance.progress)

    class Store:
        cache_uuid = "a" * 32

        def __init__(self, *args, **kwargs):
            self.db = sqlite3.connect(":memory:")
            self.db.execute("CREATE TABLE sources(active INTEGER,ready INTEGER,provider TEXT)")
            self.meta = {
                "index_progress": json.dumps({"completed_jobs": 999}),
                "index_progress_epoch": "old-epoch",
            }
            self.closed = False

        def get_meta(self, key, default=None):
            return self.meta.get(key, default)

        def set_meta(self, key, value):
            self.meta[key] = value

        def pending_count(self):
            return self.db.execute("SELECT count(*) FROM sources WHERE ready=0").fetchone()[0]

        def close(self):
            self.closed = True
            self.db.close()

    stores = []

    def open_store(*args, **kwargs):
        store = Store()
        stores.append(store)
        return store

    def worker(*args, **kwargs):
        return startup(instance, args[1])

    monkeypatch.setattr(service, "IndexStore", open_store)
    monkeypatch.setattr(service, "VectorStore", lambda *args, **kwargs: object())
    monkeypatch.setattr(service, "NeuralWorker", worker)
    monkeypatch.setattr(service, "check_resources", lambda **kwargs: {})
    monkeypatch.setattr(service.shutil, "disk_usage", lambda _: SimpleNamespace(free=500 * 1024**3))
    return instance, stores


def stopped_worker(instance, store):
    store.set_meta("index_progress_epoch", "new-epoch")
    store.set_meta("index_progress", json.dumps({"completed_jobs": 0}))

    def count(query):
        instance.stop_event.set()
        return SimpleNamespace(fetchone=lambda: (0,))

    return SimpleNamespace(
        catalogue=SimpleNamespace(execute=count), source_providers=[], close=lambda: None
    )


def test_busy_worker_startup_retries_then_reloads_reset_generation_counters(tmp_path, monkeypatch):
    attempts = 0

    def startup(instance, store):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            error = sqlite3.OperationalError("database is locked")
            error.sqlite_errorcode = sqlite3.SQLITE_BUSY
            raise error
        return stopped_worker(instance, store)

    instance, stores = fixture(tmp_path, monkeypatch, startup)
    instance._index()
    assert attempts == 2
    assert instance.progress["completed_jobs"] == 0
    assert instance.progress["index_progress_epoch"] == "new-epoch"
    assert instance.progress.get("state") != "failed"
    assert stores[0].closed
    assert json.loads((tmp_path / "progress.json").read_text())["completed_jobs"] == 0


def test_non_lock_startup_error_is_reported_without_retry(tmp_path, monkeypatch):
    attempts = 0

    def startup(instance, store):
        nonlocal attempts
        attempts += 1
        error = sqlite3.OperationalError("no such table: synthetic")
        error.sqlite_errorcode = sqlite3.SQLITE_ERROR
        raise error

    instance, stores = fixture(tmp_path, monkeypatch, startup)
    instance._index()
    assert attempts == 1 and instance.progress["state"] == "failed"
    assert "no such table" in instance.progress["reason"]
    assert stores[0].closed


def test_checkpoint_during_startup_lock_stops_retry_without_failure(tmp_path, monkeypatch):
    def startup(instance, store):
        instance.stop_event.set()
        error = sqlite3.OperationalError("database is locked")
        error.sqlite_errorcode = sqlite3.SQLITE_LOCKED
        raise error

    instance, stores = fixture(tmp_path, monkeypatch, startup)
    instance._index()
    assert instance.progress["state"] == "paused"
    assert stores[0].closed


def test_converted_resource_pressure_during_worker_startup_is_retryable(tmp_path, monkeypatch):
    attempts = 0

    def startup(instance, store):
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ResourcePressure("SQLite writer is busy; neural work remains queued for retry.")
        return stopped_worker(instance, store)

    instance, stores = fixture(tmp_path, monkeypatch, startup)
    monkeypatch.setattr(instance.stop_event, "wait", lambda timeout=None: False)
    instance._index()
    assert attempts == 2
    assert instance.progress.get("state") != "failed"
    assert instance.progress["completed_jobs"] == 0
    assert stores[0].closed


def test_restarted_worker_publishes_existing_ready_counts_before_first_batch(tmp_path, monkeypatch):
    def startup(instance, store):
        store.db.executemany(
            "INSERT INTO sources VALUES (?,?,?)",
            [(1, 1, "catalogue"), (1, 0, "catalogue"), (1, 1, "cline")],
        )

        def count(query):
            instance.stop_event.set()
            return SimpleNamespace(fetchone=lambda: (2,))

        return SimpleNamespace(
            catalogue=SimpleNamespace(execute=count), source_providers=[], close=lambda: None
        )

    instance, _ = fixture(tmp_path, monkeypatch, startup)
    instance._index()
    assert instance.progress["completed_files"] == 2
    assert instance.progress["total_eligible_files"] == 3
    assert instance.progress["pending"] == 1
    saved = json.loads((tmp_path / "progress.json").read_text())
    assert saved["completed_files"] == 2 and saved["pending"] == 1
