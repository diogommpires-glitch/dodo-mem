"""Controller fixtures contain synthetic parameters and never run a model."""

import copy
import hashlib
import json
import threading
import time
from types import SimpleNamespace

import pytest
from PySide6.QtCore import QCoreApplication, QEvent, QThreadPool, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.network_parameters import NetworkParameters
from dodo_memory_desktop.neural_client import PROJECTION_DIMS


def descriptor(block, projection):
    inputs, outputs = PROJECTION_DIMS[projection]
    return {
        "block": block,
        "projection": projection,
        "input_dim": inputs,
        "output_dim": outputs,
        "total_weight_count": inputs * outputs,
        "source": {
            "operator_id": f"synthetic-source-{block}-{projection}",
            "operation": "LayerNorm",
            "semantics": "synthetic_features",
        },
        "target": {
            "operator_id": f"synthetic-target-{block}-{projection}",
            "operation": "MatMul",
            "semantics": "synthetic_linear_output",
        },
        "operator_id": f"synthetic-projection-{block}-{projection}",
        "operation": "MatMul",
        "orientation": "output_input",
        "provenance": {
            "prepared_manifest_sha256": "a" * 64,
            "xml_sha256": "b" * 64,
            "bin_sha256": "c" * 64,
            "bin_verified": True,
            "constant_id": f"synthetic-weight-{block}-{projection}",
            "offset_bytes": 0,
            "size_bytes": inputs * outputs * 2,
            "matrix_layout": "output_input",
        },
    }


class Client:
    def __init__(self):
        self.space = "synthetic-space"
        self.calls = []
        self.before_weights = lambda _block, _projection: None
        self.transform = lambda value: value

    def envelope(self):
        return {
            "model_revision": "synthetic-revision",
            "model_space": self.space,
            "schema": 1,
            "static": True,
            "dtype": "f16",
        }

    def activity_projections(self):
        self.calls.append("descriptors")
        return self.transform(
            {
                **self.envelope(),
                "projections": [
                    descriptor(block, name) for block in range(12) for name in PROJECTION_DIMS
                ],
            }
        )

    def activity_weights(self, block, projection, *, limit):
        self.calls.append((block, projection, limit))
        self.before_weights(block, projection)
        return self.transform(
            {
                **self.envelope(),
                **descriptor(block, projection),
                "edges": [{"source": i, "target": i, "weight": 0.1} for i in range(limit)],
                "shown_count": limit,
                "output_index": None,
                "sampling": "stratified_by_output",
            }
        )


def test_default_visual_snapshot_uses_owner_cache_copy_when_legacy_drive_is_missing(
    tmp_path, monkeypatch
):
    from dodo_memory_desktop import network_parameters

    home = tmp_path / "owner-state"
    bulk = home / "bulk"
    bulk.mkdir(parents=True)
    (home / "cache-root.json").write_text(json.dumps({"root": str(bulk)}))
    legacy = tmp_path / "offline-legacy-drive"
    monkeypatch.setattr(network_parameters, "NEURAL_CACHE_ROOT", legacy)
    disk_reads = []

    def disk_usage(drive):
        disk_reads.append(drive)
        return SimpleNamespace(free=200 * 1024**3)

    monkeypatch.setattr(network_parameters.shutil, "disk_usage", disk_usage)
    client = Client()
    client.home = home
    loader = NetworkParameters(client)
    ready = []
    loader.ready.connect(ready.append)
    try:
        assert loader.cache_path == bulk / "visualisation/whole-network-cache.json"
        loader.set_model({"revision": "synthetic-revision", "space": client.space})
        loader.set_active(True)
        settle(lambda: bool(ready) and not loader.running)
        assert loader.cache_path.is_file() and not legacy.exists()
        assert disk_reads == [bulk.anchor]
        client.calls.clear()
        loader.set_active(False)
        loader.set_active(True)
        settle(lambda: len(ready) == 2)
        assert ready[-1]["from_snapshot"] and client.calls == ["descriptors"]
    finally:
        assert loader.shutdown()
        loader.deleteLater()


@pytest.mark.parametrize("raw", ['{"root":".."}', "[]", '{"root":false}', "x" * 4097])
def test_invalid_owner_cache_choice_reports_failure_without_legacy_fallback(tmp_path, raw):
    home = tmp_path / "owner-state"
    home.mkdir()
    (home / "cache-root.json").write_text(raw)
    client = Client()
    client.home = home
    loader = NetworkParameters(client)
    errors = []
    loader.failed.connect(errors.append)
    try:
        loader.set_model({"revision": "synthetic-revision", "space": client.space})
        loader.set_active(True)
        assert errors and "cache" in errors[-1].lower()
        assert not client.calls
    finally:
        assert loader.shutdown()
        loader.deleteLater()


def test_redirected_owner_cache_is_rejected_before_snapshot_reads(tmp_path):
    home = tmp_path / "owner-state"
    home.mkdir()
    elsewhere = tmp_path / "redirect-target"
    elsewhere.mkdir()
    bulk = home / "bulk"
    try:
        bulk.symlink_to(elsewhere, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"Creating a directory symlink is unavailable: {error.winerror}")
    (home / "cache-root.json").write_text(json.dumps({"root": str(bulk)}))
    client = Client()
    client.home = home
    loader = NetworkParameters(client)
    errors = []
    loader.failed.connect(errors.append)
    try:
        loader.set_model({"revision": "synthetic-revision", "space": client.space})
        loader.set_active(True)
        assert errors and "redirect" in errors[-1].lower()
        assert not client.calls and not list(elsewhere.iterdir())
    finally:
        assert loader.shutdown()
        loader.deleteLater()


def test_visual_snapshot_quota_counts_other_files_in_selected_bulk(tmp_path, monkeypatch):
    from dodo_memory_desktop import network_parameters

    home = tmp_path / "owner-state"
    bulk = home / "bulk"
    bulk.mkdir(parents=True)
    (home / "cache-root.json").write_text(json.dumps({"root": str(bulk)}))
    (bulk / "synthetic-weights.bin").write_bytes(b"x" * 512)
    monkeypatch.setattr(network_parameters, "MAX_NEURAL_BYTES", 600)
    monkeypatch.setattr(
        network_parameters.shutil, "disk_usage", lambda _: SimpleNamespace(free=200 * 1024**3)
    )
    client = Client()
    client.home = home
    loader = NetworkParameters(client)
    try:
        assert loader.cache_path == bulk / "visualisation/whole-network-cache.json"
        with pytest.raises(OSError, match="quota"):
            loader._write_snapshot({"synthetic": "x" * 200})
        assert not (bulk / "visualisation/whole-network-cache.json").exists()
    finally:
        assert loader.shutdown()
        loader.deleteLater()


def settle(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(2)
    assert predicate()


@pytest.fixture
def setup(tmp_path, monkeypatch):
    from dodo_memory_desktop import network_parameters

    monkeypatch.setattr(
        network_parameters.shutil, "disk_usage", lambda _drive: SimpleNamespace(free=200 * 1024**3)
    )
    client = Client()
    loader = NetworkParameters(client, cache_path=tmp_path / "network-cache.json")
    loader.set_model({"revision": "synthetic-revision", "space": client.space})
    ready, errors, progress = [], [], []
    loader.ready.connect(ready.append)
    loader.failed.connect(errors.append)
    loader.progress.connect(lambda done, total: progress.append((done, total)))
    yield client, loader, ready, errors, progress
    assert loader.shutdown(timeout=2)
    loader.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    QApplication.processEvents()


def test_complete_sample_is_coherent_and_cached_without_repeated_activation_reads(setup):
    client, loader, ready, errors, progress = setup
    assert client.calls == []
    loader.set_active(True)
    settle(lambda: bool(ready))
    packet = ready[0]
    assert packet["total_weight_count"] == 21_233_664
    assert packet["shown_count"] == 18_432 and len(packet["projections"]) == 72
    assert len(client.calls) == 73 and not errors
    assert progress == [(i, 72) for i in range(73)]
    assert {(p["block"], p["projection"]) for p in packet["projections"]} == {
        (block, name) for block in range(12) for name in PROJECTION_DIMS
    }
    loader.set_model({"revision": "synthetic-revision", "space": client.space})
    loader.set_active(True)
    assert len(ready) == 1
    loader.set_active(False)
    loader.set_active(True)
    settle(lambda: len(ready) == 2)
    assert ready[1] == {**packet, "from_snapshot": True}
    assert len(client.calls) == 74 and client.calls[-1] == "descriptors"


def test_pause_stops_future_calls_and_resume_keeps_validated_partial_work(setup):
    client, loader, ready, errors, _progress = setup
    started, release = threading.Event(), threading.Event()

    def wait(_block, _projection):
        if not started.is_set():
            started.set()
            assert release.wait(2)

    client.before_weights = wait
    loader.set_active(True)
    settle(started.is_set)
    loader.set_active(False)
    release.set()
    settle(lambda: not loader.running)
    assert len(client.calls) == 2 and not ready and not errors
    QTest.qWait(20)
    assert len(client.calls) == 2
    loader.set_active(True)
    settle(lambda: bool(ready))
    assert len(client.calls) == 73
    assert client.calls.count((0, "query", 256)) == 1


def test_model_change_during_a_read_discards_old_partial_generation(setup):
    client, loader, ready, errors, _progress = setup
    started, release = threading.Event(), threading.Event()

    def wait(_block, _projection):
        if not started.is_set():
            started.set()
            assert release.wait(2)

    client.before_weights = wait
    loader.set_active(True)
    settle(started.is_set)
    client.space = "new-synthetic-space"
    loader.set_model({"revision": "synthetic-revision", "space": client.space})
    assert len(client.calls) == 2
    release.set()
    settle(lambda: bool(ready))
    assert len(ready) == 1 and not errors
    assert ready[0]["model_space"] == client.space
    assert all(row["model_space"] == client.space for row in ready[0]["projections"])
    assert client.calls.count("descriptors") == 2 and len(client.calls) == 75


def test_quick_view_toggle_waits_for_the_same_inflight_request(setup):
    client, loader, ready, errors, _progress = setup
    started, release = threading.Event(), threading.Event()

    def wait(_block, _projection):
        if not started.is_set():
            started.set()
            assert release.wait(2)

    client.before_weights = wait
    loader.set_active(True)
    settle(started.is_set)
    loader.set_active(False)
    loader.set_active(True)
    QTest.qWait(20)
    assert len(client.calls) == 2
    release.set()
    settle(lambda: bool(ready))
    assert len(client.calls) == 73 and not errors


@pytest.mark.parametrize(
    "fault", ["identity", "duplicate", "missing", "projection", "artifact", "short"]
)
def test_invalid_network_sample_fails_closed_and_needs_explicit_reload(setup, fault):
    client, loader, ready, errors, _progress = setup

    def corrupt(value):
        if fault == "identity":
            if value.get("projection") == "key":
                value["model_space"] = "wrong-space"
        elif "projections" in value:
            if fault == "duplicate":
                value["projections"][-1] = copy.deepcopy(value["projections"][0])
            elif fault == "missing":
                value["projections"].pop()
        elif fault == "projection":
            value["projection"] = "key"
        elif fault == "artifact":
            value["provenance"]["bin_sha256"] = "d" * 64
        elif fault == "short":
            value["edges"].pop()
            value["shown_count"] -= 1
        return value

    client.transform = corrupt
    loader.set_active(True)
    settle(lambda: bool(errors))
    assert not ready
    count = len(client.calls)
    loader.set_active(False)
    loader.set_active(True)
    QTest.qWait(20)
    assert len(client.calls) == count
    client.transform = lambda value: value
    loader.reload()
    settle(lambda: bool(ready))
    assert ready[0]["shown_count"] == 18_432


def test_shutdown_waits_only_for_owned_request_and_does_not_start_the_next_matrix(setup):
    client, loader, ready, _errors, _progress = setup
    started, release, unrelated = threading.Event(), threading.Event(), threading.Event()

    def wait(_block, _projection):
        started.set()
        assert release.wait(2)

    client.before_weights = wait
    loader.set_active(True)
    settle(started.is_set)
    QThreadPool.globalInstance().start(lambda: unrelated.wait(2))
    try:
        assert loader.shutdown(timeout=0.01) is False
        QTimer.singleShot(20, release.set)
        assert loader.shutdown(timeout=1)
        assert len(client.calls) == 2 and not ready
        assert QThreadPool.globalInstance().activeThreadCount() >= 1
    finally:
        release.set()
        unrelated.set()


def test_deleted_controller_does_not_leave_a_worker_emitting_on_a_deleted_signal(tmp_path):
    client = Client()
    started, release = threading.Event(), threading.Event()

    def wait(_block, _projection):
        started.set()
        assert release.wait(2)

    client.before_weights = wait
    loader = NetworkParameters(client, cache_path=tmp_path / "network-cache.json")
    loader.set_model({"revision": "synthetic-revision", "space": client.space})
    loader.set_active(True)
    settle(started.is_set)
    reads = loader.reads
    loader.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not reads.pending
    release.set()
    assert QThreadPool.globalInstance().waitForDone(1000)
    QApplication.processEvents()
    assert len(client.calls) == 2


def test_current_owner_metadata_admits_a_verified_snapshot_without_weight_reads(setup):
    client, loader, ready, errors, _progress = setup
    loader.set_active(True)
    settle(lambda: bool(ready) and not loader.running)
    envelope = json.loads(loader.cache_path.read_text())
    packet = envelope["packet"]
    canonical = json.dumps(
        packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
    ).encode()
    assert envelope["sha256"] == hashlib.sha256(canonical).hexdigest()
    assert loader.cache_path.stat().st_size < 4 * 1024**2
    assert not packet.get("from_snapshot")

    def no_binary_read(_block, _projection):
        raise OSError("Less than 5 GiB RAM: binary reads remain blocked.")

    client.calls.clear()
    client.before_weights = no_binary_read
    restored = NetworkParameters(client, cache_path=loader.cache_path)
    restored_ready = []
    restored.ready.connect(restored_ready.append)
    try:
        restored.set_model({"revision": "synthetic-revision", "space": client.space})
        restored.set_active(True)
        settle(lambda: bool(restored_ready))
        assert client.calls == ["descriptors"]
        assert restored_ready[0] == {**packet, "from_snapshot": True}
        assert not errors
    finally:
        assert restored.shutdown(timeout=2)
        restored.deleteLater()


@pytest.mark.parametrize(
    "fault", ["checksum", "identity", "constant", "duplicate", "coordinate", "nan", "oversized"]
)
def test_bad_snapshots_are_not_rendered_and_fall_back_to_authenticated_reads(setup, fault):
    client, loader, ready, errors, _progress = setup
    loader.set_active(True)
    settle(lambda: bool(ready) and not loader.running)
    envelope = json.loads(loader.cache_path.read_text())
    row = envelope["packet"]["projections"][0]
    if fault == "identity":
        envelope["packet"]["model_space"] = "stale-space"
    elif fault == "constant":
        row["provenance"]["constant_id"] = "another-constant"
    elif fault == "duplicate":
        row["edges"][1] = copy.deepcopy(row["edges"][0])
    elif fault == "coordinate":
        row["edges"][0]["source"] = row["input_dim"]
    elif fault == "nan":
        row["edges"][0]["weight"] = float("nan")
    canonical = json.dumps(
        envelope["packet"], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    envelope["sha256"] = "0" * 64 if fault == "checksum" else hashlib.sha256(canonical).hexdigest()
    if fault == "oversized":
        loader.cache_path.write_bytes(b" " * (4 * 1024**2 + 1))
    else:
        loader.cache_path.write_text(json.dumps(envelope))
    client.calls.clear()
    ready.clear()
    loader.reload()
    settle(lambda: bool(ready) and not loader.running)
    assert not ready[0].get("from_snapshot") and not errors
    assert len(client.calls) == 73
    assert ready[0]["model_space"] == client.space
    assert all(
        row["provenance"]["constant_id"] != "another-constant" for row in ready[0]["projections"]
    )


@pytest.mark.parametrize("fault", ["replace", "disk", "quota"])
def test_failed_atomic_snapshot_write_preserves_last_good_and_the_memory_view(
    setup, monkeypatch, fault
):
    from dodo_memory_desktop import network_parameters

    client, loader, ready, errors, _progress = setup
    loader.set_active(True)
    settle(lambda: bool(ready) and not loader.running)
    original = loader.cache_path.read_bytes()
    warnings = []
    loader.cache_warning.connect(warnings.append)

    def locked(_source, _target):
        raise OSError("Synthetic snapshot file lock")

    if fault == "replace":
        monkeypatch.setattr(network_parameters.os, "replace", locked)
    elif fault == "disk":
        monkeypatch.setattr(
            network_parameters.shutil,
            "disk_usage",
            lambda _drive: SimpleNamespace(free=100 * 1024**3 + len(original) - 1),
        )
    else:
        monkeypatch.setattr(network_parameters, "MAX_NEURAL_BYTES", 2 * len(original) - 1)
    client.space = "new-space-with-valid-parameters"
    loader.set_model({"revision": "synthetic-revision", "space": client.space})
    settle(lambda: len(ready) == 2 and bool(warnings) and not loader.running)
    assert ready[-1]["model_space"] == client.space and not errors
    assert loader.cache_path.read_bytes() == original
    assert "not saved" in warnings[0]
    assert not list(loader.cache_path.parent.glob(".network-cache-*"))


def test_in_memory_snapshot_is_rechecked_against_current_owner_provenance(setup):
    client, loader, ready, errors, _progress = setup
    loader.set_active(True)
    settle(lambda: bool(ready) and not loader.running)

    def changed(value):
        if "projections" in value:
            value["projections"][0]["provenance"]["constant_id"] = "new-owner-constant"
        return value

    def blocked(_block, _projection):
        raise OSError("Binary reads blocked by resource gate")

    client.transform = changed
    client.before_weights = blocked
    ready.clear()
    client.calls.clear()
    loader.set_active(False)
    loader.set_active(True)
    settle(lambda: bool(errors))
    assert not ready and client.calls[0] == "descriptors"
    assert "resource gate" in errors[0]
