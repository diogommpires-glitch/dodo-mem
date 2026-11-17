"""Synthetic parameter fixtures, never model weights or inference."""

import copy
import threading
import time
from itertools import pairwise

import pytest
from PySide6.QtCore import Qt, QThreadPool, QTimer
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.neural_client import PROJECTION_DIMS
from dodo_memory_desktop.weight_projection import StaticWeightView, WeightSurface


def descriptor(projection="query"):
    return {
        "block": 0,
        "projection": projection,
        "operator_id": "synthetic-matmul",
        "operation": "MatMul",
        "input_dim": 4,
        "output_dim": 3,
        "total_weight_count": 12,
        "source": {
            "operator_id": "synthetic-input",
            "operation": "LayerNorm",
            "semantics": "synthetic_input_before_projection",
        },
        "target": {
            "operator_id": "synthetic-output",
            "operation": "MatMul",
            "semantics": "query_linear_before_bias",
        },
        "provenance": {
            "prepared_manifest_sha256": "a" * 64,
            "xml_sha256": "b" * 64,
            "bin_sha256": "c" * 64,
            "bin_verified": True,
            "matrix_layout": "output_input",
            "transpose_a": False,
            "transpose_b": True,
            "constant_id": "synthetic-constant",
            "constant_name": "synthetic-weight",
            "offset_bytes": 10,
            "size_bytes": 24,
        },
    }


class FixtureClient:
    def __init__(self):
        self.calls = []

    def activity_projections(self):
        self.calls.append("descriptors")
        return {
            "schema": 1,
            "static": True,
            "model_revision": "synthetic-revision",
            "model_space": "synthetic-space",
            "dtype": "f16",
            "projections": [descriptor(), descriptor("key")],
        }

    def activity_weights(self, block, projection, limit=512, output_index=None):
        self.calls.append((block, projection, limit, output_index))
        return {
            **copy.deepcopy(descriptor(projection)),
            "schema": 1,
            "static": True,
            "model_revision": "synthetic-revision",
            "model_space": "synthetic-space",
            "dtype": "f16",
            "edges": [
                {"source": 1, "target": 2, "weight": -0.5},
                {"source": 3, "target": 0, "weight": 0.25},
            ],
            "shown_count": 2,
            "output_index": output_index,
            "sampling": "top_absolute_incoming_weight"
            if output_index is not None
            else "top_absolute_weight",
        }


class MultiBlockClient(FixtureClient):
    def activity_projections(self):
        value = super().activity_projections()
        value["projections"] = [
            {**descriptor(projection), "block": block}
            for block in range(12)
            for projection in ("query", "key")
        ]
        return value

    def activity_weights(self, block, projection, **kwargs):
        return {**super().activity_weights(block, projection, **kwargs), "block": block}


def settle(predicate, seconds=2):
    deadline = time.monotonic() + seconds
    while not predicate() and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert predicate()


def test_static_weights_keep_exact_edges_signs_and_operator_provenance():
    client = FixtureClient()
    view = StaticWeightView(client)
    assert client.calls == []
    view.resize(1100, 700)
    view.show()
    settle(lambda: view.surface.payload is not None)
    assert client.calls == ["descriptors", (0, "query", 512, None)]
    assert len(view.surface.paths) == 2
    assert len(view.surface.inputs) == 4 and len(view.surface.outputs) == 3
    assert view.surface.payload["edges"][0] == {"source": 1, "target": 2, "weight": -0.5}
    view.items.setCurrentRow(0)
    assert "-0.5" in view.inspector.text() and "Input 1" in view.inspector.text()
    assert "synthetic-matmul" in view.inspector.text()
    assert "before_bias" in view.inspector.text()
    assert "2 / 12" in view.status.text()
    assert "Static weights" in view.legend.text() and "square-root" in view.legend.text()
    zoom = view.surface.zoom
    QTest.keyClick(view.surface, Qt.Key.Key_Plus)
    assert view.surface.zoom > zoom
    assert not view.grab().isNull()
    view.close()


def test_requested_block_is_retained_while_initial_metadata_is_loading():
    started, release = threading.Event(), threading.Event()
    client = MultiBlockClient()
    original = client.activity_projections

    def delayed():
        started.set()
        assert release.wait(2)
        return original()

    client.activity_projections = delayed
    view = StaticWeightView(client)
    view.select_block(3)
    assert client.calls == []
    view.show()
    settle(started.is_set)
    view.select_block(7)
    release.set()
    settle(lambda: view.surface.payload is not None)
    assert view.block.currentData() == view.surface.payload["block"] == 7
    assert client.calls == ["descriptors", (7, "query", 512, None)]
    view.close()


def test_fast_block_change_discards_old_result_and_keeps_projection_when_hidden():
    started, release = threading.Event(), threading.Event()
    client = MultiBlockClient()
    original = client.activity_weights

    def delayed(block, projection, **kwargs):
        if block == 3:
            started.set()
            assert release.wait(2)
        return original(block, projection, **kwargs)

    client.activity_weights = delayed
    view = StaticWeightView(client)
    view.show()
    settle(lambda: view.surface.payload is not None)
    view.projection.setCurrentIndex(view.projection.findData("key"))
    settle(lambda: not view.running)
    view.select_block(3)
    settle(started.is_set)
    view.select_block(9)
    assert view.surface.payload is None and view.items.count() == 0
    release.set()
    settle(lambda: view.surface.payload is not None and not view.running)
    assert view.surface.payload["block"] == 9
    assert view.surface.payload["projection"] == "key"
    view.hide()
    count = len(client.calls)
    view.select_block(11)
    QApplication.processEvents()
    assert len(client.calls) == count
    assert view.surface.payload is None
    view.show()
    settle(lambda: view.surface.payload is not None)
    assert view.surface.payload["block"] == 11
    assert view.surface.payload["projection"] == "key"
    assert "Selected connection" in view.inspector.text()
    assert "not an activation" in view.inspector.text()
    assert "not measured activity" in view.legend.text()
    view.close()


@pytest.mark.parametrize("block", [-1, 12, 3.0, "3", True, None])
def test_block_handoff_rejects_invalid_indices_without_requests(block):
    client = FixtureClient()
    view = StaticWeightView(client)
    with pytest.raises(ValueError, match=r"integer in \[0, 11\]"):
        view.select_block(block)
    assert client.calls == []
    view.close()


def test_missing_block_does_not_fall_back_to_the_previous_selection():
    client = FixtureClient()
    view = StaticWeightView(client)
    view.show()
    settle(lambda: view.surface.payload is not None)
    count = len(client.calls)
    view.select_block(7)
    assert view.block.currentIndex() == -1 and view.surface.payload is None
    assert "Block 8 is absent" in view.status.text()
    view.hide()
    view.show()
    settle(lambda: not view.running)
    assert client.calls[count:] == ["descriptors"] and view.surface.payload is None
    view.close()


def test_neuron_columns_keep_all_feature_ids_and_reveal_detail_without_grid_remapping():
    payload = FixtureClient().activity_weights(0, "query")
    payload.update(input_dim=1536, output_dim=384)
    surface = WeightSurface()
    surface.resize(900, 650)
    surface.set_payload(payload)
    assert len(surface.inputs) == 1536 and len(surface.outputs) == 384
    for points in (surface.inputs, surface.outputs):
        assert max(p.x() for p in points) - min(p.x() for p in points) < 45
        assert points[-1].y() - points[0].y() == 850
        assert all(a.y() < b.y() for a, b in pairwise(points))
    path = surface.paths[0]
    assert path.pointAtPercent(0) == surface.inputs[1]
    assert path.pointAtPercent(1) == surface.outputs[2]
    overview = surface.ring_indices("input")
    assert 16 <= len(overview) <= 64
    surface.select("input", 129)
    assert 129 in surface.ring_indices("input")
    surface.zoom_by(64)
    assert len(surface.ring_indices("input")) > len(overview)
    surface.zoom_by(64)
    assert len(surface.ring_indices("input")) == 1536
    surface.close()


def test_projection_requests_coalesce_and_do_not_run_hidden():
    started, release = threading.Event(), threading.Event()
    client = FixtureClient()
    original = client.activity_weights

    def blocked(*args, **kwargs):
        if not started.is_set():
            started.set()
            assert release.wait(2)
        return original(*args, **kwargs)

    client.activity_weights = blocked
    view = StaticWeightView(client)
    view.show()
    settle(started.is_set)
    view.output.setValue(0)
    view.output.setValue(1)
    view.output.setValue(2)
    release.set()
    settle(lambda: view.surface.payload is not None and not view.running)
    weights = [call for call in client.calls if isinstance(call, tuple)]
    assert weights == [(0, "query", 512, None), (0, "query", 512, 2)]
    view.hide()
    view.output.setValue(1)
    QApplication.processEvents()
    assert len(client.calls) == 3
    view.close()


def test_resource_failure_is_visible_and_does_not_retry():
    client = FixtureClient()

    def unavailable(*_args, **_kwargs):
        client.calls.append("unavailable")
        raise OSError("503: less than 5 GiB available RAM")

    client.activity_weights = unavailable
    view = StaticWeightView(client)
    view.show()
    settle(lambda: "503" in view.status.text())
    QTest.qWait(40)
    QApplication.processEvents()
    assert client.calls == ["descriptors", "unavailable"]
    assert view.surface.payload is None
    view.close()


def test_repeated_visible_activation_does_not_retry_until_explicit_reload():
    client = FixtureClient()

    def unavailable(*_args, **_kwargs):
        client.calls.append("unavailable")
        raise OSError("503: less than 5 GiB available RAM")

    client.activity_weights = unavailable
    view = StaticWeightView(client)
    view.show()
    settle(lambda: "503" in view.status.text())
    for _ in range(5):
        view.set_active(True)
        QApplication.processEvents()
    assert client.calls == ["descriptors", "unavailable"]
    view._reload()
    settle(lambda: not view.running and len(client.calls) == 4)
    assert client.calls == ["descriptors", "unavailable", "descriptors", "unavailable"]
    view.close()


def test_failed_new_selection_cannot_keep_the_previous_projection_visible():
    client = FixtureClient()
    original = client.activity_weights

    def reject_key(block, projection, **kwargs):
        if projection == "key":
            client.calls.append("key-unavailable")
            raise OSError("503: less than 5 GiB available RAM")
        return original(block, projection, **kwargs)

    client.activity_weights = reject_key
    view = StaticWeightView(client)
    view.show()
    settle(lambda: view.surface.payload is not None)
    assert view.items.count() == 2 and "synthetic-matmul" in view.inspector.text()
    view.projection.setCurrentIndex(view.projection.findData("key"))
    assert view.surface.payload is None and view.items.count() == 0
    settle(lambda: "503" in view.status.text())
    assert view.surface.payload is None
    assert not view.surface.paths and not view.surface.inputs and not view.surface.outputs
    assert view.items.count() == 0 and "synthetic-matmul" not in view.inspector.text()
    assert client.calls == ["descriptors", (0, "query", 512, None), "key-unavailable"]
    view.close()


def test_weights_from_a_changed_model_are_not_rendered_or_automatically_retried():
    client = FixtureClient()
    original = client.activity_weights

    def changed(*args, **kwargs):
        value = original(*args, **kwargs)
        value["model_space"] = "different-synthetic-space"
        return value

    client.activity_weights = changed
    view = StaticWeightView(client)
    view.show()
    settle(lambda: "model changed" in view.status.text())
    assert view.surface.payload is None and view.items.count() == 0
    assert client.calls == ["descriptors", (0, "query", 512, None)]
    view.close()


def test_shutdown_drains_owned_request_without_waiting_for_unrelated_work():
    started, release, unrelated = threading.Event(), threading.Event(), threading.Event()
    client = FixtureClient()
    original = client.activity_weights

    def blocked(*args, **kwargs):
        started.set()
        assert release.wait(2)
        return original(*args, **kwargs)

    client.activity_weights = blocked
    view = StaticWeightView(client)
    view.show()
    settle(started.is_set)
    QThreadPool.globalInstance().start(lambda: unrelated.wait(2))
    QTimer.singleShot(25, release.set)
    try:
        assert view.shutdown(timeout=1)
        assert not view.running and not view.pending and not view._tokens
        assert QThreadPool.globalInstance().activeThreadCount() >= 1
        assert view.surface.payload is None
        view.close()
    finally:
        release.set()
        unrelated.set()


def test_missing_client_is_a_quiet_unavailable_view():
    view = StaticWeightView(None)
    view.show()
    QApplication.processEvents()
    assert "unavailable" in view.status.text().lower()
    assert not view.running and view.shutdown()
    view.close()


class SnapshotClient:
    def __init__(self):
        self.calls = []
        self.space = "synthetic-space"
        self.transform = lambda value: value
        self.packet = {
            "schema": 1,
            "static": True,
            "dtype": "f16",
            "model_revision": "synthetic-revision",
            "model_space": self.space,
            "total_weight_count": 21_233_664,
            "shown_count": 18_432,
            "projections": [],
        }
        for block in range(12):
            for name, (inputs, outputs) in PROJECTION_DIMS.items():
                row = descriptor(name)
                row.update(
                    block=block,
                    input_dim=inputs,
                    output_dim=outputs,
                    total_weight_count=inputs * outputs,
                    orientation="output_input",
                    model_revision="synthetic-revision",
                    model_space=self.space,
                    schema=1,
                    static=True,
                    dtype="f16",
                    shown_count=256,
                    output_index=None,
                    sampling="stratified_by_output",
                    edges=[{"source": i, "target": i, "weight": 0.1} for i in range(256)],
                )
                row["provenance"]["size_bytes"] = inputs * outputs * 2
                self.packet["projections"].append(row)

    def activity_projections(self):
        self.calls.append("descriptors")
        return self.transform(
            {
                **copy.deepcopy(self.packet),
                "model_space": self.space,
            }
        )

    def activity_weights(self, block, projection, *, limit, output_index=None):
        self.calls.append((block, projection, limit, output_index))
        raise OSError("503: less than 5 GiB available RAM")


def test_verified_snapshot_drills_into_actual_blocks_without_new_binary_reads():
    client = SnapshotClient()
    view = StaticWeightView(client)
    view.set_parameters(client.packet)
    view.select_block(11)
    assert client.calls == []
    view.show()
    settle(lambda: view.surface.payload is not None)
    assert client.calls == ["descriptors"]
    assert view.surface.payload["block"] == 11 and view.items.count() == 256
    assert "Verified snapshot" in view.status.text() and "retained stratified" in view.status.text()
    view.projection.setCurrentIndex(view.projection.findData("ff_up"))
    view.select_block(2)
    assert view.surface.payload["block"] == 2
    assert view.surface.payload["projection"] == "ff_up"
    assert len(view.surface.outputs) == 1536 and client.calls == ["descriptors"]
    view.hide()
    view.show()
    settle(lambda: view.surface.payload is not None)
    assert client.calls == ["descriptors", "descriptors"]
    assert view.surface.payload["block"] == 2
    view.close()


@pytest.mark.parametrize("fault", ["namespace", "hash", "coordinate"])
def test_snapshot_mismatch_is_cleared_instead_of_rendered_as_current(fault):
    client = SnapshotClient()
    if fault == "namespace":
        client.space = "new-owner-space"
    elif fault == "coordinate":
        client.packet["projections"][0]["edges"][0]["source"] = 384
    else:

        def changed(value):
            value["projections"][0]["provenance"]["bin_sha256"] = "d" * 64
            return value

        client.transform = changed
    view = StaticWeightView(client)
    view.set_parameters(client.packet)
    view.show()
    settle(lambda: "503" in view.status.text())
    assert view.surface.payload is None and view._parameters is None
    assert client.calls == ["descriptors", (0, "query", 512, None)]
    view.close()


def test_strongest_output_uses_live_request_and_cannot_relabel_the_stratified_snapshot():
    client = SnapshotClient()
    started, release = threading.Event(), threading.Event()

    def blocked(block, projection, *, limit, output_index=None):
        client.calls.append((block, projection, limit, output_index))
        started.set()
        assert release.wait(2)
        raise OSError("503: less than 5 GiB available RAM")

    client.activity_weights = blocked
    view = StaticWeightView(client)
    view.set_parameters(client.packet)
    view.show()
    settle(lambda: view.surface.payload is not None)
    view.output.setValue(7)
    settle(started.is_set)
    assert view.surface.payload is None
    assert client.calls[-1] == (0, "query", 512, 7)
    view.output.setValue(-1)
    assert view.surface.payload["output_index"] is None
    assert view.surface.payload["sampling"] == "stratified_by_output"
    release.set()
    settle(lambda: not view.running)
    assert "Verified snapshot" in view.status.text()
    assert len(client.calls) == 2 and view.surface.payload is not None
    view.close()


def test_revoking_the_packet_while_metadata_loads_cannot_restore_the_old_snapshot():
    client = SnapshotClient()
    started, release = threading.Event(), threading.Event()
    original = client.activity_projections

    def delayed():
        if not started.is_set():
            started.set()
            assert release.wait(2)
        return original()

    client.activity_projections = delayed
    view = StaticWeightView(client)
    view.set_parameters(client.packet)
    view.show()
    settle(started.is_set)
    view.set_parameters(None)
    release.set()
    settle(lambda: "503" in view.status.text())
    assert client.calls == ["descriptors", "descriptors", (0, "query", 512, None)]
    assert view.surface.payload is None and view._parameters is None
    view.close()
