import json
import threading
import time

import httpx
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop.neural_client import NeuralClient, NeuralError


def architecture():
    return {
        "schema": 1,
        "model": {"name": "e5", "revision": "fixture", "space": "fixture", "device": ["NPU"]},
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "intermediate_size": 1536,
        "num_attention_heads": 12,
        "head_dim": 32,
        "max_tokens": 256,
        "capture_level": "full",
        "qualification_reason": None,
    }


def capture():
    return {
        "id": "fixture-1",
        "epoch": "fixture",
        "model_space": "fixture",
        "model_revision": "fixture",
        "device": ["NPU"],
        "level": "full",
        "started_at": 1.0,
        "completed_at": 1.1,
        "duration_ms": 100.0,
        "token_index": 0,
        "token_count": 3,
        "token_is_padding": False,
        "sampled": False,
        "source": "interactive",
        "tensors": {
            "embedding": [-0.5, 0.0, 0.5] + [0.0] * 381,
            **{f"block.{index}": [0.1] * 384 for index in range(12)},
            "heads.0": [0.2] * 384,
            "ff.0": [0.3] * 1536,
            "pooled": [0.4] * 384,
            "output": [0.05] * 384,
        },
        "unavailable": [],
    }


def client_for(tmp_path, receive):
    (tmp_path / "broker.token").write_text("x" * 48, encoding="ascii")
    return NeuralClient(tmp_path, transport=httpx.MockTransport(receive))


def test_activity_client_reads_real_signed_values_and_only_posts_capture_controls(tmp_path):
    requests = []

    def receive(request):
        requests.append(request)
        values = {
            "/activity/architecture": architecture(),
            "/activity/activations/fixture-1": capture(),
            "/activity/control": {"visible": True, "paused": False},
        }
        return httpx.Response(200, json=values[request.url.path])

    client = client_for(tmp_path, receive)
    assert callable(getattr(client, "activity_architecture", None))
    assert client.activity_architecture()["num_hidden_layers"] == 12
    assert client.activity_capture("fixture-1")["tensors"]["embedding"][:3] == [-0.5, 0, 0.5]
    client.activity_control(visible=True, paused=False, token_index=7)
    assert [request.method for request in requests] == ["GET", "GET", "POST"]
    assert requests[-1].read() == b'{"visible":true,"paused":false,"token_index":7}'
    assert all(request.headers["Authorization"] == "Bearer " + "x" * 48 for request in requests)


def test_static_architecture_keeps_paused_encoder_explicit_without_measured_values(tmp_path):
    from dodo_memory_desktop.neural_activity import NeuralActivity

    value = architecture()
    value["model"]["device"] = []
    value["capture_level"] = "final"
    value["runtime_available"] = False
    value["runtime_reason"] = "Less than 100 GiB of disk headroom."
    value["qualification_reason"] = "Static metadata only. No live activations."
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=value))
    view = NeuralActivity(None)
    view.set_architecture(client.activity_architecture())
    view.active = True
    view._snapshot(
        {
            "epoch": "fixture",
            "sequence": 0,
            "running": [],
            "latest_capture_id": None,
            "reason": value["runtime_reason"],
        }
    )
    assert view.model_status.text().startswith("Encoder paused · static parameters")
    assert view.running_status.text().startswith("Encoder paused · static parameters")
    assert view.overview.status_text().startswith("Encoder paused · static parameters")
    assert value["runtime_reason"] in view.capture_status.text()
    assert "No active encoder" in view.provenance.text()
    assert view.capture is None and not view.overview.measured_tensors()
    assert not view.surface.colours and not view.replay.isEnabled()
    view.close()


@pytest.mark.parametrize("field,value", [("runtime_available", "false"), ("runtime_reason", 100)])
def test_activity_client_rejects_invalid_runtime_metadata(tmp_path, field, value):
    metadata = architecture()
    metadata[field] = value
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=metadata))
    with pytest.raises(NeuralError):
        client.activity_architecture()


@pytest.mark.parametrize(
    "invalid",
    [
        "shape",
        "nonfinite",
        "masked",
        "unknown",
        "timing",
        "clockrange",
        "provenance",
        "overlap",
        "device",
    ],
)
def test_activity_client_rejects_invalid_scientific_values(tmp_path, invalid):
    value = capture()
    if invalid == "shape":
        value["tensors"]["block.0"].pop()
    elif invalid == "nonfinite":
        value["tensors"]["embedding"][0] = float("inf")
    elif invalid == "masked":
        value["token_is_padding"] = True
    elif invalid == "unknown":
        value["tensors"]["invented"] = [1.0] * 384
    elif invalid == "timing":
        value["completed_at"] = 0.5
    elif invalid == "clockrange":
        value["completed_at"] = 1e300
    elif invalid == "provenance":
        value["model_revision"] = None
    elif invalid == "device":
        value["device"] = []
    else:
        value["unavailable"] = ["embedding"]
    client = client_for(tmp_path, lambda _: httpx.Response(200, content=json.dumps(value)))
    assert callable(getattr(client, "activity_capture", None))
    with pytest.raises(NeuralError):
        client.activity_capture("fixture-1")


def test_activity_client_keeps_capture_ids_out_of_paths_and_validates_token_selection(tmp_path):
    client = client_for(tmp_path, lambda _: pytest.fail("Invalid controls reached the broker"))
    assert callable(getattr(client, "activity_capture", None))
    with pytest.raises(NeuralError):
        client.activity_capture("../status")
    with pytest.raises(NeuralError):
        client.activity_control(visible=True, paused=False, token_index=-1)


def test_first_unmasked_token_can_follow_left_padding(tmp_path):
    value = capture()
    value["token_index"], value["token_count"] = 2, 2
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=value))
    assert client.activity_capture("fixture-1")["token_index"] == 2


def test_masked_capture_remains_inspectable_without_fabricated_values(tmp_path):
    value = capture()
    value["token_index"], value["token_count"] = 3, 2
    value["token_is_padding"] = True
    value["unavailable"], value["tensors"] = list(value["tensors"]), {}
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=value))
    result = client.activity_capture("fixture-1")
    view = network_view()
    view.set_capture(result)
    assert "masked" in view.capture_status.text().lower()
    assert "unavailable" in view.inspector.text().lower()
    assert not view.replay.isEnabled()
    view.close()


def network_view():
    from dodo_memory_desktop import whole_workspace

    view_type = getattr(whole_workspace, "NeuralActivity", None)
    assert view_type is not None, "Graph has no real E5 network view"
    view = view_type(None)
    view.resize(1440, 900)
    view.set_architecture(architecture())
    return view


def test_network_draws_every_feature_and_expands_real_heads_and_gelu_units():
    view = network_view()
    assert [stage.identity for stage in view.surface.stages] == [
        "embedding",
        *[f"block.{index}" for index in range(12)],
        "pooled",
        "output",
    ]
    assert all(len(stage.points) == 384 for stage in view.surface.stages)
    view.surface.toggle_block(0)
    stages = {stage.identity: stage for stage in view.surface.stages}
    assert len(stages["heads.0"].points) == 384
    assert len(stages["ff.0"].points) == 1536
    assert ("embedding", "attention_residual.0") in view.surface.residual_routes
    assert ("attention_norm.0", "ff_residual.0") in view.surface.residual_routes
    view.stage.setCurrentIndex(view.stage.findData("ff.0"))
    assert view.features.count() == 1536
    assert "unavailable" in view.features.item(1535).text().lower()
    view.close()


def test_final_focus_foregrounds_measured_states_without_dropping_features():
    view = network_view()
    descriptor = architecture()
    descriptor["capture_level"] = "final"
    view.set_architecture(descriptor)
    assert view.capture is None and view.surface.focus_mode == "measured"
    value = capture()
    value["level"] = "final"
    value["tensors"] = {key: value["tensors"][key] for key in ("block.11", "pooled", "output")}
    view.set_capture(value)
    assert getattr(view.surface, "focus_mode", None) == "measured"
    assert sum(len(stage.points) for stage in view.surface.stages) == 5760
    assert [stage.identity for stage in view.surface.focused_stages()] == [
        "block.11",
        "pooled",
        "output",
    ]
    focused = view.surface.focus_bounds()
    assert focused.width() < view.surface.bounds.width() / 3
    assert all(focused.contains(stage.rect) for stage in view.surface.stages[-3:])
    view.surface.fit()
    assert view.surface.focus_mode == "all"
    assert view.surface.focus_bounds() == view.surface.bounds
    view.close()


def test_selecting_an_earlier_feature_leaves_final_focus_so_it_remains_visible():
    view = network_view()
    value = capture()
    value["level"] = "final"
    value["tensors"] = {key: value["tensors"][key] for key in ("block.11", "pooled", "output")}
    view.set_capture(value)
    view.stage.setCurrentIndex(view.stage.findData("block.0"))
    assert view.surface.focus_mode == "all"
    assert view.surface.selection == ("block.0", 0)
    assert "unavailable" in view.inspector.text().lower()
    view.measured.click()
    assert view.surface.focus_mode == "measured"
    view.close()


def test_depth_projection_preserves_captured_values_indices_and_selection():
    view = network_view()
    assert callable(getattr(view.surface, "set_depth", None))
    value = capture()
    view.set_capture(value)
    view.stage.setCurrentIndex(view.stage.findData("block.11"))
    view.features.setCurrentRow(123)
    view.depth.setChecked(False)
    stage = next(stage for stage in view.surface.stages if stage.identity == "block.11")
    flat = list(stage.points)
    view.depth.setChecked(True)
    projected = next(stage for stage in view.surface.stages if stage.identity == "block.11")
    assert len(projected.points) == len(flat) == 384
    assert projected.points[123] != flat[123]
    assert view.surface.selection == ("block.11", 123)
    assert view.capture is value
    assert view.capture["tensors"]["block.11"][123] == 0.1
    view.close()


def test_columns_keep_exact_positions_and_reveal_them_without_a_matrix_layout():
    view = network_view()
    stage = view.surface.stages[0]
    assert stage.columns == 1
    assert len(stage.points) == 384
    assert stage.rect.height() > stage.rect.width() * 8
    assert all(left.y() < right.y() for left, right in zip(stage.points, stage.points[1:]))
    view.surface.zoom = 0.7
    overview = view.surface.visible_feature_indices(stage)
    assert 25 <= len(overview) <= 45
    assert {0, 383}.issubset(overview)
    view.surface.select(stage.identity, 123)
    assert 123 in view.surface.visible_feature_indices(stage)
    view.surface.zoom = 9
    assert view.surface.visible_feature_indices(stage) == list(range(384))
    view.close()


def test_curved_bundles_follow_tensor_topology_without_feature_weight_edges():
    from PySide6.QtGui import QPainterPath

    view = network_view()
    routes = getattr(view.surface, "tensor_routes", [])
    assert len(routes) == 15
    assert routes[0] == ("input", "embedding")
    assert routes[-2:] == [("block.11", "pooled"), ("pooled", "output")]
    path = view.surface.route_path("block.11", "pooled")
    assert path.elementAt(1).type == QPainterPath.ElementType.CurveToElement
    view.surface.toggle_block(0)
    assert ("embedding", "attention_residual.0") in view.surface.residual_routes
    assert ("attention_norm.0", "ff_residual.0") in view.surface.residual_routes
    view.close()


def test_stage_labels_remain_screen_sized_when_node_geometry_is_zoomed():
    view = network_view()
    assert callable(getattr(view.surface, "label_rect", None))
    stage = view.surface.stages[0]
    view.surface.zoom = 0.25
    small = view.surface.label_rect(stage)
    view.surface.zoom = 2
    large = view.surface.label_rect(stage)
    assert small.height() == large.height()
    assert small.height() >= 60
    assert small.width() >= 64
    view.close()


def test_glow_is_driven_by_signed_captured_magnitude_and_never_fills_unavailable_states():
    view = network_view()
    value = capture()
    value["level"] = "final"
    value["tensors"] = {"block.11": [-1.0, 0.0, 4.0] + [0.0] * 381}
    view.set_capture(value)
    assert view.surface.strengths["block.11"][:3] == [0.5, 0.0, 1.0]
    assert [colour.alpha() for colour in view.surface.glows["block.11"][:3]] == [19, 0, 38]
    negative, _, positive = view.surface.colours["block.11"][:3]
    assert negative.red() > negative.blue()
    assert positive.blue() > positive.red()
    assert "block.0" not in view.surface.colours
    assert "block.0" not in view.surface.glows
    assert view.capture["tensors"]["block.11"][:3] == [-1.0, 0.0, 4.0]
    view.close()


def test_weights_page_is_lazy_and_stays_inactive_while_hidden(tmp_path):
    from dodo_memory_desktop.neural_activity import NeuralActivity

    requests = []

    def receive(request):
        requests.append(request.url.path)
        return httpx.Response(503)

    view = NeuralActivity(client_for(tmp_path, receive))
    assert view.weight_view is None
    view.view_mode.setCurrentIndex(view.view_mode.findData("weights"))
    assert view.pages.currentWidget() is view.weight_view
    view.set_active(True)
    assert not view.active and not view.weight_view.active
    QTest.qWait(30)
    assert requests == []
    assert view.shutdown(timeout=1)
    assert view.weight_view._stopping
    view.close()


def test_overview_opens_the_requested_block_without_loading_hidden_parameters():
    from dodo_memory_desktop.neural_activity import NeuralActivity

    view = NeuralActivity(None)
    view.set_architecture(architecture())
    view.view_mode.setCurrentIndex(view.view_mode.findData("overview"))
    assert view.pages.currentWidget() is view.overview
    view.overview.block_selected.emit(8)
    assert view.pages.currentWidget() is view.weight_view
    assert view.weight_view._requested_block == 8
    assert not view.weight_view.active and not view.weight_view.running
    view.view_mode.setCurrentIndex(view.view_mode.findData("overview"))
    assert view.pages.currentWidget() is view.overview
    assert view.shutdown()
    view.close()


def test_overview_receives_only_current_model_and_epoch_captures():
    import copy

    from dodo_memory_desktop.neural_activity import NeuralActivity

    view = NeuralActivity(None)
    view.set_architecture(architecture())
    view.epoch = "fixture"
    view.set_capture(capture())
    original = view.capture
    wrong_model = copy.deepcopy(original)
    wrong_model["model_space"] = "another-encoder"
    view.set_capture(wrong_model)
    assert view.capture is original and view.overview.capture is original
    view.active = True
    view._event({"epoch": "new-service", "sequence": 1, "type": "started"})
    assert view.capture is None and view.overview.capture is None
    assert view.overview.running and not view.surface.colours
    view.set_capture(original)
    assert view.overview.capture is None
    view._event({"epoch": "new-service", "sequence": 2, "type": "completed"})
    assert not view.overview.running
    view.close()


def test_changed_architecture_clears_measured_colours_and_inspector():
    import copy

    from dodo_memory_desktop.neural_activity import NeuralActivity

    view = NeuralActivity(None)
    view.set_architecture(architecture())
    view.set_capture(capture())
    changed = copy.deepcopy(architecture())
    changed["model"]["revision"] = "new-weights"
    view.set_architecture(changed)
    assert view.capture is None and view.overview.capture is None
    assert not view.surface.colours
    assert "unavailable" in view.inspector.text().lower()
    view.close()


def test_replay_is_labelled_historical_in_the_complete_network():
    from dodo_memory_desktop.neural_activity import NeuralActivity

    view = NeuralActivity(None)
    view.set_architecture(architecture())
    view.set_capture(capture())
    view._replay()
    assert view.overview.status_text().startswith("Replay ·")
    assert view.pause.isChecked()
    view.close()


def test_tensor_titles_are_drawn_in_the_header_at_normal_zoom():
    from PySide6.QtCore import QPointF, Qt
    from PySide6.QtGui import QFontDatabase

    if not QFontDatabase.families():
        assert QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf") >= 0
    view = network_view()
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.show()
    QApplication.processEvents()
    view.depth.setChecked(False)
    stage = view.surface.stages[0]
    view.surface.zoom = 0.35
    view.surface.pan = QPointF(300 - stage.rect.center().x() * view.surface.zoom, 160)
    image = view.surface.grab().toImage()
    rect = view.surface.label_rect(stage).toAlignedRect()
    header_ink = sum(
        image.pixelColor(x, y).red() > 150
        for x in range(rect.left(), rect.right())
        for y in range(rect.top(), rect.bottom())
    )
    assert header_ink > 20, "The Embeddings title was clipped out of its header"
    view.close()


def test_expansion_fits_all_1536_gelu_positions_vertically():
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QFontDatabase

    from dodo_memory_desktop.ui import theme_stylesheet

    if not QFontDatabase.families():
        assert QFontDatabase.addApplicationFont("C:/Windows/Fonts/segoeui.ttf") >= 0
    view = network_view()
    view.setStyleSheet(theme_stylesheet())
    view.resize(1280, 720)
    view.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    view.show()
    QApplication.processEvents()
    view.surface.toggle_block(0)
    stage = next(stage for stage in view.surface.stages if stage.identity == "ff.0")
    top = stage.rect.top() * view.surface.zoom + view.surface.pan.y()
    bottom = stage.rect.bottom() * view.surface.zoom + view.surface.pan.y()
    assert 0 <= top < bottom <= view.surface.height()
    view.close()


def test_head_major_coordinates_and_keyboard_match_captured_head_dimensions():
    from PySide6.QtCore import Qt

    view = network_view()
    view.surface.toggle_block(0)
    view.depth.setChecked(False)
    stage = next(stage for stage in view.surface.stages if stage.identity == "heads.0")
    origin = stage.points[0]
    assert stage.points[31].x() == origin.x()
    vertical = stage.points[1].y() - origin.y()
    horizontal = stage.points[32].x() - origin.x()
    assert vertical > 0 and horizontal > 0
    assert stage.points[31].y() == pytest.approx(origin.y() + 31 * vertical)
    assert stage.points[32].x() == origin.x() + horizontal
    assert stage.points[32].y() == origin.y()
    assert stage.points[383].x() == origin.x() + 11 * horizontal
    assert stage.points[383].y() == pytest.approx(origin.y() + 31 * vertical)
    view.surface.select("heads.0", 0)
    QTest.keyClick(view.surface, Qt.Key.Key_Right)
    assert view.surface.selection == ("heads.0", 32)
    QTest.keyClick(view.surface, Qt.Key.Key_Down)
    assert view.surface.selection == ("heads.0", 33)
    view.close()


def test_gelu_keyboard_moves_through_its_ordered_column():
    from PySide6.QtCore import Qt

    view = network_view()
    view.surface.toggle_block(0)
    view.surface.select("ff.0", 0)
    QTest.keyClick(view.surface, Qt.Key.Key_Down)
    assert view.surface.selection == ("ff.0", 1)
    view.close()


def test_inspector_preserves_signed_numbers_and_distinguishes_pooled_scope():
    view = network_view()
    view.set_capture(capture())
    view.stage.setCurrentIndex(view.stage.findData("embedding"))
    view.features.setCurrentRow(0)
    assert "-0.5" in view.inspector.text()
    assert "token 0" in view.inspector.text().lower()
    view.features.setCurrentRow(2)
    assert "+0.5" in view.inspector.text()
    view.stage.setCurrentIndex(view.stage.findData("pooled"))
    assert "all unmasked tokens" in view.inspector.text().lower()
    view.replay.click()
    assert "Replay" in view.capture_status.text()
    assert "captured values" in view.capture_status.text()
    assert view.pause.isChecked()
    view.close()


def test_new_capture_preserves_the_feature_being_inspected():
    view = network_view()
    view.set_capture(capture())
    view.stage.setCurrentIndex(view.stage.findData("block.3"))
    view.features.setCurrentRow(123)
    value = capture()
    value["id"] = "fixture-2"
    value["tensors"]["block.3"][123] = -1.25
    view.set_capture(value)
    assert view.features.currentRow() == 123
    assert "-1.25" in view.inspector.text()
    view.close()


def test_final_capture_marks_missing_states_unavailable_and_next_token_does_not_change_capture():
    view = network_view()
    value = capture()
    value["level"] = "final"
    value["tensors"] = {key: value["tensors"][key] for key in ("block.11", "pooled", "output")}
    view.set_capture(value)
    view.stage.setCurrentIndex(view.stage.findData("block.0"))
    assert "unavailable" in view.inspector.text().lower()
    view.token.setValue(7)
    assert view.capture["token_index"] == 0
    assert view.capture["tensors"] == value["tensors"]
    assert not view.lease.isActive()
    assert not view.reconnect.isActive()
    view.close()


def test_hidden_capture_control_is_serialised_after_an_inflight_visible_lease(tmp_path):
    from dodo_memory_desktop.whole_workspace import NeuralActivity

    started, release, hidden_started = threading.Event(), threading.Event(), threading.Event()
    controls = []

    def receive(request):
        value = json.loads(request.read())
        if value["visible"]:
            started.set()
            assert release.wait(3)
        else:
            hidden_started.set()
        controls.append(value)
        return httpx.Response(200, json=value)

    view = NeuralActivity(client_for(tmp_path, receive))
    view.active = True
    view.token.setValue(2)
    assert started.wait(2)
    view.set_active(False)
    hidden_overtook = hidden_started.wait(0.1)
    release.set()
    deadline = time.monotonic() + 3
    while len(controls) < 2 and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert not hidden_overtook, "A hidden control overtook the older visible lease"
    assert [value["visible"] for value in controls] == [True, False]
    assert not view.lease.isActive()
    assert not view.reconnect.isActive()
    assert view.events.reply is None
    view.close()


def test_deleted_view_finishes_its_hidden_control_after_an_inflight_request(tmp_path, capfd):
    from PySide6.QtCore import QCoreApplication, QEvent
    from shiboken6 import isValid

    from dodo_memory_desktop.whole_workspace import NeuralActivity

    started, release = threading.Event(), threading.Event()
    controls = []

    def receive(request):
        value = json.loads(request.read())
        if value["visible"]:
            started.set()
            assert release.wait(3)
        controls.append(value["visible"])
        return httpx.Response(200, json=value)

    view = NeuralActivity(client_for(tmp_path, receive))
    view.active = True
    view.token.setValue(2)
    assert started.wait(2)
    view.deleteLater()
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    assert not isValid(view)
    release.set()
    deadline = time.monotonic() + 3
    while len(controls) < 2 and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert controls == [True, False]
    assert "Signal source has been deleted" not in capfd.readouterr().err


def test_event_during_pending_snapshot_gets_one_followup_without_rolling_back_cursor(tmp_path):
    from dodo_memory_desktop.neural_activity import NeuralActivity

    started, release = threading.Event(), threading.Event()
    snapshots = []

    def receive(request):
        if request.url.path == "/activity/snapshot":
            sequence = 1 if not snapshots else 3
            snapshots.append(sequence)
            if sequence == 1:
                started.set()
                assert release.wait(3)
            return httpx.Response(
                200,
                json={
                    "epoch": "fixture",
                    "sequence": sequence,
                    "visible": True,
                    "paused": False,
                    "token_index": None,
                    "running": [],
                    "capture_level": "full",
                    "reason": None,
                    "dropped_events": 0,
                    "latest_capture_id": f"fixture-{sequence}",
                },
            )
        if request.url.path.startswith("/activity/activations/"):
            value = capture()
            value["id"] = request.url.path.rsplit("/", 1)[1]
            return httpx.Response(200, json=value)
        return httpx.Response(200, json={"visible": False, "paused": False})

    view = NeuralActivity(client_for(tmp_path, receive))
    view.set_architecture(architecture())
    view.active = True
    view.refresh()
    assert started.wait(2)
    view._event({"epoch": "fixture", "sequence": 3})
    QTest.qWait(200)
    release.set()
    deadline = time.monotonic() + 2
    while (
        view.capture is None or view.capture["id"] != "fixture-3"
    ) and time.monotonic() < deadline:
        QApplication.processEvents()
        QTest.qWait(5)
    assert view.capture is not None and view.capture["id"] == "fixture-3"
    assert view.sequence == 3
    QTest.qWait(250)
    assert snapshots == [1, 3]
    view.close()


def test_shutdown_drains_owned_visible_then_hidden_requests_without_waiting_for_unrelated_work(
    tmp_path,
):
    from PySide6.QtCore import QThreadPool, QTimer

    from dodo_memory_desktop.neural_activity import NeuralActivity

    visible_started, release_visible = threading.Event(), threading.Event()
    snapshot_started, release_snapshot = threading.Event(), threading.Event()
    unrelated = threading.Event()
    controls = []

    def receive(request):
        if request.url.path == "/activity/snapshot":
            snapshot_started.set()
            assert release_snapshot.wait(3)
            return httpx.Response(
                200,
                json={
                    "epoch": "fixture",
                    "sequence": 0,
                    "visible": False,
                    "paused": False,
                    "token_index": None,
                    "running": [],
                    "latest_capture_id": None,
                    "capture_level": "final",
                    "reason": None,
                    "dropped_events": 0,
                },
            )
        value = json.loads(request.read())
        if value["visible"]:
            visible_started.set()
            assert release_visible.wait(3)
        controls.append(value["visible"])
        return httpx.Response(200, json=value)

    view = NeuralActivity(client_for(tmp_path, receive))
    assert callable(getattr(view, "shutdown", None))
    view.active = True
    view.token.setValue(2)
    view.refresh()
    assert visible_started.wait(2) and snapshot_started.wait(2)
    QThreadPool.globalInstance().start(lambda: unrelated.wait(3))
    QTimer.singleShot(30, release_visible.set)
    QTimer.singleShot(70, release_snapshot.set)
    try:
        assert view.shutdown(timeout=1)
        assert controls == [True, False]
        assert release_snapshot.is_set()
        assert not view.controls.running and view.controls.pending is None
        assert not view.lease.isActive() and view.events.reply is None
        assert not unrelated.is_set()
        assert QThreadPool.globalInstance().activeThreadCount() >= 1
    finally:
        release_visible.set()
        release_snapshot.set()
        unrelated.set()
        view.close()
