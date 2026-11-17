"""Synthetic weight and capture fixtures verify the complete E5 diagram."""

import copy

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtTest import QTest

from dodo_memory_desktop.network_overview import PROJECTIONS, STAGES, NetworkOverview


def architecture():
    return {
        "num_hidden_layers": 12,
        "hidden_size": 384,
        "model": {"revision": "synthetic", "space": "synthetic"},
    }


def capture():
    return {
        "id": "synthetic-final",
        "model_revision": "synthetic",
        "model_space": "synthetic",
        "completed_at": 1.1,
        "tensors": {name: [0.5] * 384 for name in ("block.11", "pooled", "output")},
    }


def parameters():
    projections = []
    for block in range(12):
        for name, (_source, _target, inputs, outputs) in PROJECTIONS.items():
            projections.append(
                {
                    "block": block,
                    "projection": name,
                    "model_revision": "synthetic",
                    "model_space": "synthetic",
                    "input_dim": inputs,
                    "output_dim": outputs,
                    "total_weight_count": inputs * outputs,
                    "edges": [
                        {"source": 0, "target": 0, "weight": 0.5},
                        {"source": inputs - 1, "target": outputs - 1, "weight": -0.25},
                    ],
                }
            )
    return {"model_revision": "synthetic", "model_space": "synthetic", "projections": projections}


def view_with_parameters():
    view = NetworkOverview()
    view.resize(1000, 700)
    view.set_architecture(architecture())
    assert view.set_parameters(parameters())
    view.show()
    QTest.qWait(1)
    return view


def test_every_block_contains_all_six_true_projection_fans_and_opens_exact_detail():
    view = view_with_parameters()
    assert view.drawn_weight_count == 144
    assert view.total_weight_count == 21_233_664
    assert len(view.parameters["projections"]) == 72
    same_model = copy.deepcopy(architecture())
    same_model["model"]["device"] = ["NPU"]
    previous = view.parameters
    view.set_architecture(same_model)
    assert view.parameters is previous and view.weight_batches
    opened = []
    view.block_selected.connect(opened.append)
    for block in range(12):
        rect = view.screen_block_rect(block)
        assert view.rect().contains(rect.toRect())
        path = view.weight_batches[block, True, 3]
        assert path.elementCount() == 24  # Six independent move + cubic paths.
        point = view.feature_point(block, "input", 0)
        assert QPointF(path.elementAt(0).x, path.elementAt(0).y) == point
        QTest.mouseClick(view, Qt.MouseButton.LeftButton, pos=rect.center().toPoint())
        assert opened[-1] == block
    QTest.keyClick(view, Qt.Key.Key_Home)
    QTest.keyClick(view, Qt.Key.Key_Right)
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert opened[-1] == 1
    QTest.keyClick(view, Qt.Key.Key_End)
    QTest.keyClick(view, Qt.Key.Key_Return)
    assert opened[-1] == 11
    assert "Block 12" in view.accessibleDescription()
    view.close()


def test_partial_or_mixed_parameters_clear_all_previous_weight_edges():
    view = view_with_parameters()
    for change in ("partial", "mixed", "shape"):
        value = parameters()
        if change == "partial":
            value["projections"].pop()
        elif change == "mixed":
            value["projections"][5]["model_space"] = "another"
        else:
            value["projections"][5]["input_dim"] = 384
        assert not view.set_parameters(value)
        assert not view.weight_batches and view.parameters is None
        assert view.drawn_weight_count == view.total_weight_count == 0
        assert "unavailable" in view.parameter_status
        assert view.set_parameters(parameters())
    view.set_parameters(None)
    assert not view.weight_batches
    view.close()


def test_measured_post_norm_values_never_colour_raw_matmul_columns_or_fixed_weights():
    view = view_with_parameters()
    view.focus_block(11)
    before = view.grab().toImage()
    value = capture()
    view.set_capture(value)
    after = view.grab().toImage()
    norm = view.to_screen(view.feature_point(11, "output", 0))
    raw = view.to_screen(view.feature_point(11, "ff_output", 0))
    norm_rect = QRectF(norm.x() - 6, norm.y() - 6, 12, 12).toRect()
    raw_rect = QRectF(raw.x() - 6, raw.y() - 6, 12, 12).toRect()
    assert before.copy(norm_rect) != after.copy(norm_rect)
    assert before.copy(raw_rect) == after.copy(raw_rect)
    fan = view.to_screen(view._local(11, QPointF(970, 350)))
    assert before.copy(QRectF(fan.x(), fan.y(), 20, 20).toRect()) == after.copy(
        QRectF(fan.x(), fan.y(), 20, 20).toRect()
    )
    assert view.completed_path()
    view.set_running(True)
    assert view.capture is value and view.measured_tensors() == {"block.11", "pooled", "output"}
    view.set_replaying(True)
    assert view.status_text().startswith("Replay ·")
    view.close()


def test_virtual_feature_indices_zoom_and_pause_preserve_real_data():
    view = view_with_parameters()
    assert STAGES["ff_linear"][3] == STAGES["gelu"][3] == 1536
    assert view.feature_point(0, "ff_linear", 0) != view.feature_point(0, "gelu", 0)
    assert view.feature_point(0, "ff_output", 0) != view.feature_point(0, "output", 0)
    zoom = view.zoom
    view.focus_block(11)
    assert view.zoom > zoom
    view.zoom_by(1.5)
    assert view.zoom > zoom
    view.set_capture(capture())
    previous = view.capture
    view.set_paused(True)
    next_value = copy.deepcopy(previous)
    next_value["id"] = "next"
    view.set_capture(next_value)
    assert view.capture is previous
    view.set_paused(False)
    next_value["model_space"] = "another"
    view.set_capture(next_value)
    assert view.capture is None
    descriptor = copy.deepcopy(architecture())
    descriptor["model"]["revision"] = "changed"
    view.set_architecture(descriptor)
    assert view.parameters is None and not view.weight_batches
    view.close()
