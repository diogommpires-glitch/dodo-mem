import copy

import httpx
import pytest
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication
from test_neural_activity import architecture

from dodo_memory_desktop.neural_activity import NeuralActivity
from dodo_memory_desktop.neural_client import NeuralClient, NeuralError
from dodo_memory_desktop.neural_status import NeuralStatus


def telemetry():
    return {
        "device": ["NPU"],
        "active_inferences": 0,
        "completed_inferences": 42,
        "failed_inferences": 0,
        "last_inference_ms": 14.5,
        "last_completed_at": 1794830000.0,
        "inference_busy_percent": 12.5,
        "window_seconds": 10.0,
        "hardware_utilization_percent": None,
        "hardware_utilization_reason": "Windows NPU utilisation counter unavailable.",
    }


def client_for(tmp_path, receive):
    (tmp_path / "broker.token").write_text("x" * 48, encoding="ascii")
    return NeuralClient(tmp_path, transport=httpx.MockTransport(receive))


def test_status_distinguishes_measured_encoder_busy_from_unavailable_hardware_usage(tmp_path):
    status = {"ready": True, "device": ["NPU"], "telemetry": telemetry()}
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=status))
    widget = NeuralStatus(client)
    widget._ready(client.status())
    text = widget.summary.text()
    assert "Device: NPU" in text
    assert "Encoder busy: 12.5% (10 s)" in text
    assert "Completed: 42" in text and "Last inference: 14.5 ms" in text
    assert "NPU hardware usage: unavailable" in text
    assert "NPU hardware usage: 0" not in text
    widget.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("inference_busy_percent", 101),
        ("inference_busy_percent", True),
        ("hardware_utilization_percent", -1),
        ("last_inference_ms", "14"),
        ("completed_inferences", -1),
        ("window_seconds", 0),
    ],
)
def test_status_rejects_invalid_telemetry_instead_of_displaying_false_usage(tmp_path, field, value):
    metrics = telemetry()
    metrics[field] = value
    client = client_for(tmp_path, lambda _: httpx.Response(200, json={"telemetry": metrics}))
    with pytest.raises(NeuralError, match="telemetry"):
        client.status()


def test_status_explains_a_stopped_worker_and_formats_lag_without_a_traceback(tmp_path):
    status = {
        "ready": True,
        "state": "failed",
        "device": ["NPU"],
        "source_failures": 3,
        "background_worker_alive": False,
        "lag_seconds": 32.46238994598389,
        "reason": "Source adapter unavailable.\nTraceback details must not be rendered.",
    }
    client = client_for(tmp_path, lambda _: httpx.Response(200, json=status))
    widget = NeuralStatus(client)
    widget._ready(client.status())
    text = widget.summary.text()
    assert "Worker: stopped" in text and "Recorded source failures: 3" in text
    assert "Index lag (s): 32.5" in text and "32.4623899" not in text
    assert "Reason: Source adapter unavailable." in text and "Traceback" not in text
    widget._ready({**status, "reason": "  \n"})
    assert "Reason:" not in widget.summary.text()
    widget.close()


@pytest.mark.parametrize(
    "field,value",
    [
        ("background_worker_alive", "false"),
        ("source_failures", -1),
        ("reason", {"error": "wrong type"}),
        ("unavailable_providers", True),
    ],
)
def test_status_rejects_invalid_worker_failure_metadata(tmp_path, field, value):
    client = client_for(tmp_path, lambda _: httpx.Response(200, json={field: value}))
    with pytest.raises(NeuralError, match="status"):
        client.status()


def test_overview_keeps_live_running_status_visible():
    view = NeuralActivity(None)
    view.view_mode.setCurrentIndex(view.view_mode.findData("overview"))
    view.show()
    QApplication.processEvents()
    assert view.running_status.isVisibleTo(view)
    view.close()


def test_graph_status_remains_visible_across_both_graph_tabs(tmp_path, monkeypatch):
    from desktop_fixture import FixtureGateway
    from PySide6.QtCore import QSettings, Qt

    from dodo_memory_desktop.app import GRAPH, Desktop
    from dodo_memory_desktop.whole_index import Catalogue

    def receive(request):
        if request.url.path == "/activity/architecture":
            return httpx.Response(200, json=architecture())
        if request.url.path == "/activity/snapshot":
            return httpx.Response(
                200,
                json={
                    "epoch": "fixture",
                    "sequence": 0,
                    "running": [],
                    "visible": True,
                    "paused": False,
                    "capture_level": "full",
                    "latest_capture_id": None,
                },
            )
        if request.url.path == "/status":
            return httpx.Response(
                200, json={"ready": True, "device": ["NPU"], "telemetry": telemetry()}
            )
        if request.url.path == "/activity/control":
            return httpx.Response(200, json={"visible": True, "paused": False})
        return httpx.Response(503, json={"detail": "No synthetic parameter catalogue"})

    client = client_for(tmp_path, receive)
    monkeypatch.setattr("dodo_memory_desktop.whole_workspace.NeuralClient", lambda: client)
    window = Desktop(
        FixtureGateway(),
        QSettings(str(tmp_path / "settings.ini"), QSettings.Format.IniFormat),
        tmp_path / "boards",
        whole_catalogue=Catalogue(tmp_path / "catalogue.sqlite3"),
    )
    window.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    window.show()
    window.show_page(GRAPH)
    QApplication.processEvents()
    assert window.whole.graph_tabs.currentWidget() is window.whole.network
    assert window.whole.neural_status.isVisibleTo(window)
    window.whole.graph_tabs.setCurrentWidget(window.whole.graph)
    QApplication.processEvents()
    assert window.whole.neural_status.isVisibleTo(window)
    window.close()


def test_same_process_runtime_recovery_reloads_architecture_without_an_epoch_change(tmp_path):
    paused = architecture()
    paused["runtime_available"] = False
    paused["runtime_reason"] = "Resource gate paused the encoder."
    paused["model"]["device"] = []
    recovered = architecture()
    recovered["runtime_available"] = True
    recovered["runtime_reason"] = None
    requests = []

    def receive(request):
        requests.append(request.url.path)
        if request.url.path == "/activity/architecture":
            return httpx.Response(200, json=recovered)
        if request.url.path == "/activity/snapshot":
            return httpx.Response(
                200,
                json={
                    "epoch": "same-process",
                    "sequence": 1,
                    "running": [],
                    "visible": True,
                    "paused": False,
                    "capture_level": "full",
                    "latest_capture_id": None,
                    "runtime_available": True,
                    "runtime_reason": None,
                },
            )
        return httpx.Response(200, json={"visible": False, "paused": False})

    client = client_for(tmp_path, receive)
    view = NeuralActivity(client)
    view.set_architecture(copy.deepcopy(paused))
    view.epoch = "same-process"
    view.active = True
    view._snapshot(client.activity_snapshot())
    for _ in range(100):
        QApplication.processEvents()
        if view.architecture.get("runtime_available") is True:
            break
        QTest.qWait(5)
    assert "/activity/architecture" in requests
    assert view.architecture["runtime_available"] is True
    assert "NPU" in view.provenance.text()
    assert not view.running_status.text().startswith("Encoder paused")
    view.close()
