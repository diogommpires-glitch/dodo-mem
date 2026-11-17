import numpy as np

from dodo_mem_neural.activity import NeuralActivity

MODEL = {"name": "e5", "revision": "pinned", "space": "fixture", "device": ["NPU"]}


def test_inference_metrics_run_without_visible_capture_and_report_busy_not_hardware(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("dodo_mem_neural.activity.time.monotonic", lambda: clock[0])
    activity = NeuralActivity()
    activity.set_model(MODEL)
    assert activity.begin([1], True, MODEL) is None
    inference = activity.inference_started(MODEL)
    clock[0] += 1
    active = activity.telemetry()
    assert active["active_inferences"] == 1
    assert active["inference_busy_percent"] == 10.0
    activity.inference_finished(inference, False)
    measured = activity.telemetry()
    assert measured["completed_inferences"] == 1 and measured["last_inference_ms"] == 1000
    assert measured["hardware_utilization_percent"] is None
    assert measured["hardware_utilization_reason"]
    assert activity.snapshot()["retained_captures"] == 0
    clock[0] += 11
    assert activity.telemetry()["inference_busy_percent"] == 0.0


def test_actual_capture_requires_visible_lease_and_retains_measured_values():
    activity = NeuralActivity()
    assert activity.begin([0, 1, 1, 0], True, MODEL) is None
    activity.control(visible=True, paused=False, token_index=None)
    ticket = activity.begin([0, 1, 1, 0], True, MODEL)
    assert ticket["token_index"] == 1 and ticket["token_count"] == 2
    assert activity.snapshot()["running"][0]["id"] == ticket["id"]
    values = np.linspace(-1, 1, 384, dtype=np.float32)
    activity.complete(
        ticket, {"block.11": values, "pooled": values, "output": values}, "final", MODEL
    )
    values[:] = 42
    capture = activity.capture(ticket["id"])
    assert capture["tensors"]["block.11"][0] == -1
    assert capture["tensors"]["block.11"][-1] == 1
    assert "block.0" in capture["unavailable"]
    assert activity.snapshot()["running"] == []


def test_pause_and_invalid_token_never_invent_activations_or_change_next_input():
    activity = NeuralActivity()
    activity.control(visible=True, paused=True, token_index=3)
    assert activity.begin([1, 1, 0, 0], True, MODEL) is None
    activity.control(visible=True, paused=False, token_index=3)
    ticket = activity.begin([1, 1, 0, 0], True, MODEL)
    assert ticket["valid_token"] is False and ticket["token_index"] == 3
    activity.complete(ticket, {"block.11": np.ones(384)}, "final", MODEL)
    capture = activity.capture(ticket["id"])
    assert capture["tensors"] == {} and capture["token_is_padding"] is True
    assert capture["reason"] == "Selected token is masked or outside this inference."


def test_capture_history_is_bounded_and_background_is_sampled(monkeypatch):
    clock = [100.0]
    monkeypatch.setattr("dodo_mem_neural.activity.time.monotonic", lambda: clock[0])
    activity = NeuralActivity()
    activity.control(visible=True, paused=False, token_index=None)
    ticket = activity.begin([1], False, MODEL)
    assert ticket["sampled"] is True
    activity.complete(ticket, {"output": np.ones(384)}, "final", MODEL)
    assert activity.begin([1], False, MODEL) is None
    for _ in range(40):
        clock[0] += 0.1
        item = activity.begin([1], True, MODEL)
        activity.complete(item, {"output": np.ones(384)}, "final", MODEL)
    snapshot = activity.snapshot()
    assert snapshot["retained_captures"] == 32
    assert snapshot["activation_bytes"] <= 8 * 1024**2
    assert activity.capture(ticket["id"]) is None
    clock[0] += 31
    assert activity.begin([1], True, MODEL) is None


def test_closing_view_during_inference_retains_no_late_capture_and_duplicate_is_ignored():
    activity = NeuralActivity()
    activity.control(visible=True, paused=False, token_index=None)
    ticket = activity.begin([1], True, MODEL)
    activity.control(visible=False, paused=False, token_index=None)
    activity.complete(ticket, {"output": np.ones(384)}, "final", MODEL)
    assert activity.snapshot()["running"] == []
    assert activity.snapshot()["retained_captures"] == 0
    activity.control(visible=True, paused=False, token_index=None)
    activity.complete(ticket, {"output": np.ones(384)}, "final", MODEL)
    assert activity.snapshot()["retained_captures"] == 0
