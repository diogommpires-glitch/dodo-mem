import threading
from types import SimpleNamespace

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from dodo_mem_neural import runtime as runtime_module
from dodo_mem_neural.activity import NeuralActivity
from dodo_mem_neural.runtime import NeuralRuntime


class Observer:
    def __init__(self, token_index=None, broken=None):
        self.token_index = token_index
        self.broken = broken
        self.events = []

    def begin(self, attention_mask, interactive, model_info):
        if self.broken == "begin":
            raise RuntimeError("viewer unavailable")
        self.mask = attention_mask.copy()
        return {"id": "actual-operation", "token_index": self.token_index}

    def complete(self, ticket, tensors, level, model_info):
        self.events.append((ticket.copy(), tensors, level))
        if self.broken == "complete":
            raise RuntimeError("viewer disconnected")

    def fail(self, ticket, code):
        self.events.append((ticket, code))


def runtime(observer):
    ids = ops.parameter([1, 4], np.int64, name="input_ids")
    mask = ops.parameter([1, 4], np.int64, name="attention_mask")
    token = ops.parameter([1], np.int64, name="capture_token_index")
    hidden = ops.broadcast(ops.convert(ops.unsqueeze(ids, 2), np.float32), [1, 4, 384])
    hidden.output(0).set_names({"last_hidden_state"})
    tap = ops.gather(hidden, token, 1)
    tap.output(0).set_names({"embedding"})
    # The first returned output deliberately is not the original embedding output.
    model = ov.Model([tap, hidden], [ids, mask, token])
    value = object.__new__(NeuralRuntime)
    value.compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 2})
    value.request = value.compiled.create_infer_request()
    value.output = value.compiled.output("last_hidden_state")
    value.capture_ports = {"embedding": value.compiled.output("embedding")}
    value.capture_level = "blocks"
    value.observer = observer
    value.model_info = {"name": "e5", "space": "original"}
    value.length = 4
    value.name = "e5"
    return value


def test_actual_infer_updates_metrics_when_activation_view_is_hidden():
    activity = NeuralActivity()
    value = runtime(activity)
    value.model_info = {"name": "e5", "space": "fixture", "device": ["CPU"]}
    encoding = SimpleNamespace(ids=[3, 4, 999, 999], attention_mask=[1, 1, 0, 0], type_ids=[0] * 4)
    hidden = value._infer(encoding, interactive=True)
    assert hidden.shape == (1, 4, 384)
    measured = activity.telemetry()
    assert measured["device"] == ["CPU"] and measured["completed_inferences"] == 1
    assert measured["last_inference_ms"] > 0 and measured["active_inferences"] == 0
    assert activity.snapshot()["retained_captures"] == 0


def test_device_failure_is_reported_and_not_converted_to_observer_success():
    activity = NeuralActivity()
    value = runtime(activity)
    value.model_info = {"name": "e5", "space": "fixture", "device": ["CPU"]}

    class UnavailableRequest:
        def infer(self, *args, **kwargs):
            raise RuntimeError("synthetic device failure")

    value.request = UnavailableRequest()
    encoding = SimpleNamespace(ids=[3, 4, 999, 999], attention_mask=[1, 1, 0, 0], type_ids=[0] * 4)
    with pytest.raises(RuntimeError, match="device failure"):
        value._infer(encoding, interactive=True)
    assert activity.telemetry()["failed_inferences"] == 1
    assert activity.telemetry()["completed_inferences"] == 0


@pytest.mark.parametrize("broken", [None, "begin", "complete"])
def test_named_original_output_and_observer_failure_do_not_change_inference(broken):
    observer = Observer(token_index=1, broken=broken)
    value = runtime(observer)
    encoding = SimpleNamespace(ids=[3, 4, 999, 999], attention_mask=[1, 1, 0, 0], type_ids=[0] * 4)
    hidden = value._infer(encoding, interactive=True)
    assert hidden.shape == (1, 4, 384)
    np.testing.assert_allclose(hidden[0, :, 0], encoding.ids)
    if broken != "begin":
        ticket, tensors, level = observer.events[0]
        assert ticket["id"] == "actual-operation"
        assert ticket["valid_token"] is True
        assert level == "blocks"
        np.testing.assert_allclose(tensors["embedding"], np.full(384, 4))
        np.testing.assert_allclose(tensors["block.11"], np.full(384, 4))
        np.testing.assert_allclose(tensors["pooled"], np.full(384, 3.5))
        np.testing.assert_allclose(tensors["output"], np.full(384, 1 / np.sqrt(384)))


@pytest.mark.parametrize("selected", [2, 1000, -1])
def test_padded_and_out_of_range_token_never_fail_actual_inference(selected):
    observer = Observer(token_index=selected)
    value = runtime(observer)
    encoding = SimpleNamespace(ids=[3, 4, 999, 999], attention_mask=[1, 1, 0, 0], type_ids=[0] * 4)
    hidden = value._infer(encoding, interactive=False)
    assert hidden.shape == (1, 4, 384)
    ticket, tensors, _ = observer.events[0]
    assert ticket["token_index"] == selected
    assert ticket["valid_token"] is False
    assert tensors == {}


def test_unverified_source_contract_keeps_embedding_but_publishes_no_layer_values():
    observer = Observer(token_index=1)
    value = runtime(observer)
    value.model_info["architecture"] = {"verified": False}
    encoding = SimpleNamespace(ids=[3, 4, 999, 999], attention_mask=[1, 1, 0, 0], type_ids=[0] * 4)
    hidden = value._infer(encoding, interactive=False)
    np.testing.assert_allclose(hidden[0, :, 0], encoding.ids)
    assert observer.events[0][1] == "source_contract_unavailable"


def test_sequential_encoders_keep_one_core_on_the_owned_inference_thread(monkeypatch):
    initialised = []

    class Core:
        def __init__(self):
            initialised.append(threading.current_thread().name)

        available_devices = ("CPU",)

        def get_property(self, device, name):
            return []

        def get_versions(self, device):
            return {}

    monkeypatch.setattr(runtime_module.ov, "Core", Core)
    monkeypatch.setattr(runtime_module, "_CORE", None)
    first = object.__new__(NeuralRuntime)
    first.device = "CPU"
    runtime_module._submit(first._initialise_device)
    second = object.__new__(NeuralRuntime)
    second.device = "CPU"
    runtime_module._submit(second._initialise_device)
    assert first.core is second.core
    assert initialised == ["dodo-neural-inference"]
