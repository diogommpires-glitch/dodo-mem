from types import SimpleNamespace

import pytest

from dodo_mem_neural.service import qualified_runtime


def test_driver_failure_uses_explicit_cpu_and_reports_reason(monkeypatch):
    calls = []

    def compile_model(name, *, device):
        calls.append(device)
        if device == "NPU":
            raise RuntimeError("driver unavailable")
        return SimpleNamespace(device=device)

    monkeypatch.setattr("dodo_mem_neural.service.NeuralRuntime", compile_model)
    runtime, reason = qualified_runtime("e5", "NPU")
    assert runtime.device == "CPU" and calls == ["NPU", "CPU"]
    assert "driver unavailable" in reason


def test_invalid_artifact_is_never_retried_on_another_device(monkeypatch):
    def invalid(*args, **kwargs):
        raise ValueError("artifact hash mismatch")

    monkeypatch.setattr("dodo_mem_neural.service.NeuralRuntime", invalid)
    with pytest.raises(ValueError, match="hash mismatch"):
        qualified_runtime("e5", "NPU")
