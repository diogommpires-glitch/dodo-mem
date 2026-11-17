import json

import numpy as np
import openvino as ov
import openvino.opset13 as ops
import pytest

from dodo_mem_neural import capture_model


def e5_graph():
    data = ops.parameter([1, 256, 384], np.float32, name="data")
    hidden = data
    names = capture_model.capture_sources("full")
    for name, spec in names.items():
        if name.startswith("ff."):
            value = ops.concat([hidden] * 4, axis=2)
        elif name.startswith("heads."):
            value = ops.transpose(ops.reshape(hidden, [1, 256, 12, 32], False), [0, 2, 1, 3])
        else:
            hidden = ops.add(hidden, np.float32(1))
            value = hidden
        value.set_friendly_name(spec["operation"])
        # Keep every source reachable without changing the final hidden-state result.
        if name.startswith(("ff.", "heads.")):
            hidden = ops.add(
                hidden,
                ops.multiply(
                    ops.reduce_sum(value, list(range(len(value.shape))), False),
                    ops.constant(0, np.float32),
                ),
            )
    hidden.output(0).set_names({"last_hidden_state"})
    return ov.Model([hidden], [data])


def test_capture_gathers_one_actual_token_and_flattens_head_major():
    model = e5_graph()
    manifest = capture_model.instrument(model, "full")
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 2})
    values = np.arange(256 * 384, dtype=np.float32).reshape(1, 256, 384)
    result = compiled({"data": values, "capture_token_index": np.array([3], np.int64)})
    assert len(manifest["outputs"]) == 37
    assert result[compiled.output("embedding")].shape == (1, 1, 384)
    assert result[compiled.output("ff.0")].shape == (1, 1, 1536)
    assert result[compiled.output("heads.0")].shape == (1, 12, 1, 32)
    np.testing.assert_allclose(result[compiled.output("embedding")].ravel(), values[0, 3] + 1)
    np.testing.assert_allclose(result[compiled.output("heads.0")].ravel(), values[0, 3] + 13)
    np.testing.assert_allclose(result[compiled.output("last_hidden_state")], values + 13)


def test_capture_rejects_wrong_source_shape_before_compilation():
    model = e5_graph()
    embedding = next(
        node
        for node in model.get_ops()
        if node.friendly_name == capture_model.capture_sources("blocks")["embedding"]["operation"]
    )
    embedding.set_friendly_name("publisher_changed_operation")
    with pytest.raises(ValueError, match="embedding.*missing"):
        capture_model.instrument(model, "blocks")


def test_removing_visual_taps_restores_single_original_output_without_token_input():
    model = e5_graph()
    capture_model.instrument(model, "blocks")
    capture_model.remove_taps(model)
    assert [port.any_name for port in model.inputs] == ["data"]
    assert [port.any_name for port in model.outputs] == ["last_hidden_state"]
    compiled = ov.Core().compile_model(model, "CPU", {"INFERENCE_NUM_THREADS": 2})
    values = np.ones((1, 256, 384), np.float32)
    np.testing.assert_array_equal(
        compiled(values)[compiled.output("last_hidden_state")], values + 13
    )


@pytest.mark.parametrize("requested", [2, 256, -1])
def test_invalid_capture_token_uses_safe_gather_without_publishing_substitute(requested):
    selected, valid = capture_model.select_token(np.array([[0, 1, 0, 0]]), requested)
    assert selected == 1
    assert valid is False


def test_default_capture_token_is_first_unmasked_and_empty_mask_unavailable():
    assert capture_model.select_token(np.array([[0, 1, 1, 0]]), None) == (1, True)
    assert capture_model.select_token(np.zeros((1, 4)), None) == (0, False)


@pytest.mark.parametrize("compiler", ["default", "DRIVER"])
def test_qualification_requires_matching_contract_identity_and_all_gates(tmp_path, compiler):
    identity = {
        "schema": 1,
        "device": "NPU",
        "effective_hash": "actual-weights",
        "compiler": "default",
    }
    report = {
        "identity": {**identity, "compiler": compiler},
        "compiler": compiler,
        "state": "passed",
        "selected_level": "full",
        "capture_manifest_sha256": capture_model.manifest_for("full")["sha256"],
        "gates": {
            "encoder": {"passed": True},
            "retrieval": {"passed": True},
            "coexistence": {"passed": True},
        },
    }
    path = tmp_path / "qualification-NPU.json"
    path.write_text(json.dumps(report))
    expected = ("full", None if compiler == "default" else "DRIVER")
    assert capture_model.qualified_capture(tmp_path, identity) == expected
    for gate in report["gates"]:
        report["gates"][gate]["passed"] = False
        path.write_text(json.dumps(report))
        assert capture_model.qualified_capture(tmp_path, identity) == ("final", None)
        report["gates"][gate]["passed"] = True
    report["capture_manifest_sha256"] = "stale-layout"
    path.write_text(json.dumps(report))
    assert capture_model.qualified_capture(tmp_path, identity) == ("final", None)


@pytest.mark.parametrize("payload", [[], {"state": "passed", "gates": None}])
def test_invalid_qualification_document_shape_leaves_actual_embedding_available(tmp_path, payload):
    identity = {"schema": 1, "device": "NPU"}
    if isinstance(payload, dict):
        payload["identity"] = identity
    (tmp_path / "qualification-NPU.json").write_text(json.dumps(payload))
    assert capture_model.qualified_capture(tmp_path, identity) == ("final", None)


@pytest.mark.parametrize("compiler", ["DRIVER", ["DRIVER"]])
def test_compiler_policy_is_part_of_qualified_identity(tmp_path, compiler):
    identity = {"schema": 1, "device": "NPU", "compiler": "default"}
    report = {
        "identity": identity,
        "compiler": compiler,
        "state": "passed",
        "selected_level": "full",
        "capture_manifest_sha256": capture_model.manifest_for("full")["sha256"],
        "gates": {name: {"passed": True} for name in ("encoder", "retrieval", "coexistence")},
    }
    (tmp_path / "qualification-NPU.json").write_text(json.dumps(report))
    assert capture_model.qualified_capture(tmp_path, identity) == ("final", None)


def test_driver_compiler_cannot_replace_default_final_baseline(tmp_path):
    identity = {"schema": 1, "device": "NPU", "compiler": "default"}
    report = {
        "identity": {**identity, "compiler": "DRIVER"},
        "compiler": "DRIVER",
        "state": "passed",
        "selected_level": "final",
        "capture_manifest_sha256": capture_model.manifest_for("final")["sha256"],
        "gates": {name: {"passed": True} for name in ("encoder", "retrieval", "coexistence")},
    }
    (tmp_path / "qualification-NPU.json").write_text(json.dumps(report))
    assert capture_model.qualified_capture(tmp_path, identity) == ("final", None)
