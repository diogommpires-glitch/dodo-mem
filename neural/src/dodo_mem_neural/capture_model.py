"""Bounded token taps on the pinned E5 graph, without changing its embedding output."""

import hashlib
import json

import numpy as np
import openvino.opset13 as ops

SCHEMA_VERSION = 1
LEVELS = ("full", "blocks", "final")
TOKEN_INPUT = "capture_token_index"


def capture_sources(level):
    if level not in LEVELS:
        raise ValueError(f"Capture level must be one of {LEVELS}, got {level!r}.")
    if level == "final":
        return {}
    sources = {
        "embedding": {
            "operation": "__module.embeddings.LayerNorm/aten::layer_norm/Add",
            "shape": [1, 256, 384],
            "axis": 1,
        }
    }
    for layer in range(12):
        prefix = f"__module.encoder.layer.{layer}"
        sources[f"block.{layer}"] = {
            "operation": prefix + ".output.LayerNorm/aten::layer_norm/Add",
            "shape": [1, 256, 384],
            "axis": 1,
        }
    if level == "full":
        for layer in range(12):
            prefix = f"__module.encoder.layer.{layer}"
            sources[f"ff.{layer}"] = {
                "operation": prefix + ".intermediate.intermediate_act_fn/aten::gelu/Gelu",
                "shape": [1, 256, 1536],
                "axis": 1,
            }
            sources[f"heads.{layer}"] = {
                "operation": prefix
                + ".attention.self/aten::scaled_dot_product_attention/ScaledDotProductAttention",
                "shape": [1, 12, 256, 32],
                "axis": 2,
            }
    return sources


def validate_config(path):
    config = json.loads(path.read_text(encoding="utf-8"))
    expected = {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
    }
    for name, value in expected.items():
        if config.get(name) != value:
            raise ValueError(f"E5 config {name} must be {value!r}, got {config.get(name)!r}.")


def validate_graph(model, level="full"):
    sources = capture_sources(level)
    hidden = model.output("last_hidden_state")
    if list(hidden.shape) != [1, 256, 384]:
        raise ValueError(f"E5 last_hidden_state shape must be [1, 256, 384], got {hidden.shape}.")
    nodes = {}
    for node in model.get_ops():
        nodes.setdefault(node.friendly_name, []).append(node)
    selected = {}
    for name, spec in sources.items():
        candidates = nodes.get(spec["operation"], [])
        if len(candidates) != 1:
            raise ValueError(f"Capture source {name} missing or ambiguous: {spec['operation']}.")
        port = candidates[0].output(0)
        if list(port.shape) != spec["shape"]:
            raise ValueError(
                f"Capture source {name} shape must be {spec['shape']}, got {port.shape}."
            )
        if port.element_type.get_type_name() != "f32":
            raise ValueError(f"Capture source {name} must be f32, got {port.element_type}.")
        selected[name] = port
    return selected


def instrument(model, level):
    sources = capture_sources(level)
    selected = validate_graph(model, level)
    if sources:
        token = ops.parameter([1], np.int64, name=TOKEN_INPUT)
        token.output(0).set_names({TOKEN_INPUT})
        model.add_parameters([token])
        for name, port in selected.items():
            tap = ops.gather(port, token, sources[name]["axis"], name="capture/" + name)
            tap.output(0).set_names({name})
            model.add_outputs(tap.output(0))
        model.validate_nodes_and_infer_types()
    return manifest_for(level)


def manifest_for(level):
    sources = {
        name: {**spec, "output_port": 0, "element_type": "f32"}
        for name, spec in capture_sources(level).items()
    }
    manifest = {"schema": SCHEMA_VERSION, "level": level, "outputs": sources}
    manifest["sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return manifest


def remove_taps(model):
    for result in model.get_results():
        if "last_hidden_state" not in result.output(0).get_names():
            model.remove_result(result)
    for parameter in model.get_parameters():
        if TOKEN_INPUT in parameter.output(0).get_names():
            model.remove_parameter(parameter)
    model.validate_nodes_and_infer_types()


def select_token(mask, requested):
    mask = np.asarray(mask).reshape(-1)
    unmasked = np.flatnonzero(mask)
    safe = int(unmasked[0]) if unmasked.size else 0
    if requested is None:
        return safe, bool(unmasked.size)
    valid = (
        isinstance(requested, (int, np.integer))
        and not isinstance(requested, bool)
        and 0 <= requested < mask.size
        and bool(mask[requested])
    )
    return (int(requested), True) if valid else (safe, False)


def qualification_identity(runtime):
    return {
        "schema": SCHEMA_VERSION,
        "effective_hash": runtime.effective_hash,
        "manifest_sha256": runtime.manifest_hash,
        "openvino": runtime.openvino_version,
        "device": runtime.device,
        "device_info": runtime.device_info,
        "capture_contract_sha256": manifest_for("full")["sha256"],
        "compiler": runtime.capture_compiler or "default",
    }


def qualified_capture(home, identity):
    path = home / f"qualification-{identity['device']}.json"
    if not path.exists():
        return "final", None
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return "final", None
    if not isinstance(report, dict):
        return "final", None
    gates = report.get("gates")
    if not isinstance(gates, dict) or any(
        not isinstance(gates.get(name), dict) or gates[name].get("passed") is not True
        for name in ("encoder", "retrieval", "coexistence")
    ):
        return "final", None
    compiler = report.get("compiler")
    if compiler not in ("default", "DRIVER") or (
        compiler == "DRIVER" and identity["device"] != "NPU"
    ):
        return "final", None
    if (
        report.get("identity") != {**identity, "compiler": compiler}
        or report.get("state") != "passed"
    ):
        return "final", None
    level = report.get("selected_level")
    if (
        level not in LEVELS
        or (level == "final" and compiler != "default")
        or report.get("capture_manifest_sha256") != manifest_for(level)["sha256"]
    ):
        return "final", None
    return level, None if compiler == "default" else "DRIVER"
