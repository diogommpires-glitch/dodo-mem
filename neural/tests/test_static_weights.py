import hashlib
import json
import os
import struct

import pytest

from dodo_mem_neural.static_weights import StaticWeights, static_model_info

MATRIX = [[1, -7, 7, 2], [-9, 3, 0, 9], [0.5, -1, -2, 4]]


def fixture_descriptor(_xml, _bin_size):
    return {
        (0, "query"): {
            "block": 0,
            "projection": "query",
            "operator_id": "3",
            "operation": "fixture.query/MatMul",
            "input_dim": 4,
            "output_dim": 3,
            "orientation": "output_input",
            "total_weight_count": 12,
            "source": {
                "operator_id": "0",
                "operation": "fixture.input",
                "semantics": "input_features",
            },
            "target": {
                "operator_id": "3",
                "operation": "fixture.query/MatMul",
                "semantics": "query_linear_before_bias",
            },
            "provenance": {
                "constant_id": "1",
                "constant_name": "fixture.weight_compressed",
                "convert_id": "2",
                "convert_operation": "fixture.weight",
                "offset_bytes": 8,
                "size_bytes": 24,
                "matrix_layout": "output_input",
                "transpose_a": False,
                "transpose_b": True,
            },
        }
    }


def prepared_fixture(tmp_path, matrix=MATRIX):
    xml = b'<net><layers><layer id="3" name="fixture.query/MatMul" type="MatMul"/></layers><edges/></net>'
    values = [value for row in matrix for value in row]
    binary = b"fixture!" + struct.pack(f"<{len(values)}e", *values)
    manifest = {
        "source": "a" * 64,
        "openvino": "fixture-version",
        "transform": "compress_to_fp16",
        "xml": hashlib.sha256(xml).hexdigest(),
        "bin": hashlib.sha256(binary).hexdigest(),
    }
    (tmp_path / "model.xml").write_bytes(xml)
    (tmp_path / "model.bin").write_bytes(binary)
    raw_manifest = json.dumps(manifest).encode()
    (tmp_path / "manifest.json").write_bytes(raw_manifest)
    return {
        "name": "e5",
        "revision": "fixture-revision",
        "space": "fixture-space",
        "effective_hash": hashlib.sha256(raw_manifest).hexdigest(),
        "manifest_sha256": manifest["source"],
        "openvino": manifest["openvino"],
    }


def reader(tmp_path, matrix=MATRIX, validator=fixture_descriptor):
    info = prepared_fixture(tmp_path, matrix)
    return StaticWeights(info, _prepared=tmp_path, _projection_validator=validator)


def test_static_model_identity_checks_pinned_metadata_without_loading_weights(tmp_path):
    from dodo_mem_neural.artifacts import CANDIDATES, WEIGHT_SHA256

    source = tmp_path / "models/e5"
    prepared = tmp_path / "prepared/e5"
    source.mkdir(parents=True)
    prepared.mkdir(parents=True)
    config = {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
    }
    config_raw = json.dumps(config).encode()
    (source / "config.json").write_bytes(config_raw)
    repository, revision, stem, licence = CANDIDATES["e5"]
    manifest = {
        "repository": repository,
        "revision": revision,
        "licence": licence,
        "files": [
            {
                "path": "config.json",
                "size": len(config_raw),
                "sha256": hashlib.sha256(config_raw).hexdigest(),
            },
            {"path": stem + ".bin", "sha256": WEIGHT_SHA256["e5"]},
        ],
    }
    manifest_raw = json.dumps(manifest).encode()
    (source / "download-manifest.json").write_bytes(manifest_raw)
    (prepared / "manifest.json").write_text(json.dumps({"openvino": "fixture-version"}))
    info = static_model_info(tmp_path)
    assert info["revision"] == revision and info["device"] == []
    assert info["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert info["space"].endswith(info["effective_hash"] + ":v1")
    assert not (prepared / "model.bin").exists()
    config["hidden_size"] = 768
    (source / "config.json").write_text(json.dumps(config))
    with pytest.raises(ValueError, match="config"):
        static_model_info(tmp_path)


def test_descriptors_do_not_read_or_lock_bin_and_do_not_weaken_production_contract(tmp_path):
    value = reader(tmp_path)
    try:
        result = value.descriptors()
        assert result["model_space"] == "fixture-space"
        assert result["projections"][0]["input_dim"] == 4
        assert result["projections"][0]["provenance"]["bin_verified"] is False
        (tmp_path / "model.bin").write_bytes(b"changed!" + bytes(24))
        with pytest.raises(ValueError, match="72"):
            StaticWeights(prepared_fixture(tmp_path), _prepared=tmp_path)
    finally:
        value.close()


@pytest.mark.skipif(os.name != "nt", reason="Production lease requires Windows sharing semantics.")
@pytest.mark.parametrize(("output_dim", "limit"), [(384, 512), (1536, 512), (384, 7), (384, 1)])
def test_stratified_sample_spans_outputs_and_keeps_actual_signed_parameters(
    tmp_path, output_dim, limit
):
    matrix = [[1, -2, 3, 4] for _ in range(output_dim)]
    matrix[-1] = [-1000, 1000, 1000, 1000]

    def descriptor(xml, size):
        result = fixture_descriptor(xml, size)
        projection = result[(0, "query")]
        projection.update(output_dim=output_dim, total_weight_count=output_dim * 4)
        projection["provenance"]["size_bytes"] = output_dim * 8
        return result

    value = reader(tmp_path, matrix, descriptor)
    try:
        result = value.weights(0, "query", limit=limit)
        assert result == value.weights(0, "query", limit=limit)
        edges = result["edges"]
        assert len(edges) == limit
        assert len({(edge["source"], edge["target"]) for edge in edges}) == limit
        targets = [edge["target"] for edge in edges]
        assert len(set(targets)) == min(limit, output_dim)
        if limit >= output_dim:
            assert targets[:output_dim] == list(range(output_dim))
        elif limit > 1:
            assert targets == [index * (output_dim - 1) // (limit - 1) for index in range(limit)]
        else:
            assert targets == [output_dim // 2]
        for edge in edges:
            assert edge["weight"] == matrix[edge["target"]][edge["source"]]
        assert result["candidate_weight_count"] == output_dim * 4
        assert result["sampling"] == "stratified_by_output"
    finally:
        value.close()


@pytest.mark.skipif(os.name != "nt", reason="Production lease requires Windows sharing semantics.")
def test_signed_top_weights_keep_output_input_orientation_and_deterministic_ties(
    tmp_path, monkeypatch
):
    value = reader(tmp_path)
    try:
        metadata = value.descriptors()
        metadata["projections"][0]["provenance"]["offset_bytes"] = 999
        real_sha256 = hashlib.sha256
        verifications = []

        def count_verification(*args):
            verifications.append(True)
            return real_sha256(*args)

        monkeypatch.setattr(hashlib, "sha256", count_verification)
        result = value.weights(0, "query", limit=4)
        assert result["edges"] == [
            {"source": 1, "target": 0, "weight": -7.0},
            {"source": 0, "target": 1, "weight": -9.0},
            {"source": 3, "target": 2, "weight": 4.0},
            {"source": 2, "target": 0, "weight": 7.0},
        ]
        assert result["shown_count"] == 4 and result["total_weight_count"] == 12
        assert result["sampling"] == "stratified_by_output"
        assert result["provenance"]["bin_verified"] is True
        incoming = value.weights(0, "query", limit=512, output_index=2)
        assert incoming["edges"] == [
            {"source": 3, "target": 2, "weight": 4.0},
            {"source": 2, "target": 2, "weight": -2.0},
            {"source": 1, "target": 2, "weight": -1.0},
            {"source": 0, "target": 2, "weight": 0.5},
        ]
        assert incoming["sampling"] == "top_absolute_incoming_weight"
        assert incoming["candidate_weight_count"] == 4
        assert verifications == [True]
    finally:
        value.close()


@pytest.mark.skipif(os.name != "nt", reason="Production lease requires Windows sharing semantics.")
def test_verified_lease_prevents_write_delete_and_releases_on_close(tmp_path):
    value = reader(tmp_path)
    binary = tmp_path / "model.bin"
    value.weights(0, "query", limit=1)
    try:
        with pytest.raises(PermissionError):
            binary.write_bytes(b"changed")
        with pytest.raises(PermissionError):
            binary.unlink()
        assert value.weights(0, "query", limit=1)["edges"][0]["weight"] == -9
    finally:
        value.close()
    binary.write_bytes(b"released")
    assert binary.read_bytes() == b"released"
    with pytest.raises(RuntimeError, match="closed"):
        value.weights(0, "query")


@pytest.mark.skipif(os.name != "nt", reason="Production lease requires Windows sharing semantics.")
def test_tampered_bin_is_rejected_before_any_edges_and_failure_releases_lease(tmp_path):
    value = reader(tmp_path)
    binary = tmp_path / "model.bin"
    binary.write_bytes(b"tampered" + bytes(24))
    try:
        with pytest.raises(ValueError, match="BIN hash"):
            value.weights(0, "query")
        binary.write_bytes(b"no leaked lease")
    finally:
        value.close()


@pytest.mark.skipif(os.name != "nt", reason="Production lease requires Windows sharing semantics.")
def test_non_finite_selected_matrix_is_never_returned(tmp_path):
    value = reader(tmp_path, [[float("nan"), -7, 7, 2], *MATRIX[1:]])
    try:
        with pytest.raises(ValueError, match="non-finite"):
            value.weights(0, "query", limit=1)
    finally:
        value.close()


@pytest.mark.parametrize("change", ["manifest", "xml"])
def test_tampered_metadata_is_rejected(tmp_path, change):
    info = prepared_fixture(tmp_path)
    path = tmp_path / ("manifest.json" if change == "manifest" else "model.xml")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest identity|XML hash"):
        StaticWeights(info, _prepared=tmp_path, _projection_validator=fixture_descriptor)


def test_out_of_bounds_constant_is_rejected_before_bin_read(tmp_path):
    def invalid_span(xml, size):
        result = fixture_descriptor(xml, size)
        result[(0, "query")]["provenance"]["offset_bytes"] = size
        return result

    with pytest.raises(ValueError, match="bounds"):
        reader(tmp_path, validator=invalid_span)


def test_malformed_xml_is_a_validation_error(tmp_path):
    info = prepared_fixture(tmp_path)
    xml = b"<net><layers>"
    (tmp_path / "model.xml").write_bytes(xml)
    manifest_path = tmp_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["xml"] = hashlib.sha256(xml).hexdigest()
    raw = json.dumps(manifest).encode()
    manifest_path.write_bytes(raw)
    info["effective_hash"] = hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError, match="XML is malformed"):
        StaticWeights(info, _prepared=tmp_path, _projection_validator=fixture_descriptor)


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((True, "query", 1, None), "block"),
        ((12, "query", 1, None), "block"),
        ((0, "unknown", 1, None), "projection"),
        ((0, "query", 0, None), "limit"),
        ((0, "query", 513, None), "limit"),
        ((0, "query", True, None), "limit"),
        ((0, "query", 1, -1), "output_index"),
        ((0, "query", 1, 3), "output_index"),
        ((0, "query", 1, True), "output_index"),
    ],
)
def test_invalid_selection_is_rejected_before_verifying_bin(tmp_path, arguments, message):
    value = reader(tmp_path)
    (tmp_path / "model.bin").write_bytes(b"tampered" + bytes(24))
    try:
        block, projection, limit, output_index = arguments
        with pytest.raises(ValueError, match=message):
            value.weights(block, projection, limit=limit, output_index=output_index)
    finally:
        value.close()
