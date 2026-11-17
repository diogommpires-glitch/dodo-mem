"""Bounded, signed parameter samples from the pinned prepared E5 linear projections."""

import copy
import ctypes
import hashlib
import heapq
import json
import math
import os
import struct
import threading
import xml.etree.ElementTree as ET
from pathlib import Path

from .artifacts import CANDIDATES, WEIGHT_SHA256
from .resources import DEFAULT_CACHE

PREPARED_E5 = DEFAULT_CACHE / "prepared/e5"
PROJECTIONS = {
    "query": ("attention.self.query", 384, 384, "query_linear_before_bias"),
    "key": ("attention.self.key", 384, 384, "key_linear_before_bias"),
    "value": ("attention.self.value", 384, 384, "value_linear_before_bias"),
    "attention_output": ("attention.output.dense", 384, 384, "attention_output_linear_before_bias"),
    "ff_up": ("intermediate.dense", 384, 1536, "ff_up_linear_before_bias_before_gelu"),
    "ff_down": (
        "output.dense",
        1536,
        384,
        "ff_down_linear_before_bias_before_residual_and_layernorm",
    ),
}


def _port_shape(node, side, port_id=None):
    ports = node.findall(side + "/port")
    if port_id is not None:
        ports = [port for port in ports if port.get("id") == port_id]
    if len(ports) != 1:
        raise ValueError(f"Projection {node.get('name')} must have one {side} port.")
    return [int(value.text) for value in ports[0].findall("dim")]


def _projection_descriptors(xml, _bin_size):
    layer_list = xml.findall("layers/layer")
    layers = {node.get("id"): node for node in layer_list}
    if len(layers) != len(layer_list) or None in layers:
        raise ValueError("Prepared XML has missing or duplicate operation IDs.")
    matmuls = [node for node in layer_list if node.get("type") == "MatMul"]
    by_name = {node.get("name"): node for node in matmuls}
    if len(matmuls) != 72 or len(by_name) != 72:
        raise ValueError("Pinned E5 requires exactly 72 distinct linear MatMul projections.")
    edges = {}
    for edge in xml.findall("edges/edge"):
        target = edge.get("to-layer"), edge.get("to-port")
        if target in edges:
            raise ValueError("Prepared XML has duplicate input edges.")
        edges[target] = edge.get("from-layer"), edge.get("from-port")

    def input_node(node, port):
        source = edges.get((node.get("id"), str(port)))
        if source is None or source[0] not in layers:
            raise ValueError(f"Projection {node.get('name')} has a missing input {port}.")
        return layers[source[0]]

    result = {}
    for block in range(12):
        prefix = f"__module.encoder.layer.{block}."
        for projection, (suffix, input_dim, output_dim, semantics) in PROJECTIONS.items():
            name = prefix + suffix + "/aten::linear/MatMul"
            matmul = by_name.get(name)
            if matmul is None:
                raise ValueError(f"Pinned E5 projection is missing: {name}.")
            if matmul.find("data") is None or matmul.find("data").attrib != {
                "transpose_a": "false",
                "transpose_b": "true",
            }:
                raise ValueError(f"Projection {name} must use X @ W.T.")
            activation = input_node(matmul, 0)
            converted = input_node(matmul, 1)
            constant = input_node(converted, 0)
            weight_name = f"self.encoder.layer.{block}.{suffix}.weight"
            if (
                converted.get("type") != "Convert"
                or converted.get("name") != weight_name
                or converted.find("data") is None
                or converted.find("data").get("destination_type") != "f32"
                or constant.get("type") != "Const"
                or constant.get("name") != weight_name + "_compressed"
            ):
                raise ValueError(f"Projection {name} must use its pinned f16 Const -> f32 Convert.")
            data = constant.find("data")
            if data is None or data.get("element_type") != "f16":
                raise ValueError(f"Projection {name} has no f16 parameter data.")
            shape = [int(value) for value in data.get("shape", "").split(",")]
            if (
                shape != [output_dim, input_dim]
                or _port_shape(constant, "output") != shape
                or _port_shape(converted, "input") != shape
                or _port_shape(converted, "output") != shape
                or _port_shape(matmul, "input", "1") != shape
            ):
                raise ValueError(
                    f"Projection {name} must store [output,input]={output_dim, input_dim}."
                )
            activation_shape = _port_shape(activation, "output")
            output_shape = _port_shape(matmul, "output")
            if len(activation_shape) != 3 or activation_shape[-1] != input_dim:
                raise ValueError(f"Projection {name} has the wrong input feature axis.")
            if len(output_shape) != 3 or output_shape[-1] != output_dim:
                raise ValueError(f"Projection {name} has the wrong output feature axis.")
            if _port_shape(matmul, "input", "0") != activation_shape:
                raise ValueError(f"Projection {name} has inconsistent declared input axes.")
            if projection in {"query", "key", "value"}:
                source_name = (
                    "__module.embeddings.LayerNorm/aten::layer_norm/Add"
                    if block == 0
                    else f"__module.encoder.layer.{block - 1}.output.LayerNorm/aten::layer_norm/Add"
                )
                source_semantics = (
                    "embedding_layernorm_features"
                    if block == 0
                    else "previous_block_output_layernorm_features"
                )
            elif projection == "attention_output":
                source_name = prefix + "attention.self/aten::reshape/Reshape"
                source_semantics = "concatenated_head_context_features"
            elif projection == "ff_up":
                source_name = prefix + "attention.output.LayerNorm/aten::layer_norm/Add"
                source_semantics = "attention_output_layernorm_features"
            else:
                source_name = prefix + "intermediate.intermediate_act_fn/aten::gelu/Gelu"
                source_semantics = "gelu_features"
            if activation.get("name") != source_name:
                raise ValueError(f"Projection {name} does not consume its pinned source tensor.")
            result[block, projection] = {
                "block": block,
                "projection": projection,
                "operator_id": matmul.get("id"),
                "operation": name,
                "input_dim": input_dim,
                "output_dim": output_dim,
                "orientation": "output_input",
                "total_weight_count": input_dim * output_dim,
                "source": {
                    "operator_id": activation.get("id"),
                    "operation": source_name,
                    "semantics": source_semantics,
                },
                "target": {
                    "operator_id": matmul.get("id"),
                    "operation": name,
                    "semantics": semantics,
                },
                "provenance": {
                    "constant_id": constant.get("id"),
                    "constant_name": constant.get("name"),
                    "convert_id": converted.get("id"),
                    "convert_operation": converted.get("name"),
                    "offset_bytes": int(data.get("offset", "-1")),
                    "size_bytes": int(data.get("size", "-1")),
                    "matrix_layout": "output_input",
                    "transpose_a": False,
                    "transpose_b": True,
                },
            }
    return result


def _metadata_bytes(path, maximum):
    if path.resolve(strict=True) != path.absolute():
        raise ValueError("Prepared metadata cannot be redirected.")
    with path.open("rb") as stream:
        contents = stream.read(maximum + 1)
    if len(contents) > maximum:
        raise ValueError(f"Prepared {path.name} exceeds its bounded metadata size.")
    return contents


def static_model_info(cache=DEFAULT_CACHE):
    """Identify the installed E5 definition without loading or compiling its weights."""
    cache = Path(cache)
    target = cache / "models/e5"
    raw = _metadata_bytes(target / "download-manifest.json", 65536)
    manifest = json.loads(raw)
    repository, revision, stem, licence = CANDIDATES["e5"]
    if not isinstance(manifest, dict) or (
        manifest.get("repository"),
        manifest.get("revision"),
        manifest.get("licence"),
    ) != (repository, revision, licence):
        raise ValueError("Static E5 metadata differs from the selected pinned model.")
    entries = manifest.get("files")
    if not isinstance(entries, list) or any(
        not isinstance(item, dict) or not isinstance(item.get("path"), str) for item in entries
    ):
        raise ValueError("Static E5 artifact metadata is invalid.")
    files = {entry.get("path"): entry for entry in entries}
    if (
        len(files) != len(entries)
        or files.get(stem + ".bin", {}).get("sha256") != WEIGHT_SHA256["e5"]
    ):
        raise ValueError("Static E5 metadata has duplicate files or an unpinned weight hash.")
    config_raw = _metadata_bytes(target / "config.json", 65536)
    config_entry = files.get("config.json", {})
    if (
        config_entry.get("size") != len(config_raw)
        or config_entry.get("sha256") != hashlib.sha256(config_raw).hexdigest()
    ):
        raise ValueError("Static E5 config differs from its source manifest.")
    config = json.loads(config_raw)
    expected = {
        "hidden_size": 384,
        "num_hidden_layers": 12,
        "num_attention_heads": 12,
        "intermediate_size": 1536,
        "hidden_act": "gelu",
    }
    if not isinstance(config, dict) or any(
        config.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Static E5 config differs from the pinned architecture.")
    prepared_raw = _metadata_bytes(cache / "prepared/e5/manifest.json", 65536)
    prepared = json.loads(prepared_raw)
    if (
        not isinstance(prepared, dict)
        or not isinstance(prepared.get("openvino"), str)
        or not prepared["openvino"]
    ):
        raise ValueError("Static E5 prepared provenance is invalid.")
    effective_hash = hashlib.sha256(prepared_raw).hexdigest()
    return {
        "name": "e5",
        "revision": revision,
        "space": (
            f"{repository}@{revision}:384:masked-mean:prefix-e5:tokens256:{effective_hash}:v1"
        ),
        "device": [],
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "effective_hash": effective_hash,
        "openvino": prepared["openvino"],
    }


def _locked_binary(path):
    if os.name != "nt":
        raise OSError("Static parameter verification requires a Windows deny-write/delete lease.")
    import msvcrt
    from ctypes import wintypes

    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel.CreateFileW.restype = wintypes.HANDLE
    kernel.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel.CloseHandle.restype = wintypes.BOOL
    # FILE_SHARE_READ permits other readers, while denying modification and replacement.
    handle = kernel.CreateFileW(str(path), 0x80000000, 0x1, None, 3, 0x80, None)
    if handle == wintypes.HANDLE(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        descriptor = msvcrt.open_osfhandle(handle, os.O_RDONLY | os.O_BINARY)
    except BaseException:
        kernel.CloseHandle(handle)
        raise
    try:
        return os.fdopen(descriptor, "rb", buffering=0)
    except BaseException:
        os.close(descriptor)
        raise


class StaticWeights:
    def __init__(
        self, model_info, *, _prepared=PREPARED_E5, _projection_validator=_projection_descriptors
    ):
        self._lock = threading.RLock()
        self._closed = False
        self._binary = None
        self._failure = None
        self._prepared = Path(_prepared)
        if model_info.get("name") != "e5" or not all(
            isinstance(model_info.get(key), str) and model_info[key]
            for key in ("revision", "space", "effective_hash", "manifest_sha256", "openvino")
        ):
            raise ValueError("Qualified E5 model identity is required for static parameters.")
        manifest_path = self._prepared / "manifest.json"
        xml_path = self._prepared / "model.xml"
        self._bin_path = self._prepared / "model.bin"
        manifest_bytes = _metadata_bytes(manifest_path, 65536)
        self._manifest_hash = hashlib.sha256(manifest_bytes).hexdigest()
        if self._manifest_hash != model_info["effective_hash"]:
            raise ValueError("Prepared manifest identity does not match the qualified encoder.")
        manifest = json.loads(manifest_bytes)
        if not isinstance(manifest, dict) or set(manifest) != {
            "source",
            "openvino",
            "transform",
            "xml",
            "bin",
        }:
            raise ValueError("Prepared manifest must contain the pinned E5 provenance fields.")
        if (
            manifest["source"] != model_info["manifest_sha256"]
            or manifest["openvino"] != model_info["openvino"]
            or manifest["transform"] != "compress_to_fp16"
        ):
            raise ValueError("Prepared transform provenance differs from the qualified encoder.")
        for key in ("xml", "bin"):
            digest = manifest[key]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Prepared {key} SHA256 must be a lowercase hexadecimal digest.")
        xml_bytes = _metadata_bytes(xml_path, 2 * 1024**2)
        self._xml_hash = hashlib.sha256(xml_bytes).hexdigest()
        if self._xml_hash != manifest["xml"]:
            raise ValueError("Prepared XML hash differs from its manifest.")
        if self._bin_path.resolve(strict=True) != self._bin_path.absolute():
            raise ValueError("Prepared BIN cannot be redirected.")
        self._bin_size = self._bin_path.stat().st_size
        self._bin_hash = manifest["bin"]
        try:
            xml = ET.fromstring(xml_bytes)
        except ET.ParseError as error:
            raise ValueError("Prepared XML is malformed.") from error
        self._descriptors = _projection_validator(xml, self._bin_size)
        for descriptor in self._descriptors.values():
            provenance = descriptor["provenance"]
            offset, size = provenance["offset_bytes"], provenance["size_bytes"]
            count = descriptor["input_dim"] * descriptor["output_dim"]
            if (
                offset < 0
                or offset % 2
                or size != count * 2
                or size > 2 * 1024**2
                or offset + size > self._bin_size
            ):
                raise ValueError("Prepared Constant byte span is outside the matrix/file bounds.")
        self._model = {
            "schema": 1,
            "static": True,
            "dtype": "f16",
            "model_revision": model_info["revision"],
            "model_space": model_info["space"],
        }

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Static parameter reader is closed.")

    def _descriptor(self, descriptor):
        result = copy.deepcopy(descriptor)
        result["provenance"].update(
            prepared_manifest_sha256=self._manifest_hash,
            xml_sha256=self._xml_hash,
            bin_sha256=self._bin_hash,
            bin_verified=self._binary is not None,
        )
        return result

    def descriptors(self):
        with self._lock:
            self._check_open()
            return {
                **self._model,
                "projections": [self._descriptor(value) for value in self._descriptors.values()],
            }

    def _verify_binary(self):
        if self._failure:
            raise ValueError(self._failure)
        if self._binary is not None:
            return
        stream = _locked_binary(self._bin_path)
        try:
            if os.fstat(stream.fileno()).st_size != self._bin_size:
                self._failure = "Prepared BIN size changed before verification."
                raise ValueError(self._failure)
            digest = hashlib.sha256()
            while chunk := stream.read(1024**2):
                digest.update(chunk)
            if digest.hexdigest() != self._bin_hash:
                self._failure = "Prepared BIN hash differs from its qualified manifest."
                raise ValueError(self._failure)
        except BaseException:
            stream.close()
            raise
        self._binary = stream

    def weights(self, block, projection, *, limit=512, output_index=None):
        with self._lock:
            self._check_open()
            if type(block) is not int or not 0 <= block < 12:
                raise ValueError("block must be an integer from 0 to 11.")
            if not isinstance(projection, str) or projection not in PROJECTIONS:
                raise ValueError(f"projection must be one of {tuple(PROJECTIONS)}.")
            if type(limit) is not int or not 1 <= limit <= 512:
                raise ValueError("limit must be an integer from 1 to 512.")
            descriptor = self._descriptors.get((block, projection))
            if descriptor is None:
                raise ValueError("Selected projection is absent from this prepared model.")
            input_dim = descriptor["input_dim"]
            if output_index is not None and (
                type(output_index) is not int or not 0 <= output_index < descriptor["output_dim"]
            ):
                raise ValueError(
                    f"output_index must be null or an integer from 0 to {descriptor['output_dim'] - 1}."
                )
            self._verify_binary()
            offset = descriptor["provenance"]["offset_bytes"]
            start = 0 if output_index is None else output_index * input_dim
            count = descriptor["total_weight_count"] if output_index is None else input_dim
            if output_index is None:
                output_dim = descriptor["output_dim"]
                budget = min(limit, count)
                if budget < output_dim:
                    targets = (
                        [index * (output_dim - 1) // (budget - 1) for index in range(budget)]
                        if budget > 1
                        else [output_dim // 2]
                    )
                    quotas = dict.fromkeys(targets, 1)
                else:
                    rounds, extra = divmod(budget, output_dim)
                    quotas = {target: rounds + (target < extra) for target in range(output_dim)}
            else:
                quotas = {output_index: min(limit, input_dim)}
            self._binary.seek(offset + start * 2)
            remaining, position, heap, rows = count * 2, start, [], {}
            while remaining:
                size = min(65536, remaining)
                chunk = self._binary.read(size)
                if len(chunk) != size:
                    raise ValueError("Prepared matrix is truncated within its verified BIN lease.")
                for (weight,) in struct.iter_unpack("<e", chunk):
                    target, source = divmod(position, input_dim)
                    if not math.isfinite(weight):
                        raise ValueError(
                            f"Projection has a non-finite parameter at ({target}, {source})."
                        )
                    candidate = abs(weight), -target, -source, weight
                    quota = quotas.get(target, 0)
                    if quota:
                        if len(heap) < quota:
                            heapq.heappush(heap, candidate)
                        elif candidate > heap[0]:
                            heapq.heapreplace(heap, candidate)
                    if source == input_dim - 1 and heap:
                        rows[target] = sorted(heap, reverse=True)
                        heap = []
                    position += 1
                remaining -= size
            # One strongest edge per selected output before a second edge is shown.
            edges = [
                {"source": -source, "target": -target, "weight": weight}
                for rank in range(max(quotas.values()))
                for row in rows.values()
                if rank < len(row)
                for _, target, source, weight in [row[rank]]
            ]
            return {
                **self._model,
                **self._descriptor(descriptor),
                "edges": edges,
                "shown_count": len(edges),
                "candidate_weight_count": count,
                "output_index": output_index,
                "sampling": "stratified_by_output"
                if output_index is None
                else "top_absolute_incoming_weight",
            }

    def close(self):
        with self._lock:
            if self._binary is not None:
                self._binary.close()
                self._binary = None
            self._closed = True
