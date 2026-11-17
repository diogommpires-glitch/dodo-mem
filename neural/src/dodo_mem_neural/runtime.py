"""Fixed-shape OpenVINO inference, serialised across background and interactive work."""

import itertools
import threading
from concurrent.futures import Future
from pathlib import Path
from queue import PriorityQueue

import numpy as np
import openvino as ov
from openvino.properties.intel_npu import CompilerType, compiler_type
from tokenizers import Tokenizer

from .artifacts import CANDIDATES, atomic_json, sha256, verify
from .capture_model import (
    TOKEN_INPUT,
    instrument,
    qualification_identity,
    qualified_capture,
    remove_taps,
    select_token,
    validate_config,
    validate_graph,
)
from .resources import DEFAULT_CACHE, check_resources

_QUEUE = PriorityQueue()
_ORDER = itertools.count()
_CORE = None


def _serve_queue():
    while True:
        _, _, future, operation = _QUEUE.get()
        if future.set_running_or_notify_cancel():
            try:
                future.set_result(operation())
            except Exception as error:  # noqa: BLE001 - propagate across the executor boundary.
                # A queue boundary must resolve every submitted future, including I/O failures.
                future.set_exception(error)
        _QUEUE.task_done()
        del operation, future


threading.Thread(target=_serve_queue, name="dodo-neural-inference", daemon=True).start()


def _submit(operation, priority=10):
    future = Future()
    _QUEUE.put((priority, next(_ORDER), future, operation))
    return future.result(timeout=180)


def normalise(values):
    values = np.asarray(values, dtype=np.float32)
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    if not np.all(np.isfinite(values)) or np.any(norms < 1e-12):
        raise ValueError("Embedding contains a non-finite or zero vector.")
    return values / norms


def pool_embeddings(hidden, mask, dimensions=384, *, last_token=False):
    if hidden.ndim == 2:
        pooled = hidden
    elif last_token:
        positions = np.arange(mask.shape[1])[None, :]
        last = np.max(np.where(mask, positions, -1), axis=1)
        pooled = hidden[np.arange(hidden.shape[0]), last]
    else:
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden.astype(np.float32) * weights).sum(1) / np.maximum(weights.sum(1), 1)
    return normalise(pooled[:, :dimensions])


class NeuralRuntime:
    def __init__(
        self,
        name="e5",
        *,
        device="NPU",
        cache=DEFAULT_CACHE,
        observer=None,
        activation_home=None,
        capture_level=None,
        capture_compiler=None,
    ):
        if device not in {"NPU", "CPU"}:
            raise ValueError(f"Explicit NPU or CPU device required, got {device!r}.")
        if capture_compiler not in (None, "DRIVER") or (
            capture_compiler is not None
            and (name != "e5" or device != "NPU" or capture_level not in ("full", "blocks"))
        ):
            raise ValueError(
                "Explicit DRIVER compiler requires FULL or BLOCKS qualification on NPU E5."
            )
        check_resources(cache=cache)
        self.name, self.device = name, device
        self.observer = observer if name == "e5" else None
        self.capture_level = "final"
        self.capture_compiler = capture_compiler
        self.capture_reason = "qualification_unavailable"
        self.capture_manifest = None
        self.openvino_version = ov.__version__
        self.target = Path(cache) / "models" / name
        self.manifest = verify(self.target)
        self.manifest_hash = sha256(self.target / "download-manifest.json")
        self.tokenizer = Tokenizer.from_file(str(self.target / "tokenizer.json"))
        self.length = 512 if name == "reranker" else 256
        self.tokenizer.enable_truncation(self.length)
        pad_id = self.tokenizer.token_to_id("<pad>")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("[PAD]")
        if pad_id is None:
            pad_id = self.tokenizer.token_to_id("<|endoftext|>")
        if pad_id is None:
            raise ValueError("Pinned tokenizer has no recognised padding token.")
        self.tokenizer.enable_padding(length=self.length, pad_id=pad_id)
        core = ov.Core()
        stem = CANDIDATES[name][2]
        model = core.read_model(str(self.target / (stem + ".xml")))
        if name != "qwen":
            # The publishers' IR is FP32; qualify compact NPU weights as FP16 first.
            fp16 = Path(cache) / "prepared" / name / "model.xml"
            fp16.parent.mkdir(parents=True, exist_ok=True)
            provenance = fp16.parent / "manifest.json"
            source_hash = sha256(self.target / "download-manifest.json")
            if fp16.exists():
                import json

                identity = json.loads(provenance.read_text(encoding="utf-8"))
                if identity != {
                    "source": source_hash,
                    "openvino": ov.__version__,
                    "transform": "compress_to_fp16",
                    "xml": sha256(fp16),
                    "bin": sha256(fp16.with_suffix(".bin")),
                }:
                    raise ValueError("Prepared FP16 artifact provenance/hash mismatch.")
            else:
                check_resources(additional_disk=260 * 1024**2, cache=cache)
                ov.save_model(model, str(fp16), compress_to_fp16=True)
                atomic_json(
                    provenance,
                    {
                        "source": source_hash,
                        "openvino": ov.__version__,
                        "transform": "compress_to_fp16",
                        "xml": sha256(fp16),
                        "bin": sha256(fp16.with_suffix(".bin")),
                    },
                )
            model = core.read_model(str(fp16))
            self.effective_hash = sha256(provenance)
        else:
            self.effective_hash = sha256(self.target / "download-manifest.json")
        model.reshape({port.any_name: [1, self.length] for port in model.inputs})
        _submit(self._initialise_device)
        architecture = None
        if name == "e5":
            architecture = {
                "verified": False,
                "blocks": 12,
                "features": 384,
                "heads": 12,
                "head_features": 32,
                "ff_features": 1536,
                "tokens": self.length,
                "pooling": "masked_mean",
                "normalisation": "l2",
            }
            try:
                validate_config(self.target / "config.json")
                validate_graph(model)
            except (RuntimeError, ValueError):
                if capture_level is not None:
                    raise
                self.capture_reason = "source_contract_unavailable"
            else:
                architecture["verified"] = True
                if capture_level is not None:
                    self.capture_level = capture_level
                elif activation_home is not None:
                    self.capture_level, self.capture_compiler = qualified_capture(
                        Path(activation_home), qualification_identity(self)
                    )
                self.capture_reason = (
                    "qualification_override" if capture_level is not None else "qualified"
                )
                if self.capture_level == "final" and capture_level is None:
                    self.capture_reason = "final_only_or_qualification_unavailable"
                if self.capture_compiler and compiler_type() not in self.supported_properties:
                    if capture_level is not None:
                        raise ValueError(
                            "NPU_COMPILER_TYPE is unavailable on the selected NPU plugin."
                        )
                    self.capture_level, self.capture_compiler = "final", None
                    self.capture_reason = "compiler_policy_unavailable"
        compiled_dir = Path(cache) / "compiler" / name / device
        compiled_dir.mkdir(parents=True, exist_ok=True)
        properties = {"PERFORMANCE_HINT": "LATENCY"}
        if self.capture_compiler == "DRIVER":
            properties[compiler_type()] = CompilerType.DRIVER
        if device == "CPU":
            properties.update(INFERENCE_NUM_THREADS=2, NUM_STREAMS=1, CACHE_DIR=str(compiled_dir))
        if name == "e5":
            try:
                self.capture_manifest = instrument(model, self.capture_level)
                _submit(lambda: self._compile(model, properties))
            except (RuntimeError, ValueError):
                if (
                    self.capture_level == "final" and self.capture_compiler is None
                ) or capture_level is not None:
                    raise
                # A visual variant may degrade only to the same device's original output.
                remove_taps(model)
                self.capture_level = "final"
                self.capture_compiler = None
                properties.pop(compiler_type(), None)
                self.capture_reason = "instrumented_variant_unavailable"
                self.capture_manifest = instrument(model, "final")
                _submit(lambda: self._compile(model, properties))
        else:
            _submit(lambda: self._compile(model, properties))
        self.model_info = {
            "name": self.name,
            "revision": self.manifest["revision"],
            "space": self.space_version,
            "device": self.execution_devices,
            "level": self.capture_level,
            "compiler": self.capture_compiler or "default",
            "reason": self.capture_reason,
            "architecture": architecture,
            "manifest_sha256": self.manifest_hash,
            "effective_hash": self.effective_hash,
            "openvino": self.openvino_version,
            "capture_manifest": self.capture_manifest,
        }

    @property
    def space_version(self):
        pooling = "last" if self.name == "qwen" else "masked-mean"
        return (
            f"{self.manifest['repository']}@{self.manifest['revision']}:384:{pooling}:"
            f"prefix-{self.name}:tokens{self.length}:{self.effective_hash}:v1"
        )

    def _initialise_device(self):
        # Level Zero initialisation and inference must use the same owning thread.
        global _CORE
        if _CORE is None:
            _CORE = ov.Core()
        self.core = _CORE
        if self.device not in self.core.available_devices:
            raise RuntimeError(f"{self.device} unavailable: {self.core.available_devices}")
        supported = {
            str(value) for value in self.core.get_property(self.device, "SUPPORTED_PROPERTIES")
        }
        self.supported_properties = supported
        self.device_info = {
            "plugins": {
                name: {"build": value.build_number, "description": value.description}
                for name, value in self.core.get_versions(self.device).items()
            }
        }
        keys = {
            "FULL_DEVICE_NAME",
            "DEVICE_UUID",
            "DEVICE_ARCHITECTURE",
            "NPU_COMPILER_TYPE",
            "NPU_COMPILER_VERSION",
        }
        keys.update(key for key in supported if "DRIVER_VERSION" in key)
        for key in sorted(keys):
            if key in supported:
                self.device_info[key] = str(self.core.get_property(self.device, key))

    def _compile(self, model, properties):
        self.compiled = self.core.compile_model(model, self.device, properties)
        execution = self.compiled.get_property("EXECUTION_DEVICES")
        self.execution_devices = [execution] if isinstance(execution, str) else list(execution)
        if not all(value.startswith(self.device) for value in self.execution_devices):
            raise RuntimeError(f"Unexpected execution devices: {self.execution_devices}")
        self.request = self.compiled.create_infer_request()
        self.output = (
            self.compiled.output("last_hidden_state")
            if self.name == "e5"
            else self.compiled.output(0)
        )
        self.capture_ports = {
            name: self.compiled.output(name)
            for name in (self.capture_manifest or {}).get("outputs", {})
        }

    def close(self):
        def release():
            self.capture_ports.clear()
            self.output = None
            self.request = None
            self.compiled = None
            self.core = None

        _submit(release)

    def _observe(self, method, *args):
        if self.observer is None:
            return None
        try:
            return getattr(self.observer, method)(*args)
        except Exception:  # noqa: BLE001 - an optional viewer must never fail actual inference.
            if method == "complete":
                self._observe("fail", args[0], "observer_unavailable")
            return None

    def _complete_capture(self, ticket, result, hidden, mask, selected):
        try:
            if (self.model_info.get("architecture") or {}).get("verified") is False:
                self._observe("fail", ticket, "source_contract_unavailable")
                return
            tensors = {}
            if ticket["valid_token"]:
                tensors = {
                    name: np.array(result[port], dtype=np.float32, copy=True).reshape(-1)
                    for name, port in self.capture_ports.items()
                }
                if "block.11" not in tensors:
                    tensors["block.11"] = hidden[0, selected].astype(np.float32, copy=True)
                weights = mask[..., None].astype(np.float32)
                pooled = (hidden.astype(np.float32) * weights).sum(1) / weights.sum(1)
                tensors["pooled"] = pooled[0].copy()
                tensors["output"] = normalise(pooled)[0]
            self._observe("complete", ticket, tensors, self.capture_level, self.model_info)
        except Exception:  # noqa: BLE001 - capture extraction is isolated from the encoder result.
            self._observe("fail", ticket, "capture_unavailable")

    def _infer(self, encoding, *, interactive):
        arrays = {
            "input_ids": encoding.ids,
            "attention_mask": encoding.attention_mask,
            "token_type_ids": encoding.type_ids,
            "position_ids": list(range(self.length)),
        }

        def infer():
            mask = np.asarray([encoding.attention_mask], dtype=np.int64)
            ticket = self._observe("begin", mask, interactive, self.model_info)
            ticket = dict(ticket) if isinstance(ticket, dict) else None
            selected, valid = select_token(mask, ticket.get("token_index") if ticket else None)
            if ticket is not None:
                ticket["valid_token"] = valid
                if ticket.get("token_index") is None:
                    ticket["token_index"] = selected
            encoded = {
                port.any_name: (
                    np.asarray([selected], dtype=np.int64)
                    if port.any_name == TOKEN_INPUT
                    else np.asarray([arrays[port.any_name]], dtype=np.int64)
                )
                for port in self.compiled.inputs
            }
            inference = self._observe("inference_started", self.model_info)
            try:
                # Shared output views avoid copying dormant visual taps into Python arrays.
                result = self.request.infer(encoded, share_inputs=False, share_outputs=True)
            except Exception:  # Preserve the original failure across the queue.
                self._observe("inference_finished", inference, True)
                if ticket is not None:
                    self._observe("fail", ticket, "inference_failed")
                raise
            self._observe("inference_finished", inference, False)
            hidden = np.array(result[self.output], copy=True)
            if ticket is not None:
                self._complete_capture(ticket, result, hidden, mask, selected)
            return hidden

        return _submit(infer, priority=0 if interactive else 10)

    def encode(self, texts, query=False):
        outputs = []
        for text in texts:
            if self.name == "e5":
                text = ("query: " if query else "passage: ") + text
            encoding = self.tokenizer.encode(text)
            hidden = self._infer(encoding, interactive=query)
            outputs.append(
                pool_embeddings(
                    hidden, np.array([encoding.attention_mask]), last_token=self.name == "qwen"
                )[0]
            )
        return np.asarray(outputs, dtype=np.float32).reshape((-1, 384))

    def rerank(self, query, passages):
        if self.name != "reranker":
            raise ValueError("Reranking requires the pinned reranker.")
        return [
            float(self._infer(self.tokenizer.encode(query, passage), interactive=True).ravel()[0])
            for passage in passages
        ]
