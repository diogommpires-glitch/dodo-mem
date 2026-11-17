"""Read bounded graph metadata from the authenticated local neural broker."""

import json
import math
import re
import threading
from datetime import datetime
from pathlib import Path
from urllib.parse import urlencode

import httpx

from .configuration import runtime_path

NEURAL_HOME = runtime_path("neural")
ENDPOINT = "http://127.0.0.1:8770"
MAX_RESPONSE_BYTES = 4 * 1024**2
CAPTURE_LEVELS = {"full", "blocks", "final"}
PROJECTION_DIMS = {
    "query": (384, 384),
    "key": (384, 384),
    "value": (384, 384),
    "attention_output": (384, 384),
    "ff_up": (384, 1536),
    "ff_down": (1536, 384),
}
TENSOR_SIZES = {"embedding": 384, "pooled": 384, "output": 384}
for _block in range(12):
    TENSOR_SIZES.update({f"block.{_block}": 384, f"heads.{_block}": 384, f"ff.{_block}": 1536})


class NeuralError(OSError):
    pass


def _validate_telemetry(value):
    if value is None:
        return
    if not isinstance(value, dict):
        raise NeuralError("Local neural telemetry is invalid.")
    device = value.get("device")
    if not isinstance(device, list) or any(not isinstance(name, str) for name in device):
        raise NeuralError("Local neural telemetry device is invalid.")
    for key in ("active_inferences", "completed_inferences", "failed_inferences"):
        if type(value.get(key)) is not int or value[key] < 0:
            raise NeuralError("Local neural telemetry counts are invalid.")
    for key in (
        "inference_busy_percent",
        "hardware_utilization_percent",
        "last_inference_ms",
        "last_completed_at",
        "window_seconds",
    ):
        number = value.get(key)
        if number is None and key in {
            "hardware_utilization_percent",
            "last_inference_ms",
            "last_completed_at",
        }:
            continue
        if type(number) not in {int, float} or not math.isfinite(number) or number < 0:
            raise NeuralError("Local neural telemetry measurements are invalid.")
        if (key.endswith("percent") and number > 100) or (key == "window_seconds" and number == 0):
            raise NeuralError("Local neural telemetry measurement bounds are invalid.")
    reason = value.get("hardware_utilization_reason")
    if reason is not None and (not isinstance(reason, str) or len(reason) > 1000):
        raise NeuralError("Local neural telemetry hardware status is invalid.")


class NeuralClient:
    def __init__(self, home: Path = NEURAL_HOME, *, transport=None):
        self.home = home
        self.transport = transport
        self._weight_read = threading.Lock()

    def _token(self):
        path = self.home / "broker.token"
        try:
            if path.resolve(strict=True) != path.absolute():
                raise ValueError("Redirected credential")
            with path.open("rb") as stream:
                token = stream.read(4097).decode("ascii").strip()
            if not 32 <= len(token) <= 4096 or any(not 33 <= ord(c) <= 126 for c in token):
                raise ValueError("Invalid credential")
        except (OSError, ValueError) as error:
            raise NeuralError("Local neural credential is missing or invalid.") from error
        return token

    def _request(self, method, route, value=None, *, timeout=2):
        try:
            with (
                httpx.Client(
                    trust_env=False,
                    follow_redirects=False,
                    timeout=timeout,
                    transport=self.transport,
                ) as client,
                client.stream(
                    method,
                    ENDPOINT + route,
                    headers={"Authorization": f"Bearer {self._token()}"},
                    json=value,
                ) as response,
            ):
                body = bytearray()
                bound = 4096 if response.is_error else MAX_RESPONSE_BYTES
                for part in response.iter_bytes():
                    if len(body) + len(part) > bound:
                        raise NeuralError("Local neural response exceeds the display limit.")
                    body.extend(part)
                if route.startswith("/activity/weights") and response.status_code in (429, 503):
                    error_body = json.loads(body)
                    message = error_body.get("detail") if isinstance(error_body, dict) else None
                    if (
                        isinstance(message, str)
                        and 0 < len(message) <= 300
                        and not any(char in message for char in "<>\r\n")
                    ):
                        raise NeuralError(message)
                response.raise_for_status()
            value = json.loads(body)
        except httpx.RequestError as error:
            raise NeuralError("Local neural service is unavailable.") from error
        except httpx.HTTPStatusError as error:
            raise NeuralError("Local neural service error. Check Services.") from error
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise NeuralError("Local neural service returned invalid metadata.") from error
        if not isinstance(value, dict):
            raise NeuralError("Local neural service returned invalid metadata.")
        return value

    def _get(self, route):
        return self._request("GET", route)

    def activity_projections(self):
        value = self._get("/activity/projections")
        _parameter_model(value)
        rows = value.get("projections")
        if not isinstance(rows, list) or len(rows) != 72:
            raise NeuralError("The verified E5 projection catalogue is incomplete.")
        selected = set()
        for row in rows:
            _parameter_descriptor(row)
            selected.add((row["block"], row["projection"]))
        if len(selected) != 72:
            raise NeuralError("The E5 projection catalogue repeats a projection.")
        return value

    def activity_weights(self, block, projection, *, limit=512, output_index=None):
        if (
            type(block) is not int
            or not 0 <= block < 12
            or not isinstance(projection, str)
            or projection not in PROJECTION_DIMS
            or type(limit) is not int
            or not 1 <= limit <= 512
            or (
                output_index is not None
                and (
                    type(output_index) is not int
                    or not 0 <= output_index < PROJECTION_DIMS[projection][1]
                )
            )
        ):
            raise NeuralError("Parameter selection is outside the E5 projection bounds.")
        query = {"block": block, "projection": projection, "limit": limit}
        if output_index is not None:
            query["output_index"] = output_index
        # Overview and detail share the broker's single admitted parameter reader.
        with self._weight_read:
            value = self._request("GET", "/activity/weights?" + urlencode(query), timeout=15)
        return validate_parameter_weights(
            value, block, projection, limit=limit, output_index=output_index
        )

    def activity_architecture(self):
        value = self._get("/activity/architecture")
        dimensions = {
            "hidden_size": 384,
            "num_hidden_layers": 12,
            "intermediate_size": 1536,
            "num_attention_heads": 12,
            "head_dim": 32,
        }
        model = value.get("model")
        if (
            value.get("schema") != 1
            or any(
                type(value.get(key)) is not int or value[key] != size
                for key, size in dimensions.items()
            )
            or type(value.get("max_tokens")) is not int
            or not 1 <= value["max_tokens"] <= 512
            or value.get("capture_level") not in CAPTURE_LEVELS
            or not isinstance(model, dict)
            or any(not isinstance(model.get(key), str) for key in ("name", "revision", "space"))
            or not isinstance(model.get("device"), list)
            or ("runtime_available" in value and type(value["runtime_available"]) is not bool)
            or (
                value.get("runtime_reason") is not None
                and not isinstance(value["runtime_reason"], str)
            )
        ):
            raise NeuralError(
                "Local E5 architecture is invalid or differs from the installed model."
            )
        return value

    def activity_snapshot(self):
        value = self._get("/activity/snapshot")
        running = value.get("running")
        if (
            not isinstance(value.get("epoch"), str)
            or type(value.get("sequence")) is not int
            or value["sequence"] < 0
            or not isinstance(running, list)
            or len(running) > 64
            or any(not isinstance(item, dict) for item in running)
            or value.get("capture_level") not in CAPTURE_LEVELS
            or type(value.get("visible")) is not bool
            or type(value.get("paused")) is not bool
            or ("runtime_available" in value and type(value["runtime_available"]) is not bool)
            or (
                value.get("runtime_reason") is not None
                and not isinstance(value["runtime_reason"], str)
            )
            or (
                value.get("latest_capture_id") is not None
                and not valid_capture_id(value["latest_capture_id"])
            )
        ):
            raise NeuralError("Local neural activity metadata is invalid.")
        _validate_telemetry(value.get("telemetry"))
        return value

    def activity_capture(self, identity):
        if not valid_capture_id(identity):
            raise NeuralError(
                "Capture ID must contain only letters, numbers, dots, underscores or hyphens."
            )
        value = self._get("/activity/activations/" + identity)
        tensors = value.get("tensors")
        unavailable = value.get("unavailable")
        token = value.get("token_index")
        count = value.get("token_count")
        padding = value.get("token_is_padding")
        if (
            value.get("id") != identity
            or not isinstance(value.get("epoch"), str)
            or value.get("level") not in CAPTURE_LEVELS
            or type(token) is not int
            or type(count) is not int
            or not 0 <= token < 256
            or not 0 <= count <= 256
            or type(padding) is not bool
            or value.get("source") not in {"interactive", "background"}
            or not isinstance(tensors, dict)
            or (padding and bool(tensors))
            or (not padding and count == 0)
            or not tensors.keys() <= TENSOR_SIZES.keys()
            or not isinstance(unavailable, list)
            or any(item not in TENSOR_SIZES for item in unavailable)
            or any(item in tensors for item in unavailable)
            or any(not isinstance(value.get(key), str) for key in ("model_space", "model_revision"))
            or not isinstance(value.get("device"), list)
            or not value["device"]
            or any(not isinstance(device, str) for device in value["device"])
        ):
            raise NeuralError("Local neural capture metadata or tensor IDs are invalid.")
        for name, values in tensors.items():
            if (
                not isinstance(values, list)
                or len(values) != TENSOR_SIZES[name]
                or any(
                    type(number) not in (int, float) or not math.isfinite(number)
                    for number in values
                )
            ):
                raise NeuralError(
                    f"Capture tensor {name} must contain {TENSOR_SIZES[name]} finite values."
                )
        for name in ("started_at", "completed_at", "duration_ms"):
            number = value.get(name)
            if type(number) not in (int, float) or not math.isfinite(number) or number < 0:
                raise NeuralError(f"Capture {name} must be a finite non-negative number.")
        if value["completed_at"] < value["started_at"]:
            raise NeuralError("Capture completed_at must not precede started_at.")
        try:
            for name in ("started_at", "completed_at"):
                datetime.fromtimestamp(value[name])
        except (OverflowError, OSError, ValueError) as error:
            raise NeuralError(
                "Capture timestamp is outside the supported local clock range."
            ) from error
        return value

    def activity_control(self, *, visible, paused, token_index):
        if (
            type(visible) is not bool
            or type(paused) is not bool
            or (
                token_index is not None
                and (type(token_index) is not int or not 0 <= token_index < 256)
            )
        ):
            raise NeuralError(
                "Activity controls require booleans and a raw token index in [0, 256)."
            )
        return self._request(
            "POST",
            "/activity/control",
            {"visible": visible, "paused": paused, "token_index": token_index},
        )

    def status(self):
        value = self._get("/status")
        _validate_telemetry(value.get("telemetry"))
        if (
            "background_worker_alive" in value
            and type(value["background_worker_alive"]) is not bool
        ):
            raise NeuralError("Local neural status worker state is invalid.")
        for key in ("source_failures", "unavailable_providers"):
            if key in value and (type(value[key]) is not int or value[key] < 0):
                raise NeuralError("Local neural status source counts are invalid.")
        reason = value.get("reason")
        if reason is not None and (not isinstance(reason, str) or len(reason) > 4096):
            raise NeuralError("Local neural status reason is invalid.")
        lag = value.get("lag_seconds")
        if lag is not None and (type(lag) not in {int, float} or not math.isfinite(lag) or lag < 0):
            raise NeuralError("Local neural status index lag is invalid.")
        return value

    def topics(self):
        value = self._get("/topics")
        if "topics" not in value and isinstance(value.get("members"), list):
            groups = {}
            for member in value["members"]:
                if not isinstance(member, dict) or not isinstance(member.get("community_id"), str):
                    raise NeuralError("Local neural topic metadata is invalid.")
                identity = member["community_id"]
                topic = groups.setdefault(
                    identity, {"id": identity, "label": f"Topic {identity[:8]}", "members": []}
                )
                topic["members"].append({"node_id": member.get("source_id")})
            value = {**value, "topics": list(groups.values())}
        topics = value.get("topics")
        if not isinstance(topics, list) or len(topics) > 2000:
            raise NeuralError("Local neural topic metadata is invalid.")
        for topic in topics:
            if (
                not isinstance(topic, dict)
                or not isinstance(topic.get("id"), str)
                or not isinstance(topic.get("label"), str)
                or not isinstance(topic.get("members"), list)
                or len(topic["members"]) > 10000
            ):
                raise NeuralError("Local neural topic metadata is invalid.")
            for member in topic["members"]:
                if not isinstance(member, dict) or not isinstance(member.get("node_id"), str):
                    raise NeuralError("Local neural topic membership is invalid.")
        edges = value.get("edges", [])
        if not isinstance(edges, list) or len(edges) > 20000:
            raise NeuralError("Local neural similarity metadata is invalid.")
        for edge in edges:
            if not isinstance(edge, dict):
                raise NeuralError("Local neural similarity metadata is invalid.")
            score = edge.get("score", edge.get("weight"))
            if (
                not isinstance(edge.get("source"), str)
                or not isinstance(edge.get("target"), str)
                or not isinstance(score, (int, float))
                or not math.isfinite(score)
                or not 0 <= score <= 1
            ):
                raise NeuralError("Local neural similarity metadata is invalid.")
        return value

    def paths(self):
        value = self._get("/paths")
        if not isinstance(value.get("paths"), list) or len(value["paths"]) > 1000:
            raise NeuralError("Local neural retrieval paths are invalid.")
        if any(not isinstance(path, dict) for path in value["paths"]):
            raise NeuralError("Local neural retrieval paths are invalid.")
        return value


def valid_capture_id(identity):
    return isinstance(identity, str) and bool(
        re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", identity)
    )


def _parameter_model(value):
    if (
        not isinstance(value, dict)
        or value.get("schema") != 1
        or value.get("static") is not True
        or value.get("dtype") != "f16"
        or any(
            not isinstance(value.get(key), str) or not 0 < len(value[key]) <= 2048
            for key in ("model_revision", "model_space")
        )
    ):
        raise NeuralError("Static parameter model identity is invalid.")


def validate_parameter_weights(value, block, projection, *, limit=512, output_index=None):
    """Check the same boundary for HTTP responses and public parameter snapshots."""
    _parameter_model(value)
    _parameter_descriptor(value)
    edges = value.get("edges")
    sampling = "stratified_by_output" if output_index is None else "top_absolute_incoming_weight"
    if (
        value["block"] != block
        or value["projection"] != projection
        or value.get("sampling") != sampling
        or value.get("output_index") != output_index
        or not isinstance(edges, list)
        or len(edges) > limit
        or type(value.get("shown_count")) is not int
        or value["shown_count"] != len(edges)
        or value["provenance"].get("bin_verified") is not True
    ):
        raise NeuralError("Static weight selection or verification is invalid.")
    coordinates = set()
    for edge in edges:
        if (
            not isinstance(edge, dict)
            or type(edge.get("source")) is not int
            or type(edge.get("target")) is not int
            or not 0 <= edge["source"] < value["input_dim"]
            or not 0 <= edge["target"] < value["output_dim"]
            or (output_index is not None and edge["target"] != output_index)
            or type(edge.get("weight")) not in (int, float)
            or abs(edge["weight"]) > 65504
            or not math.isfinite(edge["weight"])
        ):
            raise NeuralError("Static parameter edge has invalid coordinates or weight.")
        coordinates.add((edge["source"], edge["target"]))
    if len(coordinates) != len(edges):
        raise NeuralError("Static parameter edges contain duplicate coordinates.")
    return value


def _parameter_descriptor(value):
    if not isinstance(value, dict) or not isinstance(value.get("projection"), str):
        raise NeuralError("Static parameter projection metadata is invalid.")
    dimensions = PROJECTION_DIMS.get(value["projection"])
    if (
        dimensions is None
        or type(value.get("block")) is not int
        or not 0 <= value["block"] < 12
        or type(value.get("input_dim")) is not int
        or type(value.get("output_dim")) is not int
        or (value["input_dim"], value["output_dim"]) != dimensions
        or type(value.get("total_weight_count")) is not int
        or value["total_weight_count"] != dimensions[0] * dimensions[1]
        or value.get("orientation") != "output_input"
    ):
        raise NeuralError("Static E5 projection dimensions or matrix axes are invalid.")
    for key in ("source", "target"):
        operator = value.get(key)
        if not isinstance(operator, dict) or any(
            not isinstance(operator.get(field), str) or not 0 < len(operator[field]) <= 2048
            for field in ("operator_id", "operation", "semantics")
        ):
            raise NeuralError("Static parameter operator identity is invalid.")
    provenance = value.get("provenance")
    if not isinstance(provenance, dict) or any(
        not isinstance(provenance.get(field), str)
        or not re.fullmatch(r"[0-9a-f]{64}", provenance[field])
        for field in ("prepared_manifest_sha256", "xml_sha256", "bin_sha256")
    ):
        raise NeuralError("Static parameter artifact provenance is invalid.")
