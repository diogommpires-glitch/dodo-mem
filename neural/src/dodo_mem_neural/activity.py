"""Bounded owner-local measurements from actual encoder inference."""

import json
import threading
import time
from collections import OrderedDict, deque
from uuid import uuid4

import numpy as np

SIZES = {"embedding": 384, "pooled": 384, "output": 384}
SIZES.update({f"block.{index}": 384 for index in range(12)})
SIZES.update({f"ff.{index}": 1536 for index in range(12)})
SIZES.update({f"heads.{index}": 384 for index in range(12)})
MAX_EVENTS = 2048
MAX_EVENT_BYTES = 2 * 1024**2
MAX_CAPTURE_BYTES = 8 * 1024**2


class NeuralActivity:
    def __init__(self):
        self._lock = threading.RLock()
        self.epoch = uuid4().hex
        self.sequence = 0
        self._events = deque()
        self._event_bytes = 0
        self._captures = OrderedDict()
        self._capture_bytes = 0
        self._running = {}
        self._visible = self._paused = False
        self._lease = 0
        self._token_index = None
        self._last_background = float("-inf")
        self._model = {"name": "e5", "revision": None, "space": None, "device": []}
        self._level = "final"
        self._reason = "Layer capture has not passed qualification."
        self._verified = False
        self._dropped = 0
        self._inference_running = {}
        self._inference_intervals = deque(maxlen=4096)
        self._completed_inferences = self._failed_inferences = 0
        self._last_inference_ms = self._last_completed_at = None

    def set_model(self, info):
        with self._lock:
            model = {key: info.get(key) for key in ("name", "revision", "space", "device")}
            if self._model["space"] and model != self._model:
                self._captures.clear()
                self._capture_bytes = 0
                self._inference_intervals.clear()
                self._inference_running.clear()
                self._completed_inferences = self._failed_inferences = 0
                self._last_inference_ms = self._last_completed_at = None
            self._model = model
            self._level = info.get("level", "final")
            self._reason = (
                "Only the final block, pooling and normalised output are measured. "
                "Earlier layer values are unavailable."
                if self._level == "final"
                else None
            )
            architecture = info.get("architecture") or {}
            self._verified = bool(architecture.get("verified", False))

    def architecture(self):
        with self._lock:
            return {
                "schema": 1,
                "model": dict(self._model),
                "hidden_size": 384,
                "num_hidden_layers": 12,
                "intermediate_size": 1536,
                "num_attention_heads": 12,
                "head_dim": 32,
                "max_tokens": 256,
                "capture_level": self._level,
                "qualification_reason": self._reason,
                "verified": self._verified,
            }

    def inference_started(self, model_info):
        with self._lock:
            self.set_model(model_info)
            identity = uuid4().hex
            self._inference_running[identity] = time.monotonic()
            return identity

    def inference_finished(self, identity, failed):
        with self._lock:
            started = self._inference_running.pop(identity, None)
            if started is None:
                return
            finished = time.monotonic()
            self._inference_intervals.append((started, finished))
            self._last_inference_ms = max(0, finished - started) * 1000
            self._last_completed_at = time.time()
            self._completed_inferences += not failed
            self._failed_inferences += bool(failed)

    def telemetry(self):
        with self._lock:
            finished = time.monotonic()
            start = finished - 10.0
            while self._inference_intervals and self._inference_intervals[0][1] <= start:
                self._inference_intervals.popleft()
            intervals = [
                *self._inference_intervals,
                *((started, finished) for started in self._inference_running.values()),
            ]
            busy = sum(max(0, end - max(begin, start)) for begin, end in intervals)
            return {
                "device": list(self._model.get("device") or []),
                "active_inferences": len(self._inference_running),
                "completed_inferences": self._completed_inferences,
                "failed_inferences": self._failed_inferences,
                "last_inference_ms": self._last_inference_ms,
                "last_completed_at": self._last_completed_at,
                "inference_busy_percent": min(100.0, busy * 10),
                "window_seconds": 10.0,
                "hardware_utilization_percent": None,
                "hardware_utilization_reason": (
                    "No verified Windows NPU utilisation counter is available. "
                    "Encoder busy time measures OpenVINO inference calls, not chip utilisation."
                ),
            }

    def control(self, *, visible, paused, token_index):
        if type(visible) is not bool or type(paused) is not bool:
            raise ValueError("Visibility and pause must be boolean.")
        if token_index is not None and (type(token_index) is not int or not 0 <= token_index < 256):
            raise ValueError("Token index must be null or an integer from 0 to 255.")
        with self._lock:
            self._visible, self._paused, self._token_index = visible, paused, token_index
            self._lease = time.monotonic() + 30 if visible else 0
            return self.snapshot()

    def snapshot(self):
        with self._lock:
            visible = self._visible and time.monotonic() < self._lease
            return {
                "epoch": self.epoch,
                "sequence": self.sequence,
                "visible": visible,
                "paused": self._paused,
                "token_index": self._token_index,
                "running": [
                    {key: ticket[key] for key in ("id", "source", "started_at")}
                    for ticket in self._running.values()
                ],
                "latest_capture_id": next(reversed(self._captures), None),
                "capture_level": self._level,
                "reason": self._reason,
                "dropped_events": self._dropped,
                "retained_captures": len(self._captures),
                "activation_bytes": self._capture_bytes,
                "telemetry": self.telemetry(),
            }

    def _emit(self, event_type, ticket):
        self.sequence += 1
        event = {
            "epoch": self.epoch,
            "sequence": self.sequence,
            "type": event_type,
            "id": ticket["id"],
            "source": ticket["source"],
            "time": time.time(),
        }
        size = len(json.dumps(event).encode())
        self._events.append((event, size))
        self._event_bytes += size
        while len(self._events) > MAX_EVENTS or self._event_bytes > MAX_EVENT_BYTES:
            _, old_size = self._events.popleft()
            self._event_bytes -= old_size
            self._dropped += 1

    def events_since(self, after=0):
        with self._lock:
            oldest = self._events[0][0]["sequence"] if self._events else self.sequence + 1
            gap = after > self.sequence or after < oldest - 1
            return {
                "epoch": self.epoch,
                "gap": gap,
                "snapshot": self.snapshot() if gap else None,
                "events": [dict(event) for event, _ in self._events if event["sequence"] > after],
            }

    def begin(self, attention_mask, interactive, model_info):
        mask = np.asarray(attention_mask).ravel()
        if len(mask) > 256 or not np.all((mask == 0) | (mask == 1)):
            raise ValueError("Capture mask must contain at most 256 binary positions.")
        with self._lock:
            now = time.monotonic()
            if not self._visible or self._paused or now >= self._lease:
                return None
            if not interactive and now - self._last_background < 1:
                return None
            if not interactive:
                self._last_background = now
            self.set_model(model_info)
            token = self._token_index
            if token is None:
                positions = np.flatnonzero(mask)
                token = int(positions[0]) if len(positions) else 0
            ticket = {
                "id": uuid4().hex,
                "token_index": token,
                "token_count": int(mask.sum()),
                "valid_token": token < len(mask) and bool(mask[token]),
                "source": "interactive" if interactive else "background",
                "sampled": not interactive,
                "started_at": time.time(),
                "started_monotonic": now,
            }
            self._running[ticket["id"]] = ticket
            self._emit("started", ticket)
            return dict(ticket)

    def complete(self, ticket, tensors, level, model_info):
        with self._lock:
            if self._running.pop(ticket["id"], None) is None:
                return
            if not self._visible or self._paused or time.monotonic() >= self._lease:
                self._emit("completed", ticket)
                return
        if level not in {"full", "blocks", "final"}:
            raise ValueError("Unknown activation capture level.")
        values = {}
        if ticket["valid_token"]:
            for name, value in tensors.items():
                array = np.asarray(value, dtype=np.float32).reshape(-1)
                if name not in SIZES or len(array) != SIZES[name] or not np.all(np.isfinite(array)):
                    raise ValueError(
                        "Capture tensor has an unknown name, shape or non-finite values."
                    )
                array = array.copy()
                array.flags.writeable = False
                values[name] = array
        size = sum(array.nbytes + 256 for array in values.values()) + 2048
        if size > MAX_CAPTURE_BYTES:
            raise ValueError("One activation capture exceeds the memory limit.")
        capture = {
            "id": ticket["id"],
            "epoch": self.epoch,
            "model_space": model_info.get("space"),
            "model_revision": model_info.get("revision"),
            "device": model_info.get("device", []),
            "level": level,
            "started_at": ticket["started_at"],
            "completed_at": time.time(),
            "duration_ms": max(0, time.monotonic() - ticket["started_monotonic"]) * 1000,
            "token_index": ticket["token_index"],
            "token_count": ticket["token_count"],
            "token_is_padding": not ticket["valid_token"],
            "sampled": ticket["sampled"],
            "source": ticket["source"],
            "tensors": values,
            "unavailable": sorted(set(SIZES) - values.keys()),
            "reason": None
            if ticket["valid_token"]
            else "Selected token is masked or outside this inference.",
        }
        with self._lock:
            if not self._visible or self._paused or time.monotonic() >= self._lease:
                self._emit("completed", ticket)
                return
            self._captures[ticket["id"]] = (capture, size)
            self._capture_bytes += size
            while len(self._captures) > 32 or self._capture_bytes > MAX_CAPTURE_BYTES:
                _, (_, old_size) = self._captures.popitem(last=False)
                self._capture_bytes -= old_size
            self._level = level
            self._emit("completed", ticket)

    def fail(self, ticket, safe_code):
        with self._lock:
            self._running.pop(ticket["id"], None)
            self._emit("failed", ticket)

    def capture(self, capture_id):
        with self._lock:
            item = self._captures.get(capture_id)
            if item is None:
                return None
            capture = item[0]
            return {
                **capture,
                "tensors": {name: values.tolist() for name, values in capture["tensors"].items()},
                "unavailable": list(capture["unavailable"]),
            }
