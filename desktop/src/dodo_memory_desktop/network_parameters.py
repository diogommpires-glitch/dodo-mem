"""Fetch one coherent, bounded owner-local sample of E5's 72 projections."""

import hashlib
import json
import os
import shutil
import tempfile
import threading
import time
from functools import partial
from pathlib import Path

from PySide6.QtCore import QEventLoop, QObject, Signal
from PySide6.QtWidgets import QApplication

from .configuration import configured_path
from .neural_client import NEURAL_HOME, PROJECTION_DIMS, validate_parameter_weights
from .whole_library import CatalogueReads

NEURAL_CACHE_ROOT = NEURAL_HOME / "bulk"
MAX_SNAPSHOT_BYTES = 4 * 1024**2
MAX_NEURAL_BYTES = 20 * 1024**3
MIN_DISK_HEADROOM = 100 * 1024**3


def _configured_cache(home):
    configuration = home / "cache-root.json"
    if not configuration.exists() and not configuration.is_symlink():
        return configured_path(str(home / "bulk"))
    if configuration.resolve(strict=True) != configuration.absolute():
        raise ValueError("Neural cache configuration cannot contain redirects.")
    with configuration.open("rb") as stream:
        raw = stream.read(4097)
    if len(raw) > 4096:
        raise ValueError("Neural cache configuration exceeds 4096 bytes.")
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"root"} or not isinstance(value["root"], str):
        raise ValueError("Neural cache configuration must contain only a root path.")
    root = Path(value["root"])
    if not root.is_absolute() or root != home / "bulk":
        raise ValueError("Neural cache root must be the configured state bulk directory.")
    if root.resolve(strict=True) != root.absolute() or not root.is_dir():
        raise ValueError("Neural cache root must be an existing directory without redirects.")
    return root


class NetworkParameters(QObject):
    ready = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, int)
    cache_warning = Signal(str)

    def __init__(
        self,
        client,
        parent=None,
        *,
        cache_path=None,
    ):
        super().__init__(parent)
        self.client = client
        self._managed_cache = cache_path is None
        self.cache_path = Path(cache_path) if cache_path is not None else None
        self.cache_root = self._cache_error = None
        self._select_cache()
        self.active = self.running = self._stopping = self._blocked = False
        self._identity = self._cache = None
        self._generation = 0
        self._cancel = threading.Event()
        self._descriptors, self._projections = [], []
        self._snapshot_checked = False
        self._cache_admitted = False
        self._tokens = set()
        self.reads = CatalogueReads(QApplication.instance())
        self.reads.done.connect(self._read_finished)
        self.destroyed.connect(
            lambda _owner=None, reads=self.reads, cancel=self._cancel: (
                cancel.set(),
                reads.pending.clear(),
            )
        )

    def set_model(self, model):
        if not isinstance(model, dict):
            raise TypeError("Parameter samples require verified model metadata.")
        identity = tuple(model.get(key) for key in ("revision", "space"))
        if any(not isinstance(value, str) or not 0 < len(value) <= 2048 for value in identity):
            raise ValueError("Parameter samples require a verified model revision and space.")
        if identity == self._identity:
            return
        self._identity = identity
        self.reload()

    def set_active(self, active):
        active = bool(active and not self._stopping)
        if active == self.active:
            return
        self.active = active
        if active:
            if self._cache is not None:
                self._cache_admitted = self._snapshot_checked = False
                self._descriptors = []
            self._start()
        else:
            self._cancel.set()

    def reload(self):
        self._generation += 1
        self._cancel.set()
        self._cache = None
        self._cache_admitted = False
        self._blocked = False
        self._descriptors, self._projections = [], []
        self._snapshot_checked = False
        self._select_cache()
        self._start()

    def _select_cache(self):
        if not self._managed_cache:
            return
        try:
            root = _configured_cache(Path(getattr(self.client, "home", NEURAL_HOME)))
        except (OSError, ValueError, RecursionError) as error:
            self.cache_path = self.cache_root = None
            self._cache_error = "Neural cache unavailable: " + str(error)
        else:
            self.cache_root = root
            self.cache_path = root / "visualisation/whole-network-cache.json"
            self._cache_error = None

    def _start(self):
        if not self.active or self._blocked or self._identity is None:
            return
        if self._cache_error:
            self._fail(self._cache_error)
            return
        if self._cache is not None and self._cache_admitted:
            self.ready.emit(self._cache)
            return
        if self.running:
            return
        if self.client is None:
            self._fail("Static network parameters require the local owner service.")
            return
        self._cancel.clear()
        generation = self._generation
        self.progress.emit(len(self._projections), 72)
        if generation == self._generation:
            self._advance()

    def _advance(self):
        if self._descriptors and not self._snapshot_checked:
            self._request(
                "snapshot",
                partial(self._read_snapshot, self._descriptors, self._identity, self._cache),
            )
        elif self._descriptors:
            row = self._descriptors[len(self._projections)]
            self._request(
                row,
                partial(self.client.activity_weights, row["block"], row["projection"], limit=256),
            )
        else:
            self._request(None, self.client.activity_projections)

    def _request(self, descriptor, operation):
        saving = descriptor == "save"
        if not saving and (not self.active or self._cancel.is_set()):
            return
        generation = self._generation

        def perform():
            if not saving and self._cancel.is_set():
                return None, None
            try:
                return True, operation()
            except (OSError, ValueError) as error:
                return False, str(error)[:1000]

        self.running = True
        self.reads.request(
            "network-parameters",
            perform,
            partial(self._received, generation, descriptor),
            partial(self._received, generation, descriptor, (False, "Parameter read failed.")),
        )
        self._tokens.add(("network-parameters", self.reads.serial))

    def _received(self, generation, descriptor, result):
        self.running = False
        if generation != self._generation:
            self._start()
            return
        succeeded, value = result
        if succeeded is None:
            self._start()
            return
        if descriptor == "save":
            if not succeeded:
                self.cache_warning.emit("Network snapshot was not saved: " + value)
            if self.active and not self._cache_admitted:
                self._start()
            return
        if descriptor == "snapshot":
            self._snapshot_checked = True
            if succeeded and value is not None:
                self._cache = value
                self._cache_admitted = True
                self._descriptors = []
                self.progress.emit(72, 72)
                if generation == self._generation and self.active and self._cache_admitted:
                    self.ready.emit(self._cache)
            elif self.active:
                self._cache = None
                self._cancel.clear()
                self._advance()
            return
        if not succeeded:
            self._fail(value)
            return
        try:
            if (value["model_revision"], value["model_space"]) != self._identity:
                raise ValueError("The parameter model changed during network sampling.")
            if descriptor is None:
                rows = value["projections"]
                expected = {(block, name) for block in range(12) for name in PROJECTION_DIMS}
                if (
                    len(rows) != 72
                    or {(row["block"], row["projection"]) for row in rows} != expected
                ):
                    raise ValueError(
                        "The network parameter catalogue is incomplete or repeats a projection."
                    )
                self._descriptors = sorted(
                    rows,
                    key=lambda row: (row["block"], list(PROJECTION_DIMS).index(row["projection"])),
                )
            else:
                self._check_projection(descriptor, value)
                self._projections.append(value)
                self.progress.emit(len(self._projections), 72)
                if generation != self._generation:
                    return
            if len(self._projections) == 72:
                self._publish()
            elif self.active:
                self._cancel.clear()
                self._advance()
        except (KeyError, TypeError, ValueError) as error:
            self._fail(str(error))

    @staticmethod
    def _check_projection(descriptor, value):
        fields = (
            "block",
            "projection",
            "input_dim",
            "output_dim",
            "total_weight_count",
            "source",
            "target",
            "operator_id",
            "operation",
            "orientation",
        )
        if any(value[field] != descriptor[field] for field in fields):
            raise ValueError("A network parameter response belongs to another projection.")
        provenance = value["provenance"]
        if provenance.get("bin_verified") is not True or any(
            provenance[field] != descriptor["provenance"][field]
            for field in (
                "prepared_manifest_sha256",
                "xml_sha256",
                "bin_sha256",
                "constant_id",
                "offset_bytes",
                "size_bytes",
                "matrix_layout",
            )
        ):
            raise ValueError("Network parameter artifact verification changed during sampling.")
        if (
            value["shown_count"] != 256
            or len(value["edges"]) != 256
            or value["output_index"] is not None
        ):
            raise ValueError("The network projection sample is incomplete.")

    def _publish(self):
        total = sum(value["total_weight_count"] for value in self._projections)
        if total != 21_233_664:
            raise ValueError("The E5 network parameter total is incomplete.")
        self._cache = {
            "schema": 1,
            "static": True,
            "dtype": "f16",
            "model_revision": self._identity[0],
            "model_space": self._identity[1],
            "projections": self._projections,
            "total_weight_count": total,
            "shown_count": 18_432,
        }
        self._projections, self._descriptors = [], []
        self._cache_admitted = True
        self._request("save", partial(self._write_snapshot, self._cache))
        if self.active:
            self.ready.emit(self._cache)

    def _read_snapshot(self, descriptors, identity, packet=None):
        try:
            if packet is None:
                if self.cache_path.resolve(strict=True) != self.cache_path.absolute():
                    return None
                with self.cache_path.open("rb") as stream:
                    raw = stream.read(MAX_SNAPSHOT_BYTES + 1)
                if len(raw) > MAX_SNAPSHOT_BYTES:
                    return None
                envelope = json.loads(raw)
                packet = envelope["packet"]
                canonical = json.dumps(
                    packet,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
                if hashlib.sha256(canonical).hexdigest() != envelope["sha256"]:
                    return None
            if (
                (packet["model_revision"], packet["model_space"]) != identity
                or packet["schema"] != 1
                or packet["static"] is not True
                or packet["dtype"] != "f16"
                or packet["shown_count"] != 18_432
                or packet["total_weight_count"] != 21_233_664
                or len(packet["projections"]) != 72
            ):
                return None
            expected = {(row["block"], row["projection"]): row for row in descriptors}
            seen = set()
            for row in packet["projections"]:
                key = row["block"], row["projection"]
                validate_parameter_weights(row, *key, limit=256, output_index=None)
                if key in seen or (row["model_revision"], row["model_space"]) != identity:
                    return None
                self._check_projection(expected[key], row)
                seen.add(key)
            if seen != expected.keys():
                return None
            return {**packet, "from_snapshot": True}
        except (OSError, ValueError, KeyError, TypeError, RecursionError):
            return None

    def _write_snapshot(self, packet):
        if self._cache_error:
            raise OSError(self._cache_error)
        canonical = json.dumps(
            packet, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False
        ).encode("utf-8")
        envelope = json.dumps(
            {"sha256": hashlib.sha256(canonical).hexdigest(), "packet": packet},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        if len(envelope) > MAX_SNAPSHOT_BYTES:
            raise ValueError("The public network snapshot exceeds 4 MiB.")
        root = self.cache_root or self.cache_path.parent

        def cannot_count(error):
            raise error

        used = sum(
            entry.stat().st_size
            for base, _, names in os.walk(root, followlinks=False, onerror=cannot_count)
            for name in names
            if (entry := Path(base) / name).is_file() and not entry.is_symlink()
        )
        # The old snapshot remains present until the flushed temporary file replaces it.
        if used + len(envelope) > MAX_NEURAL_BYTES:
            raise OSError("The network snapshot would exceed the inclusive 20 GiB neural quota.")
        if shutil.disk_usage(self.cache_path.anchor).free - len(envelope) < MIN_DISK_HEADROOM:
            raise OSError("The network snapshot would leave less than 100 GiB free disk.")
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        if self.cache_path.parent.resolve(strict=True) != self.cache_path.parent.absolute():
            raise OSError("The network snapshot directory is redirected.")
        temporary = None
        try:
            descriptor, temporary = tempfile.mkstemp(
                prefix=".network-cache-", dir=self.cache_path.parent
            )
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(envelope)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, self.cache_path)
        finally:
            if temporary is not None:
                Path(temporary).unlink(missing_ok=True)

    def _fail(self, reason):
        self._blocked = True
        self._cache = None
        self._descriptors, self._projections = [], []
        self.failed.emit(str(reason)[:1000])

    def _read_finished(self, token, _value, _succeeded):
        self._tokens.discard(token)

    def shutdown(self, timeout=5):
        self._stopping = True
        self.set_active(False)
        self._cancel.set()
        deadline = time.monotonic() + timeout
        while self.running or self._tokens:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True
