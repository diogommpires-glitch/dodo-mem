"""Bounded USearch shards with durable checksummed publication on the state drive."""

import hashlib
import json
import os
import re
import threading
import uuid
from pathlib import Path

import numpy as np
from usearch.index import Index

from .artifacts import replace_file

DIMENSIONS = 384


def _checksum(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_atomic(path, payload):
    temporary = path.with_name(path.name + ".partial")
    with temporary.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    replace_file(temporary, path)


class VectorStore:
    def __init__(
        self,
        cache_dir: Path,
        state_dir: Path,
        *,
        encoder_id="default",
        max_mutable=25_000,
        max_shard=250_000,
        resource_guard=None,
        cache_uuid="",
    ):
        if not 1 <= max_mutable <= 25_000 or not max_mutable <= max_shard <= 250_000:
            raise ValueError("USearch bounds require mutable <=25,000 and shard <=250,000")
        self.cache_dir, self.state_dir = Path(cache_dir), Path(state_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.encoder_id = encoder_id
        if cache_uuid and not re.fullmatch(r"[a-f0-9]{32}", cache_uuid):
            raise ValueError("Vector cache UUID must be 32 hexadecimal characters.")
        self.cache_uuid = cache_uuid
        self.max_mutable, self.max_shard = max_mutable, max_shard
        self.resource_guard = resource_guard
        self._lock = threading.RLock()
        self._shards = []
        self._indices = []
        self._staged = []
        self._mutable = self._new_index()
        self.generation = None
        self.recovered_from = None
        self.deferred_cleanup = 0
        self._pointer_stamp = None
        self._load_published()

    @staticmethod
    def _new_index():
        return Index(
            ndim=DIMENSIONS,
            metric="cos",
            dtype="f16",
            connectivity=16,
            expansion_add=128,
            expansion_search=1024,
        )

    def _guard(self, additional_disk=0):
        if self.resource_guard:
            self.resource_guard(background=True, additional_disk=additional_disk)

    def _load_published(self):
        failures = []
        for name in ("CURRENT", "LASTGOOD"):
            pointer = self.state_dir / name
            if not pointer.exists():
                continue
            try:
                head = json.loads(pointer.read_bytes())
                if (
                    head["encoder_id"] != self.encoder_id
                    or head.get("cache_uuid", "") != self.cache_uuid
                ):
                    continue
                manifest_name = head["manifest"]
                if Path(manifest_name).name != manifest_name:
                    raise ValueError("Manifest must be local to the state directory")
                manifest_file = self.state_dir / manifest_name
                if _checksum(manifest_file) != head["sha256"]:
                    raise ValueError("Manifest checksum mismatch")
                manifest = json.loads(manifest_file.read_bytes())
                if (
                    manifest["encoder_id"] != self.encoder_id
                    or manifest["dimensions"] != DIMENSIONS
                    or manifest.get("cache_uuid", "") != self.cache_uuid
                ):
                    raise ValueError("Vector manifest version mismatch")
                shards, indices = [], []
                for shard in manifest["shards"]:
                    if Path(shard["file"]).name != shard["file"]:
                        raise ValueError("Shard must be local to the cache directory")
                    path = self.cache_dir / shard["file"]
                    if not 0 < shard["count"] <= self.max_shard:
                        raise ValueError("Shard exceeds the configured vector bound")
                    if _checksum(path) != shard["sha256"]:
                        raise ValueError("Vector shard checksum mismatch")
                    index = Index.restore(path, view=True)
                    if index is None or len(index) != shard["count"]:
                        raise ValueError("Vector shard count mismatch")
                    index.expansion_search = 1024
                    shards.append(shard)
                    indices.append(index)
                self._shards, self._indices = shards, indices
                self.generation = manifest["generation"]
                self.recovered_from = name if name == "LASTGOOD" else None
                current = self.state_dir / "CURRENT"
                self._pointer_stamp = current.stat().st_mtime_ns if current.exists() else None
                return
            except (OSError, ValueError, KeyError, TypeError) as error:
                failures.append(f"{name}: {type(error).__name__}")
        if failures:
            raise ValueError("No valid vector publication: " + "; ".join(failures))

    def contains(self, keys):
        with self._lock:
            return [
                any(
                    bool(index.contains(int(key)))
                    for index in [self._mutable, *self._indices, *(row[1] for row in self._staged)]
                )
                for key in keys
            ]

    def get(self, keys):
        with self._lock:
            vectors = []
            for key in keys:
                found = None
                for index in [self._mutable, *self._indices, *(row[1] for row in self._staged)]:
                    if index.contains(int(key)):
                        found = index.get(int(key), dtype="f32")
                        break
                if found is None:
                    raise KeyError(f"Vector key is not present: {key}")
                vectors.append(found)
            return np.asarray(vectors, dtype=np.float32).reshape((-1, DIMENSIONS))

    def add(self, keys, vectors):
        keys = np.asarray(keys, dtype=np.uint64)
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.shape != (len(keys), DIMENSIONS):
            raise ValueError(f"Expected ({len(keys)},384) vectors, got {vectors.shape}")
        if len({int(key) for key in keys}) != len(keys):
            raise ValueError("Vector keys must be unique within a batch")
        if not np.isfinite(vectors).all():
            raise ValueError("Vectors must be finite")
        norms = np.linalg.norm(vectors, axis=1)
        if np.any(norms <= 0):
            raise ValueError("Vectors must have positive norms")
        vectors = vectors / norms[:, None]
        with self._lock:
            present = self.contains(keys)
            keys, vectors = keys[np.logical_not(present)], vectors[np.logical_not(present)]
            while len(keys):
                self._guard()
                count = min(len(keys), self.max_mutable - len(self._mutable))
                self._mutable.add(keys[:count], vectors[:count], threads=1)
                keys, vectors = keys[count:], vectors[count:]
                if len(self._mutable) == self.max_mutable:
                    self._freeze()

    def _freeze(self):
        if not len(self._mutable):
            return
        self._staged.append(self._save_shard(self._mutable))
        self._mutable = self._new_index()

    def _save_shard(self, index):
        self._guard(len(index) * DIMENSIONS * 4 + 1024**2)
        name = f"shard-{uuid.uuid4().hex}.usearch"
        path = self.cache_dir / name
        temporary = path.with_name(path.name + ".partial")
        index.save(temporary)
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        checksum = _checksum(temporary)
        os.replace(temporary, path)
        index = Index.restore(path, view=True)
        if index is None:
            raise ValueError("Published USearch shard could not be reopened")
        index.expansion_search = 1024
        return ({"file": name, "count": len(index), "sha256": checksum}, index)

    def _coalesced(self):
        shards, indices = list(self._shards), list(self._indices)
        sources = list(self._staged)
        if shards and shards[-1]["count"] < self.max_shard:
            sources.insert(0, (shards.pop(), indices.pop()))
        if len(sources) <= 1:
            return shards, indices, sources
        compacted = []
        builder = self._new_index()
        for _, index in sources:
            keys = np.sort(np.asarray(index.keys, dtype=np.uint64))
            while len(keys):
                self._guard()
                count = min(len(keys), self.max_mutable, self.max_shard - len(builder))
                batch = keys[:count]
                values = np.asarray(index.get(batch, dtype="f32"), dtype=np.float32)
                builder.add(batch, values, threads=1)
                keys = keys[count:]
                if len(builder) == self.max_shard:
                    compacted.append(self._save_shard(builder))
                    builder = self._new_index()
        if len(builder):
            compacted.append(self._save_shard(builder))
        return shards, indices, compacted

    def _prune(self):
        keep_shards, keep_manifests = set(), set()
        for name in ("CURRENT", "LASTGOOD"):
            pointer = self.state_dir / name
            if not pointer.exists():
                continue
            try:
                head = json.loads(pointer.read_bytes())
                manifest = self.state_dir / head["manifest"]
                value = json.loads(manifest.read_bytes())
            except (OSError, ValueError, KeyError, TypeError):
                return  # An unreadable retained generation must not lose its files.
            keep_manifests.add(manifest.name)
            keep_shards.update(shard["file"] for shard in value["shards"])
        self.deferred_cleanup = 0
        for directory, pattern, keep in (
            (self.cache_dir, "shard-*.usearch", keep_shards),
            (self.state_dir, "manifest-*.json", keep_manifests),
        ):
            for path in directory.glob(pattern):
                if path.name in keep or path.is_symlink():
                    continue
                try:
                    path.unlink(missing_ok=True)
                except PermissionError:
                    self.deferred_cleanup += 1  # A live Windows reader can retain a mapped shard.

    def publish(self):
        with self._lock:
            self._freeze()
            if not self._staged and self.generation:
                return {"generation": self.generation, "shards": list(self._shards)}
            self._guard(1024**2)
            shards, indices, staged = self._coalesced()
            generation = uuid.uuid4().hex
            manifest = {
                "schema": 1,
                "generation": generation,
                "encoder_id": self.encoder_id,
                "cache_uuid": self.cache_uuid,
                "dimensions": DIMENSIONS,
                "dtype": "f16",
                "metric": "cos",
                "shards": [*shards, *(row[0] for row in staged)],
            }
            path = self.state_dir / f"manifest-{generation}.json"
            _write_atomic(path, json.dumps(manifest, sort_keys=True).encode())
            head = json.dumps(
                {
                    "manifest": path.name,
                    "sha256": _checksum(path),
                    "encoder_id": self.encoder_id,
                    "cache_uuid": self.cache_uuid,
                },
                sort_keys=True,
            ).encode()
            current = self.state_dir / "CURRENT"
            if current.exists() and self.recovered_from is None:
                _write_atomic(self.state_dir / "LASTGOOD", current.read_bytes())
            _write_atomic(current, head)
            self._shards = manifest["shards"]
            self._indices = [*indices, *(row[1] for row in staged)]
            self._staged.clear()
            self.generation = generation
            self.recovered_from = None
            self._pointer_stamp = current.stat().st_mtime_ns
            self._prune()
            return manifest

    def search(self, vector, k=40):
        if not self.cache_dir.is_dir():
            raise OSError("Derived vector storage is unavailable; ordinary search remains usable.")
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.shape != (DIMENSIONS,) or not np.isfinite(vector).all():
            raise ValueError("Query must be a finite vector of dimension 384")
        if not 1 <= k <= 1000:
            raise ValueError("Vector search count must be in [1,1000]")
        norm = np.linalg.norm(vector)
        if norm <= 0:
            raise ValueError("Query vector must have a positive norm")
        with self._lock:
            pointer = self.state_dir / "CURRENT"
            stamp = pointer.stat().st_mtime_ns if pointer.exists() else None
            if stamp != self._pointer_stamp and not self._staged and not len(self._mutable):
                self._load_published()
            scores = {}
            # Only committed shards are visible until their C: manifest is published.
            for index in self._indices:
                matches = index.search(vector / norm, min(k, len(index)), threads=1)
                for key, distance in zip(matches.keys, matches.distances, strict=True):
                    key = int(key)
                    scores[key] = max(scores.get(key, -1.0), float(1.0 - distance))
            return sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:k]
