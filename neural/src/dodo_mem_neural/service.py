"""Assemble the local runtime without changing native client profiles or model servers."""

import asyncio
import hashlib
import json
import shutil
import socket
import sqlite3
import threading
import time
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import httpx
import psutil
import uvicorn
from dodo_memory_desktop.configuration import runtime_path

from .activity import NeuralActivity
from .artifacts import atomic_json
from .broker import create_broker_app
from .catalogue_adapter import AuthorisedCatalogue, ProjectionBudget, local_sources
from .catalogue_changes import defer_sqlite_busy
from .compact_store import CompactIndexStore as IndexStore
from .contracts import Credential, ModelRoute
from .organisation import publish_communities
from .proxy import create_proxy_app, model_token_count
from .resources import DEFAULT_CACHE, GIB, ResourcePressure, check_resources
from .retrieval import RetrievalEngine
from .route_gate import RouteGate
from .runtime import NeuralRuntime
from .vector_store import VectorStore
from .worker import NeuralWorker

STATE = runtime_path("neural")


def log_configuration(home):
    # pythonw has no stdout; Uvicorn's colour formatter assumes a console exists.
    return {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {"plain": {"format": "%(asctime)s %(levelname)s %(name)s %(message)s"}},
        "handlers": {
            "service": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(home / "service.log"),
                "maxBytes": 1024 * 1024,
                "backupCount": 1,
                "formatter": "plain",
            }
        },
        "loggers": {"uvicorn": {"handlers": ["service"], "level": "WARNING", "propagate": False}},
    }


def qualified_runtime(name, device, *, observer=None, activation_home=None):
    options = {}
    if observer is not None:
        options.update(observer=observer, activation_home=activation_home)
    try:
        return NeuralRuntime(name, device=device, **options), None
    except RuntimeError as error:
        if device != "NPU":
            raise
        return NeuralRuntime(
            name, device="CPU", **options
        ), f"NPU unavailable; CPU uses two threads: {error}"


def preflight_ports():
    sockets = []
    try:
        for port in (8770, 8771):
            probe = socket.socket()
            if hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
            sockets.append(probe)
    finally:
        for probe in sockets:
            probe.close()


def load_routes(home=STATE):
    document = json.loads((home / "routes.json").read_text(encoding="utf-8"))
    routes = []
    allowed = {item.name for item in fields(ModelRoute)}
    for original in document["routes"]:
        row = dict(original)
        token_file = row.pop("upstream_token_file", None)
        if token_file:
            path = Path(token_file)
            if path.resolve(strict=True) != path.absolute():
                raise ValueError("Upstream credential cannot be redirected.")
            row["upstream_token"] = path.read_text(encoding="utf-8").strip()
        if not set(row) <= allowed:
            raise ValueError("Unknown route configuration field")
        routes.append(ModelRoute(**row))
    credentials = tuple(
        Credential(**{**row, "routes": tuple(row["routes"])}) for row in document["credentials"]
    )
    return tuple(routes), credentials


class OrganisationService:
    def __init__(self, home=STATE):
        self.home = home
        self.selection = json.loads((home / "selection.json").read_text(encoding="utf-8"))
        self.catalogue, vault = local_sources()
        from dodo_memory_desktop.personal_context import PersonalProfile

        self.authorised = AuthorisedCatalogue(self.catalogue, vault, profile=PersonalProfile())
        self.activity = RouteGate()
        self.neural_activity = NeuralActivity()
        self._projection_lock = threading.RLock()
        self._projection_source = None
        self._projection_space = None
        self.runtime = self.reranker = self.store = self.vectors = None
        self.space = self.space_home = None
        self.neural_unavailable = None
        self.device_fallback = None
        try:
            self.__dict__.update(self._open_neural())
        except (ResourcePressure, OSError, RuntimeError, ValueError) as error:
            # The C: catalogue remains usable when derived storage or resource headroom is absent.
            self.neural_unavailable = f"{type(error).__name__}: {error}"
        self.progress = {
            "state": "starting",
            "embedded": 0,
            "pending": 0,
            "lag_seconds": None,
            "eta_seconds": None,
            "rate_per_second": None,
            "completed_files": 0,
            "passages": 0,
            "failures": 0,
            "eta_basis": "observed files/sec; remaining passage mix has not been sampled",
            "pending_is_discovered_only": True,
        }
        self.stop_event = threading.Event()
        self.background_paused = threading.Event()
        self.worker_thread = None
        if self.neural_unavailable:
            self.progress.update(
                state="lexical_only", phase="resource_gate", reason=self.neural_unavailable
            )

    def _open_neural(self):
        check_resources()
        runtime, fallback = qualified_runtime(
            self.selection["encoder"],
            self.selection["device"],
            observer=self.neural_activity,
            activation_home=self.home / "activation-view",
        )
        reranker = (
            NeuralRuntime("reranker", device=self.selection["device"])
            if self.selection["reranker_enabled"]
            else None
        )
        space = hashlib.sha256(runtime.space_version.encode()).hexdigest()[:24]
        space_home = self.home / "spaces" / space
        store = IndexStore(
            space_home / "index.sqlite3",
            encoder_id=runtime.space_version,
            derived_dir=DEFAULT_CACHE / "metadata" / space,
            resource_guard=check_resources,
        )
        try:
            with store.db:
                store.set_meta("semantic_threshold", str(self.selection["semantic_threshold"]))
            vectors = VectorStore(
                DEFAULT_CACHE / "indexes" / space / store.cache_uuid,
                space_home / "publication" / store.cache_uuid,
                encoder_id=runtime.space_version,
                resource_guard=check_resources,
                cache_uuid=store.cache_uuid,
            )
            check_resources()
        except (OSError, RuntimeError, ValueError):
            store.close()
            raise
        self.neural_activity.set_model(runtime.model_info)
        return {
            "runtime": runtime,
            "reranker": reranker,
            "space": space,
            "space_home": space_home,
            "store": store,
            "vectors": vectors,
            "device_fallback": fallback,
        }

    def retrieval_runtime(self):
        return (
            SimpleNamespace(
                encode=self.runtime.encode, rerank=self.reranker.rerank if self.reranker else None
            )
            if self.runtime
            else None
        )

    def _weight_reader(self):
        from .static_weights import StaticWeights, static_model_info

        runtime = self.runtime
        if runtime is None:
            if self.selection.get("encoder") != "e5":
                raise RuntimeError("Static parameters require the selected E5 model.")
            info = static_model_info()
            # Recheck current disk metadata; no compiled encoder holds this definition stable.
            reader = StaticWeights(info)
            if self._projection_source:
                self._projection_source.close()
            self._projection_source = reader
            self._projection_space = info["space"]
            return reader
        if runtime.name != "e5" or not runtime.model_info.get("architecture", {}).get("verified"):
            raise RuntimeError("The active E5 architecture is not verified.")
        info = runtime.model_info
        if self._projection_source and self._projection_space != info["space"]:
            self._projection_source.close()
            self._projection_source = None
        if self._projection_source is None:
            self._projection_source = StaticWeights(info)
            self._projection_space = info["space"]
        return self._projection_source

    def activity_architecture(self):
        architecture = self.neural_activity.architecture()
        if self.runtime is not None:
            return {**architecture, "runtime_available": True, "runtime_reason": None}
        metadata = self.projection_descriptors()
        return {
            **architecture,
            "model": {
                "name": "e5",
                "revision": metadata["model_revision"],
                "space": metadata["model_space"],
                "device": [],
            },
            "verified": True,
            "capture_level": "final",
            "qualification_reason": "Encoder unavailable; static metadata only. No live activations.",
            "runtime_available": False,
            "runtime_reason": self.neural_unavailable,
        }

    def activity_snapshot(self):
        return {
            **self.neural_activity.snapshot(),
            "runtime_available": self.runtime is not None,
            "runtime_reason": self.neural_unavailable,
        }

    def projection_descriptors(self):
        with self._projection_lock:
            return self._weight_reader().descriptors()

    def projection_weights(self, block, projection, limit, output_index):
        # Parameter inspection allocates no disk space or additional encoder runtime.
        if psutil.virtual_memory().available < 5 * GIB:
            raise ResourcePressure("Static parameter read paused: less than 5 GiB available RAM.")
        with self._projection_lock:
            return self._weight_reader().weights(
                block, projection, limit=limit, output_index=output_index
            )

    def close_projections(self):
        with self._projection_lock:
            if self._projection_source:
                self._projection_source.close()
                self._projection_source = None

    async def recover_neural(self, engine, *, index=True):
        if self.runtime is not None:
            return False
        try:
            opened = await asyncio.to_thread(self._open_neural)
        except (OSError, RuntimeError, ValueError) as error:
            self.neural_unavailable = f"{type(error).__name__}: {error}"
            self.progress.update(
                state="lexical_only", phase="resource_gate", reason=self.neural_unavailable
            )
            return False
        # Publish one matching embedding space; existing route ownership/cancellation state stays intact.
        self.__dict__.update(opened)
        engine.index, engine.vectors, engine.runtime = (
            self.store,
            self.vectors,
            self.retrieval_runtime(),
        )
        self.neural_unavailable = None
        self.progress.update(state="starting" if index else "ready", phase="neural_recovered")
        self.progress.pop("reason", None)
        if index and (self.worker_thread is None or not self.worker_thread.is_alive()):
            self.start_worker()
        return True

    async def watch_recovery(self, engine, *, index=True):
        while not self.stop_event.is_set():
            await asyncio.sleep(30)
            if not self.stop_event.is_set():
                await self.recover_neural(engine, index=index)

    def status(self, scope="full_private"):
        status = {
            **self.progress,
            "ready": True,
            "device": self.runtime.execution_devices if self.runtime else [],
            "model": self.runtime.name if self.runtime else None,
            "space": self.runtime.space_version if self.runtime else None,
            "retrieval": "hybrid" if self.runtime else "lexical_only",
            "neural_unavailable": self.neural_unavailable,
            "device_fallback": self.device_fallback,
            "local_private_preferences_available": bool(self.authorised.profile_sources),
            "profile_gap": self.authorised.profile_error,
            "reranker": "enabled" if self.reranker else "disabled: development tie",
            "memory_default": "on",
            "blocked_routes": list(self.activity.blocked),
            "quota_gib": 20,
            "minimum_available_ram_gib": 5,
            "background_worker_alive": self.worker_thread is not None
            and self.worker_thread.is_alive(),
            "telemetry": self.neural_activity.telemetry(),
        }
        if scope == "shared":
            public = {
                key: status[key]
                for key in (
                    "ready",
                    "device",
                    "model",
                    "retrieval",
                    "memory_default",
                    "quota_gib",
                    "minimum_available_ram_gib",
                    "reranker",
                    "device_fallback",
                )
            }
            return {**public, "state": "lexical_only" if self.runtime is None else "ready"}
        return status

    def topics(self, scope):
        # Community publication is optional until its independently versioned job finishes.
        if not hasattr(self.store, "topics"):
            return {"generation": 0, "topics": [], "counts": {}, "state": "building"}
        deadline = time.monotonic() + 0.8
        try:
            with defer_sqlite_busy(), self.store.read_snapshot(timeout=0.05) as reader:
                reader.db.set_progress_handler(lambda: int(time.monotonic() >= deadline), 1000)
                result = reader.topics(
                    ("shared",) if scope == "shared" else ("shared", "private"), limit=128
                )
        except sqlite3.OperationalError as error:
            if getattr(error, "sqlite_errorcode", None) == sqlite3.SQLITE_INTERRUPT:
                raise ResourcePressure("Topic metadata refresh reached its time budget.") from error
            raise
        with self.authorised.projection_reader(deadline) as authorised:
            return self._topic_projection(result, scope, authorised=authorised, deadline=deadline)

    def _topic_projection(self, result, scope, *, authorised=None, deadline=None):
        authorised = authorised or self.authorised
        groups, visible = {}, set()
        checked = 0
        limited = False
        for member in result.get("members", []):
            if deadline is not None and time.monotonic() >= deadline:
                limited = True
                break
            node_id = member["source_id"]
            try:
                checked += 1
                if not authorised.authorized(node_id, scope):
                    continue
                current = authorised.read_evidence(node_id, limit=1)
            except ProjectionBudget:
                limited = True
                break
            if not current or member["digest"] != current.get("digest"):
                continue
            visible.add(node_id)
            topic = groups.setdefault(
                member["community_id"],
                {
                    "id": member["community_id"],
                    "label": Path(member["path"]).stem,
                    "label_basis": "representative filename",
                    "members": [],
                },
            )
            topic["members"].append({"node_id": node_id})
        if limited and not visible:
            raise ResourcePressure("Topic preview is waiting for current source validation; retry.")
        eligible_edges = [
            {**edge, "score": edge["weight"]}
            for edge in result.get("edges", [])
            if edge["source"] in visible and edge["target"] in visible
        ]
        edges = eligible_edges[:200]
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    sorted(
                        (member["node_id"], topic["id"])
                        for topic in groups.values()
                        for member in topic["members"]
                    ),
                    edges,
                ],
                sort_keys=True,
                default=str,
            ).encode()
        ).hexdigest()
        progress = getattr(self, "progress", {})
        partial = bool(
            result.get("coverage", {}).get("partial")
            or limited
            or len(visible) < result.get("coverage", {}).get("covered", len(visible))
            or progress.get("pending", 0)
            or not progress.get("inventory_cycle_complete", True)
        )
        return {
            "generation": fingerprint,
            "topics": list(groups.values()),
            "edges": edges,
            "counts": {
                **result.get("coverage", {}),
                "visible": len(visible),
                "sample_limit": 128,
                "sampled_sources": len(result.get("members", [])),
                "checked_sources": checked,
                "validation_budget_limited": limited,
                "eligible_library_files": progress.get("total_eligible_files"),
                "population": "published indexed sources",
                "eligible_edges_in_projection": len(eligible_edges),
                "drawn_edges": len(edges),
            },
            "state": "partial" if partial else "ready",
        }

    def start_worker(self):
        self.worker_thread = threading.Thread(
            target=self._index, name="dodo-neural-index", daemon=True
        )
        self.worker_thread.start()

    def pause_background(self, paused):
        if paused:
            self.background_paused.set()
        else:
            self.background_paused.clear()
        return {"paused": self.background_paused.is_set()}

    def _index(self):
        store = worker = None
        stats = {}
        last_write = last_checkpoint = last_topics = 0
        batch_started = None

        def guard(background=True, additional_disk=0):
            if self.background_paused.is_set():
                raise ResourcePressure("Background indexing is paused by the local owner.")
            if self.stop_event.is_set():
                raise ResourcePressure("Service checkpoint requested.")
            result = check_resources(background=background, additional_disk=additional_disk)
            if shutil.disk_usage(self.home.anchor).free < 100 * 1024**3:
                raise ResourcePressure(
                    "Background work paused: C: state drive has less than 100 GiB free."
                )
            return result

        def checkpoint(*, persist=True):
            nonlocal last_write, last_checkpoint, stats
            self.progress["updated_at"] = time.time()
            last = stats.get("last_completed_at")
            self.progress["lag_seconds"] = max(0, time.time() - last) if last else None
            try:
                atomic_json(self.home / "progress.json", self.status())
                self.progress.pop("snapshot_error", None)
            except PermissionError:
                # This human-readable snapshot is derived; SQLite remains authoritative.
                self.progress["snapshot_error"] = "Progress file is held open by another app."
            last_write = time.monotonic()
            if persist and store is not None and last_write - last_checkpoint >= 5:
                saved = {**stats}
                if batch_started is not None:
                    saved["elapsed_active_seconds"] = (
                        stats.get("elapsed_active_seconds", 0) + time.monotonic() - batch_started
                    )
                if not persist_stats(saved):
                    stats = load_stats()
                last_checkpoint = last_write

        def startup_step(operation, phase):
            while not self.stop_event.is_set():
                try:
                    return operation()
                except (ResourcePressure, sqlite3.OperationalError) as error:
                    if isinstance(error, sqlite3.OperationalError) and getattr(
                        error, "sqlite_errorcode", 0
                    ) & 0xFF not in {
                        sqlite3.SQLITE_BUSY,
                        sqlite3.SQLITE_LOCKED,
                    }:
                        raise
                    self.progress.update(
                        state="paused",
                        phase=phase,
                        reason=str(error)
                        if isinstance(error, ResourcePressure)
                        else "Waiting for the neural index lock.",
                    )
                    # A lock retry must not try writing the same locked database.
                    checkpoint(persist=False)
                    self.stop_event.wait(10 if isinstance(error, ResourcePressure) else 1)
            return None

        def load_stats():
            saved = json.loads(store.get_meta("index_progress", "{}"))
            epoch = store.get_meta("index_progress_epoch")
            if (
                epoch is None
                or not isinstance(saved, dict)
                or saved.get("index_progress_epoch") != epoch
            ):
                saved = {}
            saved["index_progress_epoch"] = epoch
            for key in (
                "completed_jobs",
                "embedded",
                "passages",
                "chunks",
                "failures",
                "pauses",
                "stale",
                "source_failures",
            ):
                saved.setdefault(key, 0)
            saved.setdefault("elapsed_active_seconds", 0.0)
            saved.setdefault("recent_batches", [])
            self.progress.update(
                **{key: value for key, value in saved.items() if key != "recent_batches"}
            )
            return saved

        def persist_stats(saved):
            with store.db:
                if saved.get("index_progress_epoch") != store.get_meta("index_progress_epoch"):
                    return False
                store.set_meta("index_progress", json.dumps(saved))
            return True

        def report(event):
            stage = event.pop("stage")
            changed = stage != self.progress.get("phase")
            self.progress.update(state="indexing", phase=stage, reason=None, **event)
            if changed or time.monotonic() - last_write >= 1:
                checkpoint()

        self.progress.update(state="starting", phase="opening_state")
        checkpoint()
        try:
            store = startup_step(
                lambda: IndexStore(
                    self.space_home / "index.sqlite3",
                    encoder_id=self.runtime.space_version,
                    derived_dir=DEFAULT_CACHE / "metadata" / self.space,
                    resource_guard=guard,
                ),
                "opening_state",
            )
            if store is None:
                return
            stats = startup_step(load_stats, "opening_progress")
            if stats is None:
                return
            with store.db:
                store.set_meta("semantic_threshold", str(self.selection["semantic_threshold"]))
            while not self.stop_event.is_set():
                try:
                    guard()
                    break
                except ResourcePressure as error:
                    self.progress.update(state="paused", phase="resource_gate", reason=str(error))
                    checkpoint()
                    self.stop_event.wait(10)
            if self.stop_event.is_set():
                return
            self.progress.update(phase="opening_vectors")
            checkpoint()
            vectors = VectorStore(
                DEFAULT_CACHE / "indexes" / self.space / store.cache_uuid,
                self.space_home / "publication" / store.cache_uuid,
                encoder_id=self.runtime.space_version,
                resource_guard=guard,
                cache_uuid=store.cache_uuid,
            )
            worker = startup_step(
                lambda: NeuralWorker(
                    self.catalogue.database,
                    store,
                    vectors,
                    self.runtime.encode,
                    self.runtime.tokenizer,
                    resource_guard=guard,
                    scope_resolver=self.authorised.source_scope,
                    blocked_roots=(
                        DEFAULT_CACHE,
                        self.home,
                        Path(__file__).resolve().parents[2] / ".venv",
                    ),
                    progress_callback=report,
                ),
                "opening_worker",
            )
            if worker is None:
                return
            # Worker startup may invalidate counters for a changed centroid policy.
            stats = startup_step(load_stats, "opening_progress")
            if stats is None:
                return
            self.authorised.providers = worker.source_providers
            self.progress.update(phase="counting_eligible_sources")
            checkpoint()
            counts = worker.catalogue.execute(
                """SELECT count(*) FROM nodes n JOIN nodes r ON r.id=n.root_id
                WHERE n.kind='file' AND n.state='indexed' AND n.digest IS NOT NULL
                AND r.state='online'"""
            ).fetchone()[0]
            stats["total_catalogue_files"] = counts
            stats["total_catalogue_passages"] = worker.catalogue.execute(
                "SELECT count(*) FROM passages"
            ).fetchone()[0]
            self.progress.update(
                total_catalogue_files=counts,
                total_catalogue_passages=stats["total_catalogue_passages"],
                total_eligible_is_upper_bound=True,
                completed_files=store.db.execute(
                    "SELECT count(*) FROM sources WHERE active=1 AND ready=1"
                ).fetchone()[0],
                total_eligible_files=counts
                + store.db.execute(
                    "SELECT count(*) FROM sources WHERE active=1 AND provider<>'catalogue'"
                ).fetchone()[0],
                pending=store.pending_count(),
            )
            checkpoint()
            while not self.stop_event.is_set():
                if self.activity.active:
                    self.progress.update(state="yielding", phase="client_request")
                    if time.monotonic() - last_write >= 1:
                        checkpoint()
                    self.stop_event.wait(0.2)
                    continue
                try:
                    batch_started = time.monotonic()
                    report({"stage": "scanning", "pending": store.pending_count()})
                    scan = worker.scan(limit=256)
                    self.progress["unavailable_providers"] = scan["unavailable_providers"]
                    self.progress["refresh"] = worker.refresh_changed(limit=128)
                    report({"stage": "indexing", "pending": store.pending_count()})
                    batch = worker.run_pending(limit=32)
                    elapsed = time.monotonic() - batch_started
                    batch_started = None
                    stats["elapsed_active_seconds"] += elapsed
                    for key in ("embedded", "passages", "chunks", "stale"):
                        stats[key] += batch[key]
                    stats["completed_jobs"] += batch["completed"]
                    stats["source_failures"] += batch["failed"]
                    if batch["completed"]:
                        stats["last_completed_at"] = time.time()
                        stats["recent_batches"] = [
                            *stats["recent_batches"],
                            {
                                "files": batch["completed"],
                                "passages": batch["passages"],
                                "seconds": elapsed,
                            },
                        ][-20:]
                    samples = stats["recent_batches"]
                    sample_seconds = sum(row["seconds"] for row in samples)
                    rate = (
                        sum(row["files"] for row in samples) / sample_seconds
                        if sample_seconds
                        else 0
                    )
                    passage_rate = (
                        sum(row["passages"] for row in samples) / sample_seconds
                        if sample_seconds
                        else 0
                    )
                    ready = store.db.execute(
                        "SELECT count(*) FROM sources WHERE active=1 AND ready=1"
                    ).fetchone()[0]
                    external = store.db.execute(
                        "SELECT count(*) FROM sources WHERE active=1 AND provider<>'catalogue'"
                    ).fetchone()[0]
                    total = stats["total_catalogue_files"] + external
                    self.progress.update(
                        state="indexing"
                        if batch["pending"] or not scan["complete"]
                        else "watching",
                        phase="batch_complete",
                        reason=None,
                        completed_files=ready,
                        total_eligible_files=total,
                        **{key: value for key, value in stats.items() if key != "recent_batches"},
                        pending=batch["pending"],
                        rate_per_second=rate,
                        files_per_second=rate,
                        passages_per_second=passage_rate,
                        eta_seconds=max(0, total - ready) / rate if rate else None,
                        last_batch=batch,
                        inventory_cycle_complete=scan["complete"],
                    )
                    if not persist_stats(stats):
                        stats = load_stats()
                    if time.monotonic() - last_topics > 30 and batch["completed"]:
                        report({"stage": "publishing_topics"})
                        self.progress["topics"] = publish_communities(store, resource_guard=guard)
                        last_topics = time.monotonic()
                    checkpoint()
                    self.stop_event.wait(0.5 if batch["completed"] else 5)
                except ResourcePressure as error:
                    if batch_started is not None:
                        stats["elapsed_active_seconds"] += time.monotonic() - batch_started
                        batch_started = None
                    stats["pauses"] += 1
                    self.progress.update(state="paused", reason=str(error))
                    checkpoint()
                    self.stop_event.wait(10)
        except (OSError, RuntimeError, ValueError, sqlite3.Error) as error:
            stats["failures"] = stats.get("failures", 0) + 1
            self.progress.update(
                state="failed",
                failures=stats["failures"],
                reason=f"{type(error).__name__}: {error}",
            )
            checkpoint()
        finally:
            if worker is not None:
                worker.close()
            if store is not None:
                store.close()


async def serve(home=STATE, *, index=True):
    from .launcher import watch_shutdown

    preflight_ports()
    routes, credentials = load_routes(home)
    service = OrganisationService(home)

    runtime = service.retrieval_runtime()

    async def tokens(text, route_id):
        if route_id == "local-inspection":
            return len(text.encode("utf-8")) + 384
        route = next(route for route in routes if route.id == route_id)
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False, timeout=5) as client:
            return await model_token_count(route, text, client)

    engine = RetrievalEngine(service.authorised, service.store, service.vectors, runtime, tokens)
    broker = create_broker_app(
        engine,
        credentials,
        routes=routes,
        status=service.status,
        topics=service.topics,
        background_control=service.pause_background,
        activity=service.neural_activity,
        projection_source=service,
    )
    proxy = create_proxy_app(routes, credentials, activity=service.activity)
    if index and service.runtime is not None:
        service.start_worker()
    servers = [
        uvicorn.Server(
            uvicorn.Config(
                app,
                host="127.0.0.1",
                port=port,
                access_log=False,
                log_level="warning",
                log_config=log_configuration(home),
            )
        )
        for app, port in ((broker, 8770), (proxy, 8771))
    ]
    shutdown = asyncio.create_task(watch_shutdown(home, service, servers))
    recovery = asyncio.create_task(service.watch_recovery(engine, index=index))
    try:
        await asyncio.gather(*(server.serve() for server in servers))
    finally:
        shutdown.cancel()
        recovery.cancel()
        service.stop_event.set()
        await asyncio.to_thread(service.close_projections)
