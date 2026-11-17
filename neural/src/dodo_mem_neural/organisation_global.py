"""Cold global Leiden jobs use disk-backed snapshots and bounded owned processes."""

import argparse
import hashlib
import importlib.util
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import sysconfig
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import igraph
import numpy as np
import psutil

from .artifacts import atomic_json, sha256
from .resources import DEFAULT_CACHE, GIB, ResourcePressure, check_resources

VERSION = "global-mutual8-leiden-modularity-resolution1-seed42-v1"


def cold_interpreter():
    # Venv executables on Windows redirect to another PID; monitor the real process instead.
    executable = Path(sys._base_executable).resolve(strict=True)
    if executable.parent != Path(sys.base_prefix).resolve(strict=True):
        raise ValueError("Cold interpreter must be the current validated base Python executable.")
    return executable


def cold_environment(directory):
    source = Path(__file__).resolve().parents[1]
    packages = Path(sysconfig.get_path("purelib")).absolute()
    if packages.resolve(strict=True) != packages:
        raise ValueError("Cold dependency path cannot contain a redirect.")
    environment = {
        name: os.environ[name]
        for name in (
            "SYSTEMROOT",
            "WINDIR",
            "PATH",
            "USERPROFILE",
            "HOMEDRIVE",
            "HOMEPATH",
            "LOCALAPPDATA",
            "APPDATA",
            "LANG",
            "LC_ALL",
            "DODO_MEM_RUNTIME_ROOT",
            "DODO_MEM_LIBRARY_ROOTS",
        )
        if name in os.environ
    }
    desktop = Path(importlib.util.find_spec("dodo_memory_desktop").origin).resolve().parent.parent
    gateway = Path(importlib.util.find_spec("dodo_memory").origin).resolve().parent.parent
    environment.update(
        PYTHONPATH=os.pathsep.join((str(source), str(desktop), str(gateway), str(packages))),
        PYTHONNOUSERSITE="1",
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        TEMP=str(directory),
        TMP=str(directory),
    )
    return environment


@contextmanager
def _exclusive(path):
    with path.open("a+b") as lock:
        if os.name == "nt":
            import msvcrt

            lock.seek(0)
            if lock.read(1) == b"":
                lock.write(b"0")
                lock.flush()
            lock.seek(0)
            try:
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError as error:
                raise ResourcePressure("A global community job is already running.") from error
        else:
            import fcntl

            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError as error:
                raise ResourcePressure("A global community job is already running.") from error
        yield


def _snapshot(store, directory, guard):
    snapshot = directory / "graph.sqlite3"
    source = sqlite3.connect(store.path.absolute().as_uri() + "?mode=ro", uri=True, timeout=10)
    output = sqlite3.connect(snapshot)
    try:
        source.execute("BEGIN")
        if source.execute("""SELECT 1 FROM semantic_neighbours GROUP BY source
            HAVING count(*)>8 LIMIT 1""").fetchone():
            raise ValueError("Semantic graph exceeds the eight-neighbour bound.")
        output.executescript("""
            PRAGMA synchronous=FULL;
            PRAGMA journal_mode=DELETE;
            CREATE TABLE nodes(ordinal INTEGER PRIMARY KEY,source_id TEXT UNIQUE,
                source_version TEXT,prior TEXT);
            CREATE TABLE assignments(ordinal INTEGER PRIMARY KEY,membership INTEGER);
            CREATE TABLE identities(membership INTEGER PRIMARY KEY,community_id TEXT UNIQUE);
        """)
        nodes = source.execute("""
            SELECT c.source_id,c.version,p.community_id FROM centroids c
            JOIN sources s ON s.source_id=c.source_id AND s.version=c.version
            LEFT JOIN communities p ON p.source_id=c.source_id
            WHERE s.active=1 AND s.ready=1 ORDER BY c.source_id
        """)
        count = 0
        while rows := nodes.fetchmany(4096):
            guard(background=True, additional_disk=len(rows) * 512)
            output.executemany(
                "INSERT INTO nodes VALUES (?,?,?,?)",
                ((count + i, *row) for i, row in enumerate(rows)),
            )
            count += len(rows)
            output.commit()
        threshold = float(store.get_meta("semantic_threshold", "0.82"))
        edges = source.execute(
            """
            SELECT a.source,a.target,min(a.weight,b.weight) FROM semantic_neighbours a
            JOIN semantic_neighbours b ON b.source=a.target AND b.target=a.source
            JOIN sources s ON s.source_id=a.source AND s.version=a.source_version
            JOIN sources t ON t.source_id=a.target AND t.version=a.target_version
            WHERE a.source<a.target AND s.active=1 AND s.ready=1 AND t.active=1 AND t.ready=1
            AND b.source_version=a.target_version AND b.target_version=a.source_version
            AND a.weight>=? AND b.weight>=? AND a.weight<=1 AND b.weight<=1
            ORDER BY a.source,a.target
        """,
            (threshold, threshold),
        )
        edge_count = 0
        with (
            (directory / "edges.txt").open("w", encoding="ascii") as text,
            (directory / "weights.bin").open("wb") as weights,
        ):
            while rows := edges.fetchmany(4096):
                guard(background=True, additional_disk=len(rows) * 64)
                for a, b, weight in rows:
                    endpoints = output.execute(
                        "SELECT ordinal FROM nodes WHERE source_id IN (?,?) ORDER BY source_id",
                        (a, b),
                    ).fetchall()
                    if len(endpoints) != 2:
                        continue
                    text.write(f"{endpoints[0][0]} {endpoints[1][0]}\n")
                    weights.write(np.asarray([weight], dtype="<f8").tobytes())
                    edge_count += 1
            for stream in (text, weights):
                stream.flush()
                os.fsync(stream.fileno())
        output.execute("CREATE INDEX node_prior ON nodes(prior,ordinal)")
        output.commit()
        return count, edge_count
    finally:
        output.close()
        source.close()


def _cold_limits(node_count, edge_count):
    estimated_ram = 256 * 1024**2 + node_count * 256 + edge_count * 128
    estimated_disk = node_count * 512 + edge_count * 80
    if estimated_ram > 8 * GIB or psutil.virtual_memory().available - estimated_ram < 5 * GIB:
        raise ResourcePressure(
            "Global communities paused: cold job cannot retain 5 GiB RAM headroom."
        )
    return estimated_disk


def _prune(store, directory):
    retained = set()
    for key in ("organisation_global_current", "organisation_global_lastgood"):
        value = store.get_meta(key)
        if value:
            retained.add(Path(json.loads(value)["path"]).parent)
    deferred = 0
    for job in directory.iterdir():
        if (
            not job.is_dir()
            or job.is_symlink()
            or job.resolve().parent != directory
            or not re.fullmatch(r"[a-f0-9]{32}", job.name)
        ):
            continue
        names = {"edges.txt", "weights.bin", "worker.log", "request.json"}
        if job not in retained:
            names |= {"graph.sqlite3", "graph.sqlite3-journal", "result.json"}
        for name in names:
            try:
                (job / name).unlink(missing_ok=True)
            except PermissionError:
                deferred += 1
        if job not in retained:
            try:
                job.rmdir()
            except OSError:
                deferred += 1
    with store.db:
        store.set_meta("organisation_prune_deferred", deferred)


def publish_global(store, directory=None, *, resource_guard=None):
    guard = resource_guard or check_resources
    directory = Path(directory or DEFAULT_CACHE / "communities").absolute()
    if directory.resolve() != directory or directory.is_symlink():
        raise ValueError("Global community storage cannot contain redirects.")
    directory.mkdir(parents=True, exist_ok=True)
    with _exclusive(directory / "global.lock"):
        _prune(store, directory)
        count = store.centroid_count()
        edge_bound = 4 * count
        guard(background=True, additional_disk=_cold_limits(count, edge_bound))
        generation = uuid.uuid4().hex
        job = directory / generation
        job.mkdir()
        count, edges = _snapshot(store, job, guard)
        guard(background=True, additional_disk=_cold_limits(count, edges))
        atomic_json(
            job / "request.json",
            {
                "nodes": count,
                "edges": edges,
                "generation": generation,
                "parent": os.getpid(),
                "parent_started": psutil.Process().create_time(),
                "encoder_id": store.encoder_id,
                "chunker_id": store.chunker_id,
                "inputs": {
                    name: sha256(job / name)
                    for name in ("graph.sqlite3", "edges.txt", "weights.bin")
                },
            },
        )
        with store.db:
            store.set_meta("organisation_global_state", "running")
        environment = cold_environment(job)
        with (job / "worker.log").open("wb") as log:
            process = subprocess.Popen(
                [
                    str(cold_interpreter()),
                    "-S",
                    "-m",
                    "dodo_mem_neural.organisation_global",
                    str(job),
                ],
                stdout=log,
                stderr=log,
                env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
            )
            try:
                while process.poll() is None:
                    try:
                        memory = psutil.Process(process.pid).memory_info()
                    except psutil.NoSuchProcess:
                        break
                    if max(memory.rss, getattr(memory, "private", memory.rss)) > 8 * GIB:
                        raise ResourcePressure(
                            "Global communities exceeded the 8 GiB cold-process limit."
                        )
                    guard(background=True)
                    time.sleep(0.2)
                process.wait(timeout=10)
            except (OSError, ResourcePressure):
                process.terminate()
                process.wait(timeout=10)
                with store.db:
                    store.set_meta("organisation_global_state", "paused: resource gate")
                raise
        if process.returncode:
            with store.db:
                store.set_meta("organisation_global_state", "failed: child exited")
            raise ResourcePressure(
                f"Global community job exited with code {process.returncode}; last-good retained."
            )
        manifest = json.loads((job / "result.json").read_text(encoding="utf-8"))
        result = job / "graph.sqlite3"
        if (
            manifest["generation"] != generation
            or manifest["nodes"] != count
            or sha256(result) != manifest["sha256"]
        ):
            raise ValueError("Global community result checksum/count mismatch.")
        derived = sqlite3.connect(result.absolute().as_uri() + "?mode=ro", uri=True)
        try:
            if derived.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise ValueError("Global community result failed SQLite integrity validation.")
            publication = json.dumps({"path": str(result), **manifest}, separators=(",", ":"))
            with store.db:
                previous = store.get_meta("organisation_global_current")
                if previous:
                    store.set_meta("organisation_global_lastgood", previous)
                changed = 0
                cursor = derived.execute("""
                    SELECT n.source_id,n.source_version,i.community_id FROM nodes n
                    JOIN assignments a ON a.ordinal=n.ordinal
                    JOIN identities i ON i.membership=a.membership ORDER BY n.ordinal
                """)
                while rows := cursor.fetchmany(4096):
                    for source_id, version, community in rows:
                        current = store.db.execute(
                            "SELECT version,active,ready FROM sources WHERE source_id=?",
                            (source_id,),
                        ).fetchone()
                        if not current or tuple(current) != (version, 1, 1):
                            changed += 1
                            continue
                        store.db.execute(
                            """INSERT INTO communities VALUES (?,?,?,?,0)
                            ON CONFLICT(source_id) DO UPDATE SET source_version=excluded.source_version,
                            community_id=excluded.community_id,generation=excluded.generation,partial=0""",
                            (source_id, version, community, generation),
                        )
                if changed:
                    store.db.execute(
                        "UPDATE communities SET partial=1 WHERE generation=?", (generation,)
                    )
                store.set_meta("organisation_global_current", publication)
                store.set_meta("organisation_generation", generation)
                store.set_meta(
                    "organisation_global_state",
                    "partial: source versions changed" if changed else "ready",
                )
                store.set_meta("organisation_method", VERSION)
                store.set_meta("organisation_global_completed_at", time.time())
        finally:
            derived.close()
        _prune(store, directory)
        return {
            "covered": count - changed,
            "total": store.centroid_count(),
            "cohort": count,
            "partial": bool(changed),
            "global": True,
            "generation": generation,
            "reason": "source versions changed during global job" if changed else "complete graph",
            "cold_peak_rss_bytes": manifest["peak_rss_bytes"],
        }


def _watch_parent(request):
    while True:
        try:
            parent = psutil.Process(request["parent"])
            healthy = parent.create_time() == request["parent_started"]
            memory = psutil.Process().memory_info()
            healthy &= max(memory.rss, getattr(memory, "private", memory.rss)) <= 8 * GIB
            healthy &= psutil.virtual_memory().available >= 5 * GIB
            if not healthy:
                os._exit(75)
        except psutil.NoSuchProcess:
            os._exit(75)
        time.sleep(0.2)


def run_cold(directory):
    request = json.loads((directory / "request.json").read_text(encoding="utf-8"))
    if any(sha256(directory / name) != digest for name, digest in request["inputs"].items()):
        raise ValueError("Global community input checksum mismatch.")
    threading.Thread(target=_watch_parent, args=(request,), daemon=True).start()
    database = directory / "graph.sqlite3"
    db = sqlite3.connect(database)
    try:
        db.execute("PRAGMA synchronous=FULL")
        nodes, edges = request["nodes"], request["edges"]
        graph = (
            igraph.Graph.Read_Edgelist(str(directory / "edges.txt"), directed=False)
            if edges
            else igraph.Graph(n=nodes, directed=False)
        )
        if graph.ecount() != edges or graph.vcount() > nodes:
            raise ValueError("Global community graph dimensions differ from its input manifest.")
        if graph.vcount() < nodes:
            graph.add_vertices(nodes - graph.vcount())
        prior = {
            row[0]: number
            for number, row in enumerate(
                db.execute(
                    "SELECT DISTINCT prior FROM nodes WHERE prior IS NOT NULL ORDER BY prior"
                )
            )
        }
        initial = []
        next_id = len(prior)
        for (previous,) in db.execute("SELECT prior FROM nodes ORDER BY ordinal"):
            if previous is None:
                initial.append(next_id)
                next_id += 1
            else:
                initial.append(prior[previous])
        if edges:
            weights = np.memmap(directory / "weights.bin", dtype="<f8", mode="r", shape=(edges,))
            igraph.set_random_number_generator(random.Random(42))
            partition = graph.community_leiden(
                objective_function="modularity",
                weights=weights,
                resolution=1.0,
                n_iterations=-1,
                initial_membership=initial,
            )
            membership = partition.membership
        else:
            membership = range(nodes)
        db.executemany("INSERT INTO assignments VALUES (?,?)", enumerate(membership))
        db.execute("CREATE INDEX assignment_group ON assignments(membership,ordinal)")
        overlap = db.execute("""
            SELECT a.membership,n.prior,count(*) overlap FROM assignments a
            JOIN nodes n ON n.ordinal=a.ordinal WHERE n.prior IS NOT NULL
            GROUP BY a.membership,n.prior ORDER BY overlap DESC,n.prior,a.membership
        """)
        for number, previous, _ in overlap:
            db.execute("INSERT OR IGNORE INTO identities VALUES (?,?)", (number, previous))
        last = None
        digest = None
        for number, source_id in db.execute("""
            SELECT a.membership,n.source_id FROM assignments a JOIN nodes n ON n.ordinal=a.ordinal
            WHERE a.membership NOT IN (SELECT membership FROM identities)
            ORDER BY a.membership,n.source_id
        """):
            if number != last:
                if last is not None:
                    db.execute(
                        "INSERT INTO identities VALUES (?,?)", (last, digest.hexdigest()[:24])
                    )
                digest = hashlib.sha256(VERSION.encode())
                last = number
            digest.update(b"\0" + source_id.encode())
        if last is not None:
            db.execute("INSERT INTO identities VALUES (?,?)", (last, digest.hexdigest()[:24]))
        db.commit()
    finally:
        db.close()
    with database.open("r+b") as stream:
        os.fsync(stream.fileno())
    memory = psutil.Process().memory_info()
    atomic_json(
        directory / "result.json",
        {
            "generation": request["generation"],
            "nodes": nodes,
            "edges": edges,
            "sha256": sha256(database),
            "encoder_id": request["encoder_id"],
            "chunker_id": request["chunker_id"],
            "method": VERSION,
            "inputs": request["inputs"],
            "peak_rss_bytes": memory.peak_wset if os.name == "nt" else memory.rss,
            "peak_commit_bytes": getattr(memory, "peak_pagefile", None),
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    run_cold(parser.parse_args().directory)
