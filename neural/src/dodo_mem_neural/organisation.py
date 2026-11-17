"""Deterministic mutual-neighbour graphs and versioned Leiden communities."""

import hashlib
import json
import random
import threading
import time
import uuid
from collections import Counter

import igraph
import numpy as np
from usearch.index import Index

from .resources import ResourcePressure

_LEIDEN_LOCK = threading.Lock()
ORGANISATION_VERSION = "mutual8-leiden-modularity-resolution1-seed42-v1"


def _passage_support(store, vectors, source_id, version, other_vector):
    if vectors is None:
        return {"unavailable": "passage evidence backfill pending"}
    rows = store.representative_passages(source_id, version)
    available = vectors.contains([row["chunk_id"] for row in rows])
    candidates = [row for row, present in zip(rows, available, strict=True) if present]
    if not candidates:
        return {"unavailable": "passage vectors unavailable"}
    values = np.asarray(vectors.get([row["chunk_id"] for row in candidates]), dtype=np.float32)
    values /= np.maximum(np.linalg.norm(values, axis=1, keepdims=True), 1e-12)
    target = other_vector / max(float(np.linalg.norm(other_vector)), 1e-12)
    scores = values @ target
    selected = int(np.argmax(scores))
    row = candidates[selected]
    digest = store.db.execute(
        "SELECT digest FROM sources WHERE source_id=?", (source_id,)
    ).fetchone()[0]
    return {
        "source_id": source_id,
        "source_version": version,
        "source_digest": digest,
        "chunk_id": row["chunk_id"],
        "locator": row["locator"],
        "start_line": row["start"],
        "end_line": row["end"],
        "text_hash": bytes(row["text_hash"]).hex(),
        "similarity_to_other_centroid": float(np.clip(scores[selected], -1, 1)),
        "sampled_unique_passages": len(candidates),
        "total_unique_passages": row["total"],
        "selection": "closest to other centroid among up to eight longest-token unique passages",
    }


def refresh_neighbours(store, vectors, source_id, *, passage_vectors=None):
    source = store.db.execute(
        """SELECT c.* FROM centroids c JOIN sources s ON s.source_id=c.source_id
        AND s.version=c.version WHERE c.source_id=? AND s.active=1 AND s.ready=1""",
        (source_id,),
    ).fetchone()
    if source is None:
        return []
    hits = vectors.search(np.frombuffer(source["vector"], dtype="<f4"), k=128)
    neighbours = []
    threshold = float(store.get_meta("semantic_threshold", "0.82"))
    for key, score in hits:
        if score < threshold:
            continue
        target = store.db.execute(
            """SELECT c.source_id,c.version FROM centroids c
            JOIN sources s ON s.source_id=c.source_id AND s.version=c.version
            WHERE c.vector_key=? AND s.active=1 AND s.ready=1""",
            (key,),
        ).fetchone()
        if target and target["source_id"] != source_id:
            neighbours.append(
                (source_id, target["source_id"], source["version"], target["version"], score)
            )
        if len(neighbours) == 8:
            break
    with store.db:
        store.db.execute("DELETE FROM semantic_neighbours WHERE source=?", (source_id,))
        store.db.executemany("INSERT INTO semantic_neighbours VALUES (?,?,?,?,?)", neighbours)
    evidence = []
    for _, target, version, target_version, _ in neighbours:
        target_vector = store.db.execute(
            "SELECT vector FROM centroids WHERE source_id=? AND version=?", (target, target_version)
        ).fetchone()
        if target_vector is None:
            continue
        support = {
            "basis": "mutual token-weighted file centroids",
            "embedding_space": store.encoder_id,
            "chunking_version": store.chunker_id,
            "source_passage": _passage_support(
                store,
                passage_vectors,
                source_id,
                version,
                np.frombuffer(target_vector[0], dtype="<f4"),
            ),
            "target_passage": _passage_support(
                store,
                passage_vectors,
                target,
                target_version,
                np.frombuffer(source["vector"], dtype="<f4"),
            ),
        }
        evidence.append(
            (source_id, target, version, target_version, json.dumps(support, separators=(",", ":")))
        )
    store.replace_semantic_support(source_id, evidence)
    return [row[1] for row in neighbours]


def publish_communities(store, *, limit=25_000, resource_guard=None, global_dir=None):
    """Publish global communities when admitted, with bounded incremental previews."""
    global_pending = None
    if store.centroid_count() > 25_000:
        from .organisation_global import publish_global

        previous = store.get_meta("organisation_global_current")
        previous_count = json.loads(previous)["nodes"] if previous else 0
        completed = float(store.get_meta("organisation_global_completed_at", "0"))
        if (
            not previous
            or store.centroid_count() >= 1.25 * previous_count
            or time.time() - completed >= 3600
        ):
            try:
                return publish_global(store, global_dir, resource_guard=resource_guard)
            except ResourcePressure as error:
                global_pending = str(error)
                with store.db:
                    store.set_meta("organisation_global_state", "paused: " + global_pending)
        else:
            global_pending = (
                "global snapshot retained; incremental previews until hourly refresh or 25% growth"
            )
    if resource_guard:
        resource_guard(background=True)
    cursor = store.get_meta("organisation_cursor", "")
    rows = store.centroid_rows(after=cursor, limit=limit)
    if not rows and cursor:
        cursor = ""
        rows = store.centroid_rows(limit=limit)
    identities = [row["source_id"] for row in rows]
    positions = {identity: number for number, identity in enumerate(identities)}
    edges = [
        (positions[source], positions[edge["target"]], edge["weight"])
        for source in identities
        for edge in store.semantic_neighbours(source)
        if edge["target"] in positions and source < edge["target"]
    ]
    previous = {
        row["source_id"]: row["community_id"]
        for row in store.db.execute(
            """SELECT source_id,community_id FROM communities WHERE source_id>?
        ORDER BY source_id LIMIT ?""",
            (cursor, limit),
        )
        if row["source_id"] in positions
    }
    initial_ids = {value: index for index, value in enumerate(sorted(set(previous.values())))}
    initial = []
    for identity in identities:
        prior = previous.get(identity)
        initial.append(initial_ids[prior] if prior else len(initial_ids))
        if prior is None:
            initial_ids[f"new:{identity}"] = len(initial_ids)
    graph = igraph.Graph(n=len(identities), edges=[edge[:2] for edge in edges], directed=False)
    if edges:
        with _LEIDEN_LOCK:
            igraph.set_random_number_generator(random.Random(42))
            try:
                partition = graph.community_leiden(
                    objective_function="modularity",
                    weights=[edge[2] for edge in edges],
                    resolution=1.0,
                    n_iterations=-1,
                    initial_membership=initial,
                )
            finally:
                igraph.set_random_number_generator(None)
        groups = [sorted(identities[index] for index in group) for group in partition]
    else:
        groups = [[identity] for identity in identities]
    # Greedy maximum overlap retains one previous ID per community after splits/merges.
    choices = sorted(
        (
            (-count, prior, number)
            for number, members in enumerate(groups)
            for prior, count in Counter(previous.get(member) for member in members).items()
            if prior is not None
        )
    )
    assigned, used = {}, set()
    for _, prior, number in choices:
        if number not in assigned and prior not in used:
            assigned[number] = prior
            used.add(prior)
    generation = uuid.uuid4().hex
    total = store.centroid_count()
    partial = total > len(identities)
    with store.db:
        for number, members in enumerate(groups):
            community = (
                assigned.get(number)
                or hashlib.sha256(
                    json.dumps([ORGANISATION_VERSION, members], separators=(",", ":")).encode()
                ).hexdigest()[:24]
            )
            store.db.executemany(
                """INSERT INTO communities VALUES (?,?,?,?,?) ON CONFLICT(source_id)
                DO UPDATE SET source_version=excluded.source_version,
                community_id=excluded.community_id,generation=excluded.generation,
                partial=excluded.partial""",
                (
                    (
                        member,
                        rows[positions[member]]["version"],
                        community,
                        generation,
                        int(partial),
                    )
                    for member in members
                ),
            )
        store.set_meta("organisation_cursor", identities[-1] if partial and identities else "")
        store.set_meta("organisation_method", ORGANISATION_VERSION)
        store.set_meta("organisation_generation", generation)
    covered = store.db.execute(
        """SELECT count(*) FROM communities c JOIN sources s ON s.source_id=c.source_id
        AND s.version=c.source_version WHERE s.active=1 AND s.ready=1"""
    ).fetchone()[0]
    return {
        "covered": covered,
        "total": total,
        "partial": partial,
        "cohort": len(identities),
        "generation": generation,
        "reason": "bounded incremental preview" if partial else "complete graph",
        "global_pending": global_pending,
    }


def organise(identities, vectors, *, neighbours=8):
    if not 1 <= neighbours <= 8:
        raise ValueError("Mutual nearest-neighbour degree must be in [1,8]")
    if len(set(identities)) != len(identities):
        raise ValueError("Organisation identities must be unique")
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape != (len(identities), 384):
        raise ValueError("Organisation vectors must have shape (number of identities,384)")
    if not np.isfinite(vectors).all():
        raise ValueError("Organisation vectors must be finite")
    norms = np.linalg.norm(vectors, axis=1)
    if np.any(norms <= 0):
        raise ValueError("Organisation vectors must have positive norms")
    order = sorted(range(len(identities)), key=lambda index: identities[index])
    identities = [identities[index] for index in order]
    vectors = vectors[order] / norms[order, None]
    nearest = []
    if identities:
        index = Index(ndim=384, metric="cos", dtype="f32")
        index.add(np.arange(len(identities), dtype=np.uint64), vectors, threads=1)
        for number, vector in enumerate(vectors):
            matches = index.search(vector, min(neighbours + 1, len(identities)), threads=1)
            candidates = sorted(
                (
                    (int(key), max(0.0, float(1 - distance)))
                    for key, distance in zip(matches.keys, matches.distances, strict=True)
                    if int(key) != number and distance < 1
                ),
                key=lambda item: (-item[1], identities[item[0]]),
            )[:neighbours]
            nearest.append(dict(candidates))
    edges = [
        (source, target, min(weight, nearest[target][source]))
        for source, neighbours_for_source in enumerate(nearest)
        for target, weight in neighbours_for_source.items()
        if source < target and source in nearest[target]
    ]
    edges.sort()
    graph = igraph.Graph(n=len(identities), edges=[edge[:2] for edge in edges], directed=False)
    if edges:
        with _LEIDEN_LOCK:
            igraph.set_random_number_generator(random.Random(42))
            try:
                partition = graph.community_leiden(
                    objective_function="modularity",
                    weights=[edge[2] for edge in edges],
                    resolution=1.0,
                    n_iterations=-1,
                )
            finally:
                igraph.set_random_number_generator(None)
        groups = [[identities[index] for index in group] for group in partition]
    else:
        groups = [[identity] for identity in identities]
    communities = {}
    for members in groups:
        stable = hashlib.sha256(
            json.dumps([ORGANISATION_VERSION, sorted(members)], separators=(",", ":")).encode()
        ).hexdigest()[:24]
        communities.update((member, stable) for member in members)
    return {
        "version": ORGANISATION_VERSION,
        "communities": dict(sorted(communities.items())),
        "edges": [
            (identities[source], identities[target], weight) for source, target, weight in edges
        ],
    }
