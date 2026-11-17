"""Independent exact-neighbour qualification of the production index settings."""

import json
import time
from importlib.metadata import version

import numpy as np

from dodo_mem_neural.vector_store import VectorStore


def test_production_ann_recall_at_40_against_exhaustive_cosine(tmp_path):
    rng = np.random.default_rng(20261105)
    dimensions, count, query_count, k = 384, 10_000, 100, 40
    # Equal uniform and clustered cohorts exercise distinct neighbourhood geometries.
    centres = rng.normal(size=(64, dimensions)).astype(np.float32)
    centres /= np.linalg.norm(centres, axis=1, keepdims=True)
    vectors = rng.normal(size=(count, dimensions)).astype(np.float32)
    vectors[count // 2 :] = centres[np.arange(count // 2) % len(centres)] + vectors[
        count // 2 :
    ] * np.float32(0.08)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    queries = rng.normal(size=(query_count, dimensions)).astype(np.float32)
    queries[query_count // 2 :] = centres[np.arange(query_count // 2) % len(centres)] + queries[
        query_count // 2 :
    ] * np.float32(0.08)
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)

    store = VectorStore(tmp_path / "cache", tmp_path / "publication", encoder_id="ann-fixture-v1")
    store.add(np.arange(count, dtype=np.uint64), vectors)
    store.publish()
    # Reopening also qualifies the actual f16 on-disk representation.
    reader = VectorStore(tmp_path / "cache", tmp_path / "publication", encoder_id="ann-fixture-v1")
    recalls, times = [], []
    for query in queries:
        scores = np.einsum("ij,j->i", vectors, query)
        exact = set(np.argpartition(scores, -k)[-k:].tolist())
        started = time.perf_counter()
        approximate = {key for key, _ in reader.search(query, k=k)}
        times.append(1000 * (time.perf_counter() - started))
        recalls.append(len(exact & approximate) / k)

    report = {
        "synthetic": True,
        "seed": 20261105,
        "usearch": version("usearch"),
        "vectors": count,
        "dimensions": dimensions,
        "queries": query_count,
        "k": k,
        "recall_at_40": float(np.mean(recalls)),
        "uniform_recall_at_40": float(np.mean(recalls[: query_count // 2])),
        "clustered_recall_at_40": float(np.mean(recalls[query_count // 2 :])),
        "search_p95_ms": float(np.percentile(times, 95)),
    }
    print(json.dumps(report, sort_keys=True))
    assert report["recall_at_40"] >= 0.95, report
