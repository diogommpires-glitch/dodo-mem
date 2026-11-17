"""Bounded hybrid retrieval that exports only current authorised source evidence."""

import asyncio
import json
import math
from collections import deque
from dataclasses import dataclass
from time import monotonic
from typing import Protocol

from .contracts import MEMORY_INSTRUCTIONS, EvidencePacket
from .resources import ResourcePressure

EVIDENCE_EDGES = frozenset({"reference", "citation", "duplicate", "commit_parent", "changed_file"})
UNAVAILABLE = frozenset({"offline", "inaccessible", "protected", "redirected", "placeholder"})


class SourceChanged(RuntimeError):
    pass


class Catalogue(Protocol):
    def authorized(self, node_id: str, scope: str) -> bool: ...
    def node(self, node_id: str) -> dict | None: ...
    def search(self, query: str, *, limit: int, scope: str) -> list[dict]: ...
    def read_evidence(self, node_id: str, *, limit: int, start_line: int = 1) -> dict | None: ...
    def neighbours(self, node_id: str, *, limit: int, include_inferred: bool) -> list[dict]: ...


@dataclass
class Candidate:
    node_id: str
    digest: str
    start_line: int
    channels: list[str]
    trail: tuple[dict, ...] = ()


def _render(sources):
    if not sources:
        return ""
    return MEMORY_INSTRUCTIONS + json.dumps(
        {"sources": sources}, ensure_ascii=False, separators=(",", ":")
    )


def _provenance(evidence, node):
    metadata = {
        key: evidence[key]
        for key in ("historical", "role", "timestamp", "profile")
        if key in evidence
    }
    if node and (node.get("historical") or node.get("category") == "chat"):
        metadata["historical"] = True
    return metadata


class RetrievalEngine:
    def __init__(self, catalogue: Catalogue, index, vectors, runtime, token_count):
        self.catalogue = catalogue
        self.index = index
        self.vectors = vectors
        self.runtime = runtime
        self.token_count = token_count
        self.latest_paths = {"paths": [], "coverage": {}}
        self.diagnostics = {"stage": "idle", "started": monotonic(), "error_type": None}
        self._latest_sources = []

    def _stage(self, name):
        self.diagnostics.update(stage=name, started=monotonic(), error_type=None)

    def _candidate(self, row, scope):
        node_id = row.get("node_id") or row.get("source_id") or row.get("id")
        if not node_id or not self.catalogue.authorized(node_id, scope):
            return None
        node = self.catalogue.node(node_id)
        if not node or node.get("state") in UNAVAILABLE:
            return None
        digest = row.get("digest") or node.get("digest")
        if not digest or digest != node.get("digest"):
            return None
        start = row.get("start_line", row.get("start", row.get("match_start") or 1))
        if type(start) is not int or start < 1:
            return None
        return Candidate(node_id, digest, start, [])

    def _expand(self, candidates, scope):
        queue = deque(candidates.values())
        while queue and len(candidates) < 64:
            parent = queue.popleft()
            if len(parent.trail) == 3:
                continue
            rows = self.catalogue.neighbours(parent.node_id, limit=64, include_inferred=False)
            for row in rows[:64]:
                if row.get("kind") not in EVIDENCE_EDGES:
                    continue
                candidate = self._candidate(row.get("node", {}), scope)
                if candidate is None or candidate.node_id in candidates:
                    continue
                edge = {
                    "source": parent.node_id,
                    "target": candidate.node_id,
                    "source_digest": parent.digest,
                    "target_digest": candidate.digest,
                    "kind": row["kind"],
                    "evidence": row.get("evidence"),
                }
                candidate.trail = (*parent.trail, edge)
                candidate.channels.append("graph")
                candidates[candidate.node_id] = candidate
                queue.append(candidate)
                if len(candidates) == 64:
                    break

    def _verify(self, sources, scope):
        checked_edges = set()
        for source in sources:
            node_id = source["node_id"]
            if not self.catalogue.authorized(node_id, scope):
                raise SourceChanged("Source authorisation changed; retry retrieval")
            current = self.catalogue.read_evidence(
                node_id, limit=1000, start_line=source["start_line"]
            )
            if (
                not current
                or current.get("path") != source["path"]
                or current.get("digest") != source["source_version"]
                or current.get("locator") != source["locator"]
                or not current.get("text", "").startswith(source["excerpt"])
                or _provenance(current, self.catalogue.node(node_id))
                != {
                    key: source[key]
                    for key in ("historical", "role", "timestamp", "profile")
                    if key in source
                }
            ):
                raise SourceChanged("A source changed or became unavailable; retry retrieval")
            for edge in source["relationship_path"]:
                edge_key = json.dumps(edge, sort_keys=True)
                if edge_key in checked_edges:
                    continue
                for endpoint in ("source", "target"):
                    endpoint_id = edge[endpoint]
                    node = self.catalogue.node(endpoint_id)
                    if not node or not self.catalogue.authorized(endpoint_id, scope):
                        raise SourceChanged("A relationship source became unavailable")
                    evidence = self.catalogue.read_evidence(endpoint_id, limit=1)
                    if (
                        not evidence
                        or evidence.get("digest") != edge[f"{endpoint}_digest"]
                        or node.get("state") in UNAVAILABLE
                    ):
                        raise SourceChanged("A relationship source changed; retry retrieval")
                rows = self.catalogue.neighbours(edge["source"], limit=64, include_inferred=False)
                if not any(
                    row.get("node", {}).get("id") == edge["target"]
                    and row.get("kind") == edge["kind"]
                    and row.get("evidence") == edge["evidence"]
                    for row in rows
                ):
                    raise SourceChanged("A relationship changed; retry retrieval")
                checked_edges.add(edge_key)

    async def retrieve(
        self, query: str, scope: str, model_alias: str, *, node_id=None
    ) -> EvidencePacket:
        if scope not in {"shared", "full_private"} or not query.strip() or len(query) > 2000:
            raise ValueError("query must be 1..2000 characters and scope must be authorised")
        normalise_query = getattr(self.catalogue, "normalise_query", None)
        if normalise_query is not None:
            query = normalise_query(query)
        scopes = ("shared",) if scope == "shared" else ("shared", "private")
        self._stage("lexical_search")
        gaps = []
        lexical = self.catalogue.search(query, limit=8, scope=scope)
        if node_id is not None:
            anchor = self._candidate({"node_id": node_id}, scope)
            if anchor is None:
                raise PermissionError(
                    "selected context node is unavailable or outside the credential scope"
                )
            lexical.insert(0, {"node_id": node_id, "digest": anchor.digest, "start_line": 1})
        sidecar_available = self.index is not None
        if sidecar_available:
            try:
                lexical.extend(self.index.lexical_search(query, scopes, limit=8))
            except (OSError, ResourcePressure):
                sidecar_available = False
        if not sidecar_available:
            gaps.append("sidecar_unavailable")
        lexical_candidates = {}
        for row in lexical:
            candidate = self._candidate(row, scope)
            if candidate is not None and candidate.node_id not in lexical_candidates:
                lexical_candidates[candidate.node_id] = row
            if len(lexical_candidates) == 8:
                break
        lexical = list(lexical_candidates.values())
        dense = []
        dense_available = (
            self.runtime is not None and self.vectors is not None and sidecar_available
        )
        if dense_available:
            try:
                self._stage("embedding")
                embedding = await asyncio.to_thread(self.runtime.encode, [query], query=True)
                self._stage("dense_search")
                dense_hits = self.vectors.search(embedding[0], k=8)[:8]
                dense = self.index.visible_occurrences(
                    [item[0] for item in dense_hits], scopes, limit=8
                )
            except (OSError, RuntimeError):
                dense_available = False
        if not dense_available:
            gaps.append("dense_unavailable")
        candidates = {}
        self._stage("candidate_authorisation")
        for channel, rows in (("lexical", lexical), ("dense", dense)):
            for row in rows:
                candidate = self._candidate(row, scope)
                if candidate is None:
                    continue
                if candidate.node_id not in candidates:
                    candidates[candidate.node_id] = candidate
                candidate_channel = (
                    "context" if channel == "lexical" and candidate.node_id == node_id else channel
                )
                candidates[candidate.node_id].channels.append(candidate_channel)
                if len(candidates) == 16:
                    break
        self._stage("graph_expansion")
        self._expand(candidates, scope)
        seeds = [item for item in candidates.values() if not item.trail]
        linked = [item for item in candidates.values() if item.trail]
        ordered = []
        for index in range(max(len(seeds), len(linked))):
            ordered.extend(group[index] for group in (seeds, linked) if index < len(group))
        sources = []
        self._stage("source_reads")
        for candidate in ordered[:16]:
            if not self.catalogue.authorized(candidate.node_id, scope):
                continue
            evidence = self.catalogue.read_evidence(
                candidate.node_id, limit=1000, start_line=candidate.start_line
            )
            if (
                not evidence
                or evidence.get("digest") != candidate.digest
                or not evidence.get("text")
            ):
                continue
            sources.append(
                {
                    "node_id": candidate.node_id,
                    "path": evidence["path"],
                    "source_version": evidence["digest"],
                    "locator": evidence["locator"],
                    "start_line": candidate.start_line,
                    "channels": list(dict.fromkeys(candidate.channels)),
                    "relationship_path": list(candidate.trail),
                    "excerpt": evidence["text"][:1000],
                }
            )
            node = self.catalogue.node(candidate.node_id)
            sources[-1].update(_provenance(evidence, node))
        rerank = self.runtime.rerank if self.runtime is not None else None
        rerank_applied = False
        scores = None
        if sources and rerank is not None:
            try:
                scores = await asyncio.to_thread(
                    rerank, query, [item["excerpt"] for item in sources]
                )
            except (OSError, RuntimeError):
                gaps.append("reranker_unavailable")
        elif rerank is None:
            gaps.append("reranker_unavailable")
        if scores is not None:
            if len(scores) != len(sources) or not all(
                math.isfinite(float(score)) for score in scores
            ):
                raise ValueError("reranker returned invalid scores")
            sources = [
                source
                for _, source in sorted(
                    zip(scores, sources, strict=True), key=lambda item: float(item[0]), reverse=True
                )
            ]
            rerank_applied = True
        selected = []
        tokens = 0
        for source in sources:
            if len(selected) == 8:
                break
            source["cite"] = f"[{len(selected) + 1}]"
            source["excerpt"] = source["excerpt"][:600]
            while True:
                trial = _render([*selected, source])
                self._stage("answer_tokenizer")
                count = await self.token_count(trial, model_alias)
                if type(count) is not int or count < 0:
                    raise ValueError("model tokenizer returned an invalid token count")
                if len(trial) <= 6000 and count <= 1536:
                    selected.append(source)
                    tokens = count
                    break
                if len(source["excerpt"]) <= 64:
                    if "budget_exhausted" not in gaps:
                        gaps.append("budget_exhausted")
                    break
                source["excerpt"] = source["excerpt"][: max(64, len(source["excerpt"]) // 2)]
        self._stage("source_revalidation")
        self._verify(selected, scope)
        coverage = {
            "candidates": len(candidates),
            "reranked": len(sources) if rerank_applied else 0,
            "rerank_applied": rerank_applied,
            "dense_available": dense_available,
            "coverage_gaps": gaps,
            "returned": len(selected),
            "token_count": tokens,
            "memory_scope": scope,
        }
        packet = EvidencePacket(_render(selected), tuple(selected), coverage)
        self._latest_sources = selected
        self.latest_paths = {
            "paths": [
                {key: value for key, value in source.items() if key != "excerpt"}
                for source in selected
            ],
            "coverage": coverage,
        }
        self._stage("complete")
        return packet

    def verify(self, packet, scope):
        self._verify(packet.sources, scope)

    def paths(self, scope):
        previous_scope = self.latest_paths.get("coverage", {}).get("memory_scope")
        if scope == "shared" and previous_scope != "shared":
            return {"paths": [], "coverage": {}}
        self._verify(self._latest_sources, scope)
        return self.latest_paths
