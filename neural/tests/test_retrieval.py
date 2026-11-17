from pathlib import Path

import pytest

from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue
from dodo_mem_neural.resources import ResourcePressure
from dodo_mem_neural.retrieval import RetrievalEngine, SourceChanged


class Catalogue:
    def __init__(self):
        self.reads = []
        self.changed = False
        self.nodes = {
            "a": {"id": "a", "name": "a", "digest": "v1", "scope": "shared"},
            "b": {"id": "b", "name": "b", "digest": "v1", "scope": "private"},
            "c": {"id": "c", "name": "c", "digest": "v1", "scope": "shared"},
        }

    def authorized(self, node_id, scope):
        return scope == "full_private" or self.nodes[node_id]["scope"] == "shared"

    def node(self, node_id):
        return self.nodes.get(node_id)

    def search(self, query, limit, scope):
        return []

    def read_evidence(self, node_id, *, limit, start_line=1):
        self.reads.append(node_id)
        digest = "v2" if self.changed and self.reads.count(node_id) > 1 else "v1"
        return {
            "path": f"vault/{node_id}.md",
            "digest": digest,
            "locator": {"line": start_line},
            "text": ("verified " + node_id)[:limit],
        }

    def neighbours(self, node_id, *, limit, include_inferred=False):
        if node_id != "a":
            return []
        return [{"node": self.nodes["c"], "kind": "reference", "evidence": {"line": 1}}]


class Index:
    def lexical_search(self, query, scopes, limit):
        assert limit == 8
        return [{"node_id": "a", "digest": "v1", "start": 1, "text": "INDEX TEXT MUST NOT LEAK"}]

    def visible_occurrences(self, chunk_ids, scopes, *, limit):
        assert limit == 8
        return [{"node_id": "b", "digest": "v1", "start": 1, "scope": "private"}]


class Vectors:
    def search(self, vector, k):
        assert k == 8
        return [(1, 1.0)]


class Runtime:
    def encode(self, texts, query=False):
        assert query
        return [[1.0]]

    def rerank(self, query, passages):
        return list(range(len(passages), 0, -1))


async def token_count(text, model):
    return len(text)


async def test_retrieval_filters_scope_reads_current_sources_and_records_graph_path():
    catalogue = Catalogue()
    engine = RetrievalEngine(catalogue, Index(), Vectors(), Runtime(), token_count)
    packet = await engine.retrieve("question", "shared", "model")
    assert "b" not in catalogue.reads
    assert "INDEX TEXT" not in packet.memory_text
    assert "verified a" in packet.memory_text
    assert [source["node_id"] for source in packet.sources] == ["a", "c"]
    assert packet.sources[1]["relationship_path"][0]["kind"] == "reference"
    assert len(packet.memory_text) <= 1536
    assert packet.coverage["token_count"] <= 1536


async def test_retrieval_fails_closed_when_source_changes_before_export():
    catalogue = Catalogue()
    catalogue.changed = True
    engine = RetrievalEngine(catalogue, Index(), Vectors(), Runtime(), token_count)
    with pytest.raises(SourceChanged):
        await engine.retrieve("question", "shared", "model")


async def test_three_hop_commit_evidence_path_is_invalid_after_revocation():
    class ProjectCatalogue(Catalogue):
        revoked = False

        def __init__(self):
            super().__init__()
            self.nodes = {
                key: {"id": key, "digest": "v1", "scope": "shared"}
                for key in ("commit", "file", "decision", "document")
            }

        def search(self, query, *, limit, scope):
            return [self.nodes["commit"]]

        def authorized(self, node_id, scope):
            return not (self.revoked and node_id == "decision")

        def neighbours(self, node_id, *, limit, include_inferred=False):
            connections = {
                "commit": ("file", "changed_file"),
                "file": ("decision", "reference"),
                "decision": ("document", "citation"),
            }
            if node_id not in connections:
                return []
            target, kind = connections[node_id]
            return [{"node": self.nodes[target], "kind": kind, "evidence": {"line": 1}}]

    catalogue = ProjectCatalogue()
    engine = RetrievalEngine(catalogue, None, None, None, _quarter_tokens)
    packet = await engine.retrieve("trace the design", "shared", "model")
    document = next(source for source in packet.sources if source["node_id"] == "document")
    assert [edge["kind"] for edge in document["relationship_path"]] == [
        "changed_file",
        "reference",
        "citation",
    ]
    catalogue.revoked = True
    with pytest.raises(SourceChanged):
        engine._verify(packet.sources, "shared")
    refreshed = await engine.retrieve("trace the design", "shared", "model")
    assert {source["node_id"] for source in refreshed.sources} == {"commit", "file"}


async def _quarter_tokens(text, model):
    return (len(text.encode()) + 3) // 4


async def test_existing_catalogue_bootstraps_lexical_retrieval_without_sidecar_rows():
    class ExistingCatalogue(Catalogue):
        def search(self, query, limit, scope):
            assert (limit, scope) == (8, "shared")
            return [self.nodes["a"]]

    class EmptyIndex(Index):
        def lexical_search(self, query, scopes, limit):
            return []

        def visible_occurrences(self, chunk_ids, scopes, *, limit):
            return []

    engine = RetrievalEngine(ExistingCatalogue(), EmptyIndex(), Vectors(), Runtime(), token_count)
    packet = await engine.retrieve("question", "shared", "model")
    assert packet.sources[0]["node_id"] == "a"
    assert packet.sources[0]["channels"] == ["lexical"]


async def test_path_inspector_revalidates_current_source_before_export():
    catalogue = Catalogue()
    engine = RetrievalEngine(catalogue, Index(), Vectors(), Runtime(), token_count)
    await engine.retrieve("question", "shared", "model")
    catalogue.changed = True
    with pytest.raises(SourceChanged):
        engine.paths("shared")


async def test_missing_neural_backends_preserve_verified_lexical_graph_with_coverage_gap():
    engine = RetrievalEngine(Catalogue(), Index(), None, None, token_count)
    packet = await engine.retrieve("question", "shared", "model")
    assert [source["node_id"] for source in packet.sources] == ["a", "c"]
    assert packet.coverage["dense_available"] is False
    assert packet.coverage["rerank_applied"] is False
    assert packet.coverage["coverage_gaps"] == ["dense_unavailable", "reranker_unavailable"]


async def test_live_vector_drive_disconnect_preserves_verified_lexical_results():
    class DisconnectedVectors:
        def search(self, *args, **kwargs):
            raise OSError("derived drive disconnected")

    engine = RetrievalEngine(Catalogue(), Index(), DisconnectedVectors(), Runtime(), token_count)
    packet = await engine.retrieve("question", "shared", "model")
    assert [source["node_id"] for source in packet.sources] == ["a", "c"]
    assert packet.coverage["dense_available"] is False
    assert "dense_unavailable" in packet.coverage["coverage_gaps"]


@pytest.mark.parametrize("error", (OSError("drive offline"), ResourcePressure("drive offline")))
async def test_disconnected_sidecar_preserves_c_catalogue_and_skips_dense_work(error):
    class ExistingCatalogue(Catalogue):
        def search(self, query, *, limit, scope):
            return [self.nodes["a"]]

    class OfflineIndex(Index):
        def lexical_search(self, *args, **kwargs):
            raise error

    class UnusedRuntime(Runtime):
        def encode(self, *args, **kwargs):
            raise AssertionError("An offline sidecar cannot resolve dense results")

    engine = RetrievalEngine(
        ExistingCatalogue(), OfflineIndex(), Vectors(), UnusedRuntime(), token_count
    )
    packet = await engine.retrieve("question", "shared", "model")
    assert [source["node_id"] for source in packet.sources] == ["a", "c"]
    assert packet.coverage["dense_available"] is False
    assert "sidecar_unavailable" in packet.coverage["coverage_gaps"]
    assert "dense_unavailable" in packet.coverage["coverage_gaps"]


async def test_invalid_sidecar_artifact_is_not_treated_as_an_offline_drive():
    class InvalidIndex(Index):
        def lexical_search(self, *args, **kwargs):
            raise ValueError("invalid artifact")

    engine = RetrievalEngine(Catalogue(), InvalidIndex(), None, None, token_count)
    with pytest.raises(ValueError, match="invalid artifact"):
        await engine.retrieve("question", "shared", "model")


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("CoNtExTuAlIzE wItH mEm: write a short message", "write a short message"),
        ("write a short message, contextualise with mem", "write a short message"),
        ("  implement this parser  ", "  implement this parser  "),
        (
            "CONTEXTUALISE WITH MEM",
            "Summarise my personal memory and distinguish preferences from inferred interests.",
        ),
    ],
)
async def test_existing_memory_command_is_normalised_once_for_all_retrieval_stages(
    question, expected, monkeypatch
):
    monkeypatch.syspath_prepend(
        str(Path(__file__).resolve().parents[2] / "dodo-memory-desktop/src")
    )
    normalisations = []

    class CommandCatalogue(Catalogue):
        def normalise_query(self, query):
            normalisations.append(query)
            return AuthorisedCatalogue.normalise_query(self, query)

        def search(self, query, *, limit, scope):
            assert query == expected
            return [self.nodes["a"]]

    class CommandIndex(Index):
        def lexical_search(self, query, *args, **kwargs):
            assert query == expected
            return super().lexical_search(query, *args, **kwargs)

    class CommandRuntime(Runtime):
        def encode(self, texts, query=False):
            assert texts == [expected]
            return super().encode(texts, query=query)

        def rerank(self, query, passages):
            assert query == expected
            return super().rerank(query, passages)

    engine = RetrievalEngine(
        CommandCatalogue(), CommandIndex(), Vectors(), CommandRuntime(), token_count
    )
    packet = await engine.retrieve(question, "shared", "model")
    assert normalisations == [question]
    assert [source["node_id"] for source in packet.sources] == ["a", "c"]


async def test_byte_budget_shrinks_excerpt_instead_of_returning_false_no_matches():
    class UnicodeCatalogue(Catalogue):
        def read_evidence(self, node_id, *, limit, start_line=1):
            row = super().read_evidence(node_id, limit=limit, start_line=start_line)
            row["text"] = "é" * limit
            return row

    async def byte_count(text, model):
        return len(text.encode("utf-8"))

    engine = RetrievalEngine(UnicodeCatalogue(), Index(), None, None, byte_count)
    packet = await engine.retrieve("question", "shared", "local-inspection")
    assert packet.sources
    assert len(packet.sources[0]["excerpt"]) >= 64
    assert len(packet.memory_text.encode("utf-8")) <= 1536
