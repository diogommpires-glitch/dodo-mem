import hashlib
import json
import sqlite3
from types import SimpleNamespace

import numpy as np
import pytest

from dodo_mem_neural.artifacts import CANDIDATES, verify
from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue
from dodo_mem_neural.index_store import IndexStore
from dodo_mem_neural.organisation import publish_communities
from dodo_mem_neural.service import OrganisationService
from dodo_mem_neural.sqlite_history import ClineHistory


class CurrentSources:
    def __init__(self, paths):
        self.paths = paths
        self.nodes = {}

    def authorized(self, source_id, scope):
        return source_id in self.paths and scope == "full_private"

    def node(self, source_id):
        path = self.paths[source_id]
        if source_id not in self.nodes:
            self.nodes[source_id] = {
                "id": source_id,
                "path": str(path),
                "digest": hashlib.sha256(path.read_bytes()).hexdigest(),
                "state": "indexed",
            }
        return self.nodes[source_id]

    def read_evidence(self, source_id, *, limit=1000, start_line=1):
        node = self.node(source_id)
        if hashlib.sha256(self.paths[source_id].read_bytes()).hexdigest() != node["digest"]:
            return None
        return {**node, "text": self.paths[source_id].read_text()[:limit]}


def test_topics_omit_members_changed_since_community_publication(tmp_path):
    paths = {name: tmp_path / f"{name}.md" for name in ("changed", "current")}
    current = CurrentSources(paths)
    store = IndexStore(tmp_path / "state.db", encoder_id="review")
    try:
        vector = np.zeros(384, dtype=np.float32)
        vector[0] = 1
        for source_id, path in paths.items():
            path.write_text("initial source text")
            node = current.node(source_id)
            store.queue_source(source_id, node["digest"], path, "private", "v1")
            store.save_centroid(source_id, "v1", vector, 1)
            store.complete_source(source_id, "v1")
        publish_communities(store)
        service = OrganisationService.__new__(OrganisationService)
        service.store = store
        service.authorised = AuthorisedCatalogue(current, SimpleNamespace(root=tmp_path / "shared"))
        paths["changed"].write_text("new unrelated source contents")

        result = service.topics("full_private")

        assert [member["node_id"] for topic in result["topics"] for member in topic["members"]] == [
            "current"
        ]
        assert result["counts"]["visible"] == 1
    finally:
        store.close()


def test_cline_dotdot_path_cannot_escape_session_root(tmp_path):
    data = tmp_path / "cline-data"
    (data / "db").mkdir(parents=True)
    (data / "sessions").mkdir()
    outside = data / "db/outside.json"
    outside.write_text(json.dumps({"messages": [{"role": "user", "content": "outside"}]}))
    database = data / "db/sessions.db"
    db = sqlite3.connect(database)
    try:
        db.execute(
            "CREATE TABLE sessions(session_id TEXT PRIMARY KEY,started_at TEXT,"
            "updated_at TEXT,prompt TEXT,messages_path TEXT)"
        )
        escaped = data / "sessions/../db/outside.json"
        db.execute("INSERT INTO sessions VALUES ('one','a','b','',?)", (str(escaped),))
        db.commit()
        provider = ClineHistory(database, tmp_path / "snapshots")

        assert provider.node(provider.prefix + "one") is None
        assert provider.read_evidence(provider.prefix + "one") is None
    finally:
        db.close()


def test_model_manifest_cannot_omit_the_artifacts_that_runtime_loads(tmp_path):
    target = tmp_path / "e5"
    target.mkdir()
    repository, revision, _, licence = CANDIDATES["e5"]
    manifest = {"repository": repository, "revision": revision, "licence": licence, "files": []}
    (target / "download-manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="[Aa]rtifact|[Rr]equired|[Mm]anifest"):
        verify(target)
