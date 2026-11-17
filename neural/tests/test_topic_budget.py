"""Bounded previews retain source checks without multiplying catalogue connections."""

import hashlib
import sqlite3
import time
from types import SimpleNamespace

from dodo_memory_desktop.whole_index import SCHEMA, Catalogue

from dodo_mem_neural import service as service_module
from dodo_mem_neural.catalogue_adapter import AuthorisedCatalogue
from dodo_mem_neural.service import OrganisationService


def test_projection_reuses_one_read_only_connection_without_holding_a_snapshot(
    tmp_path, monkeypatch
):
    database = tmp_path / "catalogue.sqlite3"
    source = tmp_path / "source.md"
    source.write_text("synthetic source")
    original_connect = sqlite3.connect
    with original_connect(database) as db:
        db.executescript(SCHEMA)
        db.execute(
            """INSERT INTO nodes(id,root_id,path,path_key,name,kind,category,state,size,mtime_ns,file_id)
            VALUES ('root','root',?,?,'fixture','root','document','online',0,0,'r')""",
            (str(tmp_path), str(tmp_path)),
        )
        db.execute(
            """INSERT INTO nodes(id,parent_id,root_id,path,path_key,name,kind,category,state,size,mtime_ns,file_id,digest)
            VALUES ('source','root','root',?,?,'source.md','file','document','indexed',16,0,'f',?)""",
            (str(source), str(source), hashlib.sha256(source.read_bytes()).hexdigest()),
        )
        db.execute(
            "INSERT INTO passages(node_id,kind,start,end,text) VALUES ('source','lines',1,1,'synthetic source')"
        )
    calls = []

    def capture(*args, **kwargs):
        calls.append((args, kwargs))
        return original_connect(*args, **kwargs)

    monkeypatch.setattr(sqlite3, "connect", capture)
    catalogue = Catalogue(database)
    adapter = AuthorisedCatalogue(catalogue, SimpleNamespace(root=tmp_path / "vault"))
    with adapter.projection_reader(time.monotonic() + 1) as reader:
        assert reader.authorized("source", "full_private")
        assert reader.read_evidence("source", limit=1)["text"] == "s"
        # A writer can commit between checks; this is connection reuse, not stale snapshots.
        with original_connect(database) as writer:
            writer.execute("UPDATE nodes SET state='protected' WHERE id='source'")
        assert not reader.authorized("source", "full_private")
        assert reader.read_evidence("source", limit=1) is None
    assert len(calls) == 1
    assert calls[0][1]["timeout"] == 0.05 and calls[0][1]["isolation_level"] is None
    assert adapter.catalogue is catalogue


def test_topic_budget_returns_a_stable_validated_prefix_with_explicit_counts(monkeypatch):
    service = object.__new__(OrganisationService)
    checked = []
    service.authorised = SimpleNamespace(
        authorized=lambda node_id, scope: checked.append(node_id) or True,
        read_evidence=lambda node_id, **kwargs: {"digest": "digest"},
    )
    members = [
        {"source_id": name, "community_id": "group", "path": name + ".md", "digest": "digest"}
        for name in ("a", "b", "c", "d")
    ]
    clock = iter((0, 0.4, 0.9))
    monkeypatch.setattr(service_module.time, "monotonic", lambda: next(clock))
    result = service._topic_projection(
        {"members": members, "coverage": {"total": 100, "covered": 80}},
        "full_private",
        deadline=0.8,
    )
    assert checked == ["a", "b"]
    assert [m["node_id"] for m in result["topics"][0]["members"]] == ["a", "b"]
    assert result["state"] == "partial"
    assert result["counts"]["total"] == 100 and result["counts"]["visible"] == 2
    assert result["counts"]["checked_sources"] == 2
    assert result["counts"]["validation_budget_limited"] is True
