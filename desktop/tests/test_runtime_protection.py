import pytest

from dodo_memory_desktop import whole_index
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def test_runtime_inventory_is_complete_metadata_without_content(tmp_path, monkeypatch):
    root = tmp_path / "library"
    runtime = root / "runtime"
    (runtime / "nested/.cache").mkdir(parents=True)
    (runtime / "empty").mkdir()
    files = [runtime / "settings.json", runtime / "nested/.cache/cache.md"]
    for path in files:
        path.write_text("# Source\n\nThis runtime must never become searchable content.")
    monkeypatch.setattr(whole_index, "METADATA_ONLY_ROOTS", (runtime,))
    index = Indexer(tmp_path / "catalogue.sqlite3")
    reader = Catalogue(index.database)
    try:
        index.reconcile([root])
        for path in files:
            node = reader.resolve_path(path)
            assert node is not None and node["state"] == "protected"
            assert "runtime" in node["reason"].lower()
            assert reader.read_evidence(node["id"]) is None
        assert reader.resolve_path(runtime / "empty") is not None
        assert reader.resolve_path(runtime / "nested/.cache") is not None
        assert reader.search("never become searchable") == []
    finally:
        index.close()


def test_new_runtime_policy_blocks_old_cached_passages_before_rescan(tmp_path, monkeypatch):
    root = tmp_path / "library"
    root.mkdir()
    source = root / "runtime/note.md"
    source.parent.mkdir()
    source.write_text("# Earlier content\n\nA hidden runtime reference.")
    monkeypatch.setattr(whole_index, "METADATA_ONLY_ROOTS", ())
    index = Indexer(tmp_path / "catalogue.sqlite3")
    reader = Catalogue(index.database)
    try:
        index.reconcile([root])
        node = reader.resolve_path(source)
        assert reader.read_evidence(node["id"]) is not None
        monkeypatch.setattr(whole_index, "METADATA_ONLY_ROOTS", (source.parent,))
        assert reader.node(node["id"])["state"] == "protected"
        assert reader.read_evidence(node["id"]) is None
        assert reader.search("hidden runtime reference") == []
    finally:
        index.close()


@pytest.mark.parametrize("field", ["apiKey", "openAiApiKey", "accessToken", "authToken"])
def test_camel_case_credentials_never_become_evidence(tmp_path, field):
    root = tmp_path / "library"
    root.mkdir()
    source = root / "provider.json"
    source.write_text('{"' + field + '":"synthetic-credential-value-1234567890"}')
    index = Indexer(tmp_path / "catalogue.sqlite3")
    reader = Catalogue(index.database)
    try:
        index.reconcile([root])
        node = reader.resolve_path(source)
        assert node["state"] == "protected"
        assert reader.read_evidence(node["id"]) is None
    finally:
        index.close()
