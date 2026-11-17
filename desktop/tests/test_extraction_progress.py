from dodo_memory_desktop.whole_index import Catalogue, advance_name_index, path_id
from dodo_memory_desktop.whole_index_worker import Indexer


def test_pending_counts_only_include_online_roots(tmp_path):
    online, offline = (tmp_path / name for name in ("online", "offline"))
    for root in (online, offline):
        root.mkdir()
        (root / "note.md").write_text("Index me.", encoding="utf-8")

    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([online, offline])
        catalogue = Catalogue(database)
        with index.db:
            index.db.execute("UPDATE nodes SET state='pending' WHERE kind='file'")
        assert catalogue.status(summary=True)["content_pending"] == 2

        with index.db:
            index.db.execute("UPDATE nodes SET state='offline' WHERE id=?", (path_id(offline),))
        assert catalogue.status(summary=True)["content_pending"] == 1
        assert catalogue.status()["content_pending"] == 1
    finally:
        index.close()


def test_filename_index_backfills_and_follows_rename_and_delete(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    original = root / "chapter-uniquefilename.bin"
    original.write_bytes(b"\0")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([root], extract_now=False)
        catalogue = Catalogue(database)
        assert not advance_name_index(index.db, limit=1)
        assert catalogue.search("quefilen")[0]["path"] == str(original)
    finally:
        index.close()

    index = Indexer(database)
    try:
        assert advance_name_index(index.db, limit=1)
        assert catalogue.search("quefilen")[0]["path"] == str(original)
        renamed = original.with_name("chapter-replacementname.bin")
        original.rename(renamed)
        index.refresh_paths([original, renamed], [root])
        assert catalogue.search("quefilen") == []
        assert catalogue.search("placementn")[0]["path"] == str(renamed)
        renamed.unlink()
        index.refresh_paths([renamed], [root])
        assert catalogue.search("placementn") == []
    finally:
        index.close()


def test_graph_revision_tracks_visible_changes_and_ignores_extraction_progress(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "note.md"
    source.write_text("Source evidence.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([root], extract_now=False)
        catalogue = Catalogue(database)
        status = catalogue.graph_status()
        assert status["graph_revision"] == catalogue.graph_projection()["graph_revision"]
        assert index.extract_pending() == 1
        after_extract = catalogue.graph_status()
        assert after_extract["generation"] > status["generation"]
        assert after_extract["graph_revision"] == status["graph_revision"]

        node = catalogue.resolve_path(source)
        assert any(
            edge["target"] == node["id"]
            for edge in catalogue.graph_projection(path_id(root))["edges"]
        )
        with index.db:
            index.db.execute("UPDATE nodes SET state='protected' WHERE id=?", (node["id"],))
        revoked = catalogue.graph_projection(path_id(root))
        assert revoked["graph_revision"] > after_extract["graph_revision"]
        assert not any(edge["target"] == node["id"] for edge in revoked["edges"])
    finally:
        index.close()


def test_exact_filename_wins_over_content_and_stays_inside_project(tmp_path):
    root = tmp_path / "files"
    project = root / "project"
    project.mkdir(parents=True)
    (root / "README.md").write_text("Root readme.", encoding="utf-8")
    expected = project / "README.md"
    expected.write_text("Project readme.", encoding="utf-8")
    (project / "mentions.md").write_text("README.md " * 100, encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([root])
        results = Catalogue(database).search("README.md", limit=1, scope=path_id(project))
        assert [result["path"] for result in results] == [str(expected)]
    finally:
        index.close()
