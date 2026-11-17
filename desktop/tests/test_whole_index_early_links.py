from pathlib import Path
from types import SimpleNamespace

from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def test_committed_notes_publish_links_before_remaining_root_inventory(tmp_path, monkeypatch):
    fast = tmp_path / "desktop"
    slow = tmp_path / "later"
    fast.mkdir()
    slow.mkdir()
    source = fast / "a-note.md"
    target = fast / "another-note.md"
    source.write_text("See [the related note](another-note.md).", encoding="utf-8")
    target.write_text("The linked source.", encoding="utf-8")
    (slow / "later.txt").write_text("The slower drive is still pending.", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)
    walk = Indexer._directory_entries
    checked = []

    def inspect(index, path, root):
        if path == slow:
            row = catalogue.resolve_path(source)
            references = [
                edge for edge in catalogue.neighbours(row["id"]) if edge["kind"] == "reference"
            ]
            assert len(references) == 1
            assert references[0]["node"]["path"] == str(target)
            assert catalogue.status()["inventory_pending"] == 1
            assert catalogue.resolve_path(slow / "later.txt") is None
            checked.append(True)
        return walk(index, path, root)

    monkeypatch.setattr(Indexer, "_directory_entries", inspect)
    index = Indexer(database)
    try:
        index.desktop = fast
        index.reconcile([slow, fast], extract_now=False, checkpoint=lambda completed: None)
    finally:
        index.close()

    assert checked == [True]
    assert catalogue.status()["inventory_finished"] == 2


def test_restart_connects_existing_note_to_new_metadata_only_target(tmp_path):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "note.md"
    source.write_text("See [the image](figure.png).", encoding="utf-8")
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)

    def reconcile():
        index = Indexer(database)
        try:
            index.reconcile([root])
        finally:
            index.close()

    reconcile()
    note = catalogue.resolve_path(source)
    assert not any(edge["kind"] == "reference" for edge in catalogue.neighbours(note["id"]))
    target = root / "figure.png"
    target.write_bytes(b"Metadata-only fixture")
    reconcile()
    assert catalogue.resolve_path(target)["state"] == "metadata"
    assert any(
        edge["kind"] == "reference" and edge["node"]["path"] == str(target)
        for edge in catalogue.neighbours(note["id"])
    )


def test_inventory_and_watcher_preserve_file_versions_outside_sqlite_range(tmp_path, monkeypatch):
    root = tmp_path / "files"
    root.mkdir()
    source = root / "old-note.md"
    source.write_text("Original exFAT note.", encoding="utf-8")
    oversized = root / "large-note.md"
    oversized.write_text("Only its metadata is needed.", encoding="utf-8")
    oldest_windows_time = -11_644_473_600_000_000_000
    stamp = oldest_windows_time
    huge_size = (1 << 63) + 1
    original_lstat = Path.lstat
    original_entries = Indexer._directory_entries

    def lstat(path):
        info = original_lstat(path)
        if path not in {source, oversized}:
            return info
        return SimpleNamespace(
            st_dev=info.st_dev,
            st_ino=info.st_ino,
            st_mode=info.st_mode,
            st_file_attributes=getattr(info, "st_file_attributes", 0),
            st_size=huge_size if path == oversized else info.st_size,
            st_mtime_ns=stamp,
        )

    def directory_entries(index, path, scope):
        return [
            (child, lstat(child) if child in {source, oversized} else info)
            for child, info in original_entries(index, path, scope)
        ]

    monkeypatch.setattr(Path, "lstat", lstat)
    monkeypatch.setattr(Indexer, "_directory_entries", directory_entries)
    database = tmp_path / "index.sqlite3"
    catalogue = Catalogue(database)
    index = Indexer(database)
    try:
        index.reconcile([root])
        row = catalogue.resolve_path(source)
        assert row["mtime_ns"] == f"int:{oldest_windows_time}"
        assert row["state"] == "indexed"
        assert catalogue.read_evidence(row["id"])["text"] == "Original exFAT note."
        large_row = catalogue.resolve_path(oversized)
        assert large_row["size"] == f"int:{huge_size}"
        assert large_row["state"] == "oversized"

        index.reconcile([root], extract_now=False)
        assert catalogue.resolve_path(source)["digest"] == row["digest"]
        source.write_text("Modified exFAT note.", encoding="utf-8")
        stamp += 1
        index.refresh_paths([source], [root])
        assert index.extract_pending() == 1
        row = catalogue.resolve_path(source)
        assert row["mtime_ns"] == f"int:{oldest_windows_time + 1}"
        assert catalogue.read_evidence(row["id"])["text"] == "Modified exFAT note."
    finally:
        index.close()
