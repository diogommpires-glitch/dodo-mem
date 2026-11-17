import threading
import time

from PySide6.QtCore import QTimer
from PySide6.QtWidgets import QApplication

from dodo_memory_desktop import whole_library
from dodo_memory_desktop.whole_library import WholeLibrary


def row(identity, kind):
    return {
        "id": identity,
        "name": identity,
        "kind": kind,
        "category": "folder" if kind == "root" else "note",
        "state": "online" if kind == "root" else "indexed",
        "path": f"C:/fixture/{identity}",
    }


class CatalogueFixture:
    def __init__(self):
        self.generation = 1
        self.pending = 200
        self.root = row("root", "root")
        self.files = [row("note", "file")]
        self.revision_entered = threading.Event()
        self.release_revision = threading.Event()
        self.gui_thread = threading.get_ident()
        self.read_threads = set()

    def revision(self):
        self.read_threads.add(threading.get_ident())
        self.revision_entered.set()
        self.release_revision.wait(timeout=2)
        return self.generation

    def status(self, *, summary=False):
        self.read_threads.add(threading.get_ident())
        assert summary
        return {
            "generation": self.generation,
            "nodes": 1 + len(self.files),
            "content_pending": self.pending,
        }

    def extraction_counts(self):
        self.read_threads.add(threading.get_ident())
        return {"estimated": True, "extracted": 25, "skipped": 4, "failed": 2}

    def roots(self):
        self.read_threads.add(threading.get_ident())
        return [self.root]

    def children(self, identity, limit, offset):
        self.read_threads.add(threading.get_ident())
        return self.files[offset : offset + limit] if identity == "root" else []

    def search(self, _query, limit):
        return [self.root, *self.files][:limit]

    def read_evidence(self, identity, _limit, start):
        self.read_threads.add(threading.get_ident())
        if any(entry["id"] == identity for entry in self.files):
            return {"locator": {"kind": "lines", "start": start, "end": start}, "text": "Verified"}
        return None


def test_polling_and_folders_keep_selection_and_discard_removed_source(monkeypatch):
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    clock = [0]
    monkeypatch.setattr(whole_library, "monotonic", lambda: clock[0])
    app = QApplication.instance() or QApplication([])
    catalogue = CatalogueFixture()
    library = WholeLibrary(catalogue)
    library.show()

    def until(predicate):
        deadline = time.monotonic() + 3
        while not predicate():
            assert time.monotonic() < deadline
            app.processEvents()
            time.sleep(0.005)

    try:
        library.refresh()
        until(catalogue.revision_entered.is_set)
        ui_tick = threading.Event()
        QTimer.singleShot(0, ui_tick.set)
        until(ui_tick.is_set)
        assert not catalogue.release_revision.is_set()
        catalogue.release_revision.set()
        until(lambda: library.tree.topLevelItemCount() == 1)
        until(lambda: "Estimated 25 extracted, 4 skipped, 2 failed" in library.status.text())
        assert "200 online files awaiting extraction" in library.status.text()
        root = library.tree.topLevelItem(0)
        root.setExpanded(True)
        until(lambda: root.childCount() == 1)
        note = root.child(0)
        library.tree.setCurrentItem(note)
        library.open_source(catalogue.files[0])
        until(lambda: library.preview.toPlainText() == "Verified")

        catalogue.files = [row("added", "file"), *catalogue.files]
        catalogue.generation += 1
        catalogue.pending = 100
        clock[0] = 30
        library.refresh()
        until(lambda: root.childCount() == 2)
        assert "Net queue reduction 200/min, about 1 min at this rate" in library.status.text()
        assert library.tree.topLevelItem(0) is root
        assert library.tree.currentItem() is note
        assert root.isExpanded()
        assert library.preview.toPlainText() == "Verified"
        clock[0] = 90
        library.refresh()
        assert "Net queue reduction" not in library.status.text()

        catalogue.files = [catalogue.files[0]]
        catalogue.generation += 1
        library.refresh()
        until(lambda: root.childCount() == 1)
        assert library.preview.toPlainText() != "Verified"
        assert catalogue.gui_thread not in catalogue.read_threads

        library.query.setText("added")
        library.query.returnPressed.emit()
        until(lambda: library.tree.topLevelItemCount() == 2)
        root_result = library.tree.topLevelItem(0)
        root_result.setExpanded(True)
        until(lambda: root_result.childCount() == 1)
        assert root_result.child(0).text(0) == library.tree.topLevelItem(1).text(0)
    finally:
        catalogue.release_revision.set()
        library.close()
