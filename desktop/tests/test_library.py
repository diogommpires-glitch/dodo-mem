from PySide6.QtCore import QPoint, Qt, QUrl
from PySide6.QtGui import QTextDocument
from PySide6.QtTest import QTest
from PySide6.QtWidgets import (
    QApplication,
    QListWidget,
    QPlainTextEdit,
    QPushButton,
    QTextBrowser,
    QTreeWidget,
)

from dodo_memory_desktop.library import Library, RelationshipGraph

NOTE = "notes/Calibration.md"
REFERENCE = "notes/Calibration protocol.md"


def test_catalog_keyboard_open_and_more():
    app = QApplication.instance() or QApplication([])
    library = Library()
    library.set_catalog([{"path": NOTE, "title": "Calibration"}], has_more=True)
    requests = []
    library.note_requested.connect(requests.append)
    library.catalog_more_requested.connect(lambda: requests.append("next page"))
    library.show()
    app.processEvents()

    tree = library.findChild(QTreeWidget, "vaultTree")
    tree.setCurrentItem(tree.topLevelItem(0).child(0))
    QTest.keyClick(tree, Qt.Key.Key_Return)
    library.findChild(QPushButton, "moreNotes").click()
    assert requests == [NOTE, "next page"]
    library.close()


def test_reader_blocks_file_resources_and_clears_verified_content(tmp_path):
    app = QApplication.instance() or QApplication([])
    library = Library()
    fixture = tmp_path / "outside.txt"
    fixture.write_text("outside data", encoding="utf-8")
    url = fixture.as_uri()
    library.open_note({"path": NOTE, "text": f"# Calibration\n\n![outside]({url})"})
    reader = library.findChild(QTextBrowser, "noteReader")
    resource = reader.document().resource(QTextDocument.ResourceType.ImageResource, QUrl(url))
    assert resource is None or resource.isNull()
    assert "Calibration" in reader.toPlainText()

    library.set_source(
        {"note": NOTE, "sha256": "a" * 64, "page": 2, "preview_text": "Verified excerpt"}
    )
    preview = library.findChild(QPlainTextEdit, "sourcePreview")
    assert preview.toPlainText() == "Verified excerpt"
    assert library.open_paths() == [NOTE]
    page = reader.parentWidget()
    library.clear_content()
    assert library.open_paths() == []
    assert not preview.toPlainText()
    assert page.parentWidget() is None
    app.processEvents()
    library.close()


def test_relationship_list_opens_exact_evidence_with_keyboard():
    app = QApplication.instance() or QApplication([])
    library = Library()
    library.open_note({"path": NOTE, "text": "# Calibration"})
    library.set_related(
        NOTE,
        {
            "nodes": [{"path": REFERENCE, "title": "Calibration protocol"}],
            "edges": [
                {
                    "source": NOTE,
                    "target": REFERENCE,
                    "kind": "evidence",
                    "label": "Reviewed evidence",
                    "evidence": {"path": REFERENCE, "heading": "Check the reference standard"},
                },
                {
                    "source": REFERENCE,
                    "target": NOTE,
                    "kind": "link",
                    "label": "Markdown link",
                    "evidence": {"path": REFERENCE, "heading": "Bias controls"},
                },
            ],
            "count": 2,
            "truncated": False,
        },
    )
    opened = []
    library.evidence_requested.connect(
        lambda entry: opened.append((entry["path"], entry.get("heading")))
    )
    relations = library.findChild(QListWidget, "relationshipList")
    relations.setCurrentRow(0)
    assert relations.currentItem().text() == "Cites Calibration protocol"
    QTest.keyClick(relations, Qt.Key.Key_Return)
    assert opened == [(REFERENCE, "Check the reference standard")]
    library.resize(1180, 740)
    library.show()
    app.processEvents()
    graph = library.findChild(RelationshipGraph)
    QTest.mouseClick(
        graph,
        Qt.MouseButton.LeftButton,
        pos=QPoint(graph.width() // 2 + 8, int(graph.height() * 0.315)),
    )
    QTest.keyClick(relations, Qt.Key.Key_Return)
    assert opened[-1] == (REFERENCE, "Bias controls")
    library.close()


def test_narrow_source_drawer_preserves_note_and_keyboard_focus():
    app = QApplication.instance() or QApplication([])
    library = Library()
    library.resize(680, 620)
    library.show()
    library.open_note(
        {"path": NOTE, "text": "# Calibration\n\n## Bias controls\n\nCheck the controls."},
        "Bias controls",
    )
    app.processEvents()
    reader = library.findChild(QTextBrowser, "noteReader")
    assert reader.textCursor().block().text() == "Bias controls"
    library.open_note({"path": NOTE, "text": "# Calibration\n\n" + "Updated checks.\n\n" * 80})
    app.processEvents()
    assert reader.textCursor().position() == 0
    assert reader.verticalScrollBar().value() == 0
    assert not library.inspector.isVisible()

    library.set_source({"note": NOTE, "sha256": "a" * 64, "preview_text": "Original"})
    assert library.inspector.isVisible()
    QTest.keyClick(library.inspector_tabs.tabBar(), Qt.Key.Key_Escape)
    assert not library.inspector.isVisible()
    assert library.inspector_button.hasFocus()
    assert library.current_path == NOTE
    library.close()


def test_active_path_changes_before_new_inspector_requests():
    app = QApplication.instance() or QApplication([])
    library = Library()
    events = []
    library.path_changed.connect(lambda path: events.append(("selected", path)))
    library.history_requested.connect(lambda entry: events.append(("history", entry)))
    library.open_note({"path": NOTE, "text": "# Calibration", "knowledge_key": "calibration"})
    library.open_note({"path": REFERENCE, "text": "# Reference"})
    library.close_note(REFERENCE)
    assert events == [
        ("selected", NOTE),
        ("history", "calibration"),
        ("selected", REFERENCE),
        ("selected", NOTE),
        ("history", "calibration"),
    ]
    app.processEvents()
    library.close()
