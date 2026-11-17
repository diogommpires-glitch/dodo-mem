import shutil
import subprocess
import time
from typing import ClassVar

import pytest
from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QWheelEvent
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QDialog, QPushButton, QTextBrowser

from dodo_memory_desktop.canvas_store import validate_board
from dodo_memory_desktop.whole_graph import GraphSurface, WholeLibraryGraph
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


class Fixture:
    nodes: ClassVar[dict] = {
        "project": {"id": "project", "name": "Calibration", "kind": "project", "descendants": 2},
        "notes": {"id": "notes", "name": "Notes", "kind": "folder", "descendants": 1},
        "document": {
            "id": "document",
            "name": "<Calibration decision>",
            "kind": "file",
            "category": "document",
            "path": "C:/scope/decision.md",
        },
    }
    generation = 1

    def __init__(self):
        self.nodes = self.nodes.copy()

    def status(self):
        return {"generation": self.generation, "nodes": len(self.nodes)}

    def node(self, identity):
        return self.nodes.get(identity)

    def projection(self, scope, **_):
        nodes = (
            [self.nodes["project"]]
            if scope is None
            else [self.nodes[item] for item in ("notes", "document") if item in self.nodes]
        )
        edges = (
            [
                {
                    "source": "notes",
                    "target": "document",
                    "kind": "contains",
                    "evidence": "Parent folder",
                }
            ]
            if scope and len(nodes) == 2
            else []
        )
        return {
            "generation": self.generation,
            "scope": self.node(scope),
            "nodes": nodes,
            "edges": edges,
            "total": len(nodes),
        }

    def neighbours(self, identity, **_):
        if identity in {"document", "notes"}:
            other = "notes" if identity == "document" else "document"
            if other in self.nodes:
                return [
                    {"node": self.nodes[other], "kind": "contains", "evidence": "Parent folder"}
                ]
        return []

    def search(self, query, **_):
        return [row for row in self.nodes.values() if query in row["name"].casefold()]

    def git_history(self, repo, **_):
        return {
            "generation": self.generation,
            "scope": self.node(repo),
            "nodes": [
                {
                    "id": "commit",
                    "name": "Record calibration",
                    "kind": "commit",
                    "digest": "c" * 40,
                    "category": "code",
                    "refs": ["refs/heads/main"],
                    "changed_files": ["M\tdecision.md"],
                }
            ],
            "edges": [],
            "total": 1,
        }


def wait_for(app, condition):
    deadline = time.monotonic() + 4
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        QTest.qWait(10)
    assert condition()


def test_graph_keyboard_drilldown_uses_shared_projection_for_generated_views():
    app = QApplication.instance() or QApplication([])
    reader = Fixture()
    graph = WholeLibraryGraph(reader)
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.resize(1280, 720)
    graph.show()
    graph.refresh()
    wait_for(app, lambda: graph.items.count() == 1)
    assert "3 catalogue items" in graph.summary.text()
    graph.items.setCurrentRow(0)
    QTest.keyClick(graph.items, Qt.Key.Key_Return)
    wait_for(app, lambda: graph.items.count() == 2)
    assert graph.title.text() == "Calibration"

    selection = []
    graph.source_requested.connect(selection.append)
    graph.items.setCurrentRow(0)
    wait_for(app, lambda: graph.relationships.count() == 1)
    assert "Parent folder" in graph.relationships.item(0).text()
    graph.relationships.setCurrentRow(0)
    QTest.keyClick(graph.relationships, Qt.Key.Key_Return)
    assert selection == [reader.nodes["document"]]

    maps, boards = [], []
    graph.mind_map_requested.connect(lambda title, html: maps.append((title, html)))
    graph.canvas_requested.connect(boards.append)
    graph.items.setCurrentRow(1)
    graph.show_mind_map()
    assert maps[0][0] == "Calibration"
    assert "&lt;Calibration decision&gt;" in maps[0][1]
    assert "Catalogue revision 1" in maps[0][1]
    graph.show_generated_canvas()
    assert not boards
    map_dialog = next(
        child for child in graph.findChildren(QDialog) if child.windowTitle().startswith("Mind map")
    )
    dialog = next(
        child
        for child in graph.findChildren(QDialog)
        if child.windowTitle().startswith("Generated Canvas")
    )
    next(
        button
        for button in dialog.findChildren(QPushButton)
        if button.text() == "Save editable copy"
    ).click()
    board = validate_board(boards[0])
    assert len(board["nodes"]) == 2 and len(board["edges"]) == 1
    graph._select("document")
    graph.surface.zoom = 1.7
    saved = graph.diagram.positions["document"]
    reader.generation = 2
    graph.refresh()
    wait_for(app, lambda: graph.diagram.generation == 2)
    assert "Catalogue revision 2" in map_dialog.findChild(QTextBrowser).toPlainText()
    for child in graph.findChildren(QDialog):
        child.close()
    assert graph.selection == "document"
    assert graph.surface.zoom == 1.7
    assert graph.diagram.positions["document"] == saved
    graph.grab()

    graph.git.click()
    wait_for(app, lambda: graph.title.text().startswith("Git history"))
    assert graph.items.item(0).text() == "Record calibration"
    graph.items.setCurrentRow(0)
    assert "refs/heads/main" in graph.relationships.item(0).text()
    assert "decision.md" in graph.relationships.item(1).text()
    graph.close()


def test_generated_previews_refresh_independently_of_the_active_graph():
    app = QApplication.instance() or QApplication([])
    reader = Fixture()
    graph = WholeLibraryGraph(reader)
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.show()
    graph.set_scope("project")
    wait_for(app, lambda: graph.items.count() == 2)
    graph.show_mind_map()
    graph.show_generated_canvas()
    browser = graph.findChild(QTextBrowser)
    canvas = next(
        surface for surface in graph.findChildren(GraphSurface) if surface != graph.surface
    )
    assert "Calibration decision" in browser.toPlainText()
    assert "document" in canvas.diagram.nodes

    graph.set_scope(None)
    wait_for(app, lambda: graph.title.text() == "Whole library")
    reader.nodes.pop("document")
    reader.generation = 2
    graph.refresh()
    graph.refresh_previews(graph_visible=True)
    wait_for(app, lambda: canvas.diagram.generation == 2)
    assert "Catalogue revision 2" in browser.toPlainText()
    assert "Calibration decision" not in browser.toPlainText()
    assert set(canvas.diagram.nodes) == {"notes"}

    graph.hide()
    reader.nodes.pop("notes")
    reader.generation = 3
    graph.refresh_previews()
    wait_for(app, lambda: canvas.diagram.generation == 3)
    assert "Catalogue revision 3" in browser.toPlainText()
    assert not canvas.diagram.nodes
    assert graph.title.text() == "Whole library"
    graph.close()


def test_browse_cache_survives_progress_but_discards_a_revoked_node():
    class VersionedCatalogue(Fixture):
        graph_revision = 1
        message = "Ready"

        def graph_status(self):
            return {
                **self.status(),
                "graph_revision": self.graph_revision,
                "message": self.message,
            }

        def graph_projection(self, scope, **options):
            return {**self.projection(scope, **options), "graph_revision": self.graph_revision}

    app = QApplication.instance() or QApplication([])
    reader = VersionedCatalogue()
    graph = WholeLibraryGraph(reader)
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.show()
    try:
        graph.set_scope("project")
        wait_for(app, lambda: graph.items.count() == 2)
        cached = graph.diagram

        reader.generation = 2
        reader.message = "Extracting local evidence"
        graph.set_scope(None)
        wait_for(app, lambda: graph.title.text() == "Whole library")
        graph.set_scope("project")
        wait_for(app, lambda: graph.title.text() == "Calibration")
        assert graph.diagram is cached
        assert graph.diagram.generation == 1
        assert "Extracting local evidence" in graph.summary.text()

        graph.set_scope(None)
        wait_for(app, lambda: graph.title.text() == "Whole library")
        reader.nodes.pop("document")
        reader.graph_revision = 2
        graph.set_scope("project")
        wait_for(app, lambda: graph.title.text() == "Calibration")
        assert graph.diagram is not cached
        assert set(graph.diagram.nodes) == {"notes"}
        assert not graph.diagram.edges
    finally:
        graph.close()


def test_graph_keeps_aggregate_and_direct_evidence_after_async_read():
    class Aggregated(Fixture):
        def projection(self, scope, **kwargs):
            value = super().projection(scope, **kwargs)
            if scope:
                value["edges"][0].update(count=12, evidence="Cross-folder source evidence")
            return value

    app = QApplication.instance() or QApplication([])
    graph = WholeLibraryGraph(Aggregated())
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.show()
    graph.set_scope("project")
    wait_for(app, lambda: graph.items.count() == 2)
    graph.items.setCurrentRow(0)
    wait_for(
        app,
        lambda: any(
            "Parent folder" in graph.relationships.item(index).text()
            for index in range(graph.relationships.count())
        ),
    )
    assert "12 links" in graph.relationships.item(0).text()
    assert "Cross-folder source evidence" in graph.relationships.item(0).text()
    graph.close()


@pytest.mark.skipif(shutil.which("git") is None, reason="Local Git required")
def test_actual_index_drives_cross_folder_and_git_views(tmp_path):
    app = QApplication.instance() or QApplication([])
    scope = tmp_path / "scope"
    project = scope / "Project"
    reports = scope / "Reports"
    project.mkdir(parents=True)
    reports.mkdir()
    (reports / "Report.md").write_text("# Report\nFlow error under 0.1 mm.", encoding="utf-8")
    for index in range(56):
        (reports / f"Run-{index:02}.md").write_text(f"# Run {index}", encoding="utf-8")
    decision = project / "Decision.md"
    decision.write_text(
        "# Calibration decision\nSee [Report](../Reports/Report.md).", encoding="utf-8"
    )
    for command in (
        ["init", "-q", "-b", "main"],
        ["add", "Decision.md"],
        [
            "-c",
            "user.name=Dodo Fixture",
            "-c",
            "user.email=239468521+diogommpires-glitch@users.noreply.github.com",
            "commit",
            "-qm",
            "Calibration decision",
        ],
    ):
        subprocess.run(["git", *command], cwd=project, check=True, capture_output=True)
    db = tmp_path / "index.sqlite"
    worker = Indexer(db)
    try:
        worker.reconcile([scope], extract_now=False)
    finally:
        worker.close()
    reader = Catalogue(db)
    graph = WholeLibraryGraph(reader)
    graph.setAttribute(Qt.WidgetAttribute.WA_DontShowOnScreen, True)
    graph.resize(1280, 800)
    graph.show()
    graph.refresh()
    wait_for(app, lambda: graph.diagram is not None)
    assert {edge["kind"] for edge in graph.diagram.edges} == {"contains"}
    assert len(graph.diagram.hubs) == 2
    before = len(graph.diagram.nodes)
    report_hub = reader.resolve_path(reports)["id"]
    position = graph.diagram.positions[report_hub]
    graph._select(report_hub)
    pointer = graph.surface._points()[report_hub]
    QApplication.sendEvent(
        graph.surface,
        QWheelEvent(
            pointer,
            graph.surface.mapToGlobal(pointer.toPoint()).toPointF(),
            QPoint(),
            QPoint(0, 480),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        ),
    )
    wait_for(app, lambda: len(graph.diagram.nodes) > before)
    assert graph.scope is None
    assert graph.diagram.positions[report_hub] == position
    assert graph.selection == report_hub
    worker = Indexer(db)
    try:
        worker.reconcile([scope])
    finally:
        worker.close()
    graph.refresh()
    wait_for(app, lambda: graph.diagram.counts["bundled"] == 1)
    edge = next(edge for edge in graph.diagram.edges if edge.get("bundle"))
    assert edge["kind"] == "reference" and edge["count"] == 1
    points = graph.surface._points()
    centre = (points[edge["source"]] + points[edge["target"]]) / 2
    QTest.mouseClick(graph.surface, Qt.MouseButton.LeftButton, pos=centre.toPoint())
    wait_for(app, graph.open_source.isEnabled)
    source = reader.resolve_path(decision)
    opened = []
    graph.source_requested.connect(opened.append)
    graph.open_source.click()
    assert opened[0]["id"] == source["id"] and opened[0]["match_start"] == 2
    graph.set_scope(source["id"])
    wait_for(app, lambda: graph.diagram.scope and graph.diagram.scope["id"] == source["id"])
    assert "Report.md" in {row["name"] for row in graph.diagram.nodes.values()}
    graph.set_scope(reader.resolve_path(project)["id"])
    wait_for(app, lambda: graph.diagram.scope and graph.diagram.scope["name"] == "Project")
    graph.search_box.setText("Report.md")
    QTest.keyClick(graph.search_box, Qt.Key.Key_Return)
    report = reader.resolve_path(reports / "Report.md")
    wait_for(app, lambda: graph.mode == "search" and report["id"] in graph.diagram.nodes)
    graph._activate(report["id"])
    wait_for(app, lambda: graph.diagram.scope and graph.diagram.scope["id"] == report["id"])
    assert graph.selection == report["id"]
    graph.set_scope(reader.resolve_path(project)["id"])
    wait_for(app, lambda: graph.diagram.scope and graph.diagram.scope["name"] == "Project")
    graph.git.click()
    wait_for(app, lambda: graph.title.text().startswith("Git history"))
    assert graph.items.item(0).text().startswith("Calibration decision")
    graph.items.setCurrentRow(0)
    relations = [graph.relationships.item(i).text() for i in range(graph.relationships.count())]
    assert any(
        "changed file" in row.casefold() and "decision.md" in row.casefold() for row in relations
    )
    (reports / "Report.md").unlink()
    worker = Indexer(db)
    try:
        worker.reconcile([scope])
    finally:
        worker.close()
    assert not reader.graph_evidence(edge)
    graph.set_scope(None)
    wait_for(app, lambda: graph.diagram.scope is None and graph.diagram.counts["bundled"] == 0)
    assert {edge["kind"] for edge in graph.diagram.edges} == {"contains"}
    graph.close()
