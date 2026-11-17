import json
import os
import threading
import time

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from desktop_fixture import DIGEST, METHOD, PATH, SENSOR, FixtureGateway
from PySide6.QtCore import QAbstractAnimation, QSettings, Qt, QTimer, QUrl
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog, QPushButton

from dodo_memory_desktop.app import ASK, CANVAS, GRAPH, HOME, LIBRARY, PLUGINS, Desktop
from dodo_memory_desktop.gateway import GatewayError
from dodo_memory_desktop.library import MarkdownReader
from dodo_memory_desktop.plugins import EXAMPLES, PACK_SUFFIX
from dodo_memory_desktop.ui import ROLE
from dodo_memory_desktop.whole_index import Catalogue
from dodo_memory_desktop.whole_index_worker import Indexer


def until(predicate, seconds=4):
    deadline = time.monotonic() + seconds
    while not predicate():
        assert time.monotonic() < deadline
        QApplication.processEvents()
        time.sleep(0.01)


@pytest.fixture
def workspace(tmp_path):
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    backend = FixtureGateway()
    window = Desktop(backend, settings, tmp_path)
    window.show()
    until(lambda: window.online)
    yield app, window, backend
    window.close()
    window.pool.waitForDone(5000)


def test_keyboard_search_to_exact_source_and_disconnect(workspace):
    _, window, backend = workspace
    window.home.project.setCurrentIndex(1)
    QTest.keyClicks(window.home.search, "thermal calibration")
    QTest.keyClick(window.home.search, Qt.Key.Key_Return)
    until(lambda: window.home.results.count() == 1)
    window.home.results.setCurrentRow(0)
    window.home.results.setFocus()
    QTest.keyClick(window.home.results, Qt.Key.Key_Return)
    until(lambda: window.library.current_path == PATH)
    window.library.open_original.click()
    until(lambda: "20 °C" in window.library.source_preview.toPlainText())
    assert window.library.source_status.text() == "Lines 8–13"
    backend.connected = False
    window.refresh()
    until(lambda: not window.online)
    assert not window.library.open_paths()
    assert not window.library.source_preview.toPlainText()
    assert not window.home.results.count()
    backend.connected = True
    window.refresh()
    until(lambda: window.online)
    assert not window.library.open_paths()


def test_home_search_preserves_project_context_and_keeps_private_queries_local(tmp_path):
    app = QApplication.instance() or QApplication([])
    personal = tmp_path / "personal"
    alpha = personal / "alpha"
    beta = personal / "beta"
    for directory in (alpha, beta):
        directory.mkdir(parents=True)
        (directory / "guide.md").write_text(
            f"Lantern belongs to {directory.name}.", encoding="utf-8"
        )
    database = tmp_path / "catalogue.sqlite3"
    index = Indexer(database)
    try:
        index.reconcile([personal])
    finally:
        index.close()

    class RecordingGateway(FixtureGateway):
        def __init__(self):
            super().__init__()
            self.searches = []

        def call(self, name, arguments=None):
            if name == "search":
                self.searches.append(arguments["query"])
            return super().call(name, arguments)

    backend = RecordingGateway()
    settings = QSettings(str(tmp_path / "workspace.ini"), QSettings.Format.IniFormat)
    window = Desktop(backend, settings, tmp_path, whole_catalogue=Catalogue(database))
    assert window.whole.network.view_mode.currentData() == "overview"
    home = window.home
    window.show()
    try:
        until(lambda: window.online)
        assert home.search_scope.currentData() == "local"
        home.search.setText("lantern")
        QTest.keyClick(home.search, Qt.Key.Key_Return)
        until(lambda: home.results.count() == 2)
        backend.connected = False
        window.refresh()
        until(lambda: not window.online)
        assert home.results.count() == 2
        home.results.setCurrentRow(1)
        focused = home.results.currentItem().data(ROLE)["id"]
        home.graph_result.click()
        graph = window.whole.graph
        until(lambda: graph.diagram and graph.selection == focused)
        assert window.stack.currentIndex() == GRAPH
        assert graph.diagram.scope["id"] == focused
        window.show_page(HOME)
        until(lambda: home.results.count() == 2)
        assert home.results.currentItem().data(ROLE)["id"] == focused

        home.search.setText("alpha")
        QTest.keyClick(home.search, Qt.Key.Key_Return)
        until(lambda: home.results.count() == 2)
        folder = next(
            home.results.item(index)
            for index in range(home.results.count())
            if home.results.item(index).data(ROLE)["kind"] == "folder"
        )
        home.results.itemActivated.emit(folder)
        library = window.whole.library
        until(
            lambda: library.tree.topLevelItemCount() and library.tree.topLevelItem(0).childCount()
        )
        assert window.stack.currentIndex() == LIBRARY
        assert library.tree.topLevelItem(0).text(0) == "alpha"

        window.show_page(HOME)
        home.search_scope.setCurrentIndex(home.search_scope.findData("project"))
        home.search.setText("lantern")
        QTest.keyClick(home.search, Qt.Key.Key_Return)
        until(lambda: home.results.count() == 1)
        local_file = home.results.item(0).data(ROLE)
        assert local_file["path"].endswith(str(alpha / "guide.md"))
        home.results.itemActivated.emit(home.results.item(0))
        until(lambda: library.preview.toPlainText() == "Lantern belongs to alpha.")

        window.show_page(HOME)
        home.ask_search.click()
        assert window.stack.currentIndex() == ASK
        assert window.whole.ask.context.node_id == local_file["id"]
        assert window.whole.ask.question.text() == "lantern"
        assert "guide.md" in window.whole.ask.scope.text()

        backend.connected = True
        window.refresh()
        until(lambda: window.online)
        window.show_page(HOME)
        home.search_scope.setCurrentIndex(home.search_scope.findData("shared"))
        assert home.search.text() == ""
        assert not backend.searches
        home.search.setText("thermal")
        QTest.keyClick(home.search, Qt.Key.Key_Return)
        until(lambda: home.results.count() == 1)
        assert home.results.item(0).data(ROLE)["path"] == PATH
        assert backend.searches == ["thermal"]
        home.graph_result.click()
        until(lambda: window.graph.notes.currentItem() is not None)
        assert window.stack.currentIndex() == GRAPH
        assert window.whole.graph_tabs.currentWidget() is window.graph
        assert window.graph.notes.currentItem().data(ROLE)["path"] == PATH
        window.graph.all_notes.click()
        until(lambda: window.graph.snapshot is not None)
        assert window.graph.focused_path is None
    finally:
        window.close()
        window.pool.waitForDone(5000)
        app.processEvents()


def test_revoked_note_clears_open_tab_and_graph(workspace):
    _, window, backend = workspace
    window.home.project.setCurrentIndex(1)
    window.open_note(PATH)
    until(lambda: window.library.current_path == PATH)
    until(
        lambda: window.library.findChild(type(window.home.results), "relationshipList").count() == 2
    )
    backend.revoked = True
    window.refresh()
    until(lambda: window.library.current_path is None)
    assert window.library.findChild(type(window.home.results), "relationshipList").count() == 0
    assert PATH not in window.catalog


def test_project_library_checks_shared_notes_and_opens_local_locations(
    workspace, tmp_path, monkeypatch
):
    _, window, backend = workspace
    projects = window.library.projects
    projects.archive = tmp_path / "archive"
    projects.desktop = tmp_path / "Desktop"
    folder = projects.desktop / "Labs" / "Chamber study"
    folder.mkdir(parents=True)
    map_path = projects.archive / "projects" / "chamber-study" / "mind-map.html"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("<h1>Chamber study</h1>", encoding="utf-8")
    index_path = projects.archive / "index.html"
    index_path.write_text("<h1>Project maps</h1>", encoding="utf-8")
    (projects.archive / "project-catalogue.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "projects": [
                    {
                        "id": "chamber-study",
                        "title": "Chamber study",
                        "map": "projects/chamber-study/mind-map.html",
                        "roots": [
                            {
                                "path": str(folder),
                                "label": "Open chamber folder",
                                "kind": "desktop",
                            },
                            {
                                "path": str(projects.desktop / ".." / "archive"),
                                "label": "Escaping folder",
                                "kind": "desktop",
                            },
                        ],
                        "sharedNotes": [
                            {"path": PATH, "title": "Catalogue title", "sha256": DIGEST},
                            {"path": SENSOR, "title": "Stale note", "sha256": "a" * 64},
                            {"path": "private/sensitive.md", "title": "Private", "sha256": DIGEST},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    opened = []
    monkeypatch.setattr(
        "dodo_memory_desktop.project_library.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    calls = []
    call = backend.call

    def capture(name, arguments=None):
        calls.append((name, arguments))
        return call(name, arguments)

    monkeypatch.setattr(backend, "call", capture)
    window.show_page(LIBRARY)
    window.library.project_folders_button.click()
    assert window.library.projects_visible()
    assert projects.isVisible() and not window.library.vault.isVisible()
    assert projects.counts.text() == "1 project · 1 Desktop folder"
    projects.findChild(QPushButton, "openArchiveIndex").click()
    tree = projects.tree
    section = tree.topLevelItem(0)
    assert section.text(0) == "Desktop folders"
    group = section.child(0)
    assert group.text(0) == "Labs"
    project = group.child(0)
    assert project.text(0) == "Chamber study"
    group.setExpanded(True)
    project.setExpanded(True)
    rows = {project.child(i).text(0): project.child(i) for i in range(project.childCount())}
    for action in (
        "Open mind map",
        "Open chamber folder",
        "Open archive folder",
        "Escaping folder",
    ):
        tree.setCurrentItem(rows[action])
        QTest.keyClick(tree, Qt.Key.Key_Return)
    assert opened == [
        index_path.as_posix(),
        map_path.as_posix(),
        folder.as_posix(),
        map_path.parent.as_posix(),
    ]
    assert projects.status.text() == "That project location is unavailable."

    shared = rows["Shared notes"]
    until(lambda: not shared.isHidden())
    visible = [
        shared.child(i) for i in range(shared.childCount()) if not shared.child(i).isHidden()
    ]
    assert [row.text(0) for row in visible] == ["Thermal calibration"]
    assert ("note_card_refs", {"paths": [PATH, SENSOR]}) in calls
    shared.setExpanded(True)
    tree.setCurrentItem(visible[0])
    QTest.keyClick(tree, Qt.Key.Key_Return)
    until(lambda: window.library.current_path == PATH)
    assert ("get_note", {"path": PATH, "expected_sha256": DIGEST}) in calls
    assert not window.library.projects_visible()
    assert window.library.notes_button.isChecked()
    window.library.project_folders_button.click()
    project = tree.topLevelItem(0).child(0).child(0)
    shared = next(
        project.child(i)
        for i in range(project.childCount())
        if project.child(i).text(0) == "Shared notes"
    )
    until(lambda: not shared.isHidden())
    backend.revoked = True
    window.refresh()
    until(shared.isHidden)


def test_home_artwork_stays_still_and_search_survives_missing_image(workspace):
    _, window, _ = workspace
    window.home.reduce_motion = True
    window.home.select_collection("Lab notebook")
    assert window.home.art.animation.state() == QAbstractAnimation.State.Stopped
    assert window.home.art.opacity.opacity() == 1
    assert window.home.art.still.width() == 640
    window.home.art.still = type(window.home.art.still)()
    window.home._update_art()
    assert window.home.art.isHidden()
    assert window.home.search.isEnabled()


def test_vault_graph_opens_exact_evidence_and_drops_revoked_edges(workspace):
    _, window, backend = workspace
    window.show_page(GRAPH)
    window.graph.show_note_relationships()
    until(lambda: window.graph.notes.count() == 4 and window.graph.edges.count() == 2)
    first_edge = window.graph.edge_items[0]
    point = window.graph.view.mapFromScene(first_edge.path().pointAtPercent(0.5))
    QTest.mouseClick(window.graph.view.viewport(), Qt.MouseButton.LeftButton, pos=point)
    assert window.graph.lists.currentWidget() is window.graph.edges
    assert window.graph.edges.currentRow() == 0
    assert "Repeating the run" in window.graph.detail.text()
    QTest.keyClick(window.graph.edges, Qt.Key.Key_Return)
    until(lambda: window.library.current_path == PATH)
    reader = window.library._pages[PATH].findChild(MarkdownReader)
    until(lambda: reader.textCursor().block().text() == "Repeating the run")
    window.show_page(GRAPH)
    until(lambda: window.graph.notes.count() == 4)
    backend.revoked = True
    window.refresh()
    until(lambda: window.graph.notes.count() == 3)
    assert window.graph.edges.count() == 0
    until(lambda: PATH not in window.library.open_paths())


def test_project_graph_connects_real_folders_and_live_shared_notes(
    workspace, tmp_path, monkeypatch
):
    _, window, backend = workspace
    graph = window.graph.projects
    graph.archive = tmp_path / "archive"
    graph.desktop = tmp_path / "Desktop"
    folder = graph.desktop / "Labs" / "Chamber study"
    folder.mkdir(parents=True)
    map_path = graph.archive / "projects" / "chamber-study" / "mind-map.html"
    map_path.parent.mkdir(parents=True)
    map_path.write_text("<h1>Chamber study</h1>", encoding="utf-8")
    (graph.archive / "project-catalogue.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "projects": [
                    {
                        "id": "chamber-study",
                        "title": "Chamber study",
                        "map": "projects/chamber-study/mind-map.html",
                        "roots": [
                            {"path": str(folder), "label": "Chamber folder", "kind": "desktop"},
                            {
                                "path": str(graph.desktop / ".." / "archive"),
                                "label": "Escaping folder",
                            },
                        ],
                        "sharedNotes": [
                            {"path": PATH, "title": "Untrusted catalogue title", "sha256": DIGEST},
                            {"path": SENSOR, "title": "Stale note", "sha256": "a" * 64},
                            {"path": "private/sensitive.md", "title": "Private", "sha256": DIGEST},
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    opened = []
    monkeypatch.setattr(
        "dodo_memory_desktop.vault_graph.QDesktopServices.openUrl",
        lambda url: opened.append(url.toLocalFile()) or True,
    )
    calls = []
    call = backend.call

    def capture(name, arguments=None):
        calls.append((name, arguments))
        return call(name, arguments)

    monkeypatch.setattr(backend, "call", capture)
    window.show_page(GRAPH)
    assert window.graph.tabs.currentWidget() is graph
    until(lambda: graph.summary.text() == "1 folder · 1 verified shared note")
    assert graph.project.currentText() == "Chamber study"
    assert [graph.connections.item(i).text() for i in range(graph.connections.count())] == [
        "Open mind map",
        "Folder · Chamber folder",
        "Shared note · Thermal calibration",
    ]
    assert ("note_card_refs", {"paths": [PATH, SENSOR]}) in calls
    for row in range(3):
        graph.connections.setCurrentRow(row)
        QTest.keyClick(graph.connections, Qt.Key.Key_Return)
    assert opened == [map_path.as_posix(), folder.as_posix()]
    until(lambda: window.library.current_path == PATH)
    assert ("get_note", {"path": PATH, "expected_sha256": DIGEST}) in calls
    window.show_page(GRAPH)
    until(lambda: graph.summary.text() == "1 folder · 1 verified shared note")
    backend.revoked = True
    window.refresh()
    until(lambda: graph.summary.text() == "1 folder · 0 verified shared notes")
    assert graph.connections.count() == 2


def test_ambiguous_link_choice_opens_heading_and_note_save_keeps_revision(workspace):
    app, window, backend = workspace
    window.open_note(PATH)
    until(lambda: window.library.current_path == PATH)

    def pick_second():
        dialog = app.activeModalWidget()
        assert dialog is not None and dialog.windowTitle() == "Choose a note"
        choices = dialog.findChild(type(window.graph.notes))
        choices.setCurrentRow(1)
        QTest.keyClick(choices, Qt.Key.Key_Return)

    reader = window.library._pages[PATH].findChild(MarkdownReader)
    anchor = reader.document().find("the chamber procedure")
    assert anchor.charFormat().isAnchor()
    QTimer.singleShot(75, pick_second)
    reader.anchorClicked.emit(QUrl(anchor.charFormat().anchorHref()))
    until(lambda: window.library.current_path == METHOD)
    reader = window.library._pages[METHOD].findChild(MarkdownReader)
    until(lambda: reader.textCursor().block().text() == "Method")
    assert SENSOR not in window.library.open_paths()
    window.edit_note(METHOD)
    until(lambda: window.editor is not None)
    window.editor.markdown.appendPlainText("\n## Units\nTime is in seconds.")
    window.editor.save.click()
    until(lambda: backend.saved and window.editor is None)
    path, text, expected = backend.saved
    assert (path, expected) == (METHOD, DIGEST)
    assert "Time is in seconds." in text


def test_editor_completion_inserts_exact_heading(workspace):
    _, window, _ = workspace
    window.open_note(METHOD)
    until(lambda: window.library.edit_button.isEnabled())
    window.library.edit_button.click()
    until(lambda: window.editor is not None)
    editor = window.editor
    editor.markdown.setPlainText("A note 🪻 [[Chamber#Meth")
    editor.markdown.moveCursor(editor.markdown.textCursor().MoveOperation.End)
    until(lambda: editor.suggestions.isVisible())
    QTest.keyClick(editor.markdown, Qt.Key.Key_Down)
    QTest.keyClick(editor.suggestions, Qt.Key.Key_Return)
    until(lambda: editor.markdown.toPlainText().endswith(f"[[/{METHOD}#Method]]"))
    assert editor.markdown.toPlainText().startswith("A note 🪻 ")
    editor.reject()


def test_canvas_note_card_refresh_redacts_revoked_shared_title(workspace):
    _, window, backend = workspace
    window.show_page(CANVAS)
    window.canvas.add_note(PATH)
    until(lambda: PATH in window.canvas.note_cards)
    assert window.canvas.note_cards[PATH]["title"] == "Thermal calibration"
    backend.revoked = True
    window.refresh()
    until(lambda: PATH not in window.canvas.note_cards)
    node = next(node for node in window.canvas.board["nodes"] if node.get("file") == PATH)
    assert window.canvas.caption(node) == "Unavailable note"


@pytest.mark.skipif(os.name != "nt", reason="The executable host is Windows AppContainer")
def test_installed_plugin_panel_uses_shared_context_and_opens_library(workspace):
    _, window, _ = workspace
    window.open_note(PATH)
    until(lambda: window.library.current_path == PATH)
    window.show_page(PLUGINS)
    pack = min(EXAMPLES.glob(f"relationship-tools*{PACK_SUFFIX}"))
    installed = window.plugin_store.install(pack)
    window.plugin_store.enable(installed.id, installed.version, installed.permissions)
    window.plugins.refresh((installed.id, installed.version))
    window.plugins.invoke.click()
    until(lambda: "plugin-run" not in window.pending, seconds=30)
    assert window.plugins.rows.topLevelItemCount() == 2, (
        window.plugins.status.text(),
        window.status.text(),
    )
    item = window.plugins.rows.topLevelItem(0)
    assert item.text(0) in {"Sensor placement", "Chamber protocol"}
    window.plugins.rows.setCurrentItem(item)
    QTest.keyClick(window.plugins.rows, Qt.Key.Key_Return)
    until(lambda: window.library.current_path in {SENSOR, METHOD})
    assert not window.plugins.rows.topLevelItemCount()


def test_approval_uses_explicit_review_note(workspace):
    app, window, backend = workspace
    window.show_page(2)
    until(lambda: window.inbox.list.count() == 1)
    window.inbox.list.setCurrentRow(0)
    QTest.keyClick(window.inbox.list, Qt.Key.Key_Return)
    until(lambda: window.inbox.approve.isEnabled())
    assert not backend.approved

    def enter_reason():
        dialog = app.activeModalWidget()
        assert isinstance(dialog, QInputDialog)
        dialog.setTextValue("Checked the retained calibration evidence.")
        dialog.accept()

    QTimer.singleShot(30, enter_reason)
    window.inbox.approve.click()
    until(lambda: backend.approved and not window.inbox.list.count())
    assert window.inbox.proposal is None


def test_returning_home_does_not_show_old_search_excerpts(workspace):
    _, window, backend = workspace
    window.home.project.setCurrentIndex(1)
    QTest.keyClicks(window.home.search, "thermal calibration")
    QTest.keyClick(window.home.search, Qt.Key.Key_Return)
    until(lambda: window.home.results.count() == 1)
    window.show_page(1)
    backend.revoked = True
    window.show_page(0)
    assert window.home.results.count() == 0
    until(lambda: window.home.result_title.text() == "No matching notes")


def test_personal_profile_saves_local_sources_and_copies_only_approved_context(
    workspace, tmp_path, monkeypatch
):
    app, window, backend = workspace
    source = tmp_path / "personal.md"
    source.write_text("Fixture private preference", encoding="utf-8")
    monkeypatch.setattr(QFileDialog, "getOpenFileNames", lambda *_: ([str(source)], ""))
    current = {"status": "found", "revision": 3, "text": "Approved voice preference."}
    proposals = []
    get_note_calls = []
    original = backend.call

    def call(name, args=None):
        if (
            name == "read_working_knowledge"
            and args["entry_key"] == "personalization/approved-profile"
        ):
            return dict(current)
        if name == "propose_working_knowledge":
            proposals.append(args)
            return {"proposal_id": "profile-proposal"}
        if name == "get_note" and args["path"] == "personalization/approved-profile.md":
            get_note_calls.append(args["path"])
            if current["text"] == "<!-- DODO PROFILE WITHDRAWN -->":
                raise GatewayError("The approved profile has been withdrawn.")
            return {"text": current["text"]}
        return original(name, args)

    backend.call = call
    clipboard = app.clipboard()
    previous = clipboard.text()
    try:
        window.show_personal_profile()
        editor = window.profile_editor
        editor.add_file.click()
        row = editor.sources.rowCount() - 1
        editor.sources.cellWidget(row, 1).setCurrentText("Writing")
        editor.sources.cellWidget(row, 2).setCurrentText("inferred")
        editor.save_sources.click()
        saved = next(item for item in window.personal_profile.sources() if item.path == source)
        assert (saved.section, saved.basis, saved.enabled) == ("Writing", "inferred", True)
        assert not proposals and not get_note_calls

        editor.tabs.setCurrentIndex(1)
        until(editor.copy.isEnabled)
        editor.draft.setPlainText("Reviewed draft preference.")
        editor.submit.click()
        until(lambda: len(proposals) == 1 and editor.copy.isEnabled())
        assert proposals[0] == {
            "entry_key": "personalization/approved-profile",
            "text": "Reviewed draft preference.",
            "expected_revision": 3,
            "scope": "shared",
            "assertion_type": "user_statement",
            "evidence": [],
        }
        editor.copy.click()
        until(lambda: clipboard.text() == "Approved voice preference.")
        current.update(revision=4, text="Newly approved voice preference.")
        editor.copy.click()
        until(lambda: clipboard.text() == "Newly approved voice preference.")
        assert len(get_note_calls) == 2
        editor.reload.click()
        until(lambda: editor.published.text() == "Approved revision 4.")
        editor.withdraw.click()
        until(lambda: len(proposals) == 2 and editor.copy.isEnabled())
        assert proposals[1]["text"] == "<!-- DODO PROFILE WITHDRAWN -->"
        assert proposals[1]["expected_revision"] == 4
        assert not backend.approved
        current.update(revision=5, text=proposals[1]["text"])
        clipboard.setText("Clipboard sentinel")
        editor.copy.click()
        until(lambda: "has been withdrawn" in editor.hosted_status.text())
        assert clipboard.text() == "Clipboard sentinel"
        editor.reload.click()
        until(lambda: editor.published.text() == "Hosted profile withdrawn at revision 5.")
        assert not editor.copy.isEnabled()
    finally:
        clipboard.setText(previous)
        if getattr(window, "profile_editor", None):
            window.profile_editor.close()


def test_late_proposal_reply_cannot_restore_removed_proposal(workspace):
    _, window, backend = workspace
    window.show_page(2)
    until(lambda: window.inbox.list.count() == 1)
    started, release = threading.Event(), threading.Event()
    original = backend.call

    def delayed(name, arguments=None):
        if name == "read_knowledge_proposal":
            started.set()
            assert release.wait(3)
        return original(name, arguments)

    backend.call = delayed
    window.open_proposal("fixture-review")
    try:
        until(started.is_set)
        backend.approved = True
        window.refresh_inbox()
        until(lambda: not window.inbox.list.count())
    finally:
        release.set()
    until(lambda: "proposal" not in window.pending)
    assert window.inbox.proposal is None
