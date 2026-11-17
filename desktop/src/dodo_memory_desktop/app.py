"""A search-first desktop workspace for the shared memory gateway."""

import sys
from functools import partial
from pathlib import Path

from PySide6.QtCore import (
    QObject,
    QRunnable,
    QSettings,
    QSignalBlocker,
    QThreadPool,
    QTimer,
    Signal,
)
from PySide6.QtGui import QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QHBoxLayout,
    QInputDialog,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .canvas import Canvas
from .configuration import runtime_path
from .gateway import Gateway, GatewayError
from .library import Library
from .matrix import Matrix
from .note_editor import NoteEditor, new_note_path, note_title
from .personal_context import PersonalProfile
from .personal_profile_ui import PROFILE_REQUEST, PersonalProfileEditor
from .plugin_ui import Plugins
from .plugins import PluginError, PluginStore
from .ui import (
    ASSETS,
    ROLE,
    THEME,
    Home,
    Inbox,
    archive_palette,
    button,
    label,
    row_item,
    theme_stylesheet,
    windows_motion_enabled,
)
from .vault_graph import VaultGraph, arrange

HOME, LIBRARY, INBOX, GRAPH, CANVAS, PLUGINS, ASK = range(7)


class Replies(QObject):
    received = Signal(object, object, object)


class Call(QRunnable):
    def __init__(self, identity, operation, replies):
        super().__init__()
        self.identity = identity
        self.operation = operation
        self.replies = replies

    def run(self):
        try:
            result = self.operation()
        except GatewayError as error:
            self.replies.received.emit(self.identity, None, error)
        else:
            self.replies.received.emit(self.identity, result, None)


class Desktop(QMainWindow):
    def __init__(
        self,
        gateway=None,
        settings=None,
        data_directory: Path | None = None,
        *,
        whole_catalogue=None,
    ):
        super().__init__()
        self.gateway = gateway if gateway is not None else Gateway()
        self.settings = (
            settings
            if settings is not None
            else QSettings(str(runtime_path("desktop") / "window.ini"), QSettings.Format.IniFormat)
        )
        self.setWindowTitle("Dodo Mem")
        self.setWindowIcon(QIcon(str(ASSETS / "archive-mark.ico")))
        app = QApplication.instance()
        app.setPalette(archive_palette(app.palette()))
        self.setStyleSheet(THEME)
        self.resize(1280, 800)
        self.setMinimumSize(620, 490)
        geometry = self.settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        self.pool = QThreadPool(self)
        self.pool.setMaxThreadCount(4)
        self.replies = Replies(self)
        self.replies.received.connect(self._receive)
        self.epoch = 0
        self.serial = 0
        self.pending = {}
        self.last_query = ""
        self.catalog: dict[str, dict] = {}
        self.cursor = None
        self.online = False
        self.recent = self.settings.value("recent", [], type=list)
        self.open_notes: dict[str, dict] = {}
        self.pending_proposals: dict[str, int] = {}
        self.source_request: dict | None = None
        self.palette: QDialog | None = None
        self.editor: NoteEditor | None = None
        self.link_dialog: QDialog | None = None
        self.profile_editor: PersonalProfileEditor | None = None
        self.personal_profile = PersonalProfile(
            data_directory / "personal-profile.json" if data_directory else None
        )
        self.home = Home(
            self.settings.value("pins", [], type=list), whole_library=whole_catalogue is not None
        )
        self._search_scope = self.home.search_scope.currentData()
        self.home.reduce_motion = self.settings.value("reduce_motion", False, type=bool)
        self.library = Library()
        self.library.set_motion_reduced(self.home.reduce_motion or not windows_motion_enabled())
        self.inbox = Inbox()
        self.graph = VaultGraph()
        self.canvas = Canvas(data_directory / "canvases" if data_directory else None)
        self.plugin_store = PluginStore(data_directory / "plugins" if data_directory else None)
        self.plugins = Plugins(self.plugin_store)
        self.plugin_inputs = None
        self.stack = QStackedWidget()
        pages = [self.home, self.library, self.inbox, self.graph, self.canvas, self.plugins]
        self.whole = None
        if whole_catalogue is not None:
            from .whole_workspace import WholeWorkspace

            self.whole = WholeWorkspace(
                self, whole_catalogue, pages, (LIBRARY, GRAPH, CANVAS, PLUGINS, ASK)
            )
            self.home.workspace_title.setText("Whole library")
            self.home.project_folders.setText("Browse folders")
            self.home.project_connections.setText("Library graph")
            self.home.add_ins.setText("Dodo Kit and add-ins")
        self.matrix = Matrix(self)
        self.matrix_index = len(pages)
        pages.append(self.matrix)
        for page in pages:
            self.stack.addWidget(page)
        self.status = label("Connecting to Dodo Mem")
        self.status.setObjectName("status")
        self.status.setWordWrap(True)
        self.status.setMaximumWidth(185)
        self.service_button = button("Services")
        self.service_button.clicked.connect(self.show_services)
        rail = QFrame(objectName="navigation")
        rail.setFixedWidth(198)
        rail_layout = QVBoxLayout(rail)
        rail_layout.setContentsMargins(15, 29, 15, 18)
        rail_layout.setSpacing(7)
        rail_layout.addWidget(label("Dodo Mem", "brand"))
        rail_layout.addSpacing(14)
        self.navigation = []
        destinations = [
            (HOME, "Home", "magnifying-glass"),
            (LIBRARY, "Library", "books"),
            (GRAPH, "Graph", "vault-graph"),
            (CANVAS, "Canvas", "canvas"),
            (PLUGINS, "Plugins", "plugin"),
            (INBOX, "Review Inbox", "tray"),
            (self.matrix_index, "Matrix", "plugin"),
        ]
        if self.whole:
            destinations.insert(1, (ASK, "Ask Dodo", "magnifying-glass"))
        self.navigation = [None] * self.stack.count()
        for index, title, icon in destinations:
            nav = button(title, icon, "nav")
            nav.setCheckable(True)
            action = self.show_library_notes if index == LIBRARY else partial(self.show_page, index)
            nav.clicked.connect(action)
            rail_layout.addWidget(nav)
            self.navigation[index] = nav
            if index == LIBRARY:
                projects = button("Projects", "canvas", "nav")
                projects.clicked.connect(self.show_project_folders)
                rail_layout.addWidget(projects)
        personal_profile = button("Personal profile", role="nav")
        personal_profile.clicked.connect(self.show_personal_profile)
        rail_layout.addWidget(personal_profile)
        rail_layout.addStretch()
        rail_layout.addWidget(self.status)
        rail_layout.addWidget(self.service_button)
        outer = QWidget()
        layout = QHBoxLayout(outer)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        layout.addWidget(rail)
        layout.addWidget(self.stack, 1)
        self.setCentralWidget(outer)
        self.home.scope_changed.connect(self._change_scope)
        self.home.search_scope_changed.connect(self._change_search_scope)
        self.home.search_requested.connect(self.search)
        self.home.search.textChanged.connect(self._cancel_search)
        self.home.note_requested.connect(self.open_result)
        self.home.graph_requested.connect(self._open_search_graph)
        if self.whole:
            self.home.local_result_requested.connect(self.whole.open_node)
            self.home.ask_requested.connect(self._ask_search)
        self.home.project_folders.clicked.connect(self.show_project_folders)
        self.home.project_maps.clicked.connect(self.show_project_maps)
        self.home.project_connections.clicked.connect(self.show_project_connections)
        self.home.add_ins.clicked.connect(partial(self.show_page, PLUGINS))
        self.home.pinned_changed.connect(lambda values: self.settings.setValue("pins", values))
        self.library.note_requested.connect(self.open_note)
        self.library.note_heading_requested.connect(self.open_note)
        self.library.evidence_requested.connect(self.open_result)
        self.library.projects.refs_requested.connect(self._refresh_project_notes)
        self.library.source_requested.connect(self.open_source)
        self.library.relations_requested.connect(self.open_related)
        self.library.history_requested.connect(self.open_history)
        self.library.catalog_more_requested.connect(self._load_more)
        self.library.path_changed.connect(self._tab_selected)
        self.library.link_requested.connect(self.resolve_link)
        self.library.edit_requested.connect(self.edit_note)
        self.library.new_note_requested.connect(self.new_note)
        self.library.ordinary_history_requested.connect(self.open_ordinary_history)
        self.library.ordinary_revision_requested.connect(self.open_ordinary_revision)
        self.graph.scope_changed.connect(self._graph_scope_changed)
        self.graph.tabs.currentChanged.connect(self._refresh_graph_tab)
        self.graph.projects.refs_requested.connect(self._refresh_graph_project_notes)
        self.graph.note_requested.connect(self.open_result)
        self.graph.evidence_requested.connect(self.open_result)
        self.canvas.note_requested.connect(self.open_note)
        self.canvas.note_refs_requested.connect(self._resolve_canvas_cards)
        self.plugins.run_requested.connect(self._run_plugin)
        self.plugins.invocation_changed.connect(self._clear_plugin_result)
        self.plugins.apply_requested.connect(self._apply_pack_entry)
        self.plugins.changed.connect(self._plugins_changed)
        self.plugins.note_requested.connect(self.open_note)
        self._plugins_changed()
        self.inbox.proposal_requested.connect(self.open_proposal)
        self.inbox.evidence_requested.connect(self.open_result)
        self.inbox.draft_requested.connect(self.new_draft)
        self.inbox.approve_requested.connect(self.approve)
        QShortcut(QKeySequence("Ctrl+K"), self, activated=self.quick_switch)
        QShortcut(QKeySequence("Ctrl+F"), self, activated=partial(self.show_page, HOME))
        initial = (
            GRAPH
            if self.whole
            else min(self.stack.count() - 1, max(0, self.settings.value("page", 0, type=int)))
        )
        self.show_page(initial, refresh=False)
        self.refresh_timer = QTimer(self, interval=20000)
        self.refresh_timer.timeout.connect(self.refresh)
        self.refresh_timer.start()
        QTimer.singleShot(0, self.refresh)

    def closeEvent(self, event):
        if "note-save" in self.pending:
            event.ignore()
            return
        if not self.canvas.save():
            self.show_page(CANVAS)
            event.ignore()
            return
        self.settings.setValue("geometry", self.saveGeometry())
        self.settings.setValue("page", self.stack.currentIndex())
        self.refresh_timer.stop()
        self.matrix.set_active(False)
        if self.whole:
            self.whole.timer.stop()
            if not self.whole.network.shutdown():
                self.status.setText("Waiting for the E5 view to close. Try closing again.")
                event.ignore()
                return
        self.pending.clear()
        self.pool.clear()
        super().closeEvent(event)

    def request(self, lane, operation, completed, failed=None):
        self.serial += 1
        identity = (self.epoch, lane, self.serial)
        self.pending[lane] = (identity, completed, failed)
        self.pool.start(Call(identity, operation, self.replies))

    def _receive(self, identity, value, error):
        epoch, lane, _ = identity
        current = self.pending.get(lane)
        if epoch != self.epoch or current is None or current[0] != identity:
            return
        _, completed, failed = self.pending.pop(lane)
        if error is None:
            completed(value)
            return
        self.status.setText(str(error))
        if error.connection_lost:
            self._disconnect()
        elif failed is not None:
            failed()

    def _disconnect(self):
        self.epoch += 1
        self.pending.clear()
        self.online = False
        self.matrix.unavailable()
        if self.palette:
            self.palette.reject()
        if self.editor:
            QDialog.reject(self.editor)
        if self.link_dialog:
            self.link_dialog.reject()
        if self.profile_editor:
            self.profile_editor.unavailable()
        self.catalog.clear()
        self.open_notes.clear()
        self.source_request = None
        self.cursor = None
        self.home.set_recent([])
        if self._search_scope == "shared":
            self.home.set_results([])
        self.library.clear_content()
        self.library.projects.set_note_refs({})
        self.graph.clear()
        self.graph.projects.set_note_refs({})
        self.graph.filters([], [])
        self.canvas.clear_note_cards()
        self._clear_plugin_result()
        self.inbox.set_proposals([])
        self.pending_proposals.clear()
        self.inbox.clear_detail()
        self.home.set_filters([], [])
        self.navigation[INBOX].setText("Review Inbox")

    def show_page(self, index: int, *, refresh: bool = True):
        previous = self.stack.currentIndex()
        if previous != index:
            self.pending.pop("note", None)
        if previous == HOME and index != HOME:
            self._clear_search_results()
        if previous == GRAPH and index != GRAPH:
            self.pending.pop("graph", None)
            self.pending.pop("graph-project-notes", None)
            self.graph.clear("Connecting to the vault")
            self.graph.projects.set_note_refs({})
        if previous == CANVAS and index != CANVAS:
            self.pending.pop("canvas", None)
            self.canvas.clear_note_cards()
        if previous == PLUGINS and index != PLUGINS:
            self._clear_plugin_result()
        self.stack.setCurrentIndex(index)
        self.matrix.set_active(index == self.matrix_index)
        for position, nav in enumerate(self.navigation):
            nav.setChecked(position == index)
        if index == HOME:
            self.home.search.setFocus()
            if previous != HOME and self.last_query:
                self.search(self.last_query)
        if index == INBOX and refresh:
            self.refresh_inbox()
        if index == GRAPH and (refresh or self.graph.tabs.currentWidget() is self.graph.projects):
            self._refresh_graph_tab()
        if index == CANVAS and refresh:
            self.canvas.request_note_cards()
        if index == PLUGINS and self.whole and self.whole.plugins.entries is None:
            self.whole.plugins.refresh()
        if index == LIBRARY and previous != LIBRARY and self.library.projects_visible():
            self.library.projects.reload()
        if self.whole:
            self.whole.refresh()

    def show_project_folders(self):
        if self.whole:
            self.whole.show_library()
            return
        self.show_page(LIBRARY)
        self.library.show_projects()

    def show_library_notes(self):
        if self.whole:
            self.whole.show_library()
            return
        self.library.show_notes()
        self.show_page(LIBRARY)

    def show_project_connections(self):
        if self.whole:
            self.whole.graph.set_scope(None)
            self.whole.show_graph()
            return
        self.graph.show_project_connections()
        self.show_page(GRAPH)

    def show_project_maps(self):
        if self.whole:
            self.show_project_connections()
            self.whole.graph.show_mind_map()
        else:
            self.library.projects.open_archive_index()

    def _refresh_graph_tab(self, _index=None):
        if self.stack.currentIndex() != GRAPH:
            return
        if self.whole and self.whole.graph_tabs.currentWidget() is self.whole.network:
            self.whole.refresh()
            return
        if self.whole and self.whole.graph_tabs.currentWidget() is self.whole.graph:
            self.pending.pop("graph", None)
            self.pending.pop("graph-project-notes", None)
            self.whole.refresh()
            return
        if self.graph.tabs.currentWidget() is self.graph.projects:
            self.pending.pop("graph", None)
            self.graph.projects.reload()
        else:
            self.pending.pop("graph-project-notes", None)
            self.graph.projects.set_note_refs({})
            self.refresh_graph()

    def _change_scope(self):
        for lane in tuple(self.pending):
            if lane in {
                "catalog",
                "search",
                "note",
                "source",
                "relations",
                "history",
            } or lane.startswith("verify:"):
                self.pending.pop(lane)
        self.source_request = None
        self.catalog.clear()
        self.cursor = None
        self.library.set_catalog([])
        self.home.set_recent([])
        if self._search_scope == "shared":
            self.home.set_results([])
        self.refresh()
        if self._search_scope == "shared" and self.home.search.text().strip():
            self.search(self.home.search.text().strip())

    def refresh(self):
        if "catalog" not in self.pending:
            scope = self.home.scope()
            self.request(
                "catalog", partial(self.gateway.call, "browse_notes", scope), self._catalog_ready
            )
        if "inbox" not in self.pending and self.online:
            self.refresh_inbox()
        if (
            self.stack.currentIndex() == GRAPH
            and self.online
            and (not self.whole or self.whole.graph_tabs.currentWidget() is self.graph)
        ):
            if self.graph.tabs.currentWidget() is self.graph.projects:
                if "graph-project-notes" not in self.pending:
                    self._refresh_graph_project_notes()
            elif "graph" not in self.pending:
                self.refresh_graph()
        if self.stack.currentIndex() == CANVAS and self.online and "canvas" not in self.pending:
            self.canvas.request_note_cards()
        if "project-notes" not in self.pending:
            self._refresh_project_notes()
        if self.plugin_inputs and "plugin-check" not in self.pending:
            path, _context, _revisions = self.plugin_inputs
            self.request(
                "plugin-check",
                partial(self._related_plugin_input, path),
                self._check_plugin_inputs,
                self._clear_plugin_result,
            )
        for path in self.library.open_paths():
            lane = f"verify:{path}"
            if lane not in self.pending:
                self.request(
                    lane,
                    partial(self.gateway.call, "get_note", {"path": path}),
                    partial(self._verify_note, path),
                    partial(self._forget_note, path),
                )
        path = self.library.current_path
        if path and "relations" not in self.pending:
            self.open_related(path)
        if self.source_request and "source" not in self.pending:
            self.open_source(self.source_request, reveal=False)
        if (
            self.stack.currentIndex() == HOME
            and self._search_scope == "shared"
            and self.home.results.count()
            and "search" not in self.pending
        ):
            self.search(self.home.search.text().strip())

    def _catalog_ready(self, page: dict, *, append: bool = False):
        was_offline = not self.online
        self.online = True
        self.status.setText("Dodo Mem connected")
        if not append:
            self.catalog.clear()
        self.catalog.update((entry["path"], entry) for entry in page["items"])
        if self.palette:
            self.palette.reject()
        self.cursor = page.get("next_cursor")
        self.home.set_filters(page.get("projects", []), page.get("collections", []))
        if was_offline:
            self.graph.filters(page.get("projects", []), page.get("collections", []))
        self.home.set_recent([self.catalog[path] for path in self.recent if path in self.catalog])
        self.library.set_catalog(list(self.catalog.values()), has_more=bool(self.cursor))
        if was_offline:
            self._refresh_project_notes()
            self.refresh_inbox()
            if self.stack.currentIndex() == GRAPH:
                self._refresh_graph_tab()
            if self.stack.currentIndex() == CANVAS:
                self.canvas.request_note_cards()

    def _load_more(self):
        if self.cursor:
            scope = {**self.home.scope(), "cursor": self.cursor}
            self.request(
                "catalog",
                partial(self.gateway.call, "browse_notes", scope),
                partial(self._catalog_ready, append=True),
            )

    def _cancel_search(self):
        self.last_query = ""
        self._clear_search_results()

    def _change_search_scope(self, scope):
        previous = self._search_scope
        self._search_scope = scope
        # A query written for a private source must never become a shared-vault query.
        if (previous == "shared") != (scope == "shared"):
            self.home.search.clear()
            self._cancel_search()
        elif self.last_query:
            self.search(self.last_query)

    def _clear_search_results(self):
        self.pending.pop("search", None)
        if self.whole:
            self.whole.library.reads.cancel("home-search")
        self.home.results.clear()
        self.home.result_title.setText("Search results")

    def search(self, query: str):
        if not query:
            return
        self.last_query = query
        self._clear_search_results()
        if self._search_scope != "shared":
            self.whole.search_home(query, project=self._search_scope == "project")
            return
        self.request(
            "search",
            partial(
                self.gateway.call,
                "search",
                {"query": query, "limit": 20, **self.home.scope()},
            ),
            self.home.set_results,
            self._clear_search_results,
        )

    def _ask_search(self):
        query = self.home.search.text().strip()
        if not query or self._search_scope == "shared":
            return
        node = None
        if self._search_scope == "project":
            node = self.whole.selected
            if not self.whole.project_scope():
                self.home.result_title.setText("Select a project first")
                return
        if node:
            self.whole.select(node)
        else:
            self.whole.ask.set_context()
        self.whole.ask.question.setText(query)
        self.show_page(ASK)

    def _open_search_graph(self, entry):
        if self._search_scope != "shared":
            self.whole.show_graph(entry)
            return
        self.graph.focus_note(entry["path"], self.home.scope())
        if self.whole:
            self.whole.graph_tabs.setCurrentWidget(self.graph)
        self.show_page(GRAPH)

    def open_result(self, entry: dict):
        self.open_note(entry["path"], entry.get("heading"), expected_sha256=entry.get("sha256"))

    def open_note(
        self, path: str, heading: str | None = None, *, expected_sha256: str | None = None
    ):
        arguments = {"path": path}
        if expected_sha256 is not None:
            arguments["expected_sha256"] = expected_sha256
        self.request(
            "note",
            partial(self.gateway.call, "get_note", arguments),
            partial(self._note_ready, heading),
            partial(self._verify_open_note, path)
            if expected_sha256
            else partial(self._forget_note, path),
        )

    def _verify_open_note(self, path):
        if path in self.open_notes:
            self.request(
                f"verify:{path}",
                partial(self.gateway.call, "get_note", {"path": path}),
                partial(self._verify_note, path),
                partial(self._forget_note, path),
            )

    def _note_ready(self, heading, note: dict):
        path = note["path"]
        self.open_notes[path] = note
        self.recent = [path, *(entry for entry in self.recent if entry != path)][:10]
        self.settings.setValue("recent", self.recent)
        self.home.set_recent([self.catalog[item] for item in self.recent if item in self.catalog])
        if self.whole:
            self.whole.library_tabs.setCurrentWidget(self.library)
        self.show_page(LIBRARY)
        self.library.open_note(note, heading)

    def _verify_note(self, path, note):
        if path in self.library.open_paths() and note != self.open_notes.get(path):
            self.open_notes[path] = note
            self.library.open_note(note, activate=False)
            if path == self.library.current_path:
                self.library._tab_changed()

    def _forget_note(self, path):
        self.open_notes.pop(path, None)
        if self.editor and self.editor.path == path:
            QDialog.reject(self.editor)
        self.library.close_note(path)
        if self.source_request and self.source_request["path"] == path:
            self.source_request = None
            self.library.set_source(None)

    def _tab_selected(self, _index):
        self.pending.pop("source", None)
        self.pending.pop("history", None)
        self.pending.pop("revision", None)
        self.pending.pop("link", None)
        self.source_request = None
        self._clear_plugin_result()

    def open_source(self, locator: dict, *, reveal: bool = True):
        self.source_request = locator.copy()
        self.request(
            "source",
            partial(self.gateway.call, "open_source", {**locator, "preview": True}),
            partial(self.library.set_source, reveal=reveal),
            partial(self.library.set_source, None, reveal=reveal),
        )

    def open_related(self, path: str):
        self.request(
            "relations",
            partial(self.gateway.call, "related_notes", {"path": path, "broad": True}),
            partial(self.library.set_related, path),
            partial(self.library.set_related, path, {"nodes": [], "edges": [], "count": 0}),
        )

    def open_history(self, entry_key: str):
        self.request(
            "history",
            partial(self.gateway.call, "working_knowledge_history", {"entry_key": entry_key}),
            partial(self.library.set_history, entry_key),
        )

    def open_ordinary_history(self, path):
        self.request(
            "history",
            partial(self.gateway.call, "ordinary_note_history", {"path": path}),
            partial(self.library.set_ordinary_history, path),
        )

    def open_ordinary_revision(self, path, revision):
        self.request(
            "revision",
            partial(
                self.gateway.call,
                "read_ordinary_note_revision",
                {"path": path, "revision": revision},
            ),
            partial(self.library.set_ordinary_revision, path, revision),
        )

    def resolve_link(self, path, target, wiki):
        self.request(
            "link",
            partial(
                self.gateway.call,
                "resolve_note_link",
                {"path": path, "target": target, "wiki": wiki},
            ),
            partial(self._link_ready, path, target, wiki),
        )

    def _link_ready(self, source, target, wiki, result):
        if self.library.current_path != source or self.stack.currentIndex() != LIBRARY:
            return
        if result["status"] == "resolved":
            self.open_result(result)
            return
        dialog = QDialog(
            self,
            windowTitle="Choose a note" if result["status"] == "ambiguous" else "Link unavailable",
        )
        self.link_dialog = dialog
        dialog.resize(480, 200)
        layout = QVBoxLayout(dialog)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        actions.rejected.connect(dialog.reject)
        choices = result.get("choices", [])
        if choices:
            notes = QListWidget()
            notes.setAccessibleName("Notes matching the link")
            for choice in choices:
                row_item(notes, f"{choice['title']}  ·  {choice['path']}", choice)
            notes.setCurrentRow(0)
            notes.itemActivated.connect(dialog.accept)
            open_note = actions.addButton("Open note", QDialogButtonBox.ButtonRole.AcceptRole)
            open_note.clicked.connect(dialog.accept)
            layout.addWidget(notes, 1)
        else:
            layout.addWidget(label("No shared note or heading matches this link."))
            if result["status"] == "unresolved" and wiki and not target.startswith(("/", "#")):
                draft = actions.addButton("Create draft", QDialogButtonBox.ButtonRole.AcceptRole)
                draft.clicked.connect(dialog.accept)
        layout.addWidget(actions)
        accepted = dialog.exec()
        self.link_dialog = None
        if accepted and self.library.current_path == source:
            if choices:
                self.open_result(notes.currentItem().data(ROLE))
            else:
                self.new_note(note_title(target))

    def edit_note(self, path):
        if not self.editor:
            self.request(
                "editor",
                partial(self.gateway.call, "read_note_for_edit", {"path": path}),
                self._show_editor,
            )

    def new_note(self, suggested="", text=""):
        if self.editor or not self.online:
            return
        title, accepted = QInputDialog.getText(self, "New shared note", "Title", text=suggested)
        if accepted and self.online:
            try:
                path = new_note_path(title)
            except ValueError as error:
                self.status.setText(str(error))
                return
            self._show_editor(
                {"path": path, "text": text or f"# {title.strip()}\n\n", "sha256": None}
            )

    def _show_editor(self, note):
        if self.editor or not self.online:
            return
        editor = self.editor = NoteEditor(note, self)
        editor.finished.connect(self._editor_closed)
        editor.suggest_requested.connect(self._suggest_links)
        editor.save_requested.connect(self._save_note)
        editor.open()

    def _editor_closed(self):
        self.pending.pop("suggest", None)
        self.editor = None

    def _suggest_links(self, anchor):
        self.request(
            "suggest",
            partial(self.gateway.call, "suggest_note_links", {"query": anchor[1], "limit": 12}),
            lambda result: self.editor and self.editor.show_suggestions(anchor, result),
        )

    def _save_note(self, arguments):
        editor = self.editor

        def saved(_):
            if self.editor is editor:
                editor.accept()
            self.pending.pop(f"verify:{arguments['path']}", None)
            self.open_note(arguments["path"])
            self.refresh()
            self.status.setText("Note saved.")

        self.request(
            "note-save",
            partial(self.gateway.save_note, **arguments),
            saved,
            lambda: editor.save_failed(self.status.text()) if self.editor is editor else None,
        )

    def _graph_scope_changed(self):
        self.graph.clear("Loading graph")
        self.refresh_graph()

    def refresh_graph(self):
        scope = self.graph.scope()
        focus = self.graph.focused_path

        def read():
            snapshot = (
                self.gateway.call("related_notes", {"path": focus, **scope})
                if focus
                else self.gateway.call("graph_snapshot", scope)
            )
            return snapshot, arrange(snapshot)

        def ready(value):
            snapshot, _ = value
            if not any(scope.values()) and "projects" in snapshot and "collections" in snapshot:
                self.graph.filters(snapshot["projects"], snapshot["collections"])
            if scope != self.graph.scope() or focus != self.graph.focused_path:
                self._graph_scope_changed()
            else:
                self.graph.show_snapshot(value)

        self.request("graph", read, ready, self.graph.clear)

    def _resolve_canvas_cards(self, paths):
        if not paths or not self.online or self.stack.currentIndex() != CANVAS:
            self.pending.pop("canvas", None)
            self.canvas.clear_note_cards()
            return

        self.request(
            "canvas",
            partial(self._note_refs, paths),
            self.canvas.set_note_cards,
            self.canvas.clear_note_cards,
        )

    def _note_refs(self, paths):
        rows = {}
        for start in range(0, len(paths), 50):
            page = self.gateway.call("note_card_refs", {"paths": paths[start : start + 50]})
            rows.update((row["path"], row) for row in page["items"])
        return rows

    def _refresh_project_notes(self):
        projects = self.library.projects
        if (
            not self.online
            or self.stack.currentIndex() != LIBRARY
            or not self.library.projects_visible()
            or not (paths := projects.note_paths())
        ):
            return
        self.request(
            "project-notes",
            partial(self._note_refs, paths),
            projects.set_note_refs,
            partial(projects.set_note_refs, {}),
        )

    def _refresh_graph_project_notes(self):
        projects = self.graph.projects
        if (
            not self.online
            or self.stack.currentIndex() != GRAPH
            or self.graph.tabs.currentWidget() is not projects
            or (self.whole and self.whole.graph_tabs.currentWidget() is not self.graph)
        ):
            return
        if not (paths := projects.note_paths()):
            self.pending.pop("graph-project-notes", None)
            projects.set_note_refs({})
            return
        self.request(
            "graph-project-notes",
            partial(self._note_refs, paths),
            projects.set_note_refs,
            partial(projects.set_note_refs, {}),
        )

    def _clear_plugin_result(self):
        self.pending.pop("plugin-run", None)
        self.pending.pop("plugin-check", None)
        self.plugin_inputs = None
        self.plugins.show_result(None)
        if self.plugins.status.text() == "Running plugin…":
            self.plugins.status.clear()

    def _check_plugin_inputs(self, snapshot):
        if self.plugin_inputs != snapshot:
            self._clear_plugin_result()

    def _related_plugin_input(self, path):
        related = self.gateway.call("related_notes", {"path": path, "broad": True, "limit": 40})
        nodes = {row["path"]: row for row in related["nodes"]}
        if path not in nodes:
            raise GatewayError("The selected note is no longer available.")
        kinds = {}
        for edge in related["edges"]:
            other = edge["target"] if edge["source"] == path else edge["source"]
            if other in nodes:
                kinds.setdefault(other, edge["kind"])
        context = {
            "items": [
                {"title": nodes[other]["title"], "path": other, "kind": kind}
                for other, kind in sorted(kinds.items())
            ]
        }
        revisions = {source: nodes[source].get("sha256") for source in {*kinds, path}}
        return path, context, revisions

    def _run_plugin(self, plugin_id, action, sha256, generation):
        self._clear_plugin_result()
        plugin = next(
            (
                version
                for version in self._active_plugins()
                if (version.id, version.sha256, version.generation)
                == (plugin_id, sha256, generation)
            ),
            None,
        )
        if plugin is None:
            return
        entry = next((entry for entry in plugin.entries if entry["id"] == action), None)
        if entry is None or entry["kind"] not in {"command", "panel"}:
            return
        path = self.library.current_path

        def run():
            inputs = None
            context = {}
            if entry.get("permission") == "read:related":
                if not path:
                    raise GatewayError("Open a shared note in Library before running this plugin.")
                inputs = self._related_plugin_input(path)
                context = inputs[1]
            try:
                result = self.plugin_store.run(
                    plugin_id,
                    action,
                    context,
                    expected_sha256=sha256,
                    expected_generation=generation,
                )
            except PluginError as error:
                raise GatewayError(str(error)) from error
            if inputs and self._related_plugin_input(path) != inputs:
                raise GatewayError(
                    "The plugin's source notes changed. Run it again to refresh the panel."
                )
            return result, inputs

        def ready(value):
            result, self.plugin_inputs = value
            self.plugins.status.clear()
            self.plugins.show_result(result)

        self.plugins.status.setText("Running plugin…")
        self.request(
            "plugin-run",
            run,
            ready,
            lambda: self.plugins.status.setText(self.status.text()),
        )

    def _active_plugins(self):
        try:
            return self.plugin_store.active()
        except PluginError as error:
            self.plugins.status.setText(str(error))
            return []

    def _plugins_changed(self):
        self._clear_plugin_result()
        selected = self.settings.value("plugin_theme", [], type=list)
        tokens = None
        for plugin in self._active_plugins():
            for entry in plugin.entries:
                if entry["kind"] == "theme" and [plugin.id, entry["id"]] == selected:
                    tokens = entry["tokens"]
        if selected and tokens is None:
            self.settings.remove("plugin_theme")
        palette = archive_palette(QApplication.instance().palette(), tokens)
        QApplication.instance().setPalette(palette)
        self.setStyleSheet(theme_stylesheet(tokens))
        self.canvas.scene.setBackgroundBrush(palette.window())

    def _apply_pack_entry(self, entry):
        kind = entry["kind"]
        if kind == "theme":
            for plugin in self._active_plugins():
                if entry in plugin.entries:
                    self.settings.setValue("plugin_theme", [plugin.id, entry["id"]])
                    self._plugins_changed()
                    return
        elif kind == "saved_view":
            for combo, name in (
                (self.home.project, "project"),
                (self.home.collection, "collection"),
            ):
                index = combo.findData(entry.get(name))
                if index < 0 and entry.get(name):
                    self.status.setText("A filter in this saved view is no longer available.")
                    return
            with QSignalBlocker(self.home.project), QSignalBlocker(self.home.collection):
                self.home.project.setCurrentIndex(
                    max(0, self.home.project.findData(entry.get("project")))
                )
                self.home.collection.setCurrentIndex(
                    max(0, self.home.collection.findData(entry.get("collection")))
                )
            self.show_page(HOME)
            self.home.search.setText(entry.get("query", ""))
            self._change_scope()
        elif kind == "note_template":
            self.new_note(entry["note_title"], entry["text"])
        elif kind == "canvas_template":
            try:
                self.canvas.create_board(entry["title"], entry["document"])
            except (OSError, ValueError) as error:
                self.status.setText(str(error))
                return
            self.show_page(CANVAS)

    def refresh_inbox(self):
        self.request(
            "inbox",
            partial(self.gateway.call, "list_knowledge_proposals", {"limit": 100}),
            self._inbox_ready,
        )

    def _inbox_ready(self, entries):
        self.pending_proposals = {
            entry["proposal_id"]: entry["expected_revision"] for entry in entries
        }
        self.inbox.set_proposals(entries)
        suffix = f" ({len(entries)})" if entries else ""
        self.navigation[INBOX].setText(f"Review Inbox{suffix}")
        if self.inbox.proposal and not self._pending_revision(self.inbox.proposal):
            self.inbox.clear_detail()

    def _pending_revision(self, proposal: dict) -> bool:
        return self.online and (
            self.pending_proposals.get(proposal["proposal_id"]) == proposal["expected_revision"]
        )

    def open_proposal(self, proposal_id: str):
        def read():
            proposal = self.gateway.call("read_knowledge_proposal", {"proposal_id": proposal_id})
            current = self.gateway.call(
                "read_working_knowledge", {"entry_key": proposal["entry_key"]}
            )
            return proposal, current

        def ready(value):
            proposal, current = value
            if proposal["proposal_id"] == proposal_id and self._pending_revision(proposal):
                self.inbox.set_proposal(proposal, current)

        self.request("proposal", read, ready, self.inbox.clear_detail)

    def new_draft(self):
        dialog = QDialog(self, windowTitle="New draft")
        dialog.resize(530, 400)
        layout = QVBoxLayout(dialog)
        title = QLineEdit(placeholderText="Title")
        title.setAccessibleName("Draft title")
        title.setMaxLength(160)
        text = QPlainTextEdit(placeholderText="Write a note")
        text.setAccessibleName("Draft text")
        layout.addWidget(title)
        layout.addWidget(text, 1)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        submit = actions.addButton("Submit draft", QDialogButtonBox.ButtonRole.AcceptRole)
        submit.setEnabled(False)

        def update():
            content = text.toPlainText()
            submit.setEnabled(
                bool(title.text().strip() and content.strip()) and len(content) <= 100000
            )

        title.textChanged.connect(update)
        text.textChanged.connect(update)
        actions.accepted.connect(dialog.accept)
        actions.rejected.connect(dialog.reject)
        layout.addWidget(actions)
        if dialog.exec():
            self.request(
                "draft",
                partial(
                    self.gateway.call,
                    "write_note",
                    {"title": title.text().strip(), "text": text.toPlainText()},
                ),
                lambda _: self.status.setText("Draft submitted to the review inbox."),
            )

    def approve(self, proposal: dict):
        if not proposal or "approval" in self.pending:
            return
        revision = proposal["expected_revision"]
        reason, accepted = QInputDialog.getText(
            self,
            "Approve revision",
            f"Review note for {proposal['entry_key']} (revision {revision + 1})",
        )
        if not accepted or not reason.strip() or self.inbox.proposal is not proposal:
            return
        if not self._pending_revision(proposal):
            return
        if len(reason) > 2048:
            self.status.setText("The review note exceeds 2,048 characters.")
            return
        self.inbox.approve.setEnabled(False)

        def finished(_):
            self.inbox.clear_detail()
            self.refresh_inbox()
            self.status.setText("Revision approved.")

        self.request(
            "approval",
            partial(self.gateway.approve, proposal["proposal_id"], revision, reason.strip()),
            finished,
            lambda: self.open_proposal(proposal["proposal_id"]),
        )

    def quick_switch(self):
        dialog = QDialog(self, windowTitle="Quick switch")
        self.palette = dialog
        dialog.resize(570, 380)
        layout = QVBoxLayout(dialog)
        query = QLineEdit(placeholderText="Find a note, collection, or command")
        query.setAccessibleName("Quick switch query")
        results = QListWidget()
        results.setAccessibleName("Quick switch matches")
        layout.addWidget(query)
        layout.addWidget(results)
        entries = [
            ("Personal profile", self.show_personal_profile),
            (self.home.project_folders.text(), self.show_project_folders),
            ("Open mind maps", self.show_project_maps),
            (self.home.project_connections.text(), self.show_project_connections),
            *(
                (
                    nav.text(),
                    self.show_library_notes if index == LIBRARY else partial(self.show_page, index),
                )
                for index, nav in enumerate(self.navigation)
            ),
            *(
                (entry["title"], partial(self.open_note, entry["path"]))
                for entry in self.catalog.values()
            ),
            *(
                (
                    self.home.collection.itemText(index),
                    partial(self.home.select_collection, self.home.collection.itemData(index)),
                )
                for index in range(1, self.home.collection.count())
            ),
        ]
        actions = []

        def populate():
            results.clear()
            actions.clear()
            needle = query.text().strip().casefold()
            for title, action in entries:
                if needle in title.casefold():
                    row_item(results, title, len(actions))
                    actions.append(action)
            results.setCurrentRow(0)

        def choose():
            if item := results.currentItem():
                action = actions[item.data(ROLE)]
                dialog.accept()
                action()

        query.textChanged.connect(populate)
        query.returnPressed.connect(choose)
        results.itemActivated.connect(choose)
        QShortcut(QKeySequence("Down"), query, activated=lambda: results.setFocus())
        populate()
        dialog.exec()
        self.palette = None

    def show_personal_profile(self):
        if self.profile_editor is not None:
            self.profile_editor.raise_()
            return
        editor = PersonalProfileEditor(self)
        self.profile_editor = editor

        def closed(_):
            self.pending.pop(PROFILE_REQUEST, None)
            self.profile_editor = None

        editor.finished.connect(closed)
        editor.open()

    def show_services(self):
        dialog = QDialog(self, windowTitle="Services")
        dialog.resize(450, 265)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Services", "section"))
        summary = label("Checking local services")
        summary.setWordWrap(True)
        layout.addWidget(summary)
        motion = QCheckBox("Reduce motion", checked=self.home.reduce_motion)
        layout.addWidget(motion)
        layout.addStretch()
        refresh = button("Reconnect")
        start = button("Start Dodo Mem")
        actions = QHBoxLayout()
        actions.addWidget(refresh)
        actions.addWidget(start)
        layout.addLayout(actions)

        def report(status):
            summary.setText("\n".join(f"{name}   {value}" for name, value in status.items()))
            start.setEnabled(status["Dodo Mem"] != "Connected")

        def reconnect():
            self.refresh()
            self.request("services", self.gateway.service_status, report)

        def start_service():
            start.setEnabled(False)
            summary.setText("Starting Dodo Mem")
            self.request(
                "services-start",
                self.gateway.start,
                lambda _: reconnect(),
                lambda: summary.setText(self.status.text()),
            )

        def set_motion(checked):
            self.home.reduce_motion = checked
            self.library.set_motion_reduced(checked or not windows_motion_enabled())
            self.settings.setValue("reduce_motion", checked)
            if checked:
                self.home.art.stop()

        motion.toggled.connect(set_motion)
        refresh.clicked.connect(reconnect)
        start.clicked.connect(start_service)
        self.request("services", self.gateway.service_status, report)
        dialog.exec()
        self.pending.pop("services", None)
        self.pending.pop("services-start", None)


def main() -> int:
    ask_model = next(
        (arg.split("=", 1)[1] for arg in sys.argv if arg.startswith("--ask-model=")), None
    )
    args = [
        arg for arg in sys.argv if arg != "--whole-library" and not arg.startswith("--ask-model=")
    ]
    whole_catalogue = None
    if "--whole-library" in sys.argv or ask_model is not None:
        from .whole_index import Catalogue

        whole_catalogue = Catalogue()
    app = QApplication(args)
    app.setOrganizationName("Dodo")
    # Qt uses the application name in the existing Canvas storage path.
    app.setApplicationName("Dodo Shared Memory")
    app.setApplicationDisplayName("Dodo Mem")
    app.setWindowIcon(QIcon(str(ASSETS / "archive-mark.ico")))
    window = Desktop(whole_catalogue=whole_catalogue)
    if ask_model is not None:
        window.whole.show_ask()
        window.whole.ask.select_model(ask_model)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
