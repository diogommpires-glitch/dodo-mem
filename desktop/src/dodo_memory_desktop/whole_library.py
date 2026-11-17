"""Browse the private inventory without executing files or loading their markup."""

import sqlite3
from collections import deque
from functools import partial
from time import monotonic

from PySide6.QtCore import QObject, QRunnable, QSignalBlocker, QThreadPool, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .ui import ROLE, button, label

PAGE = 100
TREE_KEY = ROLE + 1
FOLDERS = frozenset({"root", "folder", "directory", "project", "volume"})
COUNT_REFRESH_SECONDS = 30
RATE_WINDOW_SECONDS = 120
MIN_RATE_SECONDS = 15


class _Read(QRunnable):
    def __init__(self, token, operation, done):
        super().__init__()
        self.token = token
        self.operation = operation
        self.done = done

    def run(self):
        try:
            result = self.operation()
        except (OSError, sqlite3.Error, ValueError):
            self.done.emit(self.token, None, False)
        else:
            self.done.emit(self.token, result, True)


class CatalogueReads(QObject):
    """Keep owner-local requests independent of the shared gateway connection."""

    done = Signal(object, object, bool)

    def __init__(self, parent):
        super().__init__(parent)
        self.serial = 0
        self.pending = {}
        self.done.connect(self._receive)

    def request(self, lane, operation, complete, failed, *, replace=False):
        if lane in self.pending and not replace:
            return
        self.serial += 1
        token = (lane, self.serial)
        self.pending[lane] = (self.serial, complete, failed)
        QThreadPool.globalInstance().start(_Read(token, operation, self.done))

    def cancel(self, lane):
        self.pending.pop(lane, None)

    def _receive(self, token, result, succeeded):
        lane, serial = token
        current = self.pending.get(lane)
        if current is None or current[0] != serial:
            return
        _, complete, failed = self.pending.pop(lane)
        if succeeded:
            complete(result)
        else:
            failed()


class WholeLibrary(QWidget):
    source_requested = Signal(object)
    selection_changed = Signal(object)
    map_requested = Signal(object)
    canvas_requested = Signal(object)
    graph_requested = Signal(object)
    ask_requested = Signal(object)

    def __init__(self, catalogue):
        super().__init__()
        self.catalogue = catalogue
        self.reads = CatalogueReads(self)
        self.generation = None
        self.current = None
        self._active_query = ""
        self._folder_scope = None
        self._view_serial = 0
        self._top_rows = None
        self._items = {}
        self._branches = {}
        self._reading = False
        self._current_key = None
        self._index_state = None
        self._extraction_counts = None
        self._counts_requested_at = None
        self._queue_history = deque()
        self.tree = QTreeWidget(accessibleName="Whole library folders and files")
        self.tree.setHeaderLabels(("Name", "Category", "Availability"))
        self.tree.setUniformRowHeights(True)
        self.tree.setColumnWidth(0, 300)
        self.tree.itemExpanded.connect(self._expand)
        self.tree.itemActivated.connect(self._activate)
        self.tree.currentItemChanged.connect(self._select)
        self.query = QLineEdit(placeholderText="Search files and indexed text")
        self.query.setAccessibleName("Search the whole library")
        self.query.setMaxLength(1000)
        self.query.returnPressed.connect(self._search)
        self.all_folders = button("All folders")
        self.all_folders.clicked.connect(self._return_to_roots)
        self.all_folders.hide()
        self.status = label("Waiting for the local index", "muted")
        self.status.setWordWrap(True)
        self.title = label("Select a file or folder", "section")
        self.title.setWordWrap(True)
        self.location = label("", "muted")
        self.location.setWordWrap(True)
        self.preview = QPlainTextEdit(readOnly=True, accessibleName="Verified source excerpt")
        actions = QHBoxLayout()
        for name, signal in (
            ("Graph", self.graph_requested),
            ("Mind map", self.map_requested),
            ("Canvas", self.canvas_requested),
            ("Ask Dodo", self.ask_requested),
        ):
            control = button(name)
            control.clicked.connect(lambda _checked=False, action=signal: action.emit(self.current))
            actions.addWidget(control)
        actions.addStretch()
        detail = QWidget()
        detail_layout = QVBoxLayout(detail)
        detail_layout.addWidget(self.title)
        detail_layout.addWidget(self.location)
        detail_layout.addLayout(actions)
        detail_layout.addWidget(self.preview, 1)
        split = QSplitter()
        split.addWidget(self.tree)
        split.addWidget(detail)
        split.setSizes([475, 585])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.addWidget(label("Whole library", "title"))
        finder = QHBoxLayout()
        finder.addWidget(self.query, 1)
        finder.addWidget(self.all_folders)
        layout.addLayout(finder)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)

    def refresh(self):
        self._load()

    def _load(self, *, force=False):
        if self._queue_history and monotonic() - self._queue_history[-1][0] > COUNT_REFRESH_SECONDS:
            self._queue_history.clear()
            self._show_progress()
        query = self._active_query
        folder = self._folder_scope
        self.reads.request(
            "view",
            partial(self._read_view, query, folder, self.generation, self._branches.copy(), force),
            partial(self._present_folder, folder, force)
            if folder is not None and not query
            else partial(self._present, query, force),
            self._load_failed,
            replace=force,
        )

    def _load_failed(self):
        self.generation = None
        self._index_state = None
        self._queue_history.clear()
        self.status.setText("The local index is unavailable.")
        if self._reading:
            self.preview.setPlainText("The excerpt could not be reverified.")

    def _read_view(self, query, folder, previous, branches, force):
        if not force and self.catalogue.revision() == previous:
            return None
        state = self.catalogue.status(summary=True)
        if query and folder is not None:
            rows = self.catalogue.search(query, 100, scope=folder)
        elif query:
            rows = self.catalogue.search(query, 100)
        elif folder is not None:
            node = self.catalogue.node(folder)
            rows = [node] if node else []
        else:
            rows = self.catalogue.roots()
        changed = []
        for key, old in branches.items():
            pages, old_rows = old
            current = self._read_children(key[-1], pages)
            if current != old_rows:
                changed.append((key, old, (pages, current)))
        return state, tuple(rows), changed

    def _present_folder(self, folder, focus, result):
        if folder != self._folder_scope:
            return
        self._present("", focus, result)
        if focus and (item := self._items.get((folder,))):
            self.tree.setCurrentItem(item)
            item.setExpanded(True)

    def _present(self, query, focus, result):
        if result is None or query != self._active_query:
            return
        state, rows, branches = result
        previous = self._index_state
        self._index_state = state
        now = monotonic()
        if previous and previous.get("inventory_offline") != state.get("inventory_offline"):
            self._queue_history.clear()
        if state.get("content_pending") and not state.get("message", "").startswith(
            "Content extraction paused"
        ):
            self._queue_history.append((now, state["content_pending"]))
            while self._queue_history[0][0] < now - RATE_WINDOW_SECONDS:
                self._queue_history.popleft()
        else:
            self._queue_history.clear()
        self._show_progress()
        self.generation = state["generation"]
        if rows != self._top_rows:
            self._top_rows = rows
            root = self.tree.invisibleRootItem()
            self._sync_rows(root, rows)
            if not rows and query:
                QTreeWidgetItem(root, ["No matching files"])
        for key, old, current in branches:
            if self._branches.get(key) is old:
                self._show_children(key, current)
        if self._reading:
            self._read_source(replace=False)
        if focus:
            self.tree.setFocus()
        if (
            self._counts_requested_at is None
            or now - self._counts_requested_at >= COUNT_REFRESH_SECONDS
        ):
            self._counts_requested_at = now
            self.reads.request(
                "extraction counts",
                self.catalogue.extraction_counts,
                self._show_extraction_counts,
                lambda: None,
            )

    def _show_extraction_counts(self, counts):
        self._extraction_counts = counts
        self._show_progress()

    def _show_progress(self):
        state = self._index_state
        if state is None:
            return
        progress = state.get("message") or "Local index"
        if not progress.startswith("Content extraction paused") and (
            unfinished := state.get("inventory_pending", 0)
        ):
            roots = "root" if unfinished == 1 else "roots"
            progress = (
                f"Scanning {unfinished:,} {roots}, "
                f"{state.get('inventory_visited', 0):,} items checked"
            )
        parts = [f"{state['nodes']:,} catalogue items", progress]
        pending = state.get("content_pending", 0)
        parts.append(f"{pending:,} online files awaiting extraction")
        if offline := state.get("inventory_offline", 0):
            roots = "root" if offline == 1 else "roots"
            parts.append(f"{offline:,} {roots} offline")
        if counts := self._extraction_counts:
            prefix = "Estimated " if counts["estimated"] else ""
            failures = f"{counts['failed']:,} failed"
            if counts["estimated"] and not counts["failed"]:
                failures = "no failures in sample"
            parts.append(
                f"{prefix}{counts['extracted']:,} extracted, "
                f"{counts['skipped']:,} skipped, {failures}"
            )
        if len(self._queue_history) > 1:
            first, original = self._queue_history[0]
            last, remaining = self._queue_history[-1]
            elapsed = last - first
            if elapsed >= MIN_RATE_SECONDS and original > remaining:
                rate = (original - remaining) * 60 / elapsed
                hours, minutes = divmod(max(1, round(remaining / rate)), 60)
                eta = f"{hours} h {minutes} min" if hours else f"{minutes} min"
                parts.append(f"Net queue reduction {rate:,.0f}/min, about {eta} at this rate")
        self.status.setText(" · ".join(parts))

    def _forget(self, item):
        if key := item.data(0, TREE_KEY):
            self._items.pop(tuple(key), None)
            self._branches.pop(tuple(key), None)
        for index in range(item.childCount()):
            self._forget(item.child(index))

    def _sync_rows(self, parent, rows):
        selected = self.tree.currentItem()
        selection = tuple(selected.data(0, TREE_KEY) or ()) if selected else ()
        parent_key = tuple(parent.data(0, TREE_KEY) or ())
        with QSignalBlocker(self.tree):
            keep = {row["id"] for row in rows}
            for index in reversed(range(parent.childCount())):
                item = parent.child(index)
                if (item.data(0, ROLE) or {}).get("id") not in keep:
                    self._forget(parent.takeChild(index))
            for index, row in enumerate(rows):
                key = (*parent_key, row["id"])
                item = self._items.get(key)
                if item is None:
                    item = QTreeWidgetItem()
                    parent.insertChild(index, item)
                    item.setData(0, TREE_KEY, key)
                    self._items[key] = item
                elif (previous := parent.indexOfChild(item)) != index:
                    expanded = item.isExpanded()
                    parent.insertChild(index, parent.takeChild(previous))
                    item.setExpanded(expanded)
                if item.data(0, ROLE) != row:
                    item.setData(0, ROLE, row)
                    for column, field in enumerate(("name", "category", "state")):
                        item.setText(column, str(row.get(field) or ""))
                    if row["kind"] in FOLDERS and key not in self._branches:
                        item.setChildIndicatorPolicy(
                            QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
                        )
            if selection:
                self.tree.setCurrentItem(self._items.get(selection))
        if selection and selection not in self._items:
            self._select(None, None)
        elif self.current and (item := self._items.get(self._current_key)):
            self.current = item.data(0, ROLE)

    def _read_children(self, identity, pages):
        rows = []
        for page in range(pages):
            found = self.catalogue.children(identity, PAGE, page * PAGE)
            rows.extend(found)
            if len(found) < PAGE:
                break
        return tuple(rows)

    def _children(self, key, pages):
        view = self._view_serial

        def read():
            return self.catalogue.revision(), self._read_children(key[-1], pages)

        def loaded(result):
            revision, rows = result
            if view != self._view_serial or key not in self._items:
                return
            if self.generation is not None and revision < self.generation:
                self._children(key, pages)
                return
            self._show_children(key, (pages, rows))

        self.reads.request(
            ("folder", key),
            read,
            loaded,
            lambda: self.status.setText("Could not load this folder. Try expanding it again."),
        )

    def _show_children(self, key, branch):
        item = self._items.get(key)
        if item is None:
            return
        pages, rows = branch
        self._branches[key] = branch
        self._sync_rows(item, rows)
        if len(rows) == PAGE * pages:
            more = QTreeWidgetItem(item, ["More files…"])
            more.setData(0, ROLE, {"parent": key, "pages": pages + 1})
        item.setChildIndicatorPolicy(
            QTreeWidgetItem.ChildIndicatorPolicy.ShowIndicator
            if rows
            else QTreeWidgetItem.ChildIndicatorPolicy.DontShowIndicator
        )

    def _expand(self, item):
        if (key := tuple(item.data(0, TREE_KEY) or ())) and key not in self._branches:
            self._children(key, 1)

    def _activate(self, item, _column):
        row = item.data(0, ROLE)
        if not row:
            return
        if "pages" in row:
            self._children(tuple(row["parent"]), row["pages"])
        elif row["kind"] in FOLDERS:
            item.setExpanded(not item.isExpanded())
        else:
            self.source_requested.emit(row)

    def _select(self, item, _previous):
        row = item.data(0, ROLE) if item else None
        if row and row.get("id"):
            self.inspect(row, tuple(item.data(0, TREE_KEY)))
        elif item is None:
            self.current = None
            self._reading = False
            self._current_key = None
            self.title.setText("Select a file or folder")
            self.location.clear()
            self.preview.clear()

    def inspect(self, row, key=None):
        self.current = row
        self._current_key = key
        self._reading = False
        self.title.setText(str(row["name"]))
        self.location.setText(str(row.get("path") or ""))
        self.preview.setPlainText(
            "Press Enter on a file to open a verified excerpt."
            if row["kind"] in FOLDERS
            else "Press Enter to verify and read this source."
        )
        self.selection_changed.emit(row)

    def open_folder(self, row):
        self.query.clear()
        self._set_view("", row["id"])

    def _return_to_roots(self):
        self.query.clear()
        self._set_view("", None)

    def open_source(self, row):
        key = self._current_key
        if not key or key[-1] != row["id"]:
            key = next((found for found in self._items if found[-1] == row["id"]), None)
        with QSignalBlocker(self.tree):
            self.tree.setCurrentItem(self._items.get(key))
        self.inspect(row, key)
        self._reading = True
        self._read_source()

    def _read_source(self, *, replace=True):
        row = self.current
        self.reads.request(
            "source",
            partial(self.catalogue.read_evidence, row["id"], 8000, row.get("match_start") or 1),
            partial(self._show_evidence, row["id"]),
            partial(self._show_evidence, row["id"], None),
            replace=replace,
        )

    def _show_evidence(self, identity, result):
        if not self._reading or not self.current or self.current["id"] != identity:
            return
        row = self.current
        if not result:
            self.preview.setPlainText(
                "No current text is available for this source. " + str(row.get("state") or "")
            )
            return
        locator = result.get("locator") or {}
        detail = ""
        if locator.get("kind") == "lines":
            detail = f" · Lines {locator['start']}–{locator['end']}"
        elif locator.get("kind") == "page":
            detail = f" · Page {locator['page']}"
        self.location.setText(str(result.get("path") or row.get("path") or "") + detail)
        self.preview.setPlainText(result["text"])

    def _search(self):
        self._set_view(self.query.text().strip(), self._folder_scope)

    def _set_view(self, query, folder):
        if (query, folder) != (self._active_query, self._folder_scope):
            self._active_query = query
            self._folder_scope = folder
            self.all_folders.setVisible(folder is not None)
            self._view_serial += 1
            self._items.clear()
            self._branches.clear()
            self._top_rows = None
            self.tree.clear()
            self._select(None, None)
        self._load(force=True)
