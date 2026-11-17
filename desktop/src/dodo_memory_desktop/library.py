"""Native note reader, vault catalogue and evidence inspector."""

import base64
import binascii
import hashlib
import math
import re
from collections import Counter
from pathlib import Path, PurePosixPath

from PySide6.QtCore import (
    QByteArray,
    QEasingCurve,
    QPoint,
    QPointF,
    QPropertyAnimation,
    QSignalBlocker,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import (
    QDesktopServices,
    QImage,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
    QShortcut,
    QTextCursor,
    QTextDocument,
)
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QTextBrowser,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .project_library import ARCHIVE, ProjectLibrary
from .ui import button

USER_DATA = Qt.ItemDataRole.UserRole
DRAWER_BREAKPOINT = 980
VAULT_BREAKPOINT = 720
WIKI = re.compile(r"(?<!\\)\[\[([^\[\]\r\n]{1,2048})\]\]")
MARKDOWN_TOKEN = re.compile(
    r"(?<!`)(`+)(?!`).*?(?<!`)\1(?!`)|\\.|\[\[[^\[\]\r\n]{1,2048}\]\]|"
    r"\[[^\]\n]*\]\([^\n)]*\)"
)


def markdown_links(text: str) -> str:
    def transform(match):
        wiki = WIKI.fullmatch(match[0])
        if wiki is None:
            return match[0]
        target, _, alias = wiki[1].partition("|")
        target = target.strip()
        if not target:
            return match[0]
        encoded = base64.urlsafe_b64encode(target.encode()).decode().rstrip("=")
        caption = re.sub(r"([\\\[\]*_`<>])", r"\\\1", alias.strip() or target)
        return f"[{caption}](dodo-wiki:{encoded})"

    fence = None
    output = []
    for line in text.splitlines(keepends=True):
        mark = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
        if mark and fence is None:
            fence = mark[1]
        elif fence and mark and mark[1].startswith(fence) and not mark[2].strip():
            fence = None
        elif fence is None and not line.startswith(("    ", "\t")):
            line = MARKDOWN_TOKEN.sub(transform, line)
        output.append(line)
    return "".join(output)


def _plain_label(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setWordWrap(True)
    return label


def _preview(name: str) -> QPlainTextEdit:
    preview = QPlainTextEdit()
    preview.setReadOnly(True)
    preview.setAccessibleName(name)
    font = preview.font()
    font.setPixelSize(17)
    preview.setFont(font)
    return preview


class MarkdownReader(QTextBrowser):
    """Render Markdown with native Qt and inert external references."""

    def __init__(self):
        super().__init__()
        self.setOpenLinks(False)
        self.setOpenExternalLinks(False)
        self.setObjectName("noteReader")
        self.setAccessibleName("Note")
        font = self.font()
        font.setPixelSize(17)
        self.document().setDefaultFont(font)
        self.document().setDocumentMargin(26)

    def loadResource(self, resource_type, _url):
        if resource_type == QTextDocument.ResourceType.ImageResource:
            return QImage()
        return QByteArray()

    def set_markdown(self, text: str) -> None:
        features = QTextDocument.MarkdownFeature
        self.document().setMarkdown(
            markdown_links(text), features.MarkdownDialectGitHub | features.MarkdownNoHTML
        )
        cursor = QTextCursor(self.document())
        self.setTextCursor(cursor)
        self.verticalScrollBar().setValue(0)

    def go_to_heading(self, heading: str) -> None:
        def normalised(text: str) -> str:
            return " ".join(re.findall(r"\w+", text.casefold()))

        target = normalised(heading)
        if not target:
            return
        block = self.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() and normalised(block.text()) == target:
                cursor = QTextCursor(block)
                self.setTextCursor(cursor)
                self.ensureCursorVisible()
                return
            block = block.next()


class RelationshipGraph(QWidget):
    selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumHeight(162)
        self.setMaximumHeight(205)
        self.setAccessibleName("Relationship graph. The list contains the same links.")
        self.setObjectName("relationshipGraph")
        self._neighbours: list[str] = []
        self._kinds: list[str] = []
        self._selected = -1

    def set_edges(self, neighbours: list[str], kinds: list[str]) -> None:
        self._neighbours = neighbours
        self._kinds = kinds
        self.select(-1)

    def select(self, index: int) -> None:
        self._selected = index
        self.update()

    def _layout(self) -> tuple[QPointF, list[QPointF], list[QPainterPath]]:
        centre = QPointF(self.width() / 2, self.height() / 2)
        nodes = sorted(set(self._neighbours), key=str.casefold)
        positions = {}
        for index, path in enumerate(nodes):
            angle = 2 * math.pi * index / len(nodes) - math.pi / 2
            positions[path] = centre + QPointF(
                self.width() * 0.36 * math.cos(angle),
                self.height() * 0.37 * math.sin(angle),
            )
        totals = Counter(self._neighbours)
        lanes: Counter[str] = Counter()
        edges = []
        for path in self._neighbours:
            point = positions[path]
            vector = point - centre
            normal = QPointF(-vector.y(), vector.x()) / max(1, math.hypot(vector.x(), vector.y()))
            lane = lanes[path] - (totals[path] - 1) / 2
            route = QPainterPath(centre)
            route.quadTo(centre + vector * 0.5 + normal * (lane * 32), point)
            edges.append(route)
            lanes[path] += 1
        return centre, list(positions.values()), edges

    def paintEvent(self, _event):
        if not self._neighbours:
            return
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        colour = self.palette().color(QPalette.ColorRole.Text)
        selected = self.palette().color(QPalette.ColorRole.Highlight)
        muted = self.palette().color(QPalette.ColorRole.Mid)
        centre, nodes, edges = self._layout()
        for index, edge in enumerate(edges):
            if index == self._selected:
                continue
            style = (
                Qt.PenStyle.DashLine if self._kinds[index] == "evidence" else Qt.PenStyle.SolidLine
            )
            painter.setPen(QPen(muted, 1.5, style))
            painter.drawPath(edge)
        if 0 <= self._selected < len(edges):
            painter.setPen(QPen(selected, 2.5))
            painter.drawPath(edges[self._selected])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(colour)
        painter.drawEllipse(centre, 8.0, 8.0)
        painter.setBrush(muted)
        for point in nodes:
            painter.drawEllipse(point, 4.5, 4.5)
        if 0 <= self._selected < len(edges):
            painter.setBrush(selected)
            painter.drawEllipse(edges[self._selected].currentPosition(), 6.0, 6.0)

    def mousePressEvent(self, event):
        if event.button() != Qt.MouseButton.LeftButton or not self._neighbours:
            super().mousePressEvent(event)
            return
        centre, _nodes, edges = self._layout()
        offset = event.position() - centre
        if math.hypot(offset.x(), offset.y()) < 12:
            return
        hit = QPainterPathStroker()
        hit.setWidth(14)
        for index, edge in enumerate(edges):
            if hit.createStroke(edge).contains(event.position()):
                self.selected.emit(index)
                event.accept()
                return


class Library(QWidget):
    path_changed = Signal(object)
    note_requested = Signal(str)
    note_heading_requested = Signal(str, str)
    evidence_requested = Signal(object)
    source_requested = Signal(object)
    relations_requested = Signal(str)
    history_requested = Signal(str)
    catalog_more_requested = Signal()
    link_requested = Signal(str, str, bool)
    edit_requested = Signal(str)
    new_note_requested = Signal()
    ordinary_history_requested = Signal(str)
    ordinary_revision_requested = Signal(str, int)

    def __init__(self, project_archive: Path = ARCHIVE):
        super().__init__()
        self.setObjectName("library")
        self.projects = ProjectLibrary(project_archive)
        self.projects.note_requested.connect(self.evidence_requested)
        self._pages: dict[str, QWidget] = {}
        self._edges: list[dict] = []
        self._links_truncated = False
        self._history: list[dict] = []
        self._motion_reduced = True
        self._drawer = False

        self.notes_button = button("Notes", role="nav")
        self.project_folders_button = button("Project folders", role="nav")
        views = QButtonGroup(self)
        for control in (self.notes_button, self.project_folders_button):
            control.setCheckable(True)
            views.addButton(control)
        self.notes_button.setChecked(True)
        self.notes_button.clicked.connect(self.show_notes)
        self.project_folders_button.clicked.connect(self.show_projects)
        navigation = QHBoxLayout()
        navigation.setContentsMargins(10, 8, 10, 0)
        navigation.addWidget(self.notes_button)
        navigation.addWidget(self.project_folders_button)
        navigation.addStretch()

        self.vault_button = QPushButton("Vault")
        self.vault_button.setCheckable(True)
        self.vault_button.setChecked(True)
        self.vault_button.toggled.connect(self._show_vault)
        self.inspector_button = QPushButton("Inspector")
        self.inspector_button.setCheckable(True)
        self.inspector_button.setChecked(True)
        self.inspector_button.toggled.connect(self._toggle_inspector)

        actions = QHBoxLayout()
        actions.addWidget(self.vault_button)
        actions.addStretch()
        new_note = QPushButton("New note")
        new_note.clicked.connect(self.new_note_requested)
        actions.addWidget(new_note)
        self.edit_button = QPushButton("Edit note")
        self.edit_button.clicked.connect(self._edit_current)
        actions.addWidget(self.edit_button)
        actions.addWidget(self.inspector_button)
        actions.setContentsMargins(10, 8, 10, 0)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.vault = self._build_vault()
        self.reader = self._build_reader()
        self.inspector = self._build_inspector()
        self.splitter.addWidget(self.vault)
        self.splitter.addWidget(self.reader)
        self.splitter.addWidget(self.inspector)
        self.splitter.setStretchFactor(1, 1)
        self.splitter.setSizes([210, 620, 330])
        self._drawer_animation = QPropertyAnimation(self.inspector, b"pos", self)
        self._drawer_animation.setDuration(145)
        self._drawer_animation.setEasingCurve(QEasingCurve.Type.OutCubic)

        self.notes_workspace = QWidget()
        notes_layout = QVBoxLayout(self.notes_workspace)
        notes_layout.setContentsMargins(0, 0, 0, 0)
        notes_layout.setSpacing(12)
        notes_layout.addLayout(actions)
        notes_layout.addWidget(self.splitter, 1)
        self.workspace = QStackedWidget()
        self.workspace.addWidget(self.notes_workspace)
        self.workspace.addWidget(self.projects)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addLayout(navigation)
        layout.addWidget(self.workspace, 1)

        close_drawer = QShortcut(QKeySequence(Qt.Key.Key_Escape), self.inspector)
        close_drawer.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        close_drawer.activated.connect(self._close_drawer)
        self._tab_changed()

    def _build_vault(self) -> QWidget:
        vault = QWidget()
        vault.setObjectName("vaultPane")
        vault.setMinimumWidth(150)
        self.tree = QTreeWidget()
        self.tree.setObjectName("vaultTree")
        self.tree.setAccessibleName("Vault notes")
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(12)
        self.tree.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.tree.itemActivated.connect(self._open_catalog_item)
        self.more_button = QPushButton("More notes")
        self.more_button.setObjectName("moreNotes")
        self.more_button.hide()
        self.more_button.clicked.connect(self.catalog_more_requested)
        layout = QVBoxLayout(vault)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.more_button)
        return vault

    def projects_visible(self) -> bool:
        return self.workspace.currentWidget() is self.projects

    def show_projects(self) -> None:
        self.project_folders_button.setChecked(True)
        if not self.projects_visible():
            self.workspace.setCurrentWidget(self.projects)
            self.projects.reload()

    def show_notes(self) -> None:
        self.notes_button.setChecked(True)
        self.workspace.setCurrentWidget(self.notes_workspace)

    def _build_reader(self) -> QWidget:
        self.tabs = QTabWidget()
        self.tabs.setObjectName("noteTabs")
        self.tabs.setDocumentMode(True)
        self.tabs.setTabsClosable(True)
        self.tabs.tabBar().setElideMode(Qt.TextElideMode.ElideRight)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.tabs.tabCloseRequested.connect(self._close_tab)
        empty = QWidget()
        empty.setObjectName("libraryEmpty")
        prompt = _plain_label("Open a note from the vault or explore your Desktop projects.")
        prompt.setAlignment(Qt.AlignmentFlag.AlignCenter)
        browse = button("Browse project folders", role="primary")
        browse.clicked.connect(self.show_projects)
        empty_layout = QVBoxLayout(empty)
        empty_layout.addStretch()
        empty_layout.addWidget(prompt)
        empty_layout.addWidget(browse, 0, Qt.AlignmentFlag.AlignHCenter)
        empty_layout.addStretch()
        stack = QStackedWidget()
        stack.addWidget(empty)
        stack.addWidget(self.tabs)
        return stack

    def _build_inspector(self) -> QWidget:
        inspector = QWidget()
        inspector.setObjectName("inspectorPane")
        inspector.setMinimumWidth(276)
        self.inspector_tabs = QTabWidget()
        self.inspector_tabs.setObjectName("inspectorTabs")
        self.inspector_tabs.setDocumentMode(True)
        self.inspector_tabs.addTab(self._build_relationships(), "Links")
        self.inspector_tabs.addTab(self._build_source(), "Source")
        self.inspector_tabs.addTab(self._build_history(), "History")
        layout = QVBoxLayout(inspector)
        layout.setContentsMargins(9, 0, 0, 0)
        layout.addWidget(self.inspector_tabs)
        return inspector

    def _build_relationships(self) -> QWidget:
        pane = QWidget()
        self.graph = RelationshipGraph()
        self.graph.selected.connect(self._select_graph_edge)
        self.relationships = QListWidget()
        self.relationships.setObjectName("relationshipList")
        self.relationships.setAccessibleName("Note relationships")
        self.relationships.setWordWrap(True)
        self.relationships.currentRowChanged.connect(self._describe_edge)
        self.relationships.itemActivated.connect(self._open_edge_evidence)
        self.edge_detail = _plain_label("Open a note to see its links.")
        self.edge_detail.setObjectName("edgeDetail")
        self.edge_open = QPushButton("Open evidence")
        self.edge_open.clicked.connect(self._open_edge_evidence)
        self.link_open = QPushButton("Open linked note")
        self.link_open.clicked.connect(self._open_linked_note)
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.graph)
        layout.addWidget(self.relationships, 1)
        layout.addWidget(self.edge_detail)
        layout.addWidget(self.link_open)
        layout.addWidget(self.edge_open)
        return pane

    def _build_source(self) -> QWidget:
        pane = QWidget()
        self.source_status = _plain_label()
        self.source_status.setObjectName("sourceStatus")
        self.source_status.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.source_preview = _preview("Verified source text")
        self.source_preview.setObjectName("sourcePreview")
        self.open_original = QPushButton("Open original")
        self.open_original.clicked.connect(self._open_original)
        self.locate_original = QPushButton()
        self.locate_original.hide()
        self.locate_original.clicked.connect(self._choose_location)
        self._original_pages = 0
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.source_status)
        layout.addWidget(self.source_preview, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.open_original)
        actions.addWidget(self.locate_original)
        layout.addLayout(actions)
        return pane

    def _build_history(self) -> QWidget:
        pane = QWidget()
        self.history_status = _plain_label()
        self.history_list = QListWidget()
        self.history_list.setAccessibleName("Note revisions")
        self.history_list.currentRowChanged.connect(self._show_revision)
        self.history_preview = _preview("Revision text")
        self.history_preview.setObjectName("revisionPreview")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(self.history_status)
        layout.addWidget(self.history_list, 1)
        layout.addWidget(self.history_preview, 2)
        return pane

    @property
    def current_path(self) -> str | None:
        page = self.tabs.currentWidget()
        return page.property("notePath") if page else None

    def open_paths(self) -> list[str]:
        return list(self._pages)

    def set_catalog(self, items: list[dict], has_more: bool = False) -> None:
        selected = self.tree.currentItem()
        selected_path = selected.data(0, USER_DATA) if selected else self.current_path
        scroll = self.tree.verticalScrollBar().value()
        self.tree.clear()
        folders: dict[tuple[str, ...], QTreeWidgetItem] = {}
        root = self.tree.invisibleRootItem()
        for item in sorted(items, key=lambda record: record["path"].casefold()):
            path = item["path"]
            parts = PurePosixPath(path).parts
            parent = root
            for depth, part in enumerate(parts[:-1], start=1):
                prefix = parts[:depth]
                if prefix not in folders:
                    folders[prefix] = QTreeWidgetItem(parent, [part])
                parent = folders[prefix]
            leaf = QTreeWidgetItem(parent, [item.get("title") or PurePosixPath(path).stem])
            leaf.setData(0, USER_DATA, path)
            leaf.setToolTip(0, path)
            if path == selected_path:
                self.tree.setCurrentItem(leaf)
        self.tree.expandToDepth(1)
        self.tree.verticalScrollBar().setValue(scroll)
        self.more_button.setVisible(has_more)

    def _open_catalog_item(self, item: QTreeWidgetItem, _column: int) -> None:
        path = item.data(0, USER_DATA)
        if path:
            self.note_requested.emit(path)

    def open_note(self, note: dict, heading: str | None = None, *, activate: bool = True) -> None:
        path = note["path"]
        text = note["text"]
        page = self._pages.get(path)
        if page is None and not activate:
            return
        if activate:
            self.show_notes()
        previous = self.current_path
        with QSignalBlocker(self.tabs):
            if page is None:
                page = QWidget()
                page.setProperty("notePath", path)
                reader = MarkdownReader()
                reader.setMaximumWidth(850)
                reader.anchorClicked.connect(lambda url: self._note_link(path, reader, url))
                layout = QHBoxLayout(page)
                layout.setContentsMargins(0, 0, 0, 0)
                layout.addStretch()
                layout.addWidget(reader, 1)
                layout.addStretch()
                self._pages[path] = page
                title = note.get("title") or PurePosixPath(path).stem
                self.tabs.addTab(page, title)
                self.tabs.setTabToolTip(self.tabs.indexOf(page), path)
            else:
                reader = page.findChild(MarkdownReader)
            signature = hashlib.sha256(text.encode()).digest()
            if page.property("noteSignature") != signature:
                reader.set_markdown(text)
                page.setProperty("noteSignature", signature)
            page.setProperty("knowledgeKey", note.get("knowledge_key"))
            page.setProperty("sourceId", note.get("source"))
            page.setProperty("editable", bool(note.get("editable")))
            if activate:
                self.tabs.setCurrentWidget(page)
        self.reader.setCurrentWidget(self.tabs)
        if previous != self.current_path:
            self._tab_changed()
        if heading:
            QTimer.singleShot(0, reader, lambda: reader.go_to_heading(heading))

    def _edit_current(self):
        if self.current_path:
            self.edit_requested.emit(self.current_path)

    def _note_link(self, path: str, _reader: MarkdownReader, url: QUrl) -> None:
        if url.scheme() == "dodo-wiki":
            token = url.path()
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,10923}", token):
                return
            try:
                target = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4)).decode()
            except (binascii.Error, UnicodeError):
                return
            self.link_requested.emit(path, target, True)
            return
        if url.scheme() in {"http", "https"} and url.host() and not url.userInfo():
            QDesktopServices.openUrl(url)
            return
        if url.scheme() or url.authority() or url.query():
            return
        self.link_requested.emit(path, bytes(url.toEncoded()).decode("ascii"), False)

    def _close_tab(self, index: int) -> None:
        page = self.tabs.widget(index)
        if page is not None:
            self.close_note(page.property("notePath"))

    def close_note(self, path: str) -> None:
        page = self._pages.pop(path, None)
        if page is None:
            return
        previous = self.current_path
        page.findChild(MarkdownReader).clear()
        with QSignalBlocker(self.tabs):
            self.tabs.removeTab(self.tabs.indexOf(page))
        # removeTab leaves the page parented to the stack. Detach it before Qt's
        # deferred delete so closing the Library cannot tear down the queued page.
        page.setParent(None)
        page.deleteLater()
        if not self._pages:
            self.reader.setCurrentIndex(0)
        if previous != self.current_path and not self.tabs.signalsBlocked():
            self._tab_changed()

    def clear_content(self) -> None:
        self.set_catalog([])
        with QSignalBlocker(self.tabs):
            for path in self.open_paths():
                self.close_note(path)
        self._tab_changed()

    def _tab_changed(self, _index: int = -1) -> None:
        page = self.tabs.currentWidget()
        path = self.current_path
        self._clear_source()
        self._clear_relationships()
        self._clear_history()
        self.edit_button.setEnabled(bool(page and page.property("editable")))
        self.open_original.setEnabled(
            bool(page and str(page.property("sourceId")).startswith("sha256:"))
        )
        self.path_changed.emit(path)
        if path:
            self.edge_detail.setText("Loading links…")
            self.relations_requested.emit(path)
        if page and page.property("knowledgeKey"):
            self.history_status.setText("Loading revisions…")
            self.history_requested.emit(page.property("knowledgeKey"))
        elif page and page.property("editable"):
            self.history_status.setText("Loading revisions…")
            self.ordinary_history_requested.emit(path)

    def _clear_source(self) -> None:
        self._verified_source = None
        self._original_pages = 0
        self.locate_original.hide()
        self.source_preview.clear()
        self.source_status.setText("Choose a verified source.")
        self.source_status.setToolTip("")

    def _open_original(self) -> None:
        if self.current_path:
            self.source_requested.emit({"path": self.current_path})

    def _choose_location(self) -> None:
        path = self.current_path
        if not path:
            return
        if self._original_pages:
            page, ok = QInputDialog.getInt(
                self,
                "Open page",
                "Page number",
                1,
                1,
                self._original_pages,
            )
            if ok:
                self.source_requested.emit({"path": path, "page": page})
            return
        value, ok = QInputDialog.getText(
            self, "Open lines", "Line number or range, for example 8–13"
        )
        if not ok:
            return
        match = re.fullmatch(r"\s*([0-9]{1,7})(?:\s*[-–]\s*([0-9]{1,7}))?\s*", value)
        if match is None:
            self.source_status.setText("Enter a line number or range.")
            return
        start = int(match[1])
        end = int(match[2]) if match[2] else start
        if 1 <= start <= end:
            self.source_requested.emit({"path": path, "lines": [start, end]})
        else:
            self.source_status.setText("Use a positive range with the first line before the last.")

    def set_source(self, source: dict | None, *, reveal: bool = True) -> None:
        if not reveal and source is not None and source == self._verified_source:
            return
        self._clear_source()
        digest = source.get("sha256", "") if source else ""
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            self.source_status.setText("Original unavailable.")
        else:
            self._verified_source = source
            location = source.get("locator", {})
            if source.get("page"):
                label = f"Page {source['page']}"
            elif location.get("kind") == "lines":
                label = f"Lines {location['start']}–{location['end']}"
            elif location.get("kind") == "messages":
                label = f"Messages {location['start']}–{location['end']}"
            else:
                label = "Verified original"
            preview_text = source.get("preview_text")
            if isinstance(preview_text, str):
                self.source_preview.setPlainText(preview_text)
                if source.get("preview_truncated"):
                    label += "\nExcerpt shortened for preview."
            else:
                label += "\n" + (source.get("preview_reason") or "No text preview is available.")
            self.source_status.setText(label)
            self.source_status.setToolTip(f"SHA-256 {digest}")
            self._original_pages = source.get("pages", 0)
            if self._original_pages or isinstance(preview_text, str):
                self.locate_original.setText("Open page" if self._original_pages else "Open lines")
                self.locate_original.show()
        if reveal:
            self.inspector_tabs.setCurrentIndex(1)
            self.inspector_button.setChecked(True)

    def _clear_relationships(self) -> None:
        self._edges.clear()
        self._links_truncated = False
        self.relationships.clear()
        self.graph.set_edges([], [])
        self.graph.hide()
        self.edge_detail.setText("Open a note to see its links.")
        self.edge_open.hide()
        self.link_open.hide()

    def set_related(self, path: str, result: dict) -> None:
        if path != self.current_path:
            return
        index = self.relationships.currentRow()
        previous = self._edges[index] if 0 <= index < len(self._edges) else None
        self._clear_relationships()
        self._links_truncated = bool(result.get("truncated"))
        titles = {node["path"]: node["title"] for node in result.get("nodes", [])}
        neighbours = []
        kinds = []
        selected = 0
        for edge in result.get("edges", []):
            if edge["source"] != path and edge["target"] != path:
                continue
            if edge == previous:
                selected = len(self._edges)
            self._edges.append(edge)
            inward = edge["target"] == path
            neighbour = edge["source"] if inward else edge["target"]
            title = titles.get(neighbour) or PurePosixPath(neighbour).stem
            kind = edge["kind"]
            if kind == "evidence":
                caption = f"Cited by {title}" if inward else f"Cites {title}"
            else:
                caption = f"Linked from {title}" if inward else f"Links to {title}"
            item = QListWidgetItem(caption)
            item.setToolTip(neighbour)
            self.relationships.addItem(item)
            neighbours.append(neighbour)
            kinds.append(kind)
        self.graph.set_edges(neighbours, kinds)
        self.graph.setVisible(bool(neighbours))
        if self._edges:
            self.relationships.setCurrentRow(selected)
        else:
            empty = (
                "No links found within this view." if self._links_truncated else "No visible links."
            )
            self.edge_detail.setText(empty)

    def _select_graph_edge(self, index: int) -> None:
        self.relationships.setCurrentRow(index)
        self.relationships.scrollToItem(self.relationships.currentItem())

    def _describe_edge(self, index: int) -> None:
        self.graph.select(index)
        self.edge_open.setVisible(0 <= index < len(self._edges))
        self.link_open.setVisible(0 <= index < len(self._edges))
        if 0 <= index < len(self._edges):
            edge = self._edges[index]
            evidence = edge["evidence"]
            heading = (evidence.get("heading") or "").lstrip("# ").strip()
            label = edge["label"]
            lines = [label]
            if heading:
                lines.append(heading)
            if self._links_truncated:
                lines.append("More links may be available.")
            self.edge_detail.setText("\n".join(lines))

    def _open_edge_evidence(self, _item=None) -> None:
        index = self.relationships.currentRow()
        if not 0 <= index < len(self._edges):
            return
        self.evidence_requested.emit(self._edges[index]["evidence"])

    def _open_linked_note(self):
        index = self.relationships.currentRow()
        if 0 <= index < len(self._edges):
            edge = self._edges[index]
            side = "source" if edge["target"] == self.current_path else "target"
            self.evidence_requested.emit(
                {"path": edge[side], "heading": edge.get(f"{side}_heading")}
            )

    def _clear_history(self) -> None:
        self._history.clear()
        self.history_list.clear()
        self.history_preview.clear()
        self.history_status.setText("Note revisions appear here.")

    def set_history(self, entry_key: str, result: dict) -> None:
        page = self.tabs.currentWidget()
        if page is None or page.property("knowledgeKey") != entry_key:
            return
        self._clear_history()
        self._history = list(result.get("records", []))
        for record in self._history:
            self.history_list.addItem(f"Revision {record['revision']}")
        if self._history:
            self.history_status.setText(
                "Recent accepted revisions" if result.get("truncated") else ""
            )
            self.history_list.setCurrentRow(0)
        else:
            self.history_status.setText("No visible revisions.")

    def _show_revision(self, index: int) -> None:
        self.history_preview.clear()
        if not 0 <= index < len(self._history):
            return
        row = self._history[index]
        if "text" in row:
            self.history_preview.setPlainText(row["text"])
        elif self.current_path:
            self.ordinary_revision_requested.emit(self.current_path, row["revision"])

    def set_ordinary_history(self, path, result):
        if path != self.current_path:
            return
        self._clear_history()
        self._history = result["records"]
        self.history_status.setText("" if self._history else "No saved revisions.")
        for row in self._history:
            self.history_list.addItem(f"Revision {row['revision']}")
        if self._history:
            self.history_list.setCurrentRow(0)

    def set_ordinary_revision(self, path, revision, result):
        index = self.history_list.currentRow()
        if (
            path == self.current_path
            and 0 <= index < len(self._history)
            and self._history[index]["revision"] == revision
        ):
            self.history_preview.setPlainText(result["text"])

    def set_motion_reduced(self, reduced: bool) -> None:
        self._motion_reduced = reduced
        if reduced:
            self._drawer_animation.stop()
            if self._drawer:
                self._place_drawer()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        narrow = self.width() < DRAWER_BREAKPOINT
        if narrow != self._drawer:
            self._drawer_animation.stop()
            self._drawer = narrow
            if narrow:
                self.inspector.setParent(self.notes_workspace)
                self.inspector_button.setChecked(False)
            else:
                self.splitter.addWidget(self.inspector)
                self.splitter.setSizes([210, 620, 330])
                self.inspector_button.setChecked(True)
        self._drawer_animation.stop()
        if self._drawer:
            self._place_drawer()
        self._show_vault(self.vault_button.isChecked())

    def _show_vault(self, visible: bool) -> None:
        self.vault.setVisible(visible)
        if visible and self.width() < VAULT_BREAKPOINT:
            self.splitter.setSizes([165, max(220, self.width() - 165)])

    def _place_drawer(self) -> QPoint:
        bounds = self.splitter.geometry()
        width = min(bounds.width(), 350)
        x = bounds.x() + bounds.width() - width
        y = bounds.y()
        self.inspector.setGeometry(x, y, width, bounds.height())
        self.inspector.raise_()
        return QPoint(x, y)

    def _toggle_inspector(self, visible: bool) -> None:
        self._drawer_animation.stop()
        self.inspector.setVisible(visible)
        if not visible or not self._drawer:
            return
        end = self._place_drawer()
        self.inspector_tabs.tabBar().setFocus()
        if not self._motion_reduced:
            self._drawer_animation.setStartValue(end + QPoint(18, 0))
            self._drawer_animation.setEndValue(end)
            self._drawer_animation.start()

    def _close_drawer(self) -> None:
        if self._drawer:
            self.inspector_button.setChecked(False)
            self.inspector_button.setFocus()
