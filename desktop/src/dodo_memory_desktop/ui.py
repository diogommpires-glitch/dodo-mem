"""The archive palette, Home search, and the proposal review pane."""

import ctypes
import re
from pathlib import Path

from PySide6.QtCore import QByteArray, QPropertyAnimation, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QIcon, QPainter, QPalette, QPixmap
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsOpacityEffect,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

ASSETS = Path(__file__).parent / "assets"
ROLE = Qt.ItemDataRole.UserRole
THEME_TOKENS = {
    "canvas": "#09070c",
    "rail": "#100d14",
    "pane": "#18131f",
    "raised": "#221b2c",
    "selection": "#382b48",
    "text": "#f2ecf8",
    "muted": "#a99ab9",
    "accent": "#c5a7ff",
}
THEME = """
QWidget { background: #09070c; color: #f2ecf8; font: 15px "Segoe UI"; }
QFrame#navigation { background: #100d14; border-right: 1px solid #3a2e49; }
QFrame#navigation QLabel { background: transparent; }
QLabel[role="brand"] { font-size: 19px; font-weight: 650; color: #f2ecf8; }
QLabel[role="title"] { font: 38px "Georgia"; color: #f2ecf8; }
QLabel[role="section"] { font-size: 17px; font-weight: 600; }
QLabel[role="muted"] { color: #a99ab9; font-size: 14px; }
QLabel#status { color: #c5a7ff; font-size: 14px; }
QPushButton, QToolButton, QComboBox { background: #18131f; border: 1px solid #514064;
    border-radius: 6px; min-height: 34px; padding: 3px 12px; text-align: left; }
QPushButton:hover, QToolButton:hover, QComboBox:hover { background: #221b2c; }
QPushButton:disabled { color: #9b8bad; border-color: #3a2e49; }
QPushButton:focus, QToolButton:focus, QComboBox:focus, QLineEdit:focus,
QListWidget:focus, QTreeWidget:focus, QTextBrowser:focus, QPlainTextEdit:focus,
QGraphicsView:focus {
    border: 1px solid #c5a7ff; }
QPushButton[role="primary"] { background: #b58aff; color: #09070c; border-color: #b58aff;
    font-weight: 650; }
QPushButton[role="primary"]:hover { background: #c5a7ff; }
QPushButton[role="primary"]:focus { border: 2px solid #f2ecf8; }
QPushButton[role="primary"]:disabled { background: #18131f; color: #9b8bad;
    border: 1px solid #3a2e49; }
QPushButton[role="nav"] { border: 1px solid transparent; background: transparent; }
QPushButton[role="nav"]:hover { background: #221b2c; }
QPushButton[role="nav"]:checked { background: #382b48; color: #e4d1ff; }
QPushButton[role="nav"]:focus { border: 1px solid #c5a7ff; }
QLineEdit { background: #18131f; border: 1px solid #645078; border-radius: 8px;
    min-height: 34px; padding: 5px 12px; selection-background-color: #493663; }
QLineEdit#search { font-size: 19px; min-height: 50px; }
QListWidget, QTreeWidget, QPlainTextEdit, QTextBrowser, QGraphicsView { background: #18131f;
    border: 1px solid #3a2e49; border-radius: 5px; }
QListWidget::item, QTreeWidget::item { padding: 6px 10px; min-height: 27px; }
QListWidget::item:selected, QTreeWidget::item:selected { background: #382b48; color: #f2ecf8; }
QListWidget::item:hover, QTreeWidget::item:hover { background: #221b2c; }
QListWidget#homePins, QListWidget#homeRecent { background: transparent; border: 1px solid transparent; }
QListWidget#homePins::item, QListWidget#homeRecent::item { border-bottom: 1px solid #3a2e49; }
QListWidget#homePins:focus, QListWidget#homeRecent:focus { border-color: #c5a7ff; }
QPlainTextEdit, QTextBrowser { font-size: 17px; padding: 8px; }
QSplitter::handle { background: #3a2e49; }
QTabWidget::pane { border: 1px solid #3a2e49; }
QTabBar::tab { background: #18131f; padding: 8px 14px; min-height: 24px;
    border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #e4d1ff; border-color: #b58aff; }
QScrollBar:vertical { background: #100d14; width: 11px; }
QScrollBar:horizontal { background: #100d14; height: 11px; }
QScrollBar::handle { background: #645078; min-height: 30px; min-width: 30px; border-radius: 4px; }
QScrollBar::add-line, QScrollBar::sub-line { width: 0; height: 0; }
QScrollBar::add-page, QScrollBar::sub-page { background: transparent; }
QToolTip { background: #221b2c; color: #f2ecf8; border: 1px solid #78648f; font-size: 14px; }
"""


def label(text: str, role: str = "") -> QLabel:
    widget = QLabel(text)
    widget.setTextFormat(Qt.TextFormat.PlainText)
    if role:
        widget.setProperty("role", role)
    return widget


def button(text: str, icon: str = "", role: str = "") -> QPushButton:
    widget = QPushButton(text)
    widget.setAccessibleName(text)
    if role:
        widget.setProperty("role", role)
    if icon:
        widget.setIcon(archive_icon(icon))
    return widget


def archive_icon(name: str) -> QIcon:
    path = ASSETS / "icons" / f"{name}.svg"
    if not path.is_file():
        return QIcon()
    renderer = QSvgRenderer(QByteArray(path.read_bytes().replace(b"currentColor", b"#c5a7ff")))
    image = QPixmap(40, 40)
    image.fill(Qt.GlobalColor.transparent)
    painter = QPainter(image)
    renderer.render(painter, QRectF(0, 0, 40, 40))
    painter.end()
    return QIcon(image)


def theme_stylesheet(tokens: dict | None = None) -> str:
    if tokens is None:
        return THEME
    mapping = {colour: tokens[key] for key, colour in THEME_TOKENS.items()}
    mapping.update({"#b58aff": tokens["accent"], "#e4d1ff": tokens["accent"]})
    return re.sub(r"#[0-9a-f]{6}\b", lambda match: mapping.get(match[0], match[0]), THEME)


def archive_palette(palette: QPalette, tokens: dict | None = None) -> QPalette:
    tokens = tokens or THEME_TOKENS
    for role, colour in (
        (QPalette.ColorRole.Window, tokens["canvas"]),
        (QPalette.ColorRole.WindowText, tokens["text"]),
        (QPalette.ColorRole.Text, tokens["text"]),
        (QPalette.ColorRole.Base, tokens["pane"]),
        (QPalette.ColorRole.AlternateBase, tokens["rail"]),
        (QPalette.ColorRole.Button, tokens["pane"]),
        (QPalette.ColorRole.ButtonText, tokens["text"]),
        (QPalette.ColorRole.HighlightedText, tokens["canvas"]),
        (QPalette.ColorRole.Link, tokens["accent"]),
        (QPalette.ColorRole.LinkVisited, tokens["accent"]),
        (QPalette.ColorRole.Highlight, tokens["accent"]),
        (QPalette.ColorRole.Mid, "#78648f" if tokens is THEME_TOKENS else tokens["muted"]),
    ):
        palette.setColor(role, QColor(colour))
    return palette


def row_item(widget: QListWidget, text: str, value) -> None:
    item = QListWidgetItem(text)
    item.setData(ROLE, value)
    widget.addItem(item)


def fit_rows(widget: QListWidget, limit: int) -> None:
    rows = min(limit, widget.count())
    height = sum(widget.sizeHintForRow(index) for index in range(rows)) if rows else 36
    widget.setFixedHeight(height + 2 * widget.frameWidth() + 2)


def windows_motion_enabled() -> bool:
    flag = ctypes.c_int()
    return bool(
        ctypes.windll.user32.SystemParametersInfoW(0x1042, 0, ctypes.byref(flag), 0) and flag.value
    )


class HomeArtwork(QWidget):
    def __init__(self):
        super().__init__()
        self.setAccessibleName("Snow leopard in the snow")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self.setMinimumSize(285, 270)
        self.setMaximumHeight(370)
        self.still = QPixmap(str(ASSETS / "drawer-still.png"))
        self.opacity = QGraphicsOpacityEffect(self)
        self.opacity.setOpacity(1)
        self.setGraphicsEffect(self.opacity)
        self.animation = QPropertyAnimation(self.opacity, b"opacity", self)
        self.animation.setDuration(160)
        self.animation.setStartValue(0.82)
        self.animation.setEndValue(1.0)

    def play(self) -> None:
        if self.isVisible() and windows_motion_enabled():
            self.animation.stop()
            self.animation.start()

    def stop(self) -> None:
        self.animation.stop()
        self.opacity.setOpacity(1)

    def hideEvent(self, event) -> None:
        self.stop()
        super().hideEvent(event)

    def paintEvent(self, _event) -> None:
        if self.still.isNull():
            return
        image = self.still.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter = QPainter(self)
        painter.drawPixmap(
            (self.width() - image.width()) // 2, (self.height() - image.height()) // 2, image
        )


class Home(QWidget):
    scope_changed = Signal()
    search_scope_changed = Signal(str)
    search_requested = Signal(str)
    note_requested = Signal(object)
    local_result_requested = Signal(object)
    graph_requested = Signal(object)
    ask_requested = Signal()
    pinned_changed = Signal(list)

    def __init__(self, pinned: list[str], *, whole_library=False):
        super().__init__()
        self.pinned = pinned
        self.reduce_motion = False
        self._collections: list[str] = []
        self._result_key = None
        root = QVBoxLayout(self)
        root.setContentsMargins(36, 38, 34, 28)
        root.setSpacing(18)
        root.addWidget(label("Dodo Mem", "title"))
        self.search = QLineEdit(placeholderText="Search your memory")
        self.search.setObjectName("search")
        self.search.setAccessibleName("Search your memory")
        self.search.setMaxLength(1000)
        self.search.returnPressed.connect(
            lambda: self.search_requested.emit(self.search.text().strip())
        )
        self.search.textChanged.connect(self._query_changed)
        self.search_scope = QComboBox(accessibleName="Search scope")
        if whole_library:
            self.search_scope.addItem("This PC", "local")
        self.search_scope.addItem("Shared vault", "shared")
        if whole_library:
            self.search_scope.addItem("Project", "project")
        self.search_scope.setVisible(whole_library)
        self.project = QComboBox()
        self.project.setAccessibleName("Project")
        self.collection = QComboBox()
        self.collection.setAccessibleName("Collection")
        for selector in (self.project, self.collection):
            selector.setMinimumContentsLength(12)
            selector.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
        self.project.currentIndexChanged.connect(self.scope_changed)
        self.collection.currentIndexChanged.connect(self._collection_changed)
        self.pin = button("Pin collection")
        self.pin.clicked.connect(self._toggle_pin)
        self.filters = QWidget()
        filters = QHBoxLayout(self.filters)
        filters.setContentsMargins(0, 0, 0, 0)
        filters.addWidget(self.project, 1)
        filters.addWidget(self.collection, 1)
        filters.addWidget(self.pin)
        self.lists = QWidget()
        lists = QVBoxLayout(self.lists)
        lists.setContentsMargins(0, 0, 0, 0)
        lists.setSpacing(12)
        search_bar = QHBoxLayout()
        search_bar.addWidget(self.search, 1)
        search_bar.addWidget(self.search_scope)
        lists.addLayout(search_bar)
        lists.addWidget(self.filters)
        self.project_context = label("Select a folder or file in Library or Graph.", "muted")
        self.project_context.setTextFormat(Qt.TextFormat.PlainText)
        self.project_context.setWordWrap(True)
        lists.addWidget(self.project_context)
        self.workspace = QWidget()
        workspace = QVBoxLayout(self.workspace)
        workspace.setContentsMargins(0, 3, 0, 5)
        self.workspace_title = label("Project workspace", "section")
        workspace.addWidget(self.workspace_title)
        actions = QGridLayout()
        self.project_folders = button("Project folders", "books", "primary")
        self.project_maps = button("Open mind maps", "link-simple")
        self.project_connections = button("Project connections", "vault-graph")
        self.add_ins = button("Add-ins", "plugin")
        for index, action in enumerate(
            (self.project_folders, self.project_maps, self.project_connections, self.add_ins)
        ):
            actions.addWidget(action, index // 2, index % 2)
        workspace.addLayout(actions)
        lists.addWidget(self.workspace)
        self.pins = QListWidget()
        self.pins.setObjectName("homePins")
        self.pins.setAccessibleName("Pinned collections")
        self.pins.setMaximumHeight(145)
        self.pins.itemActivated.connect(lambda item: self.select_collection(item.data(ROLE)))
        self.pins.itemClicked.connect(lambda item: self.select_collection(item.data(ROLE)))
        self.recent = QListWidget()
        self.recent.setObjectName("homeRecent")
        self.recent.setAccessibleName("Recent notes")
        self.recent.itemActivated.connect(lambda item: self.note_requested.emit(item.data(ROLE)))
        self.pins_title = label("Pinned collections", "section")
        self.recent_title = label("Recent notes", "section")
        lists.addWidget(self.pins_title)
        lists.addWidget(self.pins)
        lists.addWidget(self.recent_title)
        lists.addWidget(self.recent)
        lists.addStretch()
        self.art = HomeArtwork()
        self.home_row = QHBoxLayout()
        self.home_row.setSpacing(26)
        self.home_row.addWidget(self.lists, 3)
        self.home_row.addWidget(self.art, 2, Qt.AlignmentFlag.AlignTop)
        root.addLayout(self.home_row, 1)
        self.result_title = label("Search results", "section")
        self.ask_search = button("Ask Dodo")
        self.ask_search.clicked.connect(self.ask_requested)
        self.graph_result = button("Open in Graph")
        self.graph_result.setEnabled(False)
        self.graph_result.clicked.connect(self._open_result_graph)
        self.results = QListWidget()
        self.results.setAccessibleName("Search results")
        self.results.setWordWrap(True)
        self.results.itemActivated.connect(self._activate_result)
        self.results.currentItemChanged.connect(self._result_selection_changed)
        results_bar = QHBoxLayout()
        results_bar.addWidget(self.result_title, 1)
        results_bar.addWidget(self.ask_search)
        results_bar.addWidget(self.graph_result)
        root.addLayout(results_bar)
        root.addWidget(self.results, 2)
        self.set_filters([], [])
        self.search_scope.currentIndexChanged.connect(self._search_scope_changed)
        self._search_scope_changed()

    def _search_scope_changed(self) -> None:
        scope = self.search_scope.currentData()
        self.search.setPlaceholderText(
            {"local": "Search files and notes", "project": "Search the selected project"}.get(
                scope, "Search the shared vault"
            )
        )
        self._query_changed(self.search.text())
        self.search_scope_changed.emit(scope)

    def _activate_result(self, item) -> None:
        if self.search_scope.currentData() == "shared":
            self.note_requested.emit(item.data(ROLE))
        else:
            self.local_result_requested.emit(item.data(ROLE))

    def _open_result_graph(self) -> None:
        if item := self.results.currentItem():
            self.graph_requested.emit(item.data(ROLE))

    def _result_selection_changed(self, item, _previous) -> None:
        self.graph_result.setEnabled(item is not None)
        if item is not None:
            result = item.data(ROLE)
            self._result_key = result.get("id") or result["path"]

    def _restore_result_selection(self) -> None:
        for index in range(self.results.count()):
            result = self.results.item(index).data(ROLE)
            if (result.get("id") or result["path"]) == self._result_key:
                self.results.setCurrentRow(index)
                return
        if self.results.count():
            self.results.setCurrentRow(0)

    def _query_changed(self, text: str) -> None:
        self._result_key = None
        searching = bool(text.strip())
        scope = self.search_scope.currentData()
        shared = scope == "shared"
        self.workspace.setVisible(not searching)
        self.filters.setVisible(shared)
        self.project_context.setVisible(scope == "project")
        for widget in (self.pins_title, self.pins, self.recent_title, self.recent):
            widget.setVisible(shared and not searching)
        self.ask_search.setVisible(searching and not shared)
        self.graph_result.setVisible(searching)
        self.result_title.setVisible(searching)
        self.results.setVisible(searching)
        self.home_row.setStretch(0, 3 if not searching else 1)
        self.layout().setStretch(1, 0 if searching else 1)
        self._update_art()
        if not searching:
            self.results.clear()

    def _update_art(self) -> None:
        self.art.setVisible(
            self.width() >= 825 and not self.search.text().strip() and not self.art.still.isNull()
        )

    def resizeEvent(self, event) -> None:
        self._update_art()
        super().resizeEvent(event)

    def set_filters(self, projects: list[str], collections: list[str]) -> None:
        for combo, placeholder, values in (
            (self.project, "Global notes", projects),
            (self.collection, "All collections", collections),
        ):
            current = combo.currentData()
            combo.blockSignals(True)
            combo.clear()
            combo.addItem(placeholder, None)
            for name in values:
                combo.addItem(name, name)
            combo.setCurrentIndex(max(0, combo.findData(current)))
            combo.blockSignals(False)
        self._collections = collections
        self._update_pins()

    def scope(self) -> dict:
        return {"project": self.project.currentData(), "collection": self.collection.currentData()}

    def _collection_changed(self) -> None:
        self._update_pins()
        if not self.reduce_motion:
            self.art.play()
        self.scope_changed.emit()

    def select_collection(self, name: str) -> None:
        index = self.collection.findData(name)
        if index >= 0:
            self.search_scope.setCurrentIndex(self.search_scope.findData("shared"))
            self.collection.setCurrentIndex(index)

    def _toggle_pin(self) -> None:
        name = self.collection.currentData()
        if name in self.pinned:
            self.pinned.remove(name)
        elif name:
            self.pinned.append(name)
        self._update_pins()
        self.pinned_changed.emit(self.pinned.copy())

    def _update_pins(self) -> None:
        name = self.collection.currentData()
        self.pin.setEnabled(bool(name))
        self.pin.setText("Unpin collection" if name in self.pinned else "Pin collection")
        self.pins.clear()
        for pinned in self.pinned:
            if pinned in self._collections:
                row_item(self.pins, pinned, pinned)
        fit_rows(self.pins, 3)

    def set_recent(self, entries: list[dict]) -> None:
        self.recent.clear()
        for entry in entries:
            row_item(self.recent, entry["title"], entry)
        fit_rows(self.recent, 5)

    def set_results(self, entries: list[dict]) -> None:
        self.results.clear()
        for entry in entries:
            title = entry.get("title") or entry["path"]
            if self.project.currentData() and not entry.get("project"):
                title += " (Global)"
            excerpt = " ".join((entry.get("excerpt") or entry.get("body") or "").split())[:170]
            row_item(self.results, title + (f"\n{excerpt}" if excerpt else ""), entry)
        self.result_title.setText("Search results" if entries else "No matching notes")
        self._restore_result_selection()

    def set_local_results(self, entries: list[dict]) -> None:
        self.results.clear()
        for entry in entries:
            path = entry.get("path") or ""
            title = entry["name"]
            row_item(self.results, title + (f"\n{path}" if path else ""), entry)
        self.result_title.setText("Search results" if entries else "No matching files or folders")
        self._restore_result_selection()


class Inbox(QWidget):
    proposal_requested = Signal(str)
    evidence_requested = Signal(object)
    draft_requested = Signal()
    approve_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.proposal = None
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 32, 28, 24)
        root.setSpacing(16)
        header = QHBoxLayout()
        header.addWidget(label("Review Inbox", "title"), 1)
        draft = button("New draft")
        draft.clicked.connect(self.draft_requested)
        header.addWidget(draft)
        root.addLayout(header)
        self.list = QListWidget()
        self.list.setAccessibleName("Pending working knowledge")
        self.list.setWordWrap(True)
        self.list.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.list.itemActivated.connect(lambda item: self.proposal_requested.emit(item.data(ROLE)))
        self.list.itemClicked.connect(lambda item: self.proposal_requested.emit(item.data(ROLE)))
        self.list.setMinimumWidth(135)
        work = QSplitter(Qt.Orientation.Horizontal)
        work.addWidget(self.list)
        details = QWidget()
        detail = QVBoxLayout(details)
        detail.setContentsMargins(12, 0, 0, 0)
        self.title = label("Select a proposal", "section")
        detail.addWidget(self.title)
        bodies = QSplitter(Qt.Orientation.Horizontal)
        self.proposed = QPlainTextEdit(readOnly=True)
        self.proposed.setAccessibleName("Proposed text")
        self.current = QPlainTextEdit(readOnly=True)
        self.current.setAccessibleName("Current revision")
        for caption, editor in (
            ("Proposed text", self.proposed),
            ("Current revision", self.current),
        ):
            wrapper = QWidget()
            column = QVBoxLayout(wrapper)
            column.setContentsMargins(0, 0, 0, 0)
            column.addWidget(label(caption))
            column.addWidget(editor)
            bodies.addWidget(wrapper)
        detail.addWidget(bodies, 2)
        detail.addWidget(label("Evidence", "section"))
        self.evidence = QListWidget()
        self.evidence.setAccessibleName("Proposal evidence")
        self.evidence.setMaximumHeight(128)
        self.evidence.itemActivated.connect(
            lambda item: self.evidence_requested.emit(item.data(ROLE))
        )
        detail.addWidget(self.evidence, 1)
        actions = QHBoxLayout()
        self.open = button("Open source", "link-simple")
        self.open.clicked.connect(self._open_evidence)
        self.approve = button("Approve revision", role="primary")
        self.approve.clicked.connect(lambda: self.approve_requested.emit(self.proposal))
        actions.addWidget(self.open)
        actions.addStretch()
        actions.addWidget(self.approve)
        detail.addLayout(actions)
        work.addWidget(details)
        work.setSizes([200, 720])
        root.addWidget(work, 1)
        self.clear_detail()

    def _open_evidence(self) -> None:
        item = self.evidence.currentItem()
        if item:
            self.evidence_requested.emit(item.data(ROLE))

    def set_proposals(self, proposals: list[dict]) -> None:
        self.list.clear()
        for entry in proposals:
            row_item(self.list, entry["entry_key"], entry["proposal_id"])
            self.list.item(self.list.count() - 1).setToolTip(entry["entry_key"])

    def clear_detail(self) -> None:
        self.proposal = None
        self.title.setText("Select a proposal")
        self.proposed.clear()
        self.current.clear()
        self.evidence.clear()
        self.open.setEnabled(False)
        self.approve.setEnabled(False)

    def set_proposal(self, proposal: dict, current: dict) -> None:
        self.proposal = proposal
        self.title.setText(proposal["entry_key"])
        self.proposed.setPlainText(proposal.get("text", ""))
        self.current.setPlainText(current.get("text") or "No accepted revision")
        self.evidence.clear()
        for entry in proposal.get("evidence", []):
            row_item(self.evidence, entry.get("heading") or entry["path"], entry)
        self.evidence.setCurrentRow(0)
        self.open.setEnabled(self.evidence.count() > 0)
        self.approve.setEnabled(proposal.get("status") == "pending_review")
