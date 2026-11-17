"""Editable spatial boards for local notes and references."""

import hashlib
import math
from copy import deepcopy
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import (
    QAction,
    QColor,
    QDesktopServices,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPen,
    QPixmap,
    QPolygonF,
    QUndoCommand,
    QUndoStack,
)
from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsView,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMenu,
    QSplitter,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .canvas_store import (
    IMAGE_TYPES,
    MAX_IMPORT_IMAGE_BYTES,
    MAX_JSON_BYTES,
    WORLD,
    atomic_write,
    contained_file,
    image_bytes,
    load_image,
    read_board,
    read_bounded,
    relative_file,
    save_board,
    store_image,
    validate_board,
    web_url,
)
from .configuration import runtime_path
from .ui import ASSETS, label

INK = "#09070c"
PAPER = "#f2ecf8"
CARD = "#18131f"
ACCENT = "#c5a7ff"
SUBTLE = "#a99ab9"
BORDER = "#574862"
COLOURS = ("#c26860", "#ca9561", "#c2b377", "#89a47e", "#7fadaa", "#ac8cac")
ITEM_ID = 0
ITEM_KIND = 1


def _colour(record: dict, default: str) -> QColor:
    value = record.get("color", default)
    return QColor(COLOURS[int(value) - 1] if value.isdigit() else value)


def _shape(kind: str, position: QPointF, width: int, height: int, **content) -> dict:
    return {
        "id": uuid4().hex,
        "type": kind,
        "x": max(-WORLD, min(WORLD, round(position.x()))),
        "y": max(-WORLD, min(WORLD, round(position.y()))),
        "width": width,
        "height": height,
        **content,
    }


def _is_note(record: dict) -> bool:
    return record["type"] == "file" and record["file"].casefold().endswith(".md")


def _sample(storage: Path) -> dict:
    image = ASSETS / "drawer-still.png"
    group = _shape("group", QPointF(-25, -48), 800, 474, label="Early modern philosophy")
    rationalists = _shape(
        "text",
        QPointF(0, 0),
        302,
        142,
        text="Rationalists\n\nReason can yield knowledge beyond sense experience.",
    )
    empiricists = _shape(
        "text",
        QPointF(405, 0),
        340,
        142,
        text="Empiricists\n\nExperience is the primary source of our ideas and knowledge.",
    )
    cogito = _shape(
        "text",
        QPointF(0, 236),
        302,
        153,
        text="Cogito, ergo sum\n\nFor Descartes, the act of doubting confirms that the thinker exists.",
    )
    nodes = [group, rationalists, empiricists, cogito]
    if image.is_file():
        picture = store_image(storage, image_bytes(image), ".png")
        nodes.append(_shape("file", QPointF(405, 217), 145, 172, file=picture))
    reference = _shape(
        "link",
        QPointF(568, 217),
        177,
        172,
        url="https://plato.stanford.edu/entries/descartes-epistemology/",
    )
    nodes.append(reference)
    edges = [
        {
            "id": uuid4().hex,
            "fromNode": rationalists["id"],
            "toNode": empiricists["id"],
            "label": "Contrast",
        },
        {
            "id": uuid4().hex,
            "fromNode": rationalists["id"],
            "toNode": cogito["id"],
            "label": "Example",
        },
    ]
    return {"title": "Philosophy notebook", "nodes": nodes, "edges": edges}


class BoardView(QGraphicsView):
    def __init__(self, scene):
        super().__init__(scene)
        self.setObjectName("canvasView")
        self.setAccessibleName("Canvas. Use the Cards list for keyboard navigation.")
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setSceneRect(-WORLD, -WORLD, WORLD * 2, WORLD * 2)
        self._pan = None

    def wheelEvent(self, event):
        self.zoom(1.16 ** (event.angleDelta().y() / 120))
        event.accept()

    def zoom(self, factor: float) -> None:
        current = self.transform().m11()
        target = max(0.08, min(4.0, current * factor))
        self.scale(target / current, target / current)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan = event.position()
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._pan is None:
            super().mouseMoveEvent(event)
            return
        delta = self._pan - event.position()
        self._pan = event.position()
        self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() + round(delta.x()))
        self.verticalScrollBar().setValue(self.verticalScrollBar().value() + round(delta.y()))
        event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton:
            self._pan = None
            self.viewport().unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class Card(QGraphicsRectItem):
    def __init__(self, owner, record: dict, pixmap: QPixmap):
        super().__init__(0, 0, record["width"], record["height"])
        self.owner = owner
        self.record = record
        self.pixmap = pixmap
        self.setData(ITEM_ID, record["id"])
        self.setData(ITEM_KIND, "node")
        self.setPos(record["x"], record["y"])
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsMovable
        )
        self._resizing = False
        self._origin = QPointF()
        self._followers = []

    def paint(self, painter, _option, _widget=None):
        rect = self.rect()
        group = self.record["type"] == "group"
        selected = self.isSelected()
        stroke = _colour(self.record, BORDER)
        if selected:
            stroke = QColor(ACCENT)
        fill = QColor(CARD)
        if group:
            fill.setAlpha(100)
        painter.setBrush(fill)
        painter.setPen(QPen(stroke, 2 if selected else 1))
        painter.drawRoundedRect(rect, 6, 6)
        inset = rect.adjusted(14, 12, -14, -12)
        if inset.width() > 10 and inset.height() > 10:
            painter.save()
            painter.setClipRect(inset)
            if not self.pixmap.isNull():
                inner = QRectF(inset)
                source = QRectF(self.pixmap.rect())
                scale = min(inner.width() / source.width(), inner.height() / source.height())
                size = source.size() * scale
                target = QRectF(inner.center() - QPointF(size.width() / 2, size.height() / 2), size)
                painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
                painter.drawPixmap(target, self.pixmap, source)
            else:
                painter.setPen(QColor(PAPER))
                font = painter.font()
                font.setPixelSize(19 if self.record["type"] == "text" else 17)
                font.setBold(group or _is_note(self.record))
                painter.setFont(font)
                painter.drawText(
                    inset, int(Qt.TextFlag.TextWordWrap), self.owner.caption(self.record)
                )
            painter.restore()
        if selected:
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(stroke)
            painter.drawRect(QRectF(rect.right() - 9, rect.bottom() - 9, 7, 7))

    def mousePressEvent(self, event):
        if self.owner.connect_from:
            owner, identity = self.owner, self.record["id"]
            QTimer.singleShot(0, owner, lambda: owner.complete_connection(identity))
            event.accept()
            return
        rect = self.rect()
        self._resizing = (
            self.isSelected()
            and event.pos().x() >= rect.right() - 16
            and event.pos().y() >= rect.bottom() - 16
        )
        self._origin = self.pos()
        self._followers = []
        if self._resizing:
            event.accept()
            return
        if self.record["type"] == "group":
            area = self.mapRectToScene(rect)
            self._followers = [
                (card, card.pos())
                for card in self.owner.cards.values()
                if card is not self and area.contains(card.mapRectToScene(card.rect()))
            ]
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._resizing:
            position = event.pos()
            self.setRect(0, 0, max(80, min(4096, position.x())), max(58, min(4096, position.y())))
        else:
            super().mouseMoveEvent(event)
            delta = self.pos() - self._origin
            for card, origin in self._followers:
                card.setPos(origin + delta)
        self.owner.refresh_connections()

    def mouseReleaseEvent(self, event):
        if self._resizing:
            self._resizing = False
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self._followers = []
        QTimer.singleShot(0, self.owner, self.owner.commit_geometry)

    def mouseDoubleClickEvent(self, event):
        QTimer.singleShot(0, self.owner, self.owner.open_selected)
        event.accept()


class Connection(QGraphicsPathItem):
    def __init__(self, edge: dict, start: Card, end: Card):
        super().__init__()
        self.edge = edge
        self.start = start
        self.end = end
        self.setData(ITEM_ID, edge["id"])
        self.setData(ITEM_KIND, "edge")
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable)
        self.setZValue(-1)
        self.refresh()

    def refresh(self) -> None:
        start = self.start.sceneBoundingRect()
        end = self.end.sceneBoundingRect()
        a, b = start.center(), end.center()
        delta = b - a
        if abs(delta.x()) >= abs(delta.y()):
            first, last = ("right", "left") if delta.x() >= 0 else ("left", "right")
        else:
            first, last = ("bottom", "top") if delta.y() >= 0 else ("top", "bottom")

        def point(rect: QRectF, side: str) -> QPointF:
            return {
                "left": QPointF(rect.left(), rect.center().y()),
                "right": QPointF(rect.right(), rect.center().y()),
                "top": QPointF(rect.center().x(), rect.top()),
                "bottom": QPointF(rect.center().x(), rect.bottom()),
            }[side]

        a = point(start, self.edge.get("fromSide", first))
        b = point(end, self.edge.get("toSide", last))
        path = QPainterPath(a)
        path.lineTo(b)
        self.setPath(path)

    def boundingRect(self):
        return super().boundingRect().adjusted(-180, -28, 180, 28)

    def shape(self):
        stroke = QPainterPathStroker()
        stroke.setWidth(15)
        return stroke.createStroke(self.path())

    def paint(self, painter, _option, _widget=None):
        colour = _colour(self.edge, ACCENT if self.isSelected() else SUBTLE)
        painter.setPen(QPen(colour, 2.5 if self.isSelected() else 1.5))
        path = self.path()
        painter.drawPath(path)
        a, b = path.pointAtPercent(0), path.pointAtPercent(1)
        for kind, tip, origin in (("fromEnd", a, b), ("toEnd", b, a)):
            default = "arrow" if kind == "toEnd" else "none"
            if self.edge.get(kind, default) != "arrow":
                continue
            angle = math.atan2(tip.y() - origin.y(), tip.x() - origin.x())
            wings = [
                tip - QPointF(12 * math.cos(angle + turn), 12 * math.sin(angle + turn))
                for turn in (-0.42, 0.42)
            ]
            painter.setBrush(colour)
            painter.drawPolygon(QPolygonF([tip, *wings]))
        caption = self.edge.get("label")
        if caption:
            center = path.pointAtPercent(0.5)
            font = painter.font()
            font.setPixelSize(15)
            painter.setFont(font)
            text_box = QRectF(center.x() - 148, center.y() - 24, 296, 22)
            flags = Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignVCenter
            caption = painter.fontMetrics().elidedText(caption, Qt.TextElideMode.ElideRight, 286)
            width = painter.fontMetrics().horizontalAdvance(caption) + 12
            backing = QRectF(center.x() - width / 2, text_box.y(), width, text_box.height())
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(INK))
            painter.drawRoundedRect(backing, 3, 3)
            painter.setPen(QColor(SUBTLE))
            painter.drawText(text_box, int(flags), caption)


class Edit(QUndoCommand):
    def __init__(self, canvas, board: dict, name: str):
        super().__init__(name)
        self.canvas = canvas
        self.previous = canvas.board
        self.next = board

    def undo(self):
        self.canvas.apply(self.previous)

    def redo(self):
        self.canvas.apply(self.next)


class Canvas(QWidget):
    note_requested = Signal(str, str)
    note_refs_requested = Signal(object)

    def __init__(self, storage: Path | None = None):
        super().__init__()
        self.setObjectName("canvasPage")
        if storage is None:
            storage = runtime_path("desktop") / "canvases"
        self.storage = storage.resolve()
        self.current: Path | None = None
        self.digest: str | None = None
        self.board = {"title": "Untitled canvas", "nodes": [], "edges": []}
        self.cards: dict[str, Card] = {}
        self.connections: list[Connection] = []
        self.note_cards: dict[str, dict] = {}
        self.images: dict[str, QPixmap] = {}
        self.connect_from: str | None = None
        self._pending_export: tuple[Path, dict] | None = None
        self._rendering = False

        self.scene = QGraphicsScene(self)
        self.scene.setBackgroundBrush(QColor(INK))
        self.scene.selectionChanged.connect(self._sync_selection)
        self.view = BoardView(self.scene)
        self.undo = QUndoStack(self)
        self.undo.setUndoLimit(40)
        self.save_timer = QTimer(self, singleShot=True, interval=450)
        self.save_timer.timeout.connect(self.save)

        self.boards = QComboBox()
        self.boards.setAccessibleName("Open canvas")
        self.boards.setMinimumWidth(190)
        self.boards.currentIndexChanged.connect(self._board_chosen)
        self.status = label("", "muted")
        self.status.setWordWrap(True)
        self.card_list = QListWidget()
        self.card_list.setAccessibleName("Cards and connections")
        self.card_list.currentItemChanged.connect(self._choose_item)
        self.card_list.itemActivated.connect(self.open_selected)
        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        side_layout.addWidget(label("Cards", "section"))
        side_layout.addWidget(self.card_list)
        split = QSplitter(Qt.Orientation.Horizontal)
        split.addWidget(self.view)
        split.addWidget(side)
        split.setStretchFactor(0, 1)
        split.setSizes([930, 225])
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(10)
        layout.addWidget(label("Canvas", "title"))
        layout.addWidget(self._toolbar())
        layout.addWidget(split, 1)
        layout.addWidget(self.status)
        self._load_start()
        self._fit_on_show = True

    def _toolbar(self) -> QToolBar:
        bar = QToolBar("Canvas tools")
        bar.setStyleSheet("QToolBar { border: 0; spacing: 4px; } QToolButton { padding: 3px 8px; }")
        bar.addWidget(self.boards)
        board_menu = QMenu("Board", self)
        board_menu.addAction("New canvas", self._new_dialog)
        board_menu.addAction("Add philosophy example", self._add_example)
        board_menu.addAction("Import canvas", self._import_dialog)
        board_menu.addAction("Export canvas", self._export_dialog)
        add_menu = QMenu("Add", self)
        add_menu.addAction("Text card", self._text_dialog)
        add_menu.addAction("Image card", self._image_dialog)
        add_menu.addAction("URL card", self._url_dialog)
        add_menu.addAction("Note card", self._note_dialog)
        add_menu.addAction("Group", self._group_dialog)
        for menu in (board_menu, add_menu):
            action = menu.menuAction()
            bar.addAction(action)
            button = bar.widgetForAction(action)
            button.setPopupMode(button.ToolButtonPopupMode.InstantPopup)
        bar.addAction("Connect", self._start_connection)
        bar.addAction("Edit", self.edit_selected)
        delete = QAction("Delete", self)
        delete.setShortcut(QKeySequence(Qt.Key.Key_Delete))
        delete.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        delete.triggered.connect(self.delete_selected)
        self.addAction(delete)
        bar.addAction(delete)
        for action, key in (
            (self.undo.createUndoAction(self, "Undo"), QKeySequence.StandardKey.Undo),
            (self.undo.createRedoAction(self, "Redo"), QKeySequence.StandardKey.Redo),
        ):
            action.setShortcut(QKeySequence(key))
            action.setShortcutContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
            self.addAction(action)
            bar.addAction(action)
        bar.addAction("Zoom in", lambda: self.view.zoom(1.2))
        bar.addAction("Zoom out", lambda: self.view.zoom(1 / 1.2))
        bar.addAction("Fit", self.fit)
        return bar

    def _load_start(self) -> None:
        try:
            self.storage.mkdir(parents=True, exist_ok=True)
            paths = self._board_paths()
            if not paths:
                sample = self.storage / "sample.canvas"
                save_board(sample, _sample(self.storage), None)
                paths = [sample]
            self.open_board(paths[0])
        except (OSError, ValueError) as error:
            self.status.setText(str(error))

    def _board_paths(self) -> list[Path]:
        return sorted(
            path
            for path in self.storage.glob("*.canvas")
            if path.stem == "sample" or len(path.stem) == 32 and path.stem.isalnum()
        )[:200]

    def _list_boards(self) -> None:
        self.boards.blockSignals(True)
        self.boards.clear()
        for path in self._board_paths():
            try:
                board, _digest = read_board(path)
            except (OSError, ValueError):
                continue
            self.boards.addItem(board["title"], str(path))
        current = str(self.current) if self.current is not None else None
        self.boards.setCurrentIndex(self.boards.findData(current))
        self.boards.blockSignals(False)

    def _board_chosen(self, index: int) -> None:
        if index < 0:
            return
        try:
            self.open_board(Path(self.boards.itemData(index)))
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
            self._list_boards()

    def open_board(self, path: Path) -> None:
        if path == self.current:
            return
        if path.parent != self.storage or path.is_symlink() or path.resolve() != path:
            raise ValueError("Open stored canvases from the workspace.")
        if not self.save():
            raise ValueError("Resolve the save error before opening another canvas.")
        board, digest = read_board(path)
        self._pending_export = None
        self.current, self.digest = path, digest
        self.undo.clear()
        self.images.clear()
        self.clear_note_cards()
        self.apply(board)
        self.save_timer.stop()
        self.undo.setClean()
        self.status.setText("Saved on this device")
        self._list_boards()
        self.fit()
        self.request_note_cards()

    def create_board(self, title: str, board: dict | None = None) -> Path:
        if not self.save():
            raise ValueError("Resolve the save error before opening another canvas.")
        if len(self._board_paths()) >= 200:
            raise ValueError("This workspace holds at most 200 canvases.")
        path = self.storage / f"{uuid4().hex}.canvas"
        content = {"nodes": [], "edges": []} if board is None else board
        save_board(path, {**content, "title": title}, None)
        self.open_board(path)
        return path

    def import_board(self, path: Path) -> Path:
        board, _digest = read_board(path)
        assets = {}
        total_bytes = 0
        for node in board["nodes"]:
            if node["type"] != "file" or _is_note(node):
                continue
            file = node["file"]
            if file not in assets:
                source = contained_file(path.parent, file)
                content = image_bytes(source)
                total_bytes += len(content)
                if total_bytes > MAX_IMPORT_IMAGE_BYTES:
                    raise ValueError("Canvas images exceed the import size limit.")
                assets[file] = (content, source.suffix)
        stored = {
            file: store_image(self.storage, content, suffix)
            for file, (content, suffix) in assets.items()
        }
        for node in board["nodes"]:
            if node.get("file") in stored:
                node["file"] = stored[node["file"]]
        return self.create_board(path.stem[:120], board)

    def export_board(self, path: Path) -> None:
        if self.current is None:
            raise ValueError("Open a canvas before exporting.")
        if not self.save():
            raise ValueError("Resolve the save error before exporting.")
        if path.resolve().is_relative_to(self.storage):
            raise ValueError("Choose an export location outside the canvas workspace.")
        board = deepcopy(self.board)
        omitted = 0
        for node in board["nodes"]:
            if _is_note(node) and node["file"] not in self.note_cards:
                del node["file"]
                node.pop("subpath", None)
                node.update(type="text", text="Unavailable note")
                omitted += 1
        expected = None
        if path.is_symlink():
            raise ValueError("Canvas files cannot be filesystem links.")
        if path.exists():
            expected = hashlib.sha256(read_bounded(path, MAX_JSON_BYTES)).hexdigest()
        folder = path.parent / f"{path.stem}-assets"
        for node in board["nodes"]:
            if node["type"] != "file" or _is_note(node):
                continue
            original = contained_file(self.storage, node["file"])
            folder.mkdir(exist_ok=True)
            if folder.resolve(strict=True) != folder:
                raise ValueError("Canvas export media directory is redirected.")
            target = folder / original.name
            content = image_bytes(original)
            if target.is_symlink():
                raise ValueError("Canvas export images cannot be filesystem links.")
            if not target.exists():
                atomic_write(target, content)
            elif read_bounded(target, len(content)) != content:
                raise ValueError("A canvas export image already exists with different content.")
            node["file"] = f"{folder.name}/{target.name}"
        save_board(path, board, expected, portable=True)
        message = "Canvas exported."
        if omitted:
            message = f"Canvas exported with {omitted} unavailable note cards."
        self.status.setText(message)

    def _new_dialog(self) -> None:
        title, accepted = QInputDialog.getText(self, "New canvas", "Canvas name")
        if accepted and title.strip():
            self._attempt(self.create_board, title)

    def _add_example(self) -> None:
        self._attempt(lambda: self.create_board("Philosophy example", _sample(self.storage)))

    def _import_dialog(self) -> None:
        file, _filter = QFileDialog.getOpenFileName(
            self, "Import canvas", filter="Canvas (*.canvas)"
        )
        if file:
            self._attempt(self.import_board, Path(file))

    def _export_dialog(self) -> None:
        file, _filter = QFileDialog.getSaveFileName(
            self, "Export canvas", filter="Canvas (*.canvas)"
        )
        if not file:
            return
        path = Path(file).with_suffix(".canvas")
        if self.note_paths():
            self._pending_export = path, self.board
            self.request_note_cards()
        else:
            self._attempt(self.export_board, path)

    def _attempt(self, operation, *args) -> None:
        try:
            operation(*args)
        except (OSError, ValueError) as error:
            self.status.setText(str(error))

    def save(self) -> bool:
        if self.current is None or self.undo.isClean():
            return True
        try:
            self.digest = save_board(self.current, self.board, self.digest)
        except (OSError, ValueError) as error:
            self.status.setText(str(error))
            return False
        self.save_timer.stop()
        self.undo.setClean()
        self.status.setText("Saved on this device")
        return True

    def edit(self, board: dict, title: str) -> None:
        board = validate_board(board)
        if board != self.board:
            self.undo.push(Edit(self, board, title))

    def apply(self, board: dict) -> None:
        old_refs = self.note_paths()
        self.board = board
        self._render()
        self.save_timer.start()
        self.status.setText("Saving…")
        if self.note_paths() != old_refs:
            self.request_note_cards()

    def note_paths(self) -> list[str]:
        return list(dict.fromkeys(node["file"] for node in self.board["nodes"] if _is_note(node)))

    def request_note_cards(self) -> None:
        self.note_refs_requested.emit(self.note_paths())

    def set_note_cards(self, cards: dict[str, dict]) -> None:
        current = {path: cards[path] for path in self.note_paths() if path in cards}
        if current != self.note_cards:
            self.note_cards = current
            self._render()
        if self._pending_export:
            (path, board), self._pending_export = self._pending_export, None
            if board is self.board:
                self._attempt(self.export_board, path)
            else:
                self.status.setText("The canvas changed. Export again to save the new version.")

    def clear_note_cards(self) -> None:
        self.set_note_cards({})

    def caption(self, node: dict) -> str:
        kind = node["type"]
        if kind == "text":
            return node["text"]
        if kind == "link":
            url = urlsplit(node["url"])
            title = url.path.rstrip("/").rpartition("/")[2].replace("-", " ")
            return f"{url.hostname}\n\n{title}" if title else str(url.hostname)
        if kind == "group":
            return node.get("label", "Group")
        if _is_note(node):
            note = self.note_cards.get(node["file"])
            if note is None:
                return "Unavailable note"
            heading = node.get("subpath", "").removeprefix("#")
            title = str(note.get("title") or relative_file(node["file"]).stem)
            return title + (f"\n\n{heading}" if heading else "")
        return "Image unavailable"

    def _picture(self, node: dict) -> QPixmap:
        if node["type"] != "file" or _is_note(node):
            return QPixmap()
        file = node["file"]
        if file not in self.images:
            try:
                image = load_image(contained_file(self.storage, file))
                self.images[file] = QPixmap.fromImage(image)
            except (OSError, ValueError):
                return QPixmap()
        return self.images[file]

    def _render(self) -> None:
        selected = {item.data(ITEM_ID) for item in self.scene.selectedItems()}
        self._rendering = True
        self.scene.clear()
        self.cards.clear()
        self.connections.clear()
        self.card_list.clear()
        used_files = {node["file"] for node in self.board["nodes"] if node["type"] == "file"}
        self.images = {file: image for file, image in self.images.items() if file in used_files}
        for depth, record in enumerate(self.board["nodes"]):
            card = Card(self, record, self._picture(record))
            self.scene.addItem(card)
            card.setZValue(depth)
            self.cards[record["id"]] = card
            card.setSelected(record["id"] in selected)
            self._list_item(record["id"], self._card_name(record))
        for edge in self.board["edges"]:
            connection = Connection(edge, self.cards[edge["fromNode"]], self.cards[edge["toNode"]])
            self.scene.addItem(connection)
            connection.setZValue(len(self.board["nodes"]))
            connection.setSelected(edge["id"] in selected)
            self.connections.append(connection)
            self._list_item(edge["id"], edge.get("label") or "Connection")
        self._rendering = False
        self._sync_selection()

    def _card_name(self, record: dict) -> str:
        if record["type"] == "file" and not _is_note(record):
            return "Image card"
        first = self.caption(record).strip().partition("\n")[0].lstrip("# ").strip()
        return first[:150] or "Text card"

    def _list_item(self, identity: str, name: str) -> None:
        item = QListWidgetItem(name)
        item.setData(Qt.ItemDataRole.UserRole, identity)
        self.card_list.addItem(item)

    def _sync_selection(self) -> None:
        if self._rendering:
            return
        selected = self.scene.selectedItems()
        identity = selected[0].data(ITEM_ID) if selected else None
        self.card_list.blockSignals(True)
        for row in range(self.card_list.count()):
            item = self.card_list.item(row)
            if item.data(Qt.ItemDataRole.UserRole) == identity:
                self.card_list.setCurrentRow(row)
                break
        else:
            self.card_list.setCurrentRow(-1)
        self.card_list.blockSignals(False)

    def _choose_item(self, item, _previous) -> None:
        if item is None or self._rendering:
            return
        identity = item.data(Qt.ItemDataRole.UserRole)
        if self.connect_from:
            self.complete_connection(identity)
            return
        graphic = self.cards.get(identity)
        if graphic is None:
            graphic = next(
                (edge for edge in self.connections if edge.data(ITEM_ID) == identity), None
            )
        if graphic is not None:
            self.scene.clearSelection()
            graphic.setSelected(True)
            self.view.ensureVisible(graphic)

    def refresh_connections(self) -> None:
        for edge in self.connections:
            edge.refresh()

    def commit_geometry(self) -> None:
        board = deepcopy(self.board)
        for node in board["nodes"]:
            card = self.cards[node["id"]]
            node["x"] = max(-WORLD, min(WORLD, round(card.x())))
            node["y"] = max(-WORLD, min(WORLD, round(card.y())))
            node["width"] = max(1, round(card.rect().width()))
            node["height"] = max(1, round(card.rect().height()))
        self._attempt(self.edit, board, "Move or resize cards")

    def fit(self) -> None:
        bounds = self.scene.itemsBoundingRect()
        if bounds.isEmpty():
            self.view.resetTransform()
            self.view.centerOn(0, 0)
        else:
            self.view.fitInView(
                bounds.adjusted(-22, -22, 22, 22), Qt.AspectRatioMode.KeepAspectRatio
            )

    def showEvent(self, event):
        super().showEvent(event)
        if self._fit_on_show:
            self._fit_on_show = False
            QTimer.singleShot(0, self, self.fit)

    def _new_position(self) -> QPointF:
        center = self.view.mapToScene(self.view.viewport().rect().center())
        return center + QPointF(-120 + len(self.cards) % 5 * 20, -70 + len(self.cards) % 5 * 20)

    def add_node(self, node: dict) -> None:
        if self.current is None:
            raise ValueError("Create a canvas before adding cards.")
        board = deepcopy(self.board)
        if node["type"] == "group":
            board["nodes"].insert(0, node)
        else:
            board["nodes"].append(node)
        self.edit(board, "Add card")
        self.scene.clearSelection()
        card = self.cards[node["id"]]
        card.setSelected(True)
        self.view.ensureVisible(card)

    def add_note(self, path: str, heading: str | None = None) -> None:
        file = relative_file(path)
        if file.suffix.casefold() != ".md":
            raise ValueError("A note card needs a Markdown vault path.")
        node = _shape("file", self._new_position(), 245, 145, file=file.as_posix())
        if heading:
            node["subpath"] = "#" + heading.lstrip("# ")
        self.add_node(node)

    def _text_dialog(self) -> None:
        value, accepted = QInputDialog.getMultiLineText(self, "Text card", "Text")
        if accepted:
            self._attempt(self.add_node, _shape("text", self._new_position(), 270, 160, text=value))

    def _image_dialog(self) -> None:
        value, _filter = QFileDialog.getOpenFileName(
            self, "Image card", filter="Images (*.png *.jpg *.jpeg *.webp *.bmp)"
        )
        if value:
            self._attempt(self.add_image, Path(value))

    def add_image(self, path: Path) -> None:
        if path.suffix.lower() not in IMAGE_TYPES:
            raise ValueError("Choose a PNG, JPEG, WebP or BMP image.")
        file = store_image(self.storage, image_bytes(path), path.suffix)
        self.add_node(_shape("file", self._new_position(), 270, 235, file=file))

    def _url_dialog(self) -> None:
        value, accepted = QInputDialog.getText(self, "URL card", "Address")
        if accepted:
            self._attempt(self.add_url, value)

    def add_url(self, value: str) -> None:
        self.add_node(_shape("link", self._new_position(), 260, 140, url=web_url(value)))

    def _note_dialog(self) -> None:
        value, accepted = QInputDialog.getText(self, "Note card", "Vault-relative Markdown path")
        if accepted:
            self._attempt(self.add_note, value)

    def _group_dialog(self) -> None:
        name, accepted = QInputDialog.getText(self, "Group", "Group name")
        if not accepted:
            return
        selected = [item for item in self.scene.selectedItems() if item.data(ITEM_KIND) == "node"]
        bounds = QRectF(self._new_position(), QRectF(0, 0, 420, 250).size())
        if selected:
            bounds = QRectF()
            for card in selected:
                bounds = bounds.united(card.sceneBoundingRect())
            bounds.adjust(-18, -44, 18, 18)
        self._attempt(
            self.add_node,
            _shape(
                "group", bounds.topLeft(), round(bounds.width()), round(bounds.height()), label=name
            ),
        )

    def _start_connection(self) -> None:
        selected = [item for item in self.scene.selectedItems() if item.data(ITEM_KIND) == "node"]
        if len(selected) != 1:
            self.status.setText("Select one starting card, then choose Connect.")
            return
        self.connect_from = selected[0].data(ITEM_ID)
        self.status.setText("Select the card this arrow should point to.")

    def complete_connection(self, target: str) -> None:
        source, self.connect_from = self.connect_from, None
        if source == target or target not in self.cards:
            self.status.setText("Choose two different cards to connect.")
            return
        name, accepted = QInputDialog.getText(self, "Connection", "Label")
        if accepted:
            self._attempt(self.add_connection, source, target, name)
        else:
            self.status.clear()

    def add_connection(self, source: str, target: str, name: str) -> None:
        board = deepcopy(self.board)
        board["edges"].append(
            {
                "id": uuid4().hex,
                "fromNode": source,
                "toNode": target,
                "label": name,
            }
        )
        self.edit(board, "Connect cards")

    def open_selected(self, _item=None) -> None:
        selected = self.scene.selectedItems()
        if len(selected) != 1:
            return
        graphic = selected[0]
        if graphic.data(ITEM_KIND) == "edge":
            self.edit_selected()
            return
        node = graphic.record
        if _is_note(node):
            if node["file"] in self.note_cards:
                self.note_requested.emit(node["file"], node.get("subpath", "").removeprefix("#"))
        elif node["type"] == "link":
            QDesktopServices.openUrl(QUrl(node["url"]))
        elif node["type"] in {"text", "group"}:
            self.edit_selected()

    def edit_selected(self) -> None:
        selected = self.scene.selectedItems()
        if len(selected) != 1:
            return
        graphic = selected[0]
        if graphic.data(ITEM_KIND) == "edge":
            name, accepted = QInputDialog.getText(
                self, "Connection", "Label", text=graphic.edge.get("label", "")
            )
            if accepted:
                self._update(graphic.edge, "label", name, "edges")
            return
        node = graphic.record
        if _is_note(node):
            reference = node["file"] + node.get("subpath", "")
            text, accepted = QInputDialog.getText(
                self, "Note card", "Vault path and heading", text=reference
            )
            if accepted:
                self._attempt(self._relink, node, text)
        elif node["type"] == "file":
            value, _filter = QFileDialog.getOpenFileName(
                self, "Replace image", filter="Images (*.png *.jpg *.jpeg *.webp *.bmp)"
            )
            if value:
                self._attempt(self._replace_image, node, Path(value))
        elif node["type"] == "link":
            url, accepted = QInputDialog.getText(self, "URL card", "Address", text=node["url"])
            if accepted:
                self._update(node, "url", url, "nodes")
        elif node["type"] == "text":
            text, accepted = QInputDialog.getMultiLineText(self, "Text card", "Text", node["text"])
            if accepted:
                self._update(node, "text", text, "nodes")
        elif node["type"] == "group":
            name, accepted = QInputDialog.getText(
                self, "Group", "Group name", text=node.get("label", "")
            )
            if accepted:
                self._update(node, "label", name, "nodes")

    def _relink(self, original: dict, value: str) -> None:
        file, separator, heading = value.partition("#")
        if relative_file(file).suffix.casefold() != ".md":
            raise ValueError("A note card needs a Markdown vault path.")
        board = deepcopy(self.board)
        record = next(node for node in board["nodes"] if node["id"] == original["id"])
        record["file"] = file
        if separator and heading:
            record["subpath"] = "#" + heading
        else:
            record.pop("subpath", None)
        self.edit(board, "Relink note card")

    def _replace_image(self, original: dict, path: Path) -> None:
        if path.suffix.lower() not in IMAGE_TYPES:
            raise ValueError("Choose a PNG, JPEG, WebP or BMP image.")
        file = store_image(self.storage, image_bytes(path), path.suffix)
        self._update(original, "file", file, "nodes")

    def _update(self, original: dict, key: str, value: str, collection: str) -> None:
        board = deepcopy(self.board)
        record = next(item for item in board[collection] if item["id"] == original["id"])
        record[key] = value
        self._attempt(self.edit, board, "Edit canvas")

    def delete_selected(self) -> None:
        nodes, edges = set(), set()
        for item in self.scene.selectedItems():
            identities = nodes if item.data(ITEM_KIND) == "node" else edges
            identities.add(item.data(ITEM_ID))
        if not nodes and not edges:
            return
        board = deepcopy(self.board)
        board["nodes"] = [node for node in board["nodes"] if node["id"] not in nodes]
        board["edges"] = [
            edge
            for edge in board["edges"]
            if edge["id"] not in edges and not ({edge["fromNode"], edge["toNode"]} & nodes)
        ]
        self._attempt(self.edit, board, "Delete selection")
