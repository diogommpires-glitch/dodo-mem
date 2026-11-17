"""An authorised vault snapshot and its two navigable representations."""

import math
import re
from collections import Counter, defaultdict, deque
from html import escape
from pathlib import Path

from PySide6.QtCore import QRectF, QSignalBlocker, QStandardPaths, Qt, QUrl, Signal
from PySide6.QtGui import (
    QBrush,
    QDesktopServices,
    QFontMetrics,
    QPainter,
    QPainterPath,
    QPainterPathStroker,
    QPalette,
    QPen,
)
from PySide6.QtWidgets import (
    QComboBox,
    QGraphicsItem,
    QGraphicsPathItem,
    QGraphicsScene,
    QGraphicsView,
    QHBoxLayout,
    QListWidget,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .project_library import ARCHIVE, load_projects, local_path, shared_note
from .ui import ROLE, button, label, row_item


def arrange(snapshot: dict) -> dict[str, tuple[float, float]]:
    neighbours = {node["path"]: set() for node in snapshot["nodes"]}
    for edge in snapshot["edges"]:
        left, right = edge["source"], edge["target"]
        if left in neighbours and right in neighbours:
            neighbours[left].add(right)
            neighbours[right].add(left)

    visited = set()
    components = []
    for path in sorted(neighbours, key=lambda key: (-len(neighbours[key]), key)):
        if path in visited:
            continue
        queue = deque([path])
        visited.add(path)
        component = []
        while queue:
            point = queue.popleft()
            component.append(point)
            for near in sorted(neighbours[point] - visited):
                visited.add(near)
                queue.append(near)
        components.append(component)

    positions = {}
    x = y = height = 0
    for component in components:
        columns = math.ceil(math.sqrt(len(component)))
        rows = math.ceil(len(component) / columns)
        width = columns * 76
        if x and x + width > 1150:
            x, y, height = 0, y + height + 42, 0
        for i, path in enumerate(component):
            row, column = divmod(i, columns)
            positions[path] = (x + column * 76 + (row % 2) * 12, y + row * 64)
        x += width + 24
        height = max(height, rows * 64)
    return positions


class EdgeItem(QGraphicsPathItem):
    def boundingRect(self) -> QRectF:
        return super().boundingRect().adjusted(-7, -7, 7, 7)

    def shape(self) -> QPainterPath:
        hit = QPainterPathStroker()
        hit.setWidth(13)
        return hit.createStroke(self.path())


class GraphView(QGraphicsView):
    selected = Signal(object)

    def __init__(self):
        super().__init__()
        self.setScene(QGraphicsScene(self))
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setDragMode(self.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(self.ViewportAnchor.AnchorUnderMouse)
        self.setAccessibleName("Vault graph. The adjacent lists contain the same notes and links.")
        self._press = None

    def wheelEvent(self, event):
        change = 1.15 ** (event.angleDelta().y() / 120)
        if 0.12 <= self.transform().m11() * change <= 5:
            self.scale(change, change)
        event.accept()

    def mousePressEvent(self, event):
        self._press = event.position() if event.button() == Qt.MouseButton.LeftButton else None
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event):
        if self._press is not None and (event.position() - self._press).manhattanLength() < 5:
            for item in self.items(event.position().toPoint()):
                if selection := item.data(0):
                    self.selected.emit(selection)
                    break
        self._press = None
        super().mouseReleaseEvent(event)

    def fit(self):
        bounds = self.scene().itemsBoundingRect()
        if not bounds.isEmpty():
            self.resetTransform()
            self.fitInView(bounds.adjusted(-48, -48, 48, 48), Qt.AspectRatioMode.KeepAspectRatio)


class ProjectGraph(QWidget):
    note_requested = Signal(object)
    refs_requested = Signal()

    def __init__(self):
        super().__init__()
        self.archive = ARCHIVE
        self.desktop = Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        )
        self._projects = {}
        self._available = {}
        self.project = QComboBox(accessibleName="Project for connections")
        self.project.currentIndexChanged.connect(self._draw)
        self.view = GraphView()
        self.view.setAccessibleName(
            "Project connections. The adjacent list contains the same links."
        )
        self.view.selected.connect(self._activate)
        self.connections = QListWidget(
            accessibleName="Folders and shared notes linked to the project"
        )
        self.connections.setTextElideMode(Qt.TextElideMode.ElideMiddle)
        self.connections.itemActivated.connect(lambda item: self._activate(item.data(ROLE)))
        self.summary = label("Loading projects…", "muted")
        self.summary.setWordWrap(True)

        controls = QHBoxLayout()
        controls.addWidget(self.project, 1)
        fit = button("Fit")
        fit.clicked.connect(self.view.fit)
        controls.addWidget(fit)
        reload = button("Reload projects")
        reload.clicked.connect(self.reload)
        controls.addWidget(reload)
        split = QSplitter()
        split.addWidget(self.view)
        split.addWidget(self.connections)
        split.setSizes([620, 335])
        split.setStretchFactor(0, 2)
        root = QVBoxLayout(self)
        root.addLayout(controls)
        root.addWidget(split, 1)
        root.addWidget(self.summary)

    def reload(self):
        previous = self.project.currentData()
        self._projects.clear()
        self._available.clear()
        try:
            records = load_projects(self.archive)
        except (OSError, ValueError, RuntimeError):
            records = []
        with QSignalBlocker(self.project):
            self.project.clear()
            for record in records:
                if not isinstance(record, dict):
                    continue
                identity, title = record.get("id"), record.get("title")
                if (
                    not isinstance(identity, str)
                    or not re.fullmatch("[a-z0-9-]{1,150}", identity)
                    or not isinstance(title, str)
                    or not title.strip()
                    or identity in self._projects
                ):
                    continue
                self._projects[identity] = record
                self.project.addItem(title, identity)
            index = self.project.findData(previous)
            if index < 0 and self._projects:
                index = self.project.findData(max(self._projects, key=self._connection_count))
            self.project.setCurrentIndex(index)
        self._draw()
        self.refs_requested.emit()

    def _connection_count(self, identity):
        project = self._projects[identity]
        return sum(
            len(value)
            for key in ("roots", "sharedNotes")
            if isinstance(value := project.get(key), list)
        )

    def note_paths(self):
        return list(
            dict.fromkeys(
                ref["path"]
                for project in self._projects.values()
                if isinstance(notes := project.get("sharedNotes"), list)
                for entry in notes
                if (ref := shared_note(entry))
            )
        )

    def set_note_refs(self, available):
        if self._available != available:
            self._available = available
            self._draw()

    def _draw(self):
        scene = self.view.scene()
        scene.clear()
        self.connections.clear()
        identity = self.project.currentData()
        if identity not in self._projects:
            self.summary.setText(
                "Project catalogue unavailable. Reload after updating the archive."
            )
            return
        project = self._projects[identity]
        folders, notes = [], []
        if isinstance(roots := project.get("roots"), list):
            for root in roots:
                if not isinstance(root, dict) or not isinstance(value := root.get("path"), str):
                    continue
                if self._location("folder", value) is None:
                    continue
                title = root.get("label")
                folders.append(
                    (title if isinstance(title, str) and title else value, ("folder", value))
                )
        if isinstance(refs := project.get("sharedNotes"), list):
            for entry in refs:
                if ref := shared_note(entry):
                    current = self._available.get(ref["path"], {})
                    if current.get("sha256") == ref["sha256"]:
                        notes.append((current.get("title") or ref["path"], ("note", ref)))

        path = f"projects/{identity}/mind-map.html"
        map_action = (
            ("map", path) if project.get("map") == path and self._location("map", path) else None
        )
        if map_action:
            row_item(self.connections, "Open mind map", map_action)
        self._draw_node(project["title"], 0, 0, map_action, QPalette.ColorRole.Highlight)
        palette = self.palette()
        for column, kind, rows in ((-260, "Folder", folders), (260, "Shared note", notes)):
            for index, (title, action) in enumerate(rows):
                y = (index - (len(rows) - 1) / 2) * 49
                caption = f"{kind} · {title}"
                row_item(self.connections, caption, action)
                route = QPainterPath()
                route.moveTo(0, 0)
                route.lineTo(column, y)
                edge = EdgeItem(route)
                edge.setPen(QPen(palette.color(QPalette.ColorRole.Mid), 1.5))
                edge.setData(0, ("row", self.connections.count() - 1))
                edge.setToolTip(f"<span>{escape(caption)}</span>")
                scene.addItem(edge)
                colour = (
                    QPalette.ColorRole.Link if kind == "Shared note" else QPalette.ColorRole.Mid
                )
                node_title = Path(action[1]).name if kind == "Folder" else title
                self._draw_node(node_title, column, y, action, colour)
        folder_label = "folder" if len(folders) == 1 else "folders"
        note_label = "note" if len(notes) == 1 else "notes"
        self.summary.setText(
            f"{len(folders)} {folder_label} · {len(notes)} verified shared {note_label}"
        )
        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-25, -25, 25, 25))
        self.view.fit()

    def _draw_node(self, title, x, y, action, colour):
        scene, palette = self.view.scene(), self.palette()
        dot = scene.addEllipse(
            x - 7, y - 7, 14, 14, QPen(Qt.PenStyle.NoPen), QBrush(palette.color(colour))
        )
        dot.setZValue(1)
        dot.setData(0, action)
        dot.setToolTip(f"<span>{escape(title)}</span>")
        font = self.font()
        font.setPixelSize(15)
        caption = scene.addSimpleText(
            QFontMetrics(font).elidedText(title, Qt.TextElideMode.ElideRight, 205), font
        )
        caption.setBrush(palette.text())
        width, height = caption.boundingRect().width(), caption.boundingRect().height()
        if x < 0:
            caption.setPos(x - 12 - width, y - height / 2)
        elif x > 0:
            caption.setPos(x + 12, y - height / 2)
        else:
            caption.setPos(x - width / 2, y + 10)
        caption.setZValue(2)
        caption.setData(0, action)
        caption.setToolTip(dot.toolTip())

    def _location(self, kind, target):
        try:
            if kind == "map":
                path = local_path(self.archive / target, self.archive)
                return path if path.is_file() else None
            path = local_path(Path(target), Path.home(), self.desktop)
            return path if path.is_dir() else None
        except (OSError, ValueError, RuntimeError):
            return None

    def _activate(self, action):
        kind, target = action
        if kind == "row":
            self.connections.setCurrentRow(target)
        elif kind == "note":
            self.note_requested.emit(target)
        elif (path := self._location(kind, target)) is None or not QDesktopServices.openUrl(
            QUrl.fromLocalFile(str(path))
        ):
            self.summary.setText("That project location is unavailable.")


class VaultGraph(QWidget):
    scope_changed = Signal()
    note_requested = Signal(object)
    evidence_requested = Signal(object)

    def __init__(self):
        super().__init__()
        self.projects = ProjectGraph()
        self.projects.note_requested.connect(self.note_requested)
        self.snapshot = None
        self.focused_path = None
        self.edge_items = []
        self.view = GraphView()
        self.view.selected.connect(self._select)
        self.project = QComboBox()
        self.collection = QComboBox()
        self.project.setAccessibleName("Graph project")
        self.collection.setAccessibleName("Graph collection")
        for selector, title in (
            (self.project, "All projects"),
            (self.collection, "All collections"),
        ):
            selector.addItem(title, None)
            selector.currentIndexChanged.connect(self.scope_changed)
        self.notes = QListWidget()
        self.notes.setAccessibleName("Notes in the vault graph")
        self.edges = QListWidget()
        self.edges.setAccessibleName("Relationships in the vault graph")
        self.detail = label("")
        self.detail.setWordWrap(True)
        self.open_evidence = button("Open evidence")
        self.open_evidence.clicked.connect(self._open_edge)
        self.open_evidence.setEnabled(False)
        self.notes.itemActivated.connect(lambda item: self.note_requested.emit(item.data(ROLE)))
        self.edges.itemActivated.connect(self._open_edge)
        self.edges.currentItemChanged.connect(self._describe_edge)
        self.lists = QTabWidget()
        self.lists.addTab(self.notes, "Notes")
        self.lists.addTab(self.edges, "Relationships")
        inspector = QWidget()
        side = QVBoxLayout(inspector)
        side.setContentsMargins(0, 0, 0, 0)
        side.addWidget(self.lists, 1)
        side.addWidget(self.detail)
        side.addWidget(self.open_evidence)
        split = QSplitter()
        split.addWidget(self.view)
        split.addWidget(inspector)
        split.setSizes([620, 335])
        split.setStretchFactor(0, 2)
        self.summary = label("Connecting to the vault", "muted")
        controls = QHBoxLayout()
        controls.addWidget(self.project)
        controls.addWidget(self.collection)
        self.all_notes = button("All notes")
        self.all_notes.clicked.connect(self._clear_focus)
        self.all_notes.hide()
        controls.addWidget(self.all_notes)
        fit = button("Fit")
        fit.clicked.connect(self.view.fit)
        controls.addWidget(fit)
        controls.addStretch()
        self.note_page = QWidget()
        notes_layout = QVBoxLayout(self.note_page)
        notes_layout.addLayout(controls)
        notes_layout.addWidget(split, 1)
        notes_layout.addWidget(self.summary)
        self.tabs = QTabWidget(accessibleName="Graph view")
        self.tabs.addTab(self.projects, "Project connections")
        self.tabs.addTab(self.note_page, "Note relationships")
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(label("Graph", "title"))
        root.addWidget(self.tabs, 1)

    def show_project_connections(self):
        self.tabs.setCurrentWidget(self.projects)

    def show_note_relationships(self):
        self.tabs.setCurrentWidget(self.note_page)

    def focus_note(self, path, scope):
        for name, selector in (("project", self.project), ("collection", self.collection)):
            value = scope[name]
            with QSignalBlocker(selector):
                index = selector.findData(value)
                if index < 0 and value is not None:
                    selector.addItem(value, value)
                    index = selector.count() - 1
                selector.setCurrentIndex(max(0, index))
        self.focused_path = path
        self.all_notes.show()
        self.show_note_relationships()

    def _clear_focus(self):
        self.focused_path = None
        self.all_notes.hide()
        self.scope_changed.emit()

    def scope(self) -> dict:
        return {"project": self.project.currentData(), "collection": self.collection.currentData()}

    def filters(self, projects: list[str], collections: list[str]) -> None:
        for selector, values in ((self.project, projects), (self.collection, collections)):
            title, active = selector.itemText(0), selector.currentData()
            with QSignalBlocker(selector):
                selector.clear()
                selector.addItem(title, None)
                for value in values:
                    selector.addItem(value, value)
                selector.setCurrentIndex(max(0, selector.findData(active)))

    def clear(self, message="Vault unavailable") -> None:
        self.snapshot = None
        self.notes.clear()
        self.edges.clear()
        self.edge_items.clear()
        self.view.scene().clear()
        self.summary.setText(message)

    def show_snapshot(self, value: tuple[dict, dict]) -> None:
        snapshot, positions = value
        if snapshot == self.snapshot:
            return
        was_empty = self.snapshot is None
        selection = self.edges.currentItem()
        selected = selection.data(ROLE) if selection else None
        self.clear("")
        self.snapshot = snapshot
        scene = self.view.scene()
        nodes = {row["path"]: row for row in snapshot["nodes"]}
        palette = self.palette()
        counts = Counter(tuple(sorted((row["source"], row["target"]))) for row in snapshot["edges"])
        lanes = defaultdict(int)
        for edge in snapshot["edges"]:
            source, target = edge["source"], edge["target"]
            if source not in nodes or target not in nodes:
                continue
            title = f"{nodes[source]['title']} → {nodes[target]['title']}"
            row = self.edges.count()
            row_item(self.edges, title, edge)
            key = tuple(sorted((source, target)))
            lane = lanes[key] - (counts[key] - 1) / 2
            lanes[key] += 1
            x, y = positions[source]
            u, v = positions[target]
            route = QPainterPath()
            if source == target:
                radius = 18 + lanes[key] * 6
                route.addEllipse(QRectF(x, y - radius, radius, radius))
            else:
                distance = math.hypot(u - x, v - y)
                sign = 1 if source == key[0] else -1
                curve = sign * lane * 18 / max(distance, 1)
                route.moveTo(x, y)
                route.quadTo((x + u) / 2 - (v - y) * curve, (y + v) / 2 + (u - x) * curve, u, v)
            item = EdgeItem(route)
            pen = QPen(palette.color(QPalette.ColorRole.Mid), 1.4)
            if edge["kind"] == "evidence":
                pen.setStyle(Qt.PenStyle.DashLine)
            item.setPen(pen)
            item.setToolTip(f"<span>{escape(title)}</span>")
            item.setData(0, ("edge", row))
            scene.addItem(item)
            self.edge_items.append(item)
            if edge == selected:
                self.edges.setCurrentRow(row)

        linked = {path for key in counts for path in key}
        font = self.font()
        font.setPixelSize(15)
        metrics = QFontMetrics(font)
        for path, node in sorted(
            nodes.items(), key=lambda item: (item[1]["title"].casefold(), item[0])
        ):
            row_item(self.notes, node["title"], node)
            note_item = self.notes.item(self.notes.count() - 1)
            note_item.setToolTip(path)
            if path == self.focused_path:
                self.notes.setCurrentItem(note_item)
                self.lists.setCurrentWidget(self.notes)
            x, y = positions[path]
            colour = palette.color(
                QPalette.ColorRole.Highlight if path in linked else QPalette.ColorRole.Mid
            )
            item = scene.addEllipse(x - 7, y - 7, 14, 14, QPen(Qt.PenStyle.NoPen), QBrush(colour))
            item.setZValue(1)
            item.setToolTip(f"<span>{escape(node['title'])}</span>")
            item.setData(0, ("note", node))
            if len(nodes) <= 24:
                title = metrics.elidedText(node["title"], Qt.TextElideMode.ElideRight, 115)
                caption = scene.addSimpleText(title, font)
                caption.setBrush(palette.text())
                caption.setPos(x + 9, y + 7)
                caption.setZValue(2)
                caption.setData(0, ("note", node))
                caption.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIgnoresTransformations)

        scene.setSceneRect(scene.itemsBoundingRect().adjusted(-42, -42, 42, 42))
        if was_empty:
            self.view.fit()
        if self.focused_path in positions:
            self.view.centerOn(*positions[self.focused_path])
        suffix = " · Refine the filters to see more" if snapshot.get("truncated") else ""
        self.summary.setText(f"{len(nodes)} notes · {self.edges.count()} relationships{suffix}")

    def _select(self, selection):
        kind, value = selection
        if kind == "note":
            self.note_requested.emit(value)
        else:
            self.lists.setCurrentWidget(self.edges)
            self.edges.setCurrentRow(value)
            self.edges.scrollToItem(self.edges.currentItem())

    def _describe_edge(self, item, previous):
        for row, colour, width in (
            (previous, QPalette.ColorRole.Mid, 1.4),
            (item, QPalette.ColorRole.Highlight, 2.5),
        ):
            if row is None:
                continue
            index = self.edges.row(row)
            if 0 <= index < len(self.edge_items):
                edge = self.edge_items[index]
                pen = edge.pen()
                pen.setColor(self.palette().color(colour))
                pen.setWidthF(width)
                edge.setPen(pen)
        self.open_evidence.setEnabled(item is not None)
        if item is None:
            self.detail.clear()
            return
        edge = item.data(ROLE)
        heading = (edge.get("source_heading") or edge["evidence"].get("heading") or "").lstrip("# ")
        target = (edge.get("target_heading") or "").lstrip("# ")
        parts = (edge["label"], heading, f"To {target}" if target else "")
        self.detail.setText("\n".join(dict.fromkeys(part for part in parts if part)))

    def _open_edge(self, _item=None):
        if item := self.edges.currentItem():
            self.evidence_requested.emit(item.data(ROLE)["evidence"])
