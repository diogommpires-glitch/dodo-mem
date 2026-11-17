"""Private catalogue graph and generated workspace views."""

from __future__ import annotations

import math
import re
import sqlite3
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from hashlib import blake2b
from html import escape
from pathlib import PureWindowsPath

from PySide6.QtCore import (
    QIODevice,
    QLineF,
    QObject,
    QPointF,
    QRectF,
    QRunnable,
    QSaveFile,
    Qt,
    QThreadPool,
    Signal,
)
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLineEdit,
    QListWidget,
    QSplitter,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from .graph_layout import cluster_positions
from .neural_client import NeuralError
from .ui import ROLE, button, label, row_item
from .whole_index import GRAPH_BLOCKED

PAGE_SIZE = 120
GRAPH_BACKGROUND = "#0e0e15"
CLUSTER_COLOURS = ("#ad83ea", "#72bed2", "#e3a06d", "#de81b1", "#9ebd80", "#b4a0f1")
GROUPS = {
    "folder": ("Folders", "#baa5f0"),
    "topic": ("Topics", "#c291fa"),
    "code": ("Code", "#75c5ff"),
    "document": ("Documents", "#ffbd73"),
    "data": ("Data", "#84d5ad"),
    "note": ("Notes", "#f08ce2"),
    "chat": ("Chats", "#b2cd7b"),
    "image": ("Images", "#ec9f9c"),
    "media": ("Media", "#f4d674"),
    "other": ("Other", "#a4acbf"),
}
EDGE_LABELS = {
    "contains": "Contains",
    "reference": "File reference",
    "citation": "Citation",
    "duplicate": "Verified duplicate",
    "commit_parent": "Git parent",
    "changed_file": "Git changed file",
    "inferred": "Inferred topic",
    "topic": "Inferred topic",
    "semantic": "Similarity",
}
FOLDER_KINDS = frozenset({"root", "drive", "folder", "directory", "project", "repository"})
INFERRED_KINDS = frozenset({"inferred", "topic", "semantic"})
SOURCE_EDGE_KINDS = {"reference", "citation", "duplicate"} | INFERRED_KINDS
EDGE_COLOURS = {
    "contains": "#8063aa",
    "reference": "#c7a7f8",
    "citation": "#e0a9ee",
    "duplicate": "#91cdb7",
    "commit_parent": "#8ec6f2",
    "changed_file": "#b9cafa",
    "inferred": "#efb578",
    "topic": "#efb578",
    "semantic": "#b18bde",
}


def group(node):
    if node.get("kind") in FOLDER_KINDS:
        return "folder"
    return node.get("category") if node.get("category") in GROUPS else "other"


def edge_label(kind):
    return EDGE_LABELS.get(kind, str(kind).replace("_", " ").capitalize())


def node_label(node):
    name = str(node.get("name") or node.get("id"))
    count = node.get("descendants") or 0
    state = node.get("state")
    if count:
        name += f" · {count:,} items"
    if state in GRAPH_BLOCKED:
        name += f" · {str(state).replace('_', ' ')}"
    return name


def edge_evidence(edge):
    evidence = edge.get("evidence")
    return str(evidence or "").strip()[:400]


def edge_key(edge):
    return edge["source"], edge["target"], edge["kind"], bool(edge.get("bundle"))


def anchor(identity):
    return "item-" + blake2b(identity.encode(), digest_size=12).hexdigest()


def placement_group(node):
    category = group(node)
    drive = PureWindowsPath(str(node.get("path") or "")).drive if category == "folder" else ""
    return category, drive


def locate(nodes, previous):
    """Keep surviving points anchored. Allocate newcomers to free group slots."""
    groups = defaultdict(list)
    for node in nodes:
        groups[placement_group(node)].append(node["id"])
    positions = {}
    active = sorted(groups, key=lambda cluster: (list(GROUPS).index(cluster[0]), cluster[1]))
    if not active:
        return positions
    for index, category in enumerate(active):
        angle = (2 * math.pi * index / len(active)) - math.pi / 2
        radius = 0 if len(active) == 1 else 235
        centre_x = radius * math.cos(angle)
        centre_y = radius * math.sin(angle)
        occupied = set()
        for identity in groups[category]:
            if (old := previous.get(identity)) and old[2] == category:
                positions[identity] = old
                occupied.add(old[:2])
        candidates = iter(range(len(groups[category]) * 2 + 1))
        for identity in groups[category]:
            if identity in positions:
                continue
            for slot in candidates:
                theta = slot * math.pi * (3 - math.sqrt(5))
                distance = 21 * math.sqrt(slot)
                point = (
                    centre_x + distance * math.cos(theta),
                    centre_y + distance * math.sin(theta),
                )
                if point not in occupied:
                    positions[identity] = (*point, category)
                    occupied.add(point)
                    break
    return positions


def topic_projection(reader, value, scope, offset, generation):
    """Project only topic memberships whose sources still exist in the catalogue."""
    topics = value["topics"]
    topic_ids = {"neural-topic:" + topic["id"]: topic for topic in topics}
    selected = list(topic_ids) if scope is None else [scope] if scope in topic_ids else []
    selected = selected[offset : offset + PAGE_SIZE] if scope is None else selected
    nodes, edges, hubs = {}, [], []
    current_members = set()
    shown_members = set()
    scoped_total = 0
    for identity in selected:
        topic = topic_ids[identity]
        members = {}
        for member in topic["members"]:
            node = reader.node(member["node_id"])
            if node and node.get("state") not in GRAPH_BLOCKED:
                members[node["id"]] = node
        current_members.update(members)
        hub = {
            "id": identity,
            "name": topic["label"],
            "kind": "topic",
            "category": "topic",
            "descendants": len(members),
            "cluster_id": identity,
        }
        nodes[identity] = hub
        hubs.append(identity)
        scoped_total = len(members)
        page = (
            list(members.values())[offset : offset + PAGE_SIZE]
            if scope
            else list(members.values())[:8]
        )
        for node in page:
            nodes.setdefault(node["id"], {**node, "cluster_id": identity})
            shown_members.add(node["id"])
    projection = {
        "generation": generation,
        "neural_generation": value.get("generation"),
        "nodes": list(nodes.values()),
        "edges": edges,
        "hubs": hubs,
        "scope": nodes.get(scope),
        "total": scoped_total if scope else len(topic_ids),
        "counts": {
            "topics": len(hubs),
            "members": len(current_members),
            "shown": len(shown_members),
        },
    }
    return similarity_overlay(projection, value)


def similarity_overlay(projection, topics):
    visible = {node["id"] for node in projection["nodes"] if node.get("state") not in GRAPH_BLOCKED}
    edges = list(projection["edges"])
    shown = set()
    for edge in topics.get("edges", ()):
        pair = edge["source"], edge["target"]
        identity = tuple(sorted(pair))
        if not set(pair) <= visible or identity in shown or pair[0] == pair[1]:
            continue
        shown.add(identity)
        score = edge.get("score", edge.get("weight"))
        evidence = str(edge.get("evidence") or "Current indexed source similarity")
        edges.append(
            {
                **edge,
                "kind": "semantic",
                "score": score,
                "evidence": f"Similarity {score:.2f}. {evidence}. Derived link; verify source passages.",
            }
        )
        if len(shown) >= 200:
            break
    return {**projection, "edges": edges, "neural_generation": topics.get("generation")}


@dataclass
class Diagram:
    generation: int
    nodes: dict
    edges: list
    positions: dict
    total: int
    scope: dict | None
    links: dict
    hubs: tuple
    counts: dict
    graph_revision: int | None = None
    neural_generation: int | None = None

    @classmethod
    def from_projection(cls, value, previous=None):
        nodes = {node["id"]: node for node in value["nodes"]}
        edges = []
        links = defaultdict(list)
        for edge in value.get("edges", ()):
            source, target = edge["source"], edge["target"]
            if source in nodes and target in nodes:
                edges.append(edge)
                links[source].append((target, edge))
                if source != target:
                    links[target].append((source, edge))
        hubs = tuple(value.get("hubs", ()))
        scope = value.get("scope")
        if hubs and scope and scope["id"] in nodes and scope["id"] not in hubs:
            hubs = (scope["id"], *hubs)
        previous = previous or {}
        positions = (
            cluster_positions(nodes, edges, hubs, previous)
            if hubs
            else locate(nodes.values(), previous)
        )
        return cls(
            value["generation"],
            nodes,
            edges,
            positions,
            value.get("total", len(nodes)),
            value.get("scope"),
            links,
            hubs,
            value.get("counts", {}),
            value.get("graph_revision"),
            value.get("neural_generation"),
        )

    def matches_status(self, status, mode):
        if "neural_generation" in status and self.neural_generation != status["neural_generation"]:
            return False
        if mode == "topics":
            return (
                self.neural_generation is not None
                and self.neural_generation == status.get("neural_generation")
                and self.generation == status["generation"]
            )
        graph_revision = status.get("graph_revision") if mode == "browse" else None
        if graph_revision is not None and self.graph_revision is not None:
            return self.graph_revision == graph_revision
        return self.generation == status["generation"]

    def is_newer_than(self, diagram):
        if self.neural_generation != diagram.neural_generation:
            return True
        if self.graph_revision is not None and diagram.graph_revision is not None:
            return self.graph_revision > diagram.graph_revision
        return self.generation > diagram.generation


class GraphSurface(QWidget):
    selected = Signal(str)
    expanded = Signal(str)
    edge_selected = Signal(object)
    detail_requested = Signal(str)
    overview_requested = Signal()

    def __init__(self):
        super().__init__()
        self.setAccessibleName("Catalogue diagram. Use the items list for keyboard navigation.")
        self.setMinimumSize(310, 250)
        self.diagram = None
        self.selection = None
        self.selected_edge = None
        self.zoom = 1.0
        self._fit_zoom = 1.0
        self.pan = QPointF()
        self._press = None
        self._previous = None
        self._moved = False
        self._strokes = {}
        self._dots = {}
        self._labels = ()
        self._hubs = frozenset()

    def set_diagram(self, diagram, *, fit=False):
        self.diagram = diagram
        self._build_geometry()
        if fit:
            self.fit()
        else:
            self.update()

    def _build_geometry(self):
        strokes = defaultdict(QPainterPath)
        dots = defaultdict(QPolygonF)
        labels = []
        diagram = self.diagram
        self._hubs = frozenset(diagram.hubs)
        for edge in diagram.edges:
            source, target = edge["source"], edge["target"]
            count = edge.get("count", 1)
            weight = min(4, int(math.log2(count))) if edge.get("bundle") else 0
            path = strokes[edge["kind"], weight]
            path.moveTo(*diagram.positions[source][:2])
            path.lineTo(*diagram.positions[target][:2])
        for identity, node in diagram.nodes.items():
            cluster = node.get("cluster_id")
            is_hub = identity in self._hubs
            folder = node.get("kind") in FOLDER_KINDS
            count = node.get("descendants") or 0
            if cluster:
                number = int.from_bytes(blake2b(cluster.encode(), digest_size=2).digest(), "big")
                colour = CLUSTER_COLOURS[number % len(CLUSTER_COLOURS)]
                size = 7 if folder else 5
                if is_hub:
                    size = min(18, 12 + int(math.log2(1 + count)) // 2)
                    colour = QColor(colour).lighter(142).name()
            else:
                colour = GROUPS[group(node)][1]
                size = min(18, 6 + int(math.log2(1 + count)))
            dots[colour, size].append(QPointF(*diagram.positions[identity][:2]))
            labels.append((not is_hub, not folder, -count, identity))
        self._strokes = dict(strokes)
        self._dots = dict(dots)
        self._labels = tuple(identity for *_, identity in sorted(labels))

    def fit(self):
        if not self.diagram or not self.diagram.positions:
            self.zoom, self.pan = 1.0, QPointF()
        else:
            xs, ys, _ = zip(*self.diagram.positions.values(), strict=True)
            self.zoom = min(
                1.8,
                self.width() / (max(xs) - min(xs) + 140),
                self.height() / (max(ys) - min(ys) + 110),
            )
            self.pan = QPointF(-(max(xs) + min(xs)) / 2, -(max(ys) + min(ys)) / 2) * self.zoom
        self._fit_zoom = self.zoom
        self.update()

    def _centre(self):
        return QPointF(self.width() / 2, self.height() / 2) + self.pan

    def _points(self):
        centre = self._centre()
        return {
            identity: centre + QPointF(x, y) * self.zoom
            for identity, (x, y, _category) in self.diagram.positions.items()
        }

    def _hit(self, where):
        if not self.diagram:
            return None
        matches = []
        for identity, point in self._points().items():
            distance = (point - where).manhattanLength()
            if distance <= (11 if identity in self._hubs else 7):
                matches.append((distance, identity))
        return min(matches)[1] if matches else None

    def _hit_edge(self, where):
        if not self.diagram:
            return None
        world = (where - self._centre()) / self.zoom
        max_distance = (6 / self.zoom) ** 2
        best = None
        for edge in self.diagram.edges:
            ax, ay, _ = self.diagram.positions[edge["source"]]
            bx, by, _ = self.diagram.positions[edge["target"]]
            dx, dy = bx - ax, by - ay
            length = dx * dx + dy * dy
            if not length:
                continue
            ratio = ((world.x() - ax) * dx + (world.y() - ay) * dy) / length
            ratio = min(1, max(0, ratio))
            distance = (world.x() - ax - ratio * dx) ** 2 + (world.y() - ay - ratio * dy) ** 2
            if distance < max_distance:
                best, max_distance = edge, distance
        return best

    def paintEvent(self, _event):
        painter = QPainter(self)
        palette = self.palette()
        background = QColor(GRAPH_BACKGROUND)
        painter.fillRect(self.rect(), background)
        if not self.diagram or not self.diagram.nodes:
            painter.setPen(palette.text().color())
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "No catalogue items in this view"
            )
            return
        centre = self._centre()
        painter.save()
        painter.translate(centre)
        painter.scale(self.zoom, self.zoom)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        for (kind, weight), path in self._strokes.items():
            colour = QColor(EDGE_COLOURS.get(kind, "#9e8bbd"))
            colour.setAlpha(145 if kind == "contains" else 175)
            pen = QPen(colour, 1 + weight * 0.4)
            pen.setCosmetic(True)
            if kind in INFERRED_KINDS:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawPath(path)
        active = (
            [self.selected_edge]
            if self.selected_edge
            else [edge for _, edge in self.diagram.links.get(self.selection, ())]
        )
        for inferred in (False, True):
            lines = [
                QLineF(
                    *self.diagram.positions[edge["source"]][:2],
                    *self.diagram.positions[edge["target"]][:2],
                )
                for edge in active
                if (edge["kind"] in INFERRED_KINDS) == inferred
                and edge["source"] in self.diagram.positions
                and edge["target"] in self.diagram.positions
            ]
            if not lines:
                continue
            pen = QPen(QColor("#e1ceff"), 2.0)
            pen.setCosmetic(True)
            if inferred:
                pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLines(lines)
        for (colour, size), batch in self._dots.items():
            pen = QPen(QColor(colour), size)
            pen.setCosmetic(True)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.setPen(pen)
            painter.drawPoints(batch)
        if self.selection in self.diagram.positions:
            ring = QPen(QColor("#f2e9ff"), 1.4)
            ring.setCosmetic(True)
            painter.setPen(ring)
            radius = (10 if self.selection in self._hubs else 6) / self.zoom
            painter.drawEllipse(
                QPointF(*self.diagram.positions[self.selection][:2]), radius, radius
            )
        painter.restore()
        painter.setPen(palette.text().color())
        metrics = painter.fontMetrics()
        written = []
        viewport = QRectF(self.rect())
        candidates = (self.selection, *self._labels) if self.selection else self._labels
        detailed = self.zoom >= self._fit_zoom * 2.2
        for identity in candidates:
            if len(written) >= 32:
                break
            if identity not in self.diagram.nodes:
                continue
            node = self.diagram.nodes[identity]
            show_label = identity == self.selection or identity in self._hubs or detailed
            if not show_label and (
                self._hubs or (self.zoom < 1.15 and not node.get("descendants"))
            ):
                continue
            point = centre + QPointF(*self.diagram.positions[identity][:2]) * self.zoom
            if not viewport.contains(point):
                continue
            text = metrics.elidedText(node_label(node), Qt.TextElideMode.ElideRight, 175)
            rect = QRectF(
                point.x() + 12,
                point.y() - 9,
                metrics.horizontalAdvance(text) + 4,
                metrics.height() + 4,
            )
            if not any(rect.intersects(other) for other in written):
                painter.fillRect(rect, background)
                painter.drawText(rect, Qt.AlignmentFlag.AlignVCenter, text)
                written.append(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._press = self._previous = event.position()
            self._moved = False
            event.accept()

    def mouseMoveEvent(self, event):
        if self._press is not None:
            self._moved |= (event.position() - self._press).manhattanLength() > 4
            if self._moved:
                self.pan += event.position() - self._previous
                self.update()
            self._previous = event.position()

    def mouseReleaseEvent(self, event):
        if self._press is not None and not self._moved:
            if hit := self._hit(event.position()):
                self.selected.emit(hit)
            elif edge := self._hit_edge(event.position()):
                self.edge_selected.emit(edge)
        self._press = self._previous = None

    def mouseDoubleClickEvent(self, event):
        if hit := self._hit(event.position()):
            self.expanded.emit(hit)

    def wheelEvent(self, event):
        direction = event.angleDelta().y() / 120
        old_zoom = self.zoom
        zoom = max(0.12, min(8.0, self.zoom * 1.18**direction))
        self.pan += (event.position() - self._centre()) * (1 - zoom / self.zoom)
        self.zoom = zoom
        if self._hubs and direction > 0 and zoom >= self._fit_zoom * 1.55:
            where = event.position()
            points = self._points()
            folders = [
                ((point - where).manhattanLength(), identity)
                for identity, point in points.items()
                if self.diagram.nodes[identity].get("kind") in FOLDER_KINDS
            ]
            if folders:
                distance, identity = min(folders)
                if distance <= max(36, min(110, 64 * zoom)):
                    self.detail_requested.emit(identity)
        elif direction < 0 and zoom < self._fit_zoom * 1.25 <= old_zoom:
            self.overview_requested.emit()
        self.update()
        event.accept()


def mind_map_html(diagram):
    title = str((diagram.scope or {}).get("name", "Whole library"))
    groups = defaultdict(list)
    for node in diagram.nodes.values():
        groups[group(node)].append(node)
    sections = []
    for category, nodes in groups.items():
        entries = []
        for node in nodes:
            path = f"<br><small>{escape(str(node.get('path') or ''))}</small>"
            related = diagram.links.get(node["id"], ())[:12]
            links = "".join(
                f'<li>{escape(edge_label(edge["kind"]))} · <a href="#{anchor(other)}">'
                f"{escape(node_label(diagram.nodes[other]))}</a></li>"
                for other, edge in related
            )
            identity = anchor(node["id"])
            entries.append(
                f'<li id="{identity}"><b><a href="#{identity}">{escape(node_label(node))}</a></b>'
                f"{path}<ul>{links}</ul></li>"
            )
        sections.append(f"<h2>{GROUPS[category][0]}</h2><ul>{''.join(entries)}</ul>")
    page = f"{len(diagram.nodes):,} of {diagram.total:,} items in this view"
    return (
        f'<!doctype html><html lang="en"><meta charset="utf-8"><title>{escape(title)}</title>'
        f"<h1>{escape(title)}</h1><p>Catalogue revision {diagram.generation} · {page}</p>"
        f"{''.join(sections)}</html>"
    )


def canvas_board(diagram):
    """Generate an editable copy of the visible page using ordinary Canvas cards."""
    cards = []
    identities = {}
    # The authored Canvas validator permits 250 cards and 500 edges.
    for identity, node in list(diagram.nodes.items())[:250]:
        card_id = blake2b(identity.encode(), digest_size=12).hexdigest()
        identities[identity] = card_id
        x, y, _category = diagram.positions[identity]
        text = f"{node_label(node)}\n{node.get('path') or ''}"[:3000]
        cards.append(
            {
                "id": card_id,
                "type": "text",
                "x": round(x * 10),
                "y": round(y * 8),
                "width": 185,
                "height": 106,
                "text": text,
            }
        )
    edges = []
    for edge in diagram.edges:
        source, target = edge["source"], edge["target"]
        if source not in identities or target not in identities or source == target:
            continue
        edges.append(
            {
                "id": f"connection-{len(edges)}",
                "fromNode": identities[source],
                "toNode": identities[target],
                "label": edge_label(edge["kind"]),
            }
        )
        if len(edges) == 500:
            break
    return {"nodes": cards, "edges": edges}


class _Reply(QObject):
    ready = Signal(int, object, object)


class _Read(QRunnable):
    def __init__(self, serial, action, receiver):
        super().__init__()
        self.serial, self.action, self.receiver = serial, action, receiver

    def run(self):
        try:
            value, error = self.action(), None
        except (OSError, ValueError, RuntimeError, sqlite3.Error) as exc:
            value, error = None, exc
        # The application can close while an index read is in progress.
        with suppress(RuntimeError):
            self.receiver.ready.emit(self.serial, value, error)


@dataclass(eq=False)
class _Preview:
    key: tuple
    diagram: Diagram
    update: Callable[[Diagram], None]


class WholeLibraryGraph(QWidget):
    source_requested = Signal(object)
    scope_changed = Signal(object)
    canvas_requested = Signal(object)
    mind_map_requested = Signal(str, str)

    def __init__(self, reader, *, neural=None):
        super().__init__()
        self.reader = reader
        self.neural = neural
        self.diagram = None
        self.scope = None
        self.offset = 0
        self.mode = "browse"
        self.selection = None
        self._pending_selection = None
        self.expanded = []
        self._active_edge = None
        self._evidence_node = None
        self.trail = []
        self._serial = 0
        self._running = False
        self._pending = None
        self._requested_key = None
        self._reply = _Reply(self)
        self._reply.ready.connect(self._ready)
        self._relation_reply = _Reply(self)
        self._relation_reply.ready.connect(self._relations_ready)
        self._relation_serial = 0
        self._relation_running = False
        self._relation_pending = None
        self._shown_relationships = set()
        self._generated_pending = None
        self._previews = []
        self._preview_running = False
        self._preview_reply = _Reply(self)
        self._preview_reply.ready.connect(self._preview_ready)
        self._cache = {}
        self._path_reply = _Reply(self)
        self._path_reply.ready.connect(self._paths_ready)
        self._path_running = False
        self.surface = GraphSurface()
        self.surface.selected.connect(self._select)
        self.surface.expanded.connect(self._surface_activate)
        self.surface.detail_requested.connect(self._expand)
        self.surface.overview_requested.connect(self._collapse)
        self.surface.edge_selected.connect(self._inspect_edge)
        self.title = label("Whole library", "section")
        self.title.setWordWrap(True)
        self.legend = label("", "muted")
        self.legend.setTextFormat(Qt.TextFormat.RichText)
        self.legend.setWordWrap(True)
        self.summary = label("Waiting for the local catalogue.", "muted")
        self.summary.setWordWrap(True)
        self.search_box = QLineEdit(
            placeholderText="Find a project, folder or file", accessibleName="Search catalogue"
        )
        self.search_box.returnPressed.connect(self._search)
        self.inferred = QCheckBox("Show similarity links", checked=True)
        self.inferred.toggled.connect(self.refresh)
        self.grouping = QComboBox(accessibleName="Graph grouping")
        self.grouping.addItem("Projects / Folders", "browse")
        self.grouping.addItem("Topics", "topics")
        self.grouping.setItemData(
            1, "Derived locally from current indexed sources", Qt.ItemDataRole.ToolTipRole
        )
        self.grouping.currentIndexChanged.connect(self._change_grouping)
        self.question_paths = button("Question paths")
        self.question_paths.setEnabled(neural is not None)
        self.question_paths.clicked.connect(self._show_paths)
        self.items = QListWidget(accessibleName="Catalogue items")
        self.items.currentItemChanged.connect(self._current)
        self.items.itemActivated.connect(lambda item: self._activate(item.data(ROLE)))
        self.relationships = QListWidget(
            accessibleName="Evidence and relationships for the selected item"
        )
        self.relationships.itemActivated.connect(self._activate_relationship)
        self.detail = label("Select an item to inspect its relationships.", "muted")
        self.detail.setTextFormat(Qt.TextFormat.PlainText)
        self.detail.setWordWrap(True)
        self.home = button("Whole library")
        self.home.clicked.connect(self._home)
        self.back = button("Back")
        self.back.clicked.connect(self._back)
        self.previous = button("Previous page")
        self.previous.clicked.connect(lambda: self._page(-PAGE_SIZE))
        self.next = button("Next page")
        self.next.clicked.connect(lambda: self._page(PAGE_SIZE))
        self.fit = button("Fit")
        self.fit.setToolTip("Fit the graph and arrange clusters using the current links")
        self.fit.clicked.connect(self._fit_graph)
        self.maps = button("Mind map")
        self.maps.clicked.connect(self.show_mind_map)
        self.canvas = button("Generated Canvas")
        self.canvas.clicked.connect(self.show_generated_canvas)
        self.git = button("Git history")
        self.git.clicked.connect(self._git_history)
        self.focus = button("Focus connections")
        self.focus.clicked.connect(self._focus)
        self.open_source = button("Open source")
        self.open_source.clicked.connect(self._open_source)

        top = QHBoxLayout()
        for widget in (
            self.home,
            self.back,
            self.grouping,
            self.search_box,
            self.fit,
            self.inferred,
        ):
            top.addWidget(widget, int(widget is self.search_box))
        actions = QHBoxLayout()
        for widget in (
            self.maps,
            self.canvas,
            self.git,
            self.focus,
            self.open_source,
            self.question_paths,
        ):
            actions.addWidget(widget)
        sidebar = QWidget()
        side = QVBoxLayout(sidebar)
        side.setContentsMargins(4, 0, 0, 0)
        side.addWidget(label("Items", "section"))
        side.addWidget(self.items, 2)
        side.addWidget(self.detail)
        side.addWidget(self.relationships, 1)
        split = QSplitter()
        split.addWidget(self.surface)
        split.addWidget(sidebar)
        split.setSizes([660, 315])
        split.setStretchFactor(0, 3)
        bottom = QHBoxLayout()
        bottom.addWidget(self.summary, 1)
        bottom.addWidget(self.previous)
        bottom.addWidget(self.next)
        root = QVBoxLayout(self)
        root.addWidget(self.title)
        root.addLayout(top)
        root.addLayout(actions)
        root.addWidget(split, 1)
        root.addWidget(self.legend)
        root.addLayout(bottom)
        self._buttons()

    def _read(self, operation):
        self._serial += 1
        self._pending = (self._serial, operation)
        if not self._running:
            self._start()

    def _start(self):
        if self._pending is not None:
            serial, operation = self._pending
            self._pending = None
            self._running = True
            QThreadPool.globalInstance().start(_Read(serial, operation, self._reply))

    def _ready(self, serial, result, error):
        self._running = False
        if serial == self._serial:
            if error:
                if self.mode == "topics":
                    self.diagram = self.surface.diagram = None
                    self.items.clear()
                    self.relationships.clear()
                    self.surface.update()
                    self.title.setText("Topics unavailable")
                    self._buttons()
                self.summary.setText(f"Graph view unavailable. {error}")
            else:
                self._present(*result)
        self._start()

    def refresh(self):
        mode, scope, offset = self.mode, self.scope, self.offset
        query, inferred = self.search_box.text().strip(), self.inferred.isChecked()
        key = (
            mode,
            scope,
            offset,
            query if mode == "search" else "",
            inferred,
            tuple(self.expanded),
        )
        if self._running and self._requested_key == key:
            return
        self._requested_key = key
        view_key = getattr(self, "_view_key", None)
        cached = self._cache.get(key)
        previous = cached.positions if cached else {}
        if self.diagram and view_key and view_key[:4] == key[:4]:
            previous = self.diagram.positions

        def build():
            status = getattr(self.reader, "graph_status", self.reader.status)()
            topics = None
            if mode == "topics":
                if self.neural is None:
                    raise OSError("Topics need the local neural service.")
                topics = self.neural.topics()
                status = {**status, "neural_generation": topics.get("generation")}
            elif mode == "browse" and self.neural is not None and inferred:
                try:
                    topics = self.neural.topics()
                except NeuralError:
                    status = {**status, "neural_generation": None}
                else:
                    status = {**status, "neural_generation": topics.get("generation")}
            if cached and cached.matches_status(status, mode):
                return cached, key, status
            return self._diagram_for(key, previous, status["generation"], topics), key, status

        self._read(build)

    def refresh_previews(self, *, graph_visible=False):
        """Refresh open projections even when the user leaves their Graph scope."""
        if self._preview_running:
            return
        active_key = self._requested_key if graph_visible else None
        projections = {
            preview.key: preview.diagram for preview in self._previews if preview.key != active_key
        }
        if not projections:
            return
        self._preview_running = True

        def build():
            graph_status = getattr(self.reader, "graph_status", None)
            revision = getattr(self.reader, "revision", None)
            if graph_status and any(key[0] == "browse" for key in projections):
                status = graph_status()
            else:
                status = {"generation": revision()} if revision else self.reader.status()
            return {
                key: self._diagram_for(key, diagram.positions, status["generation"])
                for key, diagram in projections.items()
                if not diagram.matches_status(status, key[0])
            }

        QThreadPool.globalInstance().start(_Read(0, build, self._preview_reply))

    def _preview_ready(self, _serial, diagrams, error):
        self._preview_running = False
        if not error:
            self._update_previews(diagrams)

    def _update_previews(self, diagrams):
        for preview in self._previews[:]:
            diagram = diagrams.get(preview.key)
            if diagram and diagram.is_newer_than(preview.diagram):
                preview.diagram = diagram
                preview.update(diagram)

    def _diagram_for(self, key, previous, generation, topics=None):
        mode, scope, offset, query, inferred, expanded = key
        if mode == "topics":
            value = topic_projection(
                self.reader, topics or self.neural.topics(), scope, offset, generation
            )
        elif mode == "focus":
            value = self._neighbourhood(scope, inferred, generation)
        elif mode == "git":
            value = self.reader.git_history(scope, offset=offset, limit=PAGE_SIZE)
        elif mode == "search":
            found = self.reader.search(query, limit=PAGE_SIZE, scope=scope)
            value = {
                "generation": generation,
                "nodes": found,
                "edges": [],
                "total": len(found),
                "scope": None,
            }
        else:
            graph_projection = getattr(self.reader, "graph_projection", None)
            if graph_projection:
                value = graph_projection(
                    scope,
                    offset=offset,
                    limit=PAGE_SIZE,
                    expanded=expanded,
                    include_inferred=inferred,
                )
            else:
                value = self.reader.projection(scope, offset=offset, limit=PAGE_SIZE)
            selected = value.get("scope")
            if selected and selected.get("kind") not in FOLDER_KINDS:
                value = self._neighbourhood(selected["id"], inferred, value["generation"])
        if not inferred:
            value = {
                **value,
                "edges": [edge for edge in value["edges"] if edge["kind"] not in INFERRED_KINDS],
            }
        elif mode == "browse" and topics is not None:
            value = similarity_overlay(value, topics)
        return Diagram.from_projection(value, previous)

    def _neighbourhood(self, identity, inferred, generation):
        node = self.reader.node(identity)
        neighbours = (
            self.reader.neighbours(identity, limit=PAGE_SIZE - 1, include_inferred=inferred)
            if node
            else []
        )
        nodes = [node] if node else []
        edges = []
        for item in neighbours:
            other = item["node"]
            nodes.append(other)
            edges.append(
                {
                    "source": item.get("source", identity),
                    "target": item.get("target", other["id"]),
                    "kind": item["kind"],
                    "evidence": item.get("evidence"),
                }
            )
        return {
            "generation": generation,
            "nodes": nodes,
            "edges": edges,
            "total": len(nodes),
            "scope": node,
        }

    def _present(self, diagram, key, status):
        previous_key = getattr(self, "_view_key", None)
        unchanged = self.diagram is diagram
        fit = (
            previous_key is None
            or previous_key[:4] != key[:4]
            or not self.diagram
            or not self.diagram.nodes
        )
        self._view_key, self.diagram = key, diagram
        if self._pending_selection == key[:2]:
            self.selection = key[1]
            self._pending_selection = None
        self.grouping.blockSignals(True)
        self.grouping.setCurrentIndex(1 if key[0] == "topics" else 0)
        self.grouping.blockSignals(False)
        self._cache.pop(key, None)
        self._cache[key] = diagram
        if len(self._cache) > 12:
            self._cache.pop(next(iter(self._cache)))
        name = (diagram.scope or {}).get("name", "Whole library")
        if self.mode == "topics":
            name = (diagram.scope or {}).get("name", "Topics")
        elif self.mode == "search":
            name = f"Search · {self.search_box.text().strip()}"
        elif self.mode in {"git", "focus"}:
            name = f"{'Git history' if self.mode == 'git' else 'Connections'} · {name}"
        self.title.setText(str(name))
        total = status.get("nodes", 0)
        progress = str(status.get("message") or "").strip()
        if self.mode == "topics":
            summary = (
                f"{diagram.counts['topics']:,} topics shown · "
                f"{diagram.counts['members']:,} current topic members · "
                f"{diagram.counts['shown']:,} members shown"
            )
            if not diagram.nodes:
                summary += " · Topics are building or no current sources have topic membership"
        elif diagram.hubs:
            scope_total = diagram.counts.get("nodes", total)
            contains = diagram.counts.get("containment", 0)
            bundled = diagram.counts.get("bundled", 0)
            direct = sum(
                edge["kind"] not in INFERRED_KINDS | {"contains"} and not edge.get("bundle")
                for edge in diagram.edges
            )
            summary = (
                f"{len(diagram.nodes):,} shown · {scope_total:,} items · "
                f"{contains:,} folder links · {direct:,} direct links shown · "
                f"{bundled:,} cross-cluster links on this page"
            )
        else:
            span = (
                f"{self.offset + 1:,}–{self.offset + len(diagram.nodes):,}"
                if diagram.nodes
                else "0"
            )
            summary = f"{span} of {diagram.total:,} here · {total:,} catalogue items"
        if status.get("inventory_pending"):
            summary += f" · Inventory pending on {status['inventory_pending']:,} roots"
        if status.get("content_pending"):
            summary += f" · {status['content_pending']:,} files awaiting content extraction"
        if self.mode != "topics":
            similarities = sum(edge["kind"] in INFERRED_KINDS for edge in diagram.edges)
            summary += f" · {similarities:,} similarity links shown"
        self.summary.setText(summary + (f" · {progress}" if progress else ""))
        if unchanged:
            self._update_previews({key: diagram})
            self._show_pending()
            return
        self.items.blockSignals(True)
        visible_groups = {group(node) for node in diagram.nodes.values()}
        self.legend.setText(
            "Folders and projects &nbsp; · &nbsp; Violet lines: folder structure &nbsp; "
            "· &nbsp; Bright lines: source links &nbsp; · &nbsp; Scroll to reveal files"
            if diagram.hubs
            else " &nbsp; ".join(
                f'<span style="color:{colour}">●</span> {caption}'
                for category, (caption, colour) in GROUPS.items()
                if category in visible_groups
            )
        )
        self.legend.setText(
            self.legend.text() + " &nbsp; · &nbsp; Solid: verified structure / source links"
            " &nbsp; · &nbsp; Dashed: derived similarity, inspect before using as evidence"
        )
        self.items.clear()
        selection = self.selection if self.selection in diagram.nodes else None
        for node in diagram.nodes.values():
            row_item(self.items, node_label(node), node["id"])
            if node["id"] == selection:
                self.items.setCurrentRow(self.items.count() - 1)
        self.items.blockSignals(False)
        self.surface.set_diagram(diagram, fit=fit)
        previous_edge = edge_key(self._active_edge) if self._active_edge else None
        self._select(selection)
        if (
            previous_edge
            and not fit
            and (
                edge := next(
                    (edge for edge in diagram.edges if edge_key(edge) == previous_edge), None
                )
            )
        ):
            self._inspect_edge(edge)
        self._update_previews({key: diagram})
        self._show_pending()

    def _show_pending(self):
        if (generated := self._generated_pending) is not None:
            self._generated_pending = None
            if generated == "map":
                self.show_mind_map()
            else:
                self.show_generated_canvas()

    def _current(self, item, _previous):
        self._select(item.data(ROLE) if item else None)

    def _select(self, identity):
        self._active_edge = self.surface.selected_edge = self._evidence_node = None
        self.selection = identity
        self.surface.selection = identity
        self.surface.update()
        for index in range(self.items.count()):
            if identity and self.items.item(index).data(ROLE) == identity:
                self.items.blockSignals(True)
                self.items.setCurrentRow(index)
                self.items.blockSignals(False)
                break
        self.relationships.clear()
        self._shown_relationships.clear()
        self._relation_serial += 1
        self._relation_pending = None
        node = self.diagram.nodes.get(identity) if self.diagram else None
        if node:
            self.scope_changed.emit(node)
            self.detail.setText(str(node.get("path") or node_label(node)))
            if node.get("kind") == "commit":
                self.detail.setText(f"Commit {node.get('digest', '')}")
                for ref in node.get("refs", ()):
                    row_item(self.relationships, f"Git ref · {ref}", None)
                for path in node.get("changed_files", ()):
                    row_item(self.relationships, f"Changed file · {path}", None)
            for other, edge in self.diagram.links.get(identity, ()):
                self._show_relationship(self.diagram.nodes[other], edge)
            if self.mode != "git" and node.get("kind") != "topic":
                self._relation_pending = (
                    self._relation_serial,
                    identity,
                    self.inferred.isChecked(),
                )
                self._start_relations()
        else:
            self.detail.setText("Select an item to inspect its relationships.")
        self._buttons()

    def _inspect_edge(self, edge):
        if not self.diagram or not edge:
            return
        source = self.diagram.nodes.get(edge["source"])
        target = self.diagram.nodes.get(edge["target"])
        if other := edge.get("other_node"):
            if other["id"] == edge["source"]:
                source = other
            elif other["id"] == edge["target"]:
                target = other
        if not source or not target:
            return
        self._active_edge = self.surface.selected_edge = edge
        self._evidence_node = None
        self.surface.selection = None
        self.surface.update()
        self._relation_serial += 1
        self._relation_pending = None
        self.relationships.clear()
        count = edge.get("count", 1)
        self.detail.setText(
            f"{edge_label(edge['kind'])} · {count:,} link{'s' if count != 1 else ''}\n"
            f"{source['name']} → {target['name']}"
        )
        if edge["kind"] == "semantic":
            row_item(self.relationships, edge_evidence(edge), None)
            row_item(self.relationships, "Open " + node_label(target), target)
            self._evidence_node = target
        elif edge["kind"] in SOURCE_EDGE_KINDS and hasattr(self.reader, "graph_evidence"):
            row_item(self.relationships, "Checking current sources…", None)
            self._relation_pending = self._relation_serial, edge, False
            self._start_relations()
        else:
            row_item(self.relationships, "Open " + node_label(target), target)
        self._buttons()

    def _start_relations(self):
        if self._relation_running or self._relation_pending is None:
            return
        serial, identity, include_inferred = self._relation_pending
        self._relation_pending = None
        self._relation_running = True

        def read():
            if isinstance(identity, dict):
                return self.reader.graph_evidence(identity)
            return self.reader.neighbours(identity, limit=40, include_inferred=include_inferred)

        QThreadPool.globalInstance().start(_Read(serial, read, self._relation_reply))

    def _relations_ready(self, serial, rows, error):
        self._relation_running = False
        if serial == self._relation_serial:
            if self._active_edge:
                self._show_edge_sources(rows if not error else [])
            elif error:
                row_item(self.relationships, "Direct relationships unavailable", None)
            else:
                for row in rows:
                    self._show_relationship(row["node"], row)
        self._start_relations()

    def _show_edge_sources(self, rows):
        self.relationships.clear()
        self._evidence_node = None
        if not rows:
            row_item(self.relationships, "No current source evidence is available.", None)
        for row in rows:
            node, target = row["node"], row["target"]
            source = row.get("source")
            if source:
                locator = source.get("locator") or {}
                node = {**node, "match_start": locator.get("start") or locator.get("page") or 1}
                self._evidence_node = self._evidence_node or node
            caption = f"{node['name']} → {target['name']}"
            if evidence := edge_evidence(row):
                caption += f"\n{evidence}"
            row_item(self.relationships, caption, node if source else None)
        self._buttons()

    def _show_relationship(self, node, edge):
        kind, evidence = edge["kind"], edge_evidence(edge)
        identity = node["id"], kind, evidence
        if identity in self._shown_relationships:
            return
        self._shown_relationships.add(identity)
        text = f"{edge_label(kind)} · {node_label(node)}"
        if (count := edge.get("count", 1)) > 1:
            text += f" · {count:,} links"
        if evidence:
            text += f"\n{evidence}"
        if "source" in edge and (
            kind == "semantic"
            or kind in SOURCE_EDGE_KINDS
            and hasattr(self.reader, "graph_evidence")
        ):
            row_item(self.relationships, text, {**edge, "other_node": node})
        else:
            row_item(self.relationships, text, node)

    def _activate_relationship(self, item):
        value = item.data(ROLE)
        if not value:
            return
        if "source" in value and "target" in value and "kind" in value:
            self._inspect_edge(value)
        else:
            self._open_relation(value)

    def _buttons(self):
        node = self.diagram.nodes.get(self.selection, {}) if self.diagram else {}
        self.back.setEnabled(bool(self.trail))
        self.previous.setEnabled(self.offset > 0)
        self.next.setEnabled(
            bool(
                self.diagram
                and self.mode in {"browse", "git", "topics"}
                and self.offset + PAGE_SIZE < self.diagram.total
            )
        )
        can_open = bool(node and node.get("kind") not in FOLDER_KINDS | {"topic"})
        if self._active_edge:
            can_open = self._evidence_node is not None
        self.open_source.setEnabled(bool(can_open))
        self.focus.setEnabled(bool(node) and node.get("kind") != "topic")
        current = self.diagram.scope or {} if self.diagram else {}
        self.git.setEnabled(
            bool(
                node.get("project_id")
                or current.get("project_id")
                or node.get("kind") in {"project", "repository"}
                or current.get("kind") in {"project", "repository"}
            )
        )
        self.maps.setEnabled(self.diagram is not None)
        self.canvas.setEnabled(self.maps.isEnabled())

    def _navigate(self, mode, scope):
        self.trail.append((self.mode, self.scope, self.offset, self.search_box.text()))
        self.mode, self.scope, self.offset, self.selection = mode, scope, 0, None
        self._pending_selection = None
        self.expanded.clear()
        self.refresh()

    def set_scope(self, node_id):
        self.grouping.blockSignals(True)
        self.grouping.setCurrentIndex(0)
        self.grouping.blockSignals(False)
        self._navigate("browse", node_id)

    def show_node(self, node):
        mode = "browse" if node.get("kind") in FOLDER_KINDS else "focus"
        self._navigate(mode, node["id"])
        self._pending_selection = mode, node["id"]

    def _home(self):
        self.search_box.clear()
        self._navigate(self.grouping.currentData(), None)

    def _change_grouping(self, _index):
        self.search_box.clear()
        self._navigate(self.grouping.currentData(), None)

    def _show_paths(self):
        if self.neural is None or self._path_running:
            return
        self._path_running = True
        self.question_paths.setEnabled(False)

        def read():
            packet = self.neural.paths()
            rows = []
            for path in packet["paths"]:
                identity = path.get("node_id")
                node = self.reader.node(identity) if isinstance(identity, str) else None
                if node and node.get("state") not in GRAPH_BLOCKED:
                    rows.append((node, path))
            return rows, len(packet["paths"]), packet.get("coverage", {})

        QThreadPool.globalInstance().start(_Read(0, read, self._path_reply))

    def _paths_ready(self, _serial, result, error):
        self._path_running = False
        self.question_paths.setEnabled(True)
        self._relation_serial += 1
        self._relation_pending = None
        self._active_edge = self.surface.selected_edge = self._evidence_node = None
        self.selection = self.surface.selection = None
        self.items.blockSignals(True)
        self.items.setCurrentRow(-1)
        self.items.blockSignals(False)
        self.surface.update()
        self.relationships.clear()
        if error:
            self.detail.setText("Question paths unavailable. Check the local neural service.")
            return
        rows, total, coverage = result
        self.detail.setText(
            f"Latest question · {len(rows):,} current sources of {total:,} retrieved"
        )
        for node, path in rows:
            channels = ", ".join(str(channel) for channel in path.get("channels", ()))
            route = " → ".join(str(step) for step in path.get("relationship_path", ()))
            caption = node_label(node)
            if channels:
                caption += "\nRetrieved via " + channels
            if route:
                caption += "\n" + route
            line = path.get("start_line") or (path.get("locator") or {}).get("start")
            if isinstance(line, int) and line > 0:
                node = {**node, "match_start": line}
                caption += f" · line {line:,}"
            row_item(self.relationships, caption, node)
        if not rows:
            row_item(
                self.relationships,
                "No current retrieval sources. Ask a question in Ask Dodo.",
                None,
            )
        if coverage:
            row_item(
                self.relationships,
                "Coverage · " + ", ".join(f"{key}: {value}" for key, value in coverage.items()),
                None,
            )
        self._buttons()

    def _back(self):
        if self.trail:
            self.mode, self.scope, self.offset, query = self.trail.pop()
            self.search_box.setText(query)
            self.selection = None
            self.expanded.clear()
            self.refresh()

    def _page(self, direction):
        self.offset = max(0, self.offset + direction)
        self.selection = None
        self.expanded.clear()
        self.refresh()

    def _search(self):
        if self.search_box.text().strip():
            self._navigate("search", None)

    def _focus(self):
        if self.selection:
            self._navigate("focus", self.selection)

    def _git_history(self):
        if not self.diagram:
            return
        for node in (self.diagram.nodes.get(self.selection) or {}, self.diagram.scope or {}):
            project = node.get("project_id")
            if not project and node.get("kind") in {"project", "repository"}:
                project = node.get("id")
            if project:
                self._navigate("git", project)
                return

    def _activate(self, identity):
        if self.diagram and (node := self.diagram.nodes.get(identity)):
            if self.mode == "search":
                self.show_node(node)
            else:
                self._open_relation(node)

    def _surface_activate(self, identity):
        if (
            self.diagram
            and self.mode == "browse"
            and self.diagram.hubs
            and self.diagram.nodes[identity].get("kind") in FOLDER_KINDS
        ):
            if identity in self.expanded:
                self.expanded.remove(identity)
                self.refresh()
            else:
                self._expand(identity)
            return
        self._activate(identity)

    def _expand(self, identity):
        if self.mode == "browse" and identity not in self.expanded:
            self.expanded = [*self.expanded[-3:], identity]
            self.refresh()

    def _collapse(self):
        if self.expanded:
            self.expanded.clear()
            self.refresh()

    def _fit_graph(self):
        self._cache.pop(self._requested_key, None)
        self._view_key = self._requested_key = None
        self.refresh()

    def _open_relation(self, node):
        if not node:
            return
        if node.get("kind") == "topic":
            self._navigate("topics", node["id"])
        elif node.get("kind") in FOLDER_KINDS:
            self.set_scope(node["id"])
        else:
            self.source_requested.emit(node)

    def _open_source(self):
        if self._active_edge and self._evidence_node:
            self.source_requested.emit(self._evidence_node)
        elif not self._active_edge and self.diagram and self.selection in self.diagram.nodes:
            self.source_requested.emit(self.diagram.nodes[self.selection])

    def _generated_diagram(self, kind):
        selected = self.diagram.nodes.get(self.selection) if self.diagram else None
        if selected and selected["id"] != self.scope and selected.get("kind") in FOLDER_KINDS:
            self._generated_pending = kind
            self.set_scope(selected["id"])
            return None
        if self.diagram is None or self._running:
            self._generated_pending = kind
            if not self._running:
                self.refresh()
            return None
        return self.diagram

    def show_mind_map(self):
        diagram = self._generated_diagram("map")
        if diagram is None:
            return
        html = mind_map_html(diagram)
        title = str((diagram.scope or {}).get("name", "Whole library"))
        self.mind_map_requested.emit(title, html)
        dialog = QDialog(self, windowTitle=f"Mind map · {title}")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(740, 640)
        browser = QTextBrowser()
        browser.setOpenLinks(False)
        current = {}

        def update(value):
            current["html"] = mind_map_html(value)
            current["targets"] = {anchor(identity): node for identity, node in value.nodes.items()}
            browser.setHtml(current["html"])

        browser.anchorClicked.connect(
            lambda url: self._open_relation(current["targets"].get(url.fragment()))
        )
        save = button("Save HTML")
        save.clicked.connect(lambda: self._save_html(title, current["html"]))
        layout = QVBoxLayout(dialog)
        layout.addWidget(browser)
        layout.addWidget(save)
        self._show_preview(dialog, update, diagram)

    def _show_preview(self, dialog, update, diagram):
        entry = _Preview(self._view_key, diagram, update)
        self._previews.append(entry)
        dialog.finished.connect(lambda _result: self._previews.remove(entry))
        dialog.show()
        update(diagram)

    def _save_html(self, title, html):
        safe_title = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", title).strip(" .") or "Mind map"
        filename, _ = QFileDialog.getSaveFileName(
            self, "Save generated mind map", f"{safe_title}.html", "HTML (*.html)"
        )
        if filename:
            encoded = html.encode("utf-8")
            file = QSaveFile(filename)
            if not file.open(QIODevice.OpenModeFlag.WriteOnly):
                self.summary.setText(f"Could not save the mind map. {file.errorString()}")
            elif file.write(encoded) != len(encoded):
                self.summary.setText(f"Could not save the mind map. {file.errorString()}")
                file.cancelWriting()
            elif not file.commit():
                self.summary.setText(f"Could not save the mind map. {file.errorString()}")

    def show_generated_canvas(self):
        diagram = self._generated_diagram("canvas")
        if diagram is None:
            return
        title = str((diagram.scope or {}).get("name", "Whole library"))
        dialog = QDialog(self, windowTitle=f"Generated Canvas · {title}")
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        dialog.resize(800, 640)
        surface = GraphSurface()
        current = {"diagram": diagram}
        revision = label("", "muted")

        def update(value):
            current["diagram"] = value
            revision.setText(f"Revision {value.generation} · {len(value.nodes):,} visible items")
            surface.set_diagram(value, fit=surface.diagram is None)

        surface.expanded.connect(
            lambda identity: self._open_relation(current["diagram"].nodes[identity])
        )
        copy = button("Save editable copy", role="primary")
        copy.clicked.connect(
            lambda: self.canvas_requested.emit(
                {"title": title[:120], **canvas_board(current["diagram"])}
            )
        )
        layout = QVBoxLayout(dialog)
        layout.addWidget(revision)
        layout.addWidget(surface, 1)
        layout.addWidget(copy)
        self._show_preview(dialog, update, diagram)
