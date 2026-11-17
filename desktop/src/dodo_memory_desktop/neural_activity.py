"""Draw the installed E5 architecture and owner-local captured activations."""

import json
import math
import time
from dataclasses import dataclass
from functools import partial
from itertools import pairwise

from PySide6.QtCore import (
    QEventLoop,
    QObject,
    QPointF,
    QRectF,
    QSignalBlocker,
    Qt,
    QTimer,
    QUrl,
    Signal,
)
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPolygonF
from PySide6.QtNetwork import QNetworkAccessManager, QNetworkProxy, QNetworkRequest
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QListWidget,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .network_overview import NetworkOverview
from .network_parameters import NetworkParameters
from .neural_client import ENDPOINT, NeuralError
from .ui import button, label, row_item
from .whole_library import CatalogueReads


@dataclass
class TensorStage:
    identity: str
    title: str
    rect: QRectF
    points: list[QPointF]
    columns: int
    block: int | None = None


class ActivationSurface(QWidget):
    selected = Signal(str, int)
    structure_changed = Signal()

    def __init__(self):
        super().__init__()
        self.setAccessibleName(
            "E5 tensor routes. Pan by dragging, zoom with the wheel; use the feature list for keyboard inspection."
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(400, 360)
        self.architecture = None
        self.capture = None
        self.stages = []
        self.expanded = set()
        self.residual_routes = []
        self.tensor_routes = []
        self.colours = {}
        self.strengths = {}
        self.glows = {}
        self.depth = True
        self.focus_mode = "all"
        self.focus_block = None
        self.input_rect = QRectF(15, 150, 88, 650)
        self.selection = ("embedding", 0)
        self.zoom = 1.0
        self.pan = QPointF()
        self.bounds = QRectF()
        self._drag = None
        self._moved = False
        self._manual = False

    def set_architecture(self, value):
        self.architecture = value
        self.focus_mode = "measured" if value["capture_level"] == "final" else "all"
        self._build()
        self._fit_rect(self.focus_bounds())

    def _build(self):
        hidden = self.architecture["hidden_size"]
        stages = [("embedding", "Embeddings", hidden, None)]
        self.residual_routes = []
        previous = "embedding"
        for block in range(self.architecture["num_hidden_layers"]):
            identity = f"block.{block}"
            if block in self.expanded:
                stages.extend(
                    [
                        (f"heads.{block}", "12 heads x 32\nContext", hidden, block),
                        (f"attention_projection.{block}", "Attention\nprojection", hidden, block),
                        (f"attention_residual.{block}", "Residual\naddition", hidden, block),
                        (f"attention_norm.{block}", "Attention\nLayerNorm", hidden, block),
                        (
                            f"ff.{block}",
                            "Dense → GELU",
                            self.architecture["intermediate_size"],
                            block,
                        ),
                        (f"ff_projection.{block}", "Output dense", hidden, block),
                        (f"ff_residual.{block}", "Residual\naddition", hidden, block),
                        (identity, f"Block {block + 1}\nFinal LayerNorm", hidden, block),
                    ]
                )
                self.residual_routes.extend(
                    [
                        (previous, f"attention_residual.{block}"),
                        (f"attention_norm.{block}", f"ff_residual.{block}"),
                    ]
                )
            else:
                stages.append((identity, f"Block {block + 1}\nTransformer", hidden, block))
            previous = identity
        stages.extend(
            [
                ("pooled", "Masked mean\npooling", hidden, None),
                ("output", "L2 normalised\noutput", hidden, None),
            ]
        )
        self.stages = []
        x = 190
        for identity, title, size, block in stages:
            columns = (
                self.architecture["num_attention_heads"] if identity.startswith("heads.") else 1
            )
            points = []
            for index in range(size):
                if identity.startswith("heads."):
                    column, position = divmod(index, self.architecture["head_dim"])
                    fraction = position / (self.architecture["head_dim"] - 1)
                else:
                    column, fraction = 0, index / (size - 1)
                offset = column * 28
                if self.depth:
                    offset += 26 * math.sin(math.pi * fraction) + 24 * (fraction - 0.5)
                points.append(QPointF(x + offset, 150 + fraction * 650))
            left, right = min(point.x() for point in points), max(point.x() for point in points)
            top, bottom = min(point.y() for point in points), max(point.y() for point in points)
            rect = QRectF(left - 20, top - 20, right - left + 40, bottom - top + 40)
            self.stages.append(TensorStage(identity, title, rect, points, columns, block))
            x = rect.right() + 270
        identities = ["input", *[stage.identity for stage in self.stages]]
        self.tensor_routes = list(pairwise(identities))
        bounds = QRectF(self.input_rect)
        for stage in self.stages:
            bounds = bounds.united(stage.rect)
        self.bounds = bounds.adjusted(-35, -85, 35, 35)
        self.structure_changed.emit()
        self.update()

    def set_depth(self, enabled):
        if enabled == self.depth:
            return
        self.depth = enabled
        selected = self.selection
        if self.architecture is not None:
            self._build()
            self.selection = selected
            self._manual = False
            self._fit_rect(self.focus_bounds())

    def focus_bounds(self):
        if self.focus_mode == "all" or not self.stages:
            return QRectF(self.bounds)
        bounds = QRectF()
        for stage in self.focused_stages():
            bounds = bounds.united(stage.rect)
        return bounds.adjusted(-45, -45, 45, 35) if not bounds.isEmpty() else QRectF(self.bounds)

    def focused_stages(self):
        if self.focus_mode == "all":
            return self.stages
        names = {"block.11", "pooled", "output"}
        if self.focus_mode == "block":
            names = {stage.identity for stage in self.stages if stage.block == self.focus_block}
        elif self.capture and self.capture["level"] != "final":
            names = self.capture["tensors"].keys()
        return [stage for stage in self.stages if stage.identity in names]

    def fit_measured(self):
        self.focus_mode = "measured"
        self._manual = False
        self._fit_rect(self.focus_bounds())

    def toggle_block(self, block):
        if self.architecture is None or not 0 <= block < self.architecture["num_hidden_layers"]:
            return
        if block in self.expanded:
            self.expanded.remove(block)
        else:
            self.expanded.add(block)
        self._build()
        self.focus_mode, self.focus_block = "block", block
        self._fit_rect(self.focus_bounds())

    def set_capture(self, value):
        first = self.capture is None
        self.capture = value
        self.colours = {}
        self.strengths = {}
        self.glows = {}
        for name, values in value["tensors"].items():
            maximum = max(map(abs, values)) or 1.0
            colours, strengths, glows = [], [], []
            for number in values:
                strength = math.sqrt(abs(number) / maximum)
                target = (197, 167, 255) if number >= 0 else (245, 132, 180)
                colours.append(
                    QColor(*(round(35 + (channel - 35) * strength) for channel in target))
                )
                strengths.append(strength)
                glows.append(QColor(*target, round(38 * strength)))
            self.colours[name] = colours
            self.strengths[name], self.glows[name] = strengths, glows
        if first and value["level"] == "final" and not self._manual:
            self.fit_measured()
        self.update()

    def fit(self):
        self.focus_mode = "all"
        self._manual = False
        self._fit_rect(self.bounds)

    def _fit_rect(self, rect):
        if rect.isEmpty():
            return
        height = max(100, self.height() - 125)
        self.zoom = min((self.width() - 40) / rect.width(), height / rect.height())
        self.pan = QPointF(self.width() / 2, 105 + height / 2) - rect.center() * self.zoom
        self.update()

    def label_rect(self, stage):
        centre = stage.rect.center().x() * self.zoom + self.pan.x()
        width = max(64, min(180, (stage.rect.width() + 150) * self.zoom - 16))
        return QRectF(centre - width / 2, 10, width, 86)

    def visible_feature_indices(self, stage):
        # A ring always represents one actual position, never an averaged group.
        rows = len(stage.points) // stage.columns
        spacing = 650 * self.zoom / max(1, rows - 1)
        stride = max(1, math.ceil(14 / max(0.01, spacing)))
        indices = {
            column * rows + row for column in range(stage.columns) for row in range(0, rows, stride)
        }
        indices.update(column * rows + rows - 1 for column in range(stage.columns))
        if self.selection[0] == stage.identity:
            indices.add(self.selection[1])
        return sorted(indices)

    def route_path(self, source, target, *, residual=False):
        rects = {stage.identity: stage.rect for stage in self.stages}
        rects["input"] = self.input_rect
        left, right = rects[source], rects[target]
        if residual:
            start, end = (
                QPointF(left.center().x(), left.top()),
                QPointF(right.center().x(), right.top()),
            )
            height = min(start.y(), end.y()) - 105
            controls = QPointF(start.x(), height), QPointF(end.x(), height)
        else:
            start, end = (
                QPointF(left.right(), left.center().y()),
                QPointF(right.left(), right.center().y()),
            )
            distance = (end.x() - start.x()) * 0.45
            controls = start + QPointF(distance, -110), end + QPointF(-distance, 110)
        path = QPainterPath(start)
        path.cubicTo(*controls, end)
        return path

    def zoom_by(self, factor, anchor=None):
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = (anchor - self.pan) / self.zoom
        self.zoom = max(0.12, min(64.0, self.zoom * factor))
        self.pan = anchor - before * self.zoom
        self._manual = True
        self.update()

    def select(self, identity, index, *, centre=False):
        self.selection = identity, index
        if centre:
            if self.focus_mode == "measured" and identity not in {
                stage.identity for stage in self.focused_stages()
            }:
                self.focus_mode = "all"
            stage = next((stage for stage in self.stages if stage.identity == identity), None)
            if stage:
                point = stage.points[index] * self.zoom + self.pan
                if not self.rect().adjusted(20, 20, -20, -20).contains(point.toPoint()):
                    self.pan = (
                        QPointF(self.width() / 2, self.height() / 2)
                        - stage.points[index] * self.zoom
                    )
        self.update()

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#09070c"))
        if not self.stages:
            painter.setPen(QColor("#a99ab9"))
            painter.drawText(
                self.rect(),
                Qt.AlignmentFlag.AlignCenter,
                "Waiting for verified local E5 architecture",
            )
            return
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.save()
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)
        by_id = {stage.identity: stage for stage in self.stages}
        level = self.capture["level"] if self.capture else self.architecture["capture_level"]
        final_focus = self.focus_mode == "measured" and level == "final"
        shown = self.focused_stages() if final_focus else self.stages
        shown_ids = {stage.identity for stage in shown}
        for source, target in self.tensor_routes:
            if final_focus and (source not in shown_ids or target not in shown_ids):
                continue
            path = self.route_path(source, target)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            source_rect = self.input_rect if source == "input" else by_id[source].rect
            target_rect = by_id[target].rect
            # These strands depict the tensor as a whole, not feature weights.
            for strand in range(32):
                fraction = strand / 31
                start = QPointF(source_rect.center().x(), source_rect.top() + 20 + fraction * 650)
                end = QPointF(target_rect.center().x(), target_rect.top() + 20 + fraction * 650)
                bend = (end.x() - start.x()) * 0.48
                sweep = math.sin(fraction * math.pi) * 130
                strand_path = QPainterPath(start)
                strand_path.cubicTo(start + QPointF(bend, -sweep), end - QPointF(bend, -sweep), end)
                painter.setPen(QPen(QColor(166, 131, 222, 56), 0.75 / self.zoom))
                painter.drawPath(strand_path)
            painter.setPen(QPen(QColor("#b99ae6"), 1.3 / self.zoom))
            painter.drawPath(path)
            end, before = path.pointAtPercent(1), path.pointAtPercent(0.97)
            direction = end - before
            length = math.hypot(direction.x(), direction.y()) or 1
            direction /= length
            side = QPointF(-direction.y(), direction.x())
            painter.setBrush(QColor("#9473c6"))
            painter.drawPolygon(
                QPolygonF([end, end - direction * 8 + side * 4, end - direction * 8 - side * 4])
            )
        for source, target in self.residual_routes:
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#80679f"), 1.3, Qt.PenStyle.DashLine))
            painter.drawPath(self.route_path(source, target, residual=True))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor("#786291"), 1.4))
        if not final_focus:
            painter.drawEllipse(self.input_rect.center(), 18, 18)
        viewport = QRectF(
            -self.pan.x() / self.zoom,
            -self.pan.y() / self.zoom,
            self.width() / self.zoom,
            self.height() / self.zoom,
        )
        for stage in shown:
            if not viewport.intersects(stage.rect):
                continue
            values = self.colours.get(stage.identity)
            indices = self.visible_feature_indices(stage)
            if values is None:
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor("#746386"), 1.0 / self.zoom))
                for index in indices:
                    painter.drawEllipse(stage.points[index], 5.2 / self.zoom, 5.2 / self.zoom)
                continue
            for index in indices:
                point = stage.points[index]
                strength = self.strengths[stage.identity][index]
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(self.glows[stage.identity][index])
                painter.drawEllipse(
                    point, (8 + 3 * strength) / self.zoom, (8 + 3 * strength) / self.zoom
                )
                painter.setBrush(QColor("#09070c"))
                painter.setPen(QPen(values[index], 1.2 / self.zoom))
                painter.drawEllipse(
                    point, (5.2 + strength) / self.zoom, (5.2 + strength) / self.zoom
                )
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(values[index])
                painter.drawEllipse(
                    point, (2.5 + strength) / self.zoom, (2.5 + strength) / self.zoom
                )
        stage = by_id.get(self.selection[0])
        if stage and 0 <= self.selection[1] < len(stage.points):
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.setPen(QPen(QColor("#ffffff"), 1.2 / self.zoom))
            painter.drawEllipse(stage.points[self.selection[1]], 8.5 / self.zoom, 8.5 / self.zoom)
        painter.restore()
        painter.fillRect(QRectF(0, 0, self.width(), 100), QColor("#09070c"))
        font, metadata = QFont("Segoe UI"), QFont("Segoe UI")
        font.setPixelSize(17)
        metadata.setPixelSize(14)
        for stage in shown:
            rect = self.label_rect(stage)
            if rect.right() < 0 or rect.left() > self.width():
                continue
            available = stage.identity in self.colours
            painter.setPen(QColor("#e5d6ff") if available else QColor("#a89bb8"))
            painter.setFont(font)
            title = stage.title.replace("\n", " ")
            if rect.width() < 140:
                names = {
                    "embedding": "Embed.",
                    "pooled": "Mean pool",
                    "output": "L2 norm.",
                    "heads": "Heads",
                    "attention_projection": "Attn proj.",
                    "attention_residual": "Residual",
                    "attention_norm": "Attn norm.",
                    "ff": "GELU",
                    "ff_projection": "Dense",
                    "ff_residual": "Residual",
                }
                title = (
                    f"Block {stage.block + 1}"
                    if stage.identity.startswith("block.")
                    else names[stage.identity.split(".")[0]]
                )
            painter.drawText(
                QRectF(rect.left(), rect.top(), rect.width(), 44),
                Qt.AlignmentFlag.AlignHCenter | Qt.TextFlag.TextWordWrap,
                title,
            )
            painter.setFont(metadata)
            count = str(len(stage.points))
            if rect.width() >= 110:
                count += " positions"
            painter.drawText(
                QRectF(rect.left(), rect.top() + 46, rect.width(), 18),
                Qt.AlignmentFlag.AlignHCenter,
                count,
            )
            if rect.width() >= 105:
                painter.drawText(
                    QRectF(rect.left(), rect.top() + 66, rect.width(), 18),
                    Qt.AlignmentFlag.AlignHCenter,
                    "Measured" if available else "Unavailable",
                )
        centre = self.input_rect.center().x() * self.zoom + self.pan.x()
        if not final_focus and 0 <= centre <= self.width():
            painter.setFont(font)
            painter.setPen(QColor("#a89bb8"))
            painter.drawText(
                QRectF(centre - 32, 10, 64, 40), Qt.AlignmentFlag.AlignHCenter, "Input"
            )
            painter.setFont(metadata)
            painter.drawText(
                QRectF(centre - 32, 56, 64, 20), Qt.AlignmentFlag.AlignHCenter, "IDs/mask"
            )
        if final_focus:
            painter.setFont(metadata)
            painter.setPen(QColor("#a99ab9"))
            painter.drawText(
                QRectF(14, self.height() - 24, self.width() - 28, 20),
                "FINAL states · earlier layers unavailable · Whole model shows all 12 blocks",
            )

    def resizeEvent(self, event):
        if not self._manual:
            self._fit_rect(self.focus_bounds())
        super().resizeEvent(event)

    def wheelEvent(self, event):
        self.zoom_by(1.2 ** (event.angleDelta().y() / 120), event.position())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self._moved = False
            self.setFocus()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._moved |= delta.manhattanLength() > 2
            self.pan += delta
            self._drag = event.position()
            self._manual = True
            self.update()

    def mouseReleaseEvent(self, event):
        self._drag = None
        if self._moved:
            return
        point = (event.position() - self.pan) / self.zoom
        for stage in self.stages:
            if self.label_rect(stage).contains(event.position()) or stage.rect.contains(point):
                index = min(
                    range(len(stage.points)),
                    key=lambda index: (stage.points[index] - point).manhattanLength(),
                )
                self.select(stage.identity, index)
                self.selected.emit(stage.identity, index)
                break

    def mouseDoubleClickEvent(self, event):
        point = (event.position() - self.pan) / self.zoom
        for stage in self.stages:
            if stage.block is not None and (
                self.label_rect(stage).contains(event.position()) or stage.rect.contains(point)
            ):
                self.toggle_block(stage.block)
                break

    def keyPressEvent(self, event):
        identity, index = self.selection
        stage = next((stage for stage in self.stages if stage.identity == identity), None)
        if stage is None:
            return
        horizontal, vertical = 1, 1
        if identity.startswith("heads."):
            horizontal, vertical = self.architecture["head_dim"], 1
        movement = {
            Qt.Key.Key_Left: -horizontal,
            Qt.Key.Key_Right: horizontal,
            Qt.Key.Key_Up: -vertical,
            Qt.Key.Key_Down: vertical,
        }
        if event.key() in movement:
            index = max(0, min(len(stage.points) - 1, index + movement[event.key()]))
            self.select(identity, index, centre=True)
            self.selected.emit(identity, index)
        elif event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Space} and stage.block is not None:
            self.toggle_block(stage.block)
        else:
            super().keyPressEvent(event)


class ActivityEvents(QObject):
    changed = Signal(object)
    disconnected = Signal()

    def __init__(self, client, parent):
        super().__init__(parent)
        self.client = client
        self.network = QNetworkAccessManager(self)
        self.network.setProxy(QNetworkProxy(QNetworkProxy.ProxyType.NoProxy))
        self.reply = None
        self.buffer = b""

    def start(self, sequence):
        self.stop()
        try:
            token = self.client._token()
        except NeuralError:
            self.disconnected.emit()
            return
        request = QNetworkRequest(QUrl(f"{ENDPOINT}/activity/events?after={sequence}"))
        request.setRawHeader(b"Authorization", f"Bearer {token}".encode("ascii"))
        request.setRawHeader(b"Accept", b"text/event-stream")
        request.setAttribute(
            QNetworkRequest.Attribute.RedirectPolicyAttribute,
            QNetworkRequest.RedirectPolicy.ManualRedirectPolicy,
        )
        reply = self.network.get(request)
        self.reply = reply
        self.buffer = b""
        reply.readyRead.connect(self._read)
        reply.finished.connect(partial(self._finished, reply))

    def _read(self):
        if self.reply is None:
            return
        self.buffer += bytes(self.reply.readAll()).replace(b"\r\n", b"\n")
        if len(self.buffer) > 65536:
            self.stop()
            self.disconnected.emit()
            return
        while b"\n\n" in self.buffer:
            event, self.buffer = self.buffer.split(b"\n\n", 1)
            data = b"\n".join(
                line[5:].lstrip() for line in event.splitlines() if line.startswith(b"data:")
            )
            if not data:
                continue
            try:
                value = json.loads(data)
            except (UnicodeDecodeError, json.JSONDecodeError):
                self.stop()
                self.disconnected.emit()
                return
            if (
                not isinstance(value, dict)
                or type(value.get("sequence")) is not int
                or not isinstance(value.get("epoch"), str)
            ):
                self.stop()
                self.disconnected.emit()
                return
            self.changed.emit(value)

    def _finished(self, reply):
        if self.reply is reply:
            self.reply = None
            self.disconnected.emit()
        reply.deleteLater()

    def stop(self):
        reply, self.reply = self.reply, None
        if reply is not None:
            reply.abort()
        self.buffer = b""


class CaptureControls(QObject):
    """Finish the final hidden lease even after the view's native widget is deleted."""

    failed = Signal()

    def __init__(self, client, parent):
        super().__init__(parent)
        self.client = client
        self.reads = CatalogueReads(self)
        self.last = None
        self.pending = None
        self.running = False

    def post(self, value):
        if self.client is None:
            return
        self.last, self.pending = value, value
        self._send()

    def hide(self, _owner=None):
        if self.last and self.last["visible"]:
            self.post({**self.last, "visible": False})

    def _send(self):
        if self.running or self.pending is None:
            return
        value, self.pending = self.pending, None
        self.running = True
        self.reads.request(
            "activity-control",
            partial(self.client.activity_control, **value),
            self._ready,
            self._failed,
        )

    def _ready(self, _value=None):
        self.running = False
        self._send()

    def _failed(self):
        self._ready()
        self.failed.emit()


class NeuralActivity(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.capture = None
        self.architecture = None
        self.active = False
        self._visible = False
        self.weight_view = None
        self._network_packet = None
        self.overview = NetworkOverview()
        self.overview.block_selected.connect(self._open_block)
        self.parameters = NetworkParameters(client, self)
        self.parameters.ready.connect(self._network_ready)
        self.parameters.failed.connect(self._network_failed)
        self.parameters.progress.connect(self._network_progress)
        self.parameters.cache_warning.connect(self._network_cache_warning)
        self.parameter_status = label("", "muted")
        self.parameter_status.setWordWrap(True)
        self.parameter_status.setTextFormat(Qt.TextFormat.PlainText)
        self.parameter_status.hide()
        self.replaying = False
        self.sequence = 0
        self.epoch = None
        self._requested_capture = None
        self._snapshot_requested = None
        self._snapshot_again = False
        self._stopping = False
        self._read_tokens = set()
        self._shown_tensor = None
        self.reads = CatalogueReads(QApplication.instance())
        self.reads.done.connect(self._read_finished)
        self.controls = CaptureControls(client, QApplication.instance())
        self.controls.failed.connect(self._disconnected)
        self.destroyed.connect(self.controls.hide)
        self.destroyed.connect(lambda _owner=None, reads=self.reads: reads.pending.clear())
        self.events = ActivityEvents(client, self)
        self.events.changed.connect(self._event)
        self.events.disconnected.connect(self._disconnected)
        self.lease = QTimer(self, interval=10000)
        self.lease.timeout.connect(self._control)
        self.reconnect = QTimer(self, interval=5000)
        self.reconnect.timeout.connect(self._retry)
        self.debounce = QTimer(self, interval=150, singleShot=True)
        self.debounce.timeout.connect(self.refresh)
        self.surface = ActivationSurface()
        self.surface.structure_changed.connect(self._stages_changed)
        self.surface.selected.connect(self._surface_selected)
        self.model_status = label("E5 network · waiting for verified local architecture", "section")
        self.model_status.setWordWrap(True)
        self.provenance = label("", "muted")
        self.provenance.setWordWrap(True)
        self.running_status = label("Idle · no captured inference reported.", "muted")
        self.running_status.setWordWrap(True)
        self.capture_status = label("Values unavailable until the next real inference.", "muted")
        self.capture_status.setWordWrap(True)
        self.pause = QCheckBox("Pause capture")
        self.pause.setStyleSheet(
            "QCheckBox::indicator { width: 15px; height: 15px; border: 1px solid #645078; "
            "background: #18131f; } QCheckBox::indicator:checked { background: #c5a7ff; }"
        )
        self.pause.toggled.connect(self._pause_changed)
        self.replay = button("Replay captured values")
        self.replay.setEnabled(False)
        self.replay.clicked.connect(self._replay)
        self.depth = QCheckBox("Perspective", checked=True)
        self.depth.setStyleSheet(self.pause.styleSheet())
        self.depth.setToolTip(
            "Schematic perspective only; feature identities and values are unchanged."
        )
        self.depth.toggled.connect(self.surface.set_depth)
        self.token = QSpinBox(minimum=-1, maximum=255, value=-1)
        self.token.setSpecialValueText("First unmasked")
        self.token.setAccessibleName("Token index for the next real inference")
        self.token.setToolTip(
            "Applies to the next real inference. This control never runs the model. Masked or out-of-range selections reject that capture."
        )
        self.token.valueChanged.connect(self._control)
        self.stage = QComboBox(accessibleName="Tensor to inspect")
        self.stage.currentIndexChanged.connect(self._stage_changed)
        self.features = QListWidget(
            accessibleName="All feature positions and captured numeric values"
        )
        self.features.currentRowChanged.connect(self._feature_changed)
        self.inspector = label("Select a feature position.")
        self.inspector.setWordWrap(True)
        self.inspector.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.inspector.setStyleSheet("font-size: 17px;")
        self.expand = button("Expand selected block")
        self.expand.clicked.connect(self._expand_selected)
        fit = button("Whole model")
        fit.clicked.connect(self.surface.fit)
        self.measured = button("Final states", role="primary")
        self.measured.clicked.connect(self.surface.fit_measured)
        smaller, larger = button("-"), button("+")
        smaller.setAccessibleName("Zoom out")
        larger.setAccessibleName("Zoom in")
        smaller.clicked.connect(lambda: self.surface.zoom_by(1 / 1.3))
        larger.clicked.connect(lambda: self.surface.zoom_by(1.3))
        controls = QHBoxLayout()
        for control in (
            self.measured,
            fit,
            smaller,
            larger,
            self.depth,
            self.expand,
            self.pause,
            self.replay,
        ):
            controls.addWidget(control)
        controls.addStretch()
        token_row = QHBoxLayout()
        token_row.addWidget(label("Token for next inference"))
        token_row.addWidget(self.token)
        token_row.addWidget(
            label("Selected token states; pooling uses all unmasked tokens.", "muted"), 1
        )
        side = QWidget()
        sidebar = QVBoxLayout(side)
        sidebar.setContentsMargins(10, 0, 0, 0)
        sidebar.addWidget(label("Numeric inspector", "section"))
        sidebar.addWidget(self.stage)
        sidebar.addWidget(self.inspector)
        sidebar.addWidget(self.features, 1)
        split = QSplitter()
        split.addWidget(self.surface)
        split.addWidget(side)
        split.setSizes([1050, 330])
        split.setStretchFactor(0, 3)
        legend = label(
            "Ring intensity = sqrt(|value| / tensor maximum). Violet positive · Rose negative · Hollow unavailable.\nOverview samples real feature positions; zoom or use the list for every position. Curves are bundled tensor flow, not neuron weights.",
            "muted",
        )
        legend.setWordWrap(True)
        self.activation_page = QWidget()
        self.pages = QStackedWidget()
        self.pages.addWidget(self.activation_page)
        self.pages.addWidget(self.overview)
        self.view_mode = QComboBox(accessibleName="E5 visualisation")
        self.view_mode.addItem("Whole network", "overview")
        self.view_mode.addItem("Activations", "activations")
        self.view_mode.addItem("Weights", "weights")
        self.view_mode.setCurrentIndex(self.view_mode.findData("activations"))
        self.view_mode.currentIndexChanged.connect(self._change_view)
        heading = QHBoxLayout()
        heading.addWidget(label("E5 network", "section"))
        heading.addStretch()
        overview_button = button("Whole network")
        overview_button.clicked.connect(self._show_network)
        heading.addWidget(overview_button)
        self.reload_connections = button("Reload connections")
        self.reload_connections.clicked.connect(self._reload_network)
        self.reload_connections.hide()
        heading.addWidget(self.reload_connections)
        heading.addWidget(self.view_mode)
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)
        root.addLayout(heading)
        root.addWidget(self.running_status)
        root.addWidget(self.parameter_status)
        root.addWidget(self.pages, 1)
        layout = QVBoxLayout(self.activation_page)
        layout.setContentsMargins(18, 14, 18, 14)
        layout.addWidget(self.model_status)
        layout.addWidget(self.provenance)
        layout.addLayout(controls)
        layout.addLayout(token_row)
        layout.addWidget(self.capture_status)
        layout.addWidget(split, 1)
        layout.addWidget(legend)

    def set_architecture(self, value):
        changed = self.architecture is not None and any(
            self.architecture["model"][key] != value["model"][key] for key in ("space", "revision")
        )
        capture_mismatch = self.capture is not None and (
            self.capture["model_space"] != value["model"]["space"]
            or self.capture["model_revision"] != value["model"]["revision"]
        )
        if changed or capture_mismatch:
            self._clear_capture()
        if changed:
            self._clear_parameters()
        self.architecture = value
        self.overview.set_architecture(value)
        self.parameters.set_model(value["model"])
        self.token.setMaximum(value["max_tokens"] - 1)
        self.measured.setText(
            "Final states" if value["capture_level"] == "final" else "Measured states"
        )
        self.model_status.setText("12 transformer blocks · 384 features")
        model = value["model"]
        device = ", ".join(model["device"]) or "No active encoder"
        self.provenance.setText(
            f"12 heads x 32 · 1536 GELU units · {device} · revision {model['revision'][:16]} · capture {value['capture_level']}"
        )
        self.provenance.setToolTip(f"Model space: {model['space']}\nRevision: {model['revision']}")
        self.surface.set_architecture(value)
        reason = value.get("qualification_reason")
        self.capture_status.setText(
            f"Capture level: {value['capture_level']}. {reason or 'Values arrive only after real inference.'}"
        )
        if value.get("runtime_available") is False:
            self.model_status.setText(
                "Encoder paused · static parameters · 12 transformer blocks · 384 features"
            )
            self.running_status.setText("Encoder paused · static parameters.")
            self.capture_status.setText(
                "No live activations. " + (value.get("runtime_reason") or reason or "")
            )
        self.refresh()

    def set_capture(self, value):
        if (self.epoch is not None and value["epoch"] != self.epoch) or (
            self.architecture is not None
            and (
                value["model_space"] != self.architecture["model"]["space"]
                or value["model_revision"] != self.architecture["model"]["revision"]
            )
        ):
            self._capture_failed()
            return
        first = self.capture is None
        self.capture = value
        self.overview.set_capture(value)
        self.surface.set_capture(value)
        self.replay.setEnabled(bool(value["tensors"]))
        self._capture_caption()
        if first and value["level"] == "final":
            self.stage.setCurrentIndex(self.stage.findData("block.11"))
        self._stage_changed()

    def _clear_capture(self):
        self.capture = self.surface.capture = None
        self.surface.colours.clear()
        self.surface.strengths.clear()
        self.surface.glows.clear()
        self.overview.set_capture(None)
        self._requested_capture = None
        self.replay.setEnabled(False)
        self.capture_status.setText("Values unavailable until the next real inference.")
        self.surface.update()
        self._stage_changed(centre=False)

    def _weights(self):
        if self.weight_view is None:
            from .weight_projection import StaticWeightView

            self.weight_view = StaticWeightView(self.client)
            self.weight_view.set_parameters(self._network_packet)
            self.pages.addWidget(self.weight_view)
        return self.weight_view

    def _open_block(self, block):
        self._weights().select_block(block)
        self.view_mode.setCurrentIndex(self.view_mode.findData("weights"))

    def _show_network(self):
        self.view_mode.setCurrentIndex(self.view_mode.findData("overview"))
        self.overview.fit()

    def _reload_network(self):
        self._clear_parameters()
        self.parameters.reload()

    def _network_ready(self, value):
        if self.overview.set_parameters(value):
            self._network_packet = value
            if self.weight_view is not None:
                self.weight_view.set_parameters(value)
            self.parameter_status.hide()
        else:
            self._network_failed("The connection packet does not match this encoder.")

    def _network_failed(self, reason):
        self._clear_parameters()
        self.parameter_status.setText("Connections unavailable: " + reason)
        self.parameter_status.setVisible(self.view_mode.currentData() == "overview")

    def _clear_parameters(self):
        self._network_packet = None
        self.overview.set_parameters(None)
        if self.weight_view is not None:
            self.weight_view.set_parameters(None)

    def _network_progress(self, completed, total):
        self.parameter_status.setText(
            f"Reading real connections: {completed} / {total} projections"
        )
        self.parameter_status.setVisible(self.view_mode.currentData() == "overview")

    def _change_view(self, _index=None):
        if self._stopping:
            return
        overview = self.view_mode.currentData() == "overview"
        self.reload_connections.setVisible(overview)
        self.parameter_status.setVisible(overview and bool(self.parameter_status.text()))
        if self.view_mode.currentData() == "weights":
            self.pages.setCurrentWidget(self._weights())
        elif self.view_mode.currentData() == "overview":
            self.pages.setCurrentWidget(self.overview)
        else:
            self.pages.setCurrentWidget(self.activation_page)
        self.set_active(self._visible)

    def _network_cache_warning(self, reason):
        self.parameter_status.setText("Connections loaded; snapshot could not be saved: " + reason)
        self.parameter_status.setVisible(self.view_mode.currentData() == "overview")

    def _recheck_encoder(self):
        self._clear_capture()
        self._clear_parameters()
        self.parameters.reload()
        self.architecture = None
        if self.client is not None:
            self._read(
                "activity-architecture",
                self.client.activity_architecture,
                self.set_architecture,
                self._disconnected,
                replace=True,
            )

    def _capture_caption(self):
        if self.capture is None:
            return
        value = self.capture
        prefix = "Replay · captured values" if self.replaying else "Captured values"
        caption = f"{prefix} · {value['id']} · {value['source']} · raw token index {value['token_index']} · {value['token_count']} unmasked · {value['level']} · {value['duration_ms']:.1f} ms"
        if value["token_is_padding"]:
            caption += " · Selected position masked or out of range; values unavailable."
        self.capture_status.setText(caption)

    def _stages_changed(self):
        selected = self.stage.currentData()
        with QSignalBlocker(self.stage):
            self.stage.clear()
            for stage in self.surface.stages:
                self.stage.addItem(
                    stage.title.replace("\n", " ") + f" · {stage.identity}", stage.identity
                )
            self.stage.setCurrentIndex(max(0, self.stage.findData(selected)))
        self._stage_changed(centre=False)

    def _stage_changed(self, _index=None, *, centre=True):
        identity = self.stage.currentData()
        stage = next((stage for stage in self.surface.stages if stage.identity == identity), None)
        preserve = identity == self._shown_tensor
        selected = self.features.currentRow() if preserve else 0
        self._shown_tensor = identity
        with QSignalBlocker(self.features):
            self.features.clear()
            if stage:
                values = self.capture["tensors"].get(identity) if self.capture else None
                for index in range(len(stage.points)):
                    number = f"{values[index]:+.7g}" if values else "unavailable"
                    name = (
                        f"h{index // 32} / d{index % 32}"
                        if identity.startswith("heads.")
                        else f"Feature {index}"
                    )
                    row_item(self.features, f"{name}    {number}", index)
                self.features.setCurrentRow(max(0, min(len(stage.points) - 1, selected)))
        self.expand.setEnabled(stage is not None and stage.block is not None)
        expanded = stage is not None and stage.block in self.surface.expanded
        self.expand.setText("Collapse selected block" if expanded else "Expand selected block")
        self._feature_changed(self.features.currentRow(), centre=centre and not preserve)

    def _feature_changed(self, index, *, centre=True):
        identity = self.stage.currentData()
        if index < 0 or identity is None:
            return
        values = self.capture["tensors"].get(identity) if self.capture else None
        scope = "No token captured"
        if identity in {"pooled", "output"}:
            scope = "All unmasked tokens"
        elif self.capture:
            scope = f"Selected token {self.capture['token_index']}"
        number = f"{values[index]:+.9g}" if values else "Values unavailable"
        scale = (
            f"\nTensor max |value|: {max(map(abs, values)):.7g}"
            if values
            else "\nThis component was not captured."
        )
        position = (
            f"Head {index // 32} · dimension {index % 32}"
            if identity.startswith("heads.")
            else f"Feature {index}"
        )
        self.inspector.setText(f"{identity}\n{position}\n{scope}\n{number}{scale}")
        self.surface.select(identity, index, centre=centre)

    def _surface_selected(self, identity, index):
        self.stage.setCurrentIndex(self.stage.findData(identity))
        self.features.setCurrentRow(index)

    def _expand_selected(self):
        identity = self.stage.currentData()
        stage = next((stage for stage in self.surface.stages if stage.identity == identity), None)
        if stage and stage.block is not None:
            self.surface.toggle_block(stage.block)

    def _replay(self):
        self.replaying = True
        self.overview.set_replaying(True)
        self.pause.setChecked(True)
        self._capture_caption()

    def _pause_changed(self, paused):
        self.overview.set_paused(paused)
        if paused:
            self.reads.cancel("activity-capture")
            self._requested_capture = None
        else:
            self.replaying = False
            self.overview.set_replaying(False)
            self._capture_caption()
            self.refresh()
        self._control()

    def set_active(self, active):
        visible = active
        if visible and self._stopping:
            return
        self._visible = visible
        weights = self.view_mode.currentData() == "weights"
        self.parameters.set_active(visible and self.view_mode.currentData() == "overview")
        if self.weight_view is not None:
            self.weight_view.set_active(visible and weights)
        active = visible and not weights
        if active == self.active:
            return
        self.active = active
        if self.client is None:
            return
        self._control(force=True)
        if active:
            self.lease.start()
            if self.architecture is None:
                self._read(
                    "activity-architecture",
                    self.client.activity_architecture,
                    self.set_architecture,
                    self._disconnected,
                )
            self.events.start(self.sequence)
            self.refresh()
        else:
            self.overview.set_running(False)
            self.lease.stop()
            self.reconnect.stop()
            self.debounce.stop()
            self.events.stop()
            self._snapshot_requested = None
            self._snapshot_again = False
            for lane in ("activity-architecture", "activity-snapshot", "activity-capture"):
                self.reads.cancel(lane)

    def _control(self, _value=None, *, force=False):
        if self.client is None or not (self.active or force):
            return
        index = self.token.value()
        self.controls.post(
            {
                "visible": self.active,
                "paused": self.pause.isChecked(),
                "token_index": None if index == -1 else index,
            }
        )

    def _read(self, lane, operation, complete, failed, *, replace=False):
        serial = self.reads.serial
        self.reads.request(lane, operation, complete, failed, replace=replace)
        if self.reads.serial != serial:
            self._read_tokens.add((lane, self.reads.serial))

    def _read_finished(self, token, _value, _succeeded):
        self._read_tokens.discard(token)

    def shutdown(self, timeout=5):
        deadline = time.monotonic() + timeout
        self._stopping = True
        self.set_active(False)
        self.lease.stop()
        self.reconnect.stop()
        self.debounce.stop()
        self.events.stop()
        self.controls.hide()
        while self._read_tokens or self.controls.running or self.controls.pending is not None:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        if not self.parameters.shutdown(timeout=max(0, deadline - time.monotonic())):
            return False
        if self.weight_view is not None:
            return self.weight_view.shutdown(timeout=max(0, deadline - time.monotonic()))
        return True

    def refresh(self):
        if self.active and self.client is not None:
            if "activity-snapshot" in self.reads.pending:
                self._snapshot_again = True
                return
            self._snapshot_requested = self.epoch, self.sequence
            self._read(
                "activity-snapshot",
                self.client.activity_snapshot,
                self._snapshot,
                self._disconnected,
            )

    def _snapshot(self, value):
        if not self.active:
            return
        requested, self._snapshot_requested = self._snapshot_requested, None
        again, self._snapshot_again = self._snapshot_again, False
        if again and not self.debounce.isActive():
            self.debounce.start()
        stale_epoch = (
            requested is not None and requested[0] != self.epoch and value["epoch"] != self.epoch
        )
        stale_sequence = value["epoch"] == self.epoch and value["sequence"] < self.sequence
        if stale_epoch or stale_sequence:
            return
        if self.epoch is not None and self.epoch != value["epoch"]:
            self._recheck_encoder()
        self.sequence, self.epoch = value["sequence"], value["epoch"]
        if (
            self.architecture is not None
            and "runtime_available" in value
            and value["runtime_available"] != self.architecture.get("runtime_available", True)
        ):
            if not value["runtime_available"]:
                self._clear_capture()
            self._read(
                "activity-architecture",
                self.client.activity_architecture,
                self.set_architecture,
                self._disconnected,
            )
        running = len(value["running"])
        runtime_available = value.get(
            "runtime_available",
            self.architecture is None or self.architecture.get("runtime_available") is not False,
        )
        self.overview.set_running(bool(running) and runtime_available)
        if not runtime_available:
            state = "Encoder paused · static parameters."
        elif running:
            state = f"Running · {running} real inference(s); captured values remain displayed."
        else:
            state = (
                "Idle · last captured values remain displayed."
                if self.capture
                else "Idle · waiting for the next real inference."
            )
        reason = value.get("runtime_reason") or value.get("reason")
        if self.reconnect.isActive():
            state = "Reconnecting · snapshots every 5 seconds. " + state
        self.running_status.setText(state + (f" {reason}" if reason else ""))
        identity = value["latest_capture_id"]
        if (
            self.architecture is not None
            and runtime_available
            and identity
            and identity != self._requested_capture
            and not self.pause.isChecked()
        ):
            self._requested_capture = identity
            self._read(
                "activity-capture",
                partial(self.client.activity_capture, identity),
                self.set_capture,
                self._capture_failed,
                replace=True,
            )

    def _capture_failed(self):
        self._requested_capture = None
        self.capture_status.setText("Capture unavailable. Reconnecting to owner-local values.")
        self._disconnected()

    def _event(self, value):
        if not self.active:
            return
        self.reconnect.stop()
        if value["epoch"] != self.epoch or value["sequence"] > self.sequence:
            if self.epoch is not None and value["epoch"] != self.epoch:
                self._recheck_encoder()
            if value.get("type") == "started":
                self.overview.set_running(True)
            elif value.get("type") in {"completed", "failed"}:
                self.overview.set_running(False)
            self.epoch, self.sequence = value["epoch"], value["sequence"]
            if "activity-snapshot" in self.reads.pending:
                self._snapshot_again = True
            elif not self.debounce.isActive():
                self.debounce.start()

    def _disconnected(self):
        if self.active:
            self.overview.set_running(False)
            self.running_status.setText(
                "Reconnecting · activity stream unavailable; checking snapshots every 5 seconds."
            )
            self.reconnect.start()

    def _retry(self):
        if not self.active:
            return
        if self.architecture is None:
            self._read(
                "activity-architecture",
                self.client.activity_architecture,
                self.set_architecture,
                self._disconnected,
            )
        self.refresh()
        if self.events.reply is None:
            self.events.start(self.sequence)

    def showEvent(self, event):
        self.set_active(True)
        super().showEvent(event)

    def hideEvent(self, event):
        self.set_active(False)
        super().hideEvent(event)

    def closeEvent(self, event):
        if not self.shutdown():
            event.ignore()
            return
        super().closeEvent(event)
