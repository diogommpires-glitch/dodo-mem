"""Inspect bounded, actual E5 parameters without running the encoder."""

import math
import time
from functools import partial

from PySide6.QtCore import QEventLoop, QPointF, QRectF, QSignalBlocker, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPainterPathStroker, QPen
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QHBoxLayout,
    QListWidget,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .neural_client import validate_parameter_weights
from .ui import button, label
from .whole_library import CatalogueReads

PROJECTIONS = {
    "query": "Query",
    "key": "Key",
    "value": "Value",
    "attention_output": "Attention output",
    "ff_up": "Feed-forward up",
    "ff_down": "Feed-forward down",
}
SEMANTICS = {
    "embedding_layernorm_features": "Normalised embedding features",
    "previous_block_output_layernorm_features": "Previous block's normalised output",
    "concatenated_head_context_features": "Combined attention-head features",
    "attention_output_layernorm_features": "Normalised attention output",
    "gelu_features": "GELU output features",
    "query_linear_before_bias": "Query projection before bias",
    "key_linear_before_bias": "Key projection before bias",
    "value_linear_before_bias": "Value projection before bias",
    "attention_output_linear_before_bias": "Attention output projection before bias",
    "ff_up_linear_before_bias_before_gelu": "Linear output before bias and GELU",
    "ff_down_linear_before_bias_before_residual_and_layernorm": "Linear output before bias, residual and layer normalisation",
}


class WeightSurface(QWidget):
    edge_selected = Signal(int)
    node_selected = Signal(str, int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(420, 340)
        self.setAccessibleName(
            "Static parameter connections; equivalent feature and edge list alongside"
        )
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.payload = None
        self.inputs, self.outputs, self.paths = [], [], []
        self.selection = None
        self.zoom = 1.0
        self.pan = QPointF()
        self._drag = None
        self._moved = False
        self.bounds = QRectF(0, 0, 1400, 1040)

    @staticmethod
    def _points(count, centre):
        # A column preserves feature order; its bow is camera layout, not model data.
        return [
            QPointF(
                centre + 44 * math.sin(math.pi * index / max(1, count - 1)),
                100 + index * 850 / max(1, count - 1),
            )
            for index in range(count)
        ]

    def set_payload(self, payload):
        self.payload = payload
        self.inputs = self._points(payload["input_dim"], 100)
        self.outputs = self._points(payload["output_dim"], 1250)
        self.paths = []
        for edge in payload["edges"]:
            start, end = self.inputs[edge["source"]], self.outputs[edge["target"]]
            path = QPainterPath(start)
            bend = (end.x() - start.x()) * 0.43
            # Unequal vertical tangents create the reference's sweeping fan.
            sweep = (end.y() - start.y()) * 0.12
            path.cubicTo(start + QPointF(bend, sweep), end - QPointF(bend, sweep), end)
            self.paths.append(path)
        self.selection = None
        self.bounds = QRectF()
        for point in self.inputs + self.outputs:
            self.bounds = self.bounds.united(QRectF(point.x(), point.y(), 1, 1))
        self.bounds = self.bounds.adjusted(-45, -90, 45, 55)
        self.fit()

    def fit(self):
        self.zoom = max(
            0.15,
            min(
                (self.width() - 36) / self.bounds.width(),
                (self.height() - 70) / self.bounds.height(),
            ),
        )
        self.pan = (
            QPointF(self.width() / 2, self.height() / 2 + 18) - self.bounds.center() * self.zoom
        )
        self.update()

    def select(self, kind, index):
        self.selection = kind, index
        self.update()

    def zoom_by(self, factor, anchor=None):
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = (anchor - self.pan) / self.zoom
        self.zoom = min(64.0, max(0.15, self.zoom * factor))
        self.pan = anchor - before * self.zoom
        self.update()

    def ring_indices(self, kind):
        points = self.inputs if kind == "input" else self.outputs
        if not points:
            return set()
        pitch = 850 * self.zoom / max(1, len(points) - 1)
        stride = 2 ** max(0, math.ceil(math.log2(18 / max(0.01, pitch))))
        indices = set(range(0, len(points), stride)) | {len(points) - 1}
        if self.selection is not None:
            selected_kind, index = self.selection
            if selected_kind == kind:
                indices.add(index)
            elif selected_kind == "edge":
                edge = self.payload["edges"][index]
                indices.add(edge["source" if kind == "input" else "target"])
        return indices

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#09070c"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        font = QFont("Segoe UI")
        font.setPixelSize(15)
        painter.setFont(font)
        if self.payload is None:
            painter.setPen(QColor("#a99ab9"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Static parameters not loaded"
            )
            return
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)
        maximum = max((abs(edge["weight"]) for edge in self.payload["edges"]), default=1) or 1
        for index, (edge, path) in enumerate(zip(self.payload["edges"], self.paths, strict=True)):
            colour = QColor("#ac7af4" if edge["weight"] >= 0 else "#e78aca")
            colour.setAlpha(100)
            width = (0.35 + 0.7 * math.sqrt(abs(edge["weight"]) / maximum)) / self.zoom
            if self.selection == ("edge", index):
                colour, width = QColor("#f2ecf8"), 2.4 / self.zoom
            painter.setPen(QPen(colour, width))
            painter.drawPath(path)
        viewport = self.rect().adjusted(-24, -24, 24, 24)
        for kind, points in (("input", self.inputs), ("output", self.outputs)):
            rings = self.ring_indices(kind)
            for index, point in enumerate(points):
                screen_point = point * self.zoom + self.pan
                if not viewport.contains(screen_point.toPoint()):
                    continue
                selected = self.selection == (kind, index)
                if index not in rings:
                    painter.setPen(Qt.PenStyle.NoPen)
                    painter.setBrush(QColor("#78618c"))
                    painter.drawEllipse(point, 0.85 / self.zoom, 0.85 / self.zoom)
                    continue
                radius = 5.7 / self.zoom
                painter.setPen(QPen(QColor("#f5edff" if selected else "#d9b9ff"), 1.3 / self.zoom))
                painter.setBrush(QColor("#09070c"))
                painter.drawEllipse(point, radius, radius)
                painter.setPen(QPen(QColor("#51376a"), 0.8 / self.zoom))
                painter.drawEllipse(point, radius + 3 / self.zoom, radius + 3 / self.zoom)
                painter.setPen(Qt.PenStyle.NoPen)
                painter.setBrush(QColor("#dbb8ff" if selected else "#9774b9"))
                painter.drawEllipse(point, 2 / self.zoom, 2 / self.zoom)
        painter.resetTransform()
        painter.setPen(QColor("#f2ecf8"))
        for title, points in (("Input", self.inputs), ("Output before bias", self.outputs)):
            x = sum(p.x() for p in points) / len(points) * self.zoom + self.pan.x()
            y = min(p.y() for p in points) * self.zoom + self.pan.y() - 42
            width = min(340, self.width() - 20)
            left = min(self.width() - width - 10, max(10, x - width / 2))
            painter.drawText(
                QRectF(left, y, width, 30),
                Qt.AlignmentFlag.AlignCenter,
                f"{title} · {len(points)} features",
            )
        painter.setPen(QColor("#a99ab9"))
        painter.drawText(14, 24, "Static model parameters · FP16")
        painter.drawText(
            14,
            self.height() - 16,
            "Zoom to reveal every feature · curved lines are actual parameters",
        )

    def resizeEvent(self, event):
        self.fit()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        self.zoom_by(1.2 ** (event.angleDelta().y() / 120), event.position())
        event.accept()

    def keyPressEvent(self, event):
        key = event.key()
        if key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.2)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.2)
        elif key == Qt.Key.Key_Home:
            self.fit()
        elif key in (Qt.Key.Key_Left, Qt.Key.Key_Right, Qt.Key.Key_Up, Qt.Key.Key_Down):
            delta = {
                Qt.Key.Key_Left: QPointF(40, 0),
                Qt.Key.Key_Right: QPointF(-40, 0),
                Qt.Key.Key_Up: QPointF(0, 40),
                Qt.Key.Key_Down: QPointF(0, -40),
            }
            self.pan += delta[key]
            self.update()
        else:
            super().keyPressEvent(event)
            return
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self._drag = event.position()
            self._moved = False

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._moved |= abs(delta.x()) + abs(delta.y()) > 2
            self.pan += delta
            self._drag = event.position()
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        self._drag = None
        if self._moved:
            return
        point = (event.position() - self.pan) / self.zoom
        for kind, points in (("input", self.inputs), ("output", self.outputs)):
            if not points:
                continue
            nearest = min(
                enumerate(points),
                key=lambda item: (item[1] - point).manhattanLength(),
            )
            if (nearest[1] - point).manhattanLength() <= 11 / self.zoom:
                self.node_selected.emit(kind, nearest[0])
                return
        stroker = QPainterPathStroker()
        stroker.setWidth(9 / self.zoom)
        for index, path in enumerate(self.paths):
            if stroker.createStroke(path).contains(point):
                self.edge_selected.emit(index)
                return


class StaticWeightView(QWidget):
    def __init__(self, client=None):
        super().__init__()
        self.client = client
        self.active = self.running = self._stopping = False
        self.pending = self._inflight = None
        self.descriptors = []
        self._requested_block = None
        self._model_identity = None
        self._parameters = None
        self._generation = 0
        self._fresh_metadata = False
        self._tokens = set()
        self._list_mode = 0
        self.reads = CatalogueReads(QApplication.instance())
        self.reads.done.connect(self._read_finished)
        self.destroyed.connect(lambda _owner=None, reads=self.reads: reads.pending.clear())
        self.block = QComboBox(accessibleName="Parameter transformer block")
        self.projection = QComboBox(accessibleName="Actual parameter projection")
        self.output = QSpinBox(minimum=-1, maximum=-1, value=-1)
        self.output.setSpecialValueText("All output features")
        self.output.setAccessibleName("Show strongest incoming parameters for output feature")
        reload = button("Reload")
        reload.clicked.connect(self._reload)
        fit = button("Fit")
        self.surface = WeightSurface()
        fit.clicked.connect(self.surface.fit)
        self.mode = QComboBox(accessibleName="Keyboard parameter navigation")
        self.mode.addItems(["Connections", "Input features", "Output features"])
        self.items = QListWidget(accessibleName="Exact parameter edges and feature indices")
        self.inspector = label("Select an actual parameter or feature.")
        self.inspector.setWordWrap(True)
        self.inspector.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.inspector.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
            | Qt.TextInteractionFlag.TextSelectableByKeyboard
        )
        self.inspector.setStyleSheet("font-size: 17px;")
        self.status = label("Static parameters · waiting for the local owner service.", "muted")
        self.status.setWordWrap(True)
        self.legend = label(
            "Static weights · learned parameters, not measured activity. Violet + / pink − · thickness = larger |weight| (square-root scale). White means selected connection, not activation.",
            "muted",
        )
        self.legend.setWordWrap(True)
        controls = QHBoxLayout()
        for widget in (self.block, self.projection, self.output, reload, fit):
            controls.addWidget(widget)
        side = QWidget()
        sidebar = QVBoxLayout(side)
        sidebar.addWidget(self.mode)
        sidebar.addWidget(self.items, 1)
        details = QScrollArea(widgetResizable=True)
        details.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        details.setWidget(self.inspector)
        details.setMinimumHeight(180)
        sidebar.addWidget(details, 1)
        split = QSplitter()
        split.addWidget(self.surface)
        split.addWidget(side)
        split.setSizes([850, 320])
        layout = QVBoxLayout(self)
        layout.addLayout(controls)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)
        layout.addWidget(self.legend)
        self.block.currentIndexChanged.connect(self._block_changed)
        self.projection.currentIndexChanged.connect(self._selection_changed)
        self.output.valueChanged.connect(self._selection_changed)
        self.mode.currentIndexChanged.connect(self._populate_items)
        self.items.currentRowChanged.connect(self._inspect)
        self.surface.edge_selected.connect(self._edge_selected)
        self.surface.node_selected.connect(self._node_selected)

    def set_active(self, active):
        active = bool(active and self.isVisible() and not self._stopping)
        if active == self.active:
            return
        self.active = active
        if not self.active:
            self.pending = None
            return
        if self.client is None:
            self.status.setText("Static parameters unavailable: no local owner client.")
        else:
            self._refresh_metadata()

    def set_parameters(self, packet):
        if packet is not None and (
            not isinstance(packet, dict)
            or not isinstance(packet.get("projections"), list)
            or len(packet["projections"]) != 72
        ):
            raise ValueError("Detail snapshots require the complete verified E5 parameter packet.")
        if packet is not None and packet is self._parameters:
            return
        self._parameters = packet
        self._generation += 1
        self._fresh_metadata = False
        self.pending = None
        self._clear_display()
        if self.active and self.client is not None:
            self._refresh_metadata()

    def _reload(self):
        if not self.active or self.client is None:
            return
        self._refresh_metadata()

    def _refresh_metadata(self):
        self._generation += 1
        self._fresh_metadata = False
        self._clear_display()
        self.pending = ("descriptors", self._generation)
        self._send()

    def select_block(self, block):
        if type(block) is not int or not 0 <= block < 12:
            raise ValueError(f"Transformer block must be an integer in [0, 11], got {block!r}")
        self._requested_block = block
        index = self.block.findData(block)
        if self.descriptors and index >= 0:
            self.block.setCurrentIndex(index)
            return
        self._clear_display()
        if self.descriptors:
            with QSignalBlocker(self.block):
                self.block.setCurrentIndex(-1)
            self.pending = None
            self.status.setText(
                f"Static parameters unavailable: Block {block + 1} is absent from this model."
            )
            return
        self.status.setText(f"Waiting for Block {block + 1} parameter metadata.")
        if self.active and not self.descriptors and self.client is not None:
            self.pending = ("descriptors", self._generation)
            self._send()

    def _block_changed(self, _index=None):
        self._requested_block = self.block.currentData()
        selected = self.projection.currentData()
        with QSignalBlocker(self.projection):
            self.projection.clear()
            for item in self.descriptors:
                if item["block"] == self.block.currentData():
                    self.projection.addItem(PROJECTIONS[item["projection"]], item["projection"])
            self.projection.setCurrentIndex(max(0, self.projection.findData(selected)))
        self._selection_changed()

    def _selection_changed(self, _value=None):
        descriptor = next(
            (
                item
                for item in self.descriptors
                if item["block"] == self.block.currentData()
                and item["projection"] == self.projection.currentData()
            ),
            None,
        )
        if descriptor is None:
            return
        self._clear_display()
        with QSignalBlocker(self.output):
            self.output.setMaximum(descriptor["output_dim"] - 1)
        if self.active:
            if not self._fresh_metadata:
                return
            if (
                self.output.value() == -1
                and (snapshot := self._snapshot_projection(descriptor)) is not None
            ):
                self.pending = None
                self.surface.set_payload(snapshot)
                self.status.setText(
                    f"Verified snapshot · retained stratified sample · {snapshot['shown_count']} / {snapshot['total_weight_count']:,} parameters shown"
                )
                self._populate_items()
                return
            self.pending = (
                "weights",
                descriptor["block"],
                descriptor["projection"],
                None if self.output.value() == -1 else self.output.value(),
                self._generation,
            )
            self._send()

    def _snapshot_projection(self, descriptor):
        packet = self._parameters
        if packet is None:
            return None
        try:
            if (packet["model_revision"], packet["model_space"]) != self._model_identity:
                raise ValueError("The detail snapshot belongs to another model.")
            rows = [
                row
                for row in packet["projections"]
                if (row["block"], row["projection"])
                == (descriptor["block"], descriptor["projection"])
            ]
            if len(rows) != 1:
                raise ValueError("The detail snapshot projection is missing or repeated.")
            value = validate_parameter_weights(
                rows[0], descriptor["block"], descriptor["projection"], limit=256, output_index=None
            )
            if any(
                value[field] != descriptor[field]
                for field in (
                    "block",
                    "projection",
                    "input_dim",
                    "output_dim",
                    "total_weight_count",
                    "source",
                    "target",
                    "operator_id",
                    "operation",
                    "orientation",
                )
            ):
                raise ValueError("The current projection differs from the detail snapshot.")
            if any(
                value["provenance"][field] != descriptor["provenance"][field]
                for field in (
                    "prepared_manifest_sha256",
                    "xml_sha256",
                    "bin_sha256",
                    "constant_id",
                    "offset_bytes",
                    "size_bytes",
                    "matrix_layout",
                )
            ):
                raise ValueError("The detail snapshot artifact version changed.")
            return value
        except (OSError, ValueError, KeyError, TypeError):
            self._parameters = None
            return None

    def _clear_display(self):
        self.surface.payload = None
        self.surface.inputs, self.surface.outputs, self.surface.paths = [], [], []
        self.surface.selection = None
        self.surface.update()
        with QSignalBlocker(self.items):
            self.items.clear()
        self.inspector.setText("Select an actual parameter or feature.")

    def _send(self):
        if not self.active or self.running or self.pending is None:
            return
        key = self.pending
        operation = (
            self.client.activity_projections
            if key[0] == "descriptors"
            else partial(
                self.client.activity_weights, key[1], key[2], limit=512, output_index=key[3]
            )
        )

        def perform():
            try:
                return True, operation()
            except (OSError, ValueError) as error:
                return False, str(error)

        self.running, self._inflight = True, key
        self.status.setText("Reading model parameters locally.")
        self.reads.request("parameters", perform, self._received, self._failed)
        self._tokens.add(("parameters", self.reads.serial))

    def _received(self, result):
        key, self._inflight = self._inflight, None
        self.running = False
        current = key == self.pending
        if current:
            self.pending = None
        if self.active and current:
            succeeded, value = result
            if not succeeded:
                self._clear_display()
                self.status.setText("Static parameters unavailable: " + value)
            elif key[0] == "descriptors":
                self._model_identity = value["model_revision"], value["model_space"]
                self.descriptors = value["projections"]
                self._fresh_metadata = True
                if (
                    self._parameters is not None
                    and (
                        self._parameters.get("model_revision"),
                        self._parameters.get("model_space"),
                    )
                    != self._model_identity
                ):
                    self._parameters = None
                with QSignalBlocker(self.block):
                    self.block.clear()
                    for block in sorted({item["block"] for item in self.descriptors}):
                        self.block.addItem(f"Block {block + 1}", block)
                    index = self.block.findData(self._requested_block)
                    if self._requested_block is not None:
                        self.block.setCurrentIndex(index)
                if self.block.currentIndex() < 0:
                    self._clear_display()
                    self.status.setText(
                        f"Static parameters unavailable: Block {self._requested_block + 1} is absent from this model."
                    )
                else:
                    self._block_changed()
            elif (value["model_revision"], value["model_space"]) != self._model_identity:
                self.descriptors = []
                self._parameters = None
                self._fresh_metadata = False
                self._clear_display()
                self.inspector.setText(
                    "Parameter identity changed; previous values are unavailable."
                )
                self.status.setText("Static parameters unavailable: model changed. Choose Reload.")
            else:
                self.surface.set_payload(value)
                self.status.setText(
                    f"{value['shown_count']} / {value['total_weight_count']:,} parameters shown · {value['sampling'].replace('_', ' ')}"
                )
                self._populate_items()
        self._send()

    def _failed(self):
        self._received((False, "Local parameter read failed."))

    def _read_finished(self, token, _value, _succeeded):
        self._tokens.discard(token)

    def _populate_items(self, _index=None):
        payload = self.surface.payload
        mode = self.mode.currentIndex()
        selected = self.items.currentRow() if mode == self._list_mode else -1
        self._list_mode = mode
        with QSignalBlocker(self.items):
            self.items.clear()
            if payload is None:
                return
            if mode == 0:
                for edge in payload["edges"]:
                    self.items.addItem(
                        f"Input {edge['source']} → output {edge['target']}    {edge['weight']:+.7g}"
                    )
            else:
                count = payload["input_dim"] if mode == 1 else payload["output_dim"]
                self.items.addItems(
                    [f"{'Input' if mode == 1 else 'Output'} feature {i}" for i in range(count)]
                )
        if mode == 0 and selected < 0:
            selected = 0
        if selected >= self.items.count():
            selected = 0 if mode == 0 else -1
        self.items.setCurrentRow(selected)

    def _inspect(self, index):
        payload = self.surface.payload
        if index < 0 or payload is None:
            return
        mode = self.mode.currentIndex()
        kind = ("edge", "input", "output")[mode]
        self.surface.select(kind, index)
        if mode == 0:
            edge = payload["edges"][index]
            selected = f"Selected connection\nInput {edge['source']} → output {edge['target']}\nW[{edge['target']}, {edge['source']}] = {edge['weight']!r}\nLearned weight, not an activation or measured contribution."
        else:
            selected = f"{kind.title()} feature {index}"
            if mode == 2 and self.output.value() != index:
                self.output.setValue(index)
                return
        provenance = payload["provenance"]
        source, target = payload["source"], payload["target"]
        source_label = SEMANTICS.get(source["semantics"], source["semantics"].replace("_", " "))
        target_label = SEMANTICS.get(target["semantics"], target["semantics"].replace("_", " "))
        self.inspector.setText(
            f"{selected}\n\n{source_label}\n→ {target_label}\n\nProjection operator {payload['operator_id']}\n{payload['operation']}\nSource operator {source['operator_id']}\n{source['operation']}\nRaw source semantics: {source['semantics']}\nTarget operator {target['operator_id']}\n{target['operation']}\nRaw target semantics: {target['semantics']}\n\n{payload['dtype']} parameters\nRevision {payload['model_revision']}\nConstant {provenance['constant_id']}\nBytes {provenance['offset_bytes']} + {provenance['size_bytes']}\nMatrix layout: {provenance['matrix_layout']}\nBIN verified: {provenance['bin_verified']}\nPrepared SHA-256 {provenance['prepared_manifest_sha256']}\nXML SHA-256 {provenance['xml_sha256']}\nBIN SHA-256 {provenance['bin_sha256']}"
        )

    def _edge_selected(self, index):
        self.mode.setCurrentIndex(0)
        self.items.setCurrentRow(index)

    def _node_selected(self, kind, index):
        self.mode.setCurrentIndex(1 if kind == "input" else 2)
        self.items.setCurrentRow(index)

    def shutdown(self, timeout=5):
        self._stopping = True
        self.set_active(False)
        deadline = time.monotonic() + timeout
        while self._tokens or self.running:
            QApplication.processEvents(QEventLoop.ProcessEventsFlag.ExcludeUserInputEvents)
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.005)
        return True

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
