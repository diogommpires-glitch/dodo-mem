"""The complete E5 network, using verified real projection weights."""

import math
from datetime import datetime

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QWidget

PROJECTIONS = {
    "query": ("input", "query", 384, 384),
    "key": ("input", "key", 384, 384),
    "value": ("input", "value", 384, 384),
    "attention_output": ("context", "attention", 384, 384),
    "ff_up": ("ff_input", "ff_linear", 384, 1536),
    "ff_down": ("gelu", "ff_output", 1536, 384),
}
STAGES = {
    "input": (40, 430, 700, 384, "Input"),
    "query": (260, 180, 220, 384, "Q linear"),
    "key": (260, 430, 220, 384, "K linear"),
    "value": (260, 680, 220, 384, "V linear"),
    "context": (440, 430, 650, 384, "Head context"),
    "attention": (580, 430, 650, 384, "Attn projection"),
    "ff_input": (670, 430, 650, 384, "Norm output"),
    "ff_linear": (830, 430, 760, 1536, "FF up linear"),
    "gelu": (920, 430, 760, 1536, "GELU output"),
    "ff_output": (1080, 430, 650, 384, "FF down linear"),
    "output": (1180, 430, 650, 384, "Block output"),
}


class NetworkOverview(QWidget):
    block_selected = Signal(int)

    def __init__(self):
        super().__init__()
        self.setMinimumSize(720, 440)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMouseTracking(True)
        self.setAccessibleName(
            "Whole E5 network: twelve blocks and seventy-two real parameter fans"
        )
        self.architecture = self.capture = self.parameters = None
        self.running = self.paused = self.replaying = False
        self.selected_block = 0
        self.parameter_status = "Real parameter connections have not loaded"
        self.weight_batches = {}
        self.drawn_weight_count = self.total_weight_count = 0
        self.zoom, self.pan = 1.0, QPointF()
        self._drag, self._moved = None, False
        self.block_rects, self.block_centres, self._origins = {}, {}, {}
        for block in range(12):
            row, column = divmod(block, 4)
            if row == 1:
                column = 3 - column
            origin = QPointF(column * 1360, row * 1060)
            self._origins[block] = origin
            self.block_rects[block] = QRectF(origin.x(), origin.y(), 1220, 880)
            self.block_centres[block] = origin + QPointF(610, 440)
        self.world_bounds = QRectF(-100, -160, 5500, 3320)
        self.fit()

    def set_architecture(self, value):
        if value["num_hidden_layers"] != 12 or value["hidden_size"] != 384:
            raise ValueError("The E5 diagram requires twelve blocks and 384 hidden features")
        previous = self.architecture["model"] if self.architecture else None
        if previous is None or (value["model"]["revision"], value["model"]["space"]) != (
            previous["revision"],
            previous["space"],
        ):
            self.capture = None
            self.set_parameters(None)
        self.architecture = value
        self._describe_selection()
        self.update()

    def set_parameters(self, value):
        self.parameters = None
        self.weight_batches.clear()
        self.drawn_weight_count = self.total_weight_count = 0
        self.parameter_status = "Real parameter connections have not loaded"
        valid = value is not None and self.architecture is not None
        if valid:
            model = self.architecture["model"]
            identity = value["model_revision"], value["model_space"]
            projections = value["projections"]
            expected = {(block, name) for block in range(12) for name in PROJECTIONS}
            supplied = {(item["block"], item["projection"]) for item in projections}
            valid = (
                identity == (model["revision"], model["space"])
                and len(projections) == 72
                and supplied == expected
            )
            for item in projections:
                if item["projection"] not in PROJECTIONS:
                    valid = False
                    break
                _source, _target, inputs, outputs = PROJECTIONS[item["projection"]]
                if (item["model_revision"], item["model_space"]) != identity or (
                    item["input_dim"],
                    item["output_dim"],
                ) != (inputs, outputs):
                    valid = False
                    break
        if not valid:
            if value is not None:
                self.parameter_status = "Incomplete or changed model parameters are unavailable"
            self.update()
            return False
        for item in projections:
            source, target, _inputs, _outputs = PROJECTIONS[item["projection"]]
            maximum = max((abs(edge["weight"]) for edge in item["edges"]), default=0) or 1
            for edge in item["edges"]:
                start = self.feature_point(item["block"], source, edge["source"])
                end = self.feature_point(item["block"], target, edge["target"])
                strength = min(3, int(math.sqrt(abs(edge["weight"]) / maximum) * 3))
                key = item["block"], edge["weight"] >= 0, strength
                self.weight_batches.setdefault(key, QPainterPath()).addPath(self._curve(start, end))
            self.drawn_weight_count += len(item["edges"])
            self.total_weight_count += item["total_weight_count"]
        self.parameters = value
        self.parameter_status = (
            f"{self.drawn_weight_count:,} / {self.total_weight_count:,} dense weights drawn"
        )
        if value.get("from_snapshot"):
            self.parameter_status += " · verified snapshot"
        self.update()
        return True

    def set_capture(self, value):
        if value is None:
            self.capture = None
        elif self.paused:
            return
        elif self.architecture is not None:
            model = self.architecture["model"]
            self.capture = (
                value
                if (value["model_revision"], value["model_space"])
                == (model["revision"], model["space"])
                else None
            )
        self._describe_selection()
        self.update()

    def set_running(self, value):
        self.running = bool(value)
        self.update()

    def set_paused(self, value):
        self.paused = bool(value)
        self.update()

    def set_replaying(self, value):
        self.replaying = bool(value)
        self.update()

    def measured_tensors(self):
        return set(self.capture["tensors"]) if self.capture is not None else set()

    def completed_path(self):
        return "block.11" in self.measured_tensors() and not self.capture.get(
            "token_is_padding", False
        )

    def status_text(self):
        if self.replaying and self.capture:
            return "Replay · " + self._capture_label()
        if self.architecture and self.architecture.get("runtime_available") is False:
            status = "Encoder paused · static parameters"
            return status + " · last captured values" if self.capture else status
        if self.paused:
            return "Paused display"
        if self.running:
            return "Encoder running · layer timings unavailable"
        return "Captured · " + self._capture_label() if self.capture else "No captured values"

    def _capture_label(self):
        timestamp = datetime.fromtimestamp(self.capture["completed_at"]).strftime("%H:%M:%S")
        return f"{timestamp} · {self.capture['id'][:8]}"

    def _local(self, block, point):
        if block // 4 == 1:
            point = QPointF(1220 - point.x(), point.y())
        return self._origins[block] + point

    def feature_point(self, block, stage, index):
        x, y, length, count, _label = STAGES[stage]
        fraction = index / max(1, count - 1)
        return self._local(
            block, QPointF(x + 10 * math.sin(math.pi * fraction), y + (fraction - 0.5) * length)
        )

    def to_screen(self, point):
        return point * self.zoom + self.pan

    def screen_block_rect(self, index):
        rect = self.block_rects[index]
        return QRectF(self.to_screen(rect.topLeft()), rect.size() * self.zoom)

    def fit(self):
        self.zoom = max(
            0.05,
            min(
                (self.width() - 30) / self.world_bounds.width(),
                (self.height() - 142) / self.world_bounds.height(),
            ),
        )
        self.pan = (
            QPointF(self.width() / 2, 85 + (self.height() - 142) / 2)
            - self.world_bounds.center() * self.zoom
        )
        self.update()

    def zoom_by(self, factor, anchor=None):
        anchor = anchor or QPointF(self.width() / 2, self.height() / 2)
        before = (anchor - self.pan) / self.zoom
        self.zoom = max(0.05, min(20, self.zoom * factor))
        self.pan = anchor - before * self.zoom
        self.update()

    def focus_block(self, index):
        self.selected_block = index
        rect = self.block_rects[index].adjusted(-35, -90, 35, 25)
        self.zoom = min((self.width() - 30) / rect.width(), (self.height() - 142) / rect.height())
        self.pan = (
            QPointF(self.width() / 2, 85 + (self.height() - 142) / 2) - rect.center() * self.zoom
        )
        self.update()

    @staticmethod
    def _curve(start, end):
        path = QPainterPath(start)
        bend = (end.x() - start.x()) * 0.43
        sweep = (end.y() - start.y()) * 0.12
        path.cubicTo(start + QPointF(bend, sweep), end - QPointF(bend, sweep), end)
        return path

    def _describe_selection(self):
        state = (
            "measured output"
            if f"block.{self.selected_block}" in self.measured_tensors()
            else "output unavailable"
        )
        self.setAccessibleDescription(
            f"Block {self.selected_block + 1}, {state}. Left/right select; Enter opens detail; F fits the block; Home fits all twelve; plus/minus zoom. Solid fans are verified fixed model weights. Dashed paths are tensor operations. Ring density changes with zoom; features retain their real indices."
        )

    @staticmethod
    def _font(painter, size, *, bold=False):
        font = QFont("Segoe UI")
        font.setPixelSize(size)
        font.setBold(bold)
        painter.setFont(font)

    def _bridge(self, painter, block, source, target, *, residual=False):
        start = self.feature_point(block, source, STAGES[source][3] // 2)
        end = self.feature_point(block, target, STAGES[target][3] // 2)
        path = self._curve(start, end)
        if residual:
            path = QPainterPath(start)
            floor = self._origins[block].y() + 855
            path.cubicTo(QPointF(start.x(), floor), QPointF(end.x(), floor), end)
        colour = QColor("#8f76ac" if self.completed_path() else "#5b4d69")
        pen = QPen(colour, 0.7, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _stage(self, painter, block, name):
        _x, _y, length, count, _label = STAGES[name]
        values = (
            self.capture["tensors"].get(f"block.{block}")
            if name == "output" and self.capture
            else None
        )
        maximum = max(map(abs, values), default=0) if values is not None else 0
        pitch = length * self.zoom / max(1, count - 1)
        stride = 2 ** max(0, math.ceil(math.log2(7 / max(0.01, pitch))))
        for index in sorted(set(range(0, count, stride)) | {count - 1}):
            point = self.feature_point(block, name, index)
            painter.setPen(QPen(QColor("#b594d2"), 0.7 / self.zoom))
            painter.setBrush(QColor("#09070c"))
            if values is not None:
                magnitude = math.sqrt(abs(values[index]) / maximum) if maximum else 0
                colour = QColor("#ae83ef" if values[index] >= 0 else "#db89b9")
                colour.setAlpha(round(35 + 220 * magnitude))
                painter.setBrush(colour)
            radius = min(4.6, max(1.45, self.zoom * 7)) / self.zoom
            painter.drawEllipse(point, radius, radius)

    def paintEvent(self, _event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#09070c"))
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        self._font(painter, 16)
        if self.architecture is None:
            painter.setPen(QColor("#dcc9ee"))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Waiting for verified E5 architecture"
            )
            return
        visible = {
            block
            for block in range(12)
            if self.screen_block_rect(block).intersects(QRectF(self.rect()))
        }
        painter.save()
        painter.setClipRect(QRectF(0, 64, self.width(), self.height() - 120))
        painter.translate(self.pan)
        painter.scale(self.zoom, self.zoom)
        for (block, positive, strength), path in self.weight_batches.items():
            if block not in visible:
                continue
            colour = QColor("#ad7dec" if positive else "#dc87c4")
            colour.setAlpha(95 + strength * 13)
            pen = QPen(colour, 0.42 + strength * 0.12)
            pen.setCosmetic(True)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawPath(path)
        for block in visible:
            for source in ("query", "key", "value"):
                self._bridge(painter, block, source, "context")
            for source, target in (
                ("attention", "ff_input"),
                ("ff_linear", "gelu"),
                ("ff_output", "output"),
            ):
                self._bridge(painter, block, source, target)
            self._bridge(painter, block, "input", "ff_input", residual=True)
            self._bridge(painter, block, "ff_input", "output", residual=True)
            for name in STAGES:
                self._stage(painter, block, name)
            if block < 11:
                self._between_blocks(painter, block)
        self._terminals(painter)
        painter.restore()
        self._labels(painter, visible)
        self._header(painter)

    def _between_blocks(self, painter, block):
        start = self.feature_point(block, "output", 192)
        end = self.feature_point(block + 1, "input", 192)
        path = self._curve(start, end)
        if block in (3, 7):
            side = self.world_bounds.right() - 25 if block == 3 else self.world_bounds.left() + 25
            path = QPainterPath(start)
            path.cubicTo(QPointF(side, start.y()), QPointF(side, end.y()), end)
        pen = QPen(
            QColor("#9273b4" if self.completed_path() else "#685271"), 1, Qt.PenStyle.DashLine
        )
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawPath(path)

    def _terminals(self, painter):
        painter.setBrush(Qt.BrushStyle.NoBrush)
        pen = QPen(QColor("#5b4d69"), 0.9, Qt.PenStyle.DashLine)
        pen.setCosmetic(True)
        painter.setPen(pen)
        painter.drawPath(self._curve(QPointF(450, -105), self.feature_point(0, "input", 192)))
        painter.drawPath(self._curve(self.feature_point(11, "output", 192), QPointF(5000, 3140)))
        if {"pooled", "output"}.issubset(self.measured_tensors()) and not self.capture.get(
            "token_is_padding", False
        ):
            pen.setColor(QColor("#c5a4ed"))
            pen.setWidthF(1.4)
            painter.setPen(pen)
        painter.drawLine(QPointF(5000, 3140), QPointF(5290, 3140))

    def _labels(self, painter, visible):
        for block in visible:
            rect = self.screen_block_rect(block)
            self._font(painter, 16, bold=block == self.selected_block)
            painter.setPen(QColor("#e2c9f3" if block == self.selected_block else "#bfa6d4"))
            painter.drawText(
                QRectF(rect.x() - 8, rect.y() - 27, rect.width() + 16, 24),
                Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
                f"Block {block + 1}",
            )
            if self.zoom < 0.33:
                continue
            self._font(painter, 14)
            for name, (_x, _y, _length, count, label) in STAGES.items():
                top = self.to_screen(self.feature_point(block, name, 0))
                painter.setPen(QColor("#bda5cf"))
                width, offset = 116, 0
                if self.zoom < 0.7:
                    label = {
                        "input": "Input",
                        "query": "Q",
                        "key": "K",
                        "value": "V",
                        "context": "Ctx",
                        "attention": "Attn",
                        "ff_input": "Norm",
                        "ff_linear": "FF ↑",
                        "gelu": "GELU",
                        "ff_output": "FF ↓",
                        "output": "Out",
                    }[name]
                    width = 56
                elif name in ("attention", "gelu", "ff_output"):
                    offset = 44
                label_rect = QRectF(top.x() - width / 2, top.y() - 44 - offset, width, 42)
                if label_rect.left() < 8 or label_rect.right() > self.width() - 8:
                    continue
                painter.drawText(
                    label_rect,
                    Qt.AlignmentFlag.AlignCenter | Qt.TextFlag.TextWordWrap,
                    f"{label}\n{count}",
                )
        self._font(painter, 14)
        painter.setPen(QColor("#d2b9e5"))
        for point, text in (
            (QPointF(60, -105), "Input → Embeddings · 384"),
            (QPointF(4530, 3140), "Masked mean · 384"),
            (QPointF(5120, 3140), "L2 output · 384"),
        ):
            screen = self.to_screen(point)
            width = 200
            margin = 0
            if text.startswith("Masked"):
                width, margin = 170, 190
            elif text.startswith("L2"):
                width = 150
            left = min(self.width() - width - margin, max(8, screen.x()))
            painter.drawText(QRectF(left, screen.y() - 22, width, 22), text)

    def _header(self, painter):
        painter.fillRect(QRectF(0, 0, self.width(), 64), QColor("#09070c"))
        painter.fillRect(QRectF(0, self.height() - 55, self.width(), 55), QColor("#09070c"))
        self._font(painter, 14)
        painter.setPen(QColor("#e0caef"))
        painter.drawText(
            QRectF(14, 8, self.width() - 28, 23),
            "12 blocks · 72 projections · " + self.parameter_status,
        )
        painter.setPen(QColor("#baa4cb"))
        painter.drawText(QRectF(14, 31, self.width() - 28, 23), self.status_text())
        painter.drawText(
            QRectF(14, self.height() - 50, self.width() - 28, 46),
            Qt.TextFlag.TextWordWrap,
            "Solid: learned weights · dashed: operations/residuals · fill: measured output. Wheel: zoom · drag: pan · F: fit block · Home: all.",
        )

    def resizeEvent(self, event):
        self.fit()
        super().resizeEvent(event)

    def wheelEvent(self, event):
        self.zoom_by(1.2 ** (event.angleDelta().y() / 120), event.position())
        event.accept()

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.setFocus()
            self._drag, self._moved = event.position(), False
            event.accept()

    def mouseMoveEvent(self, event):
        if self._drag is not None:
            delta = event.position() - self._drag
            self._moved |= abs(delta.x()) + abs(delta.y()) > 2
            self.pan += delta
            self._drag = event.position()
            self.update()
            return
        for block in range(12):
            if self.screen_block_rect(block).contains(event.position()):
                self.setToolTip(
                    f"Block {block + 1}: six real parameter fans. Q/K/V linear → bias, 12-head attention and concatenation → context → attention projection → bias, residual and normalisation → FF up → bias and GELU → FF down → bias, residual and normalisation → block output. Click for exact operator detail; scroll for feature columns."
                )
                return
        self.setToolTip("")

    def mouseReleaseEvent(self, event):
        if self._drag is None:
            return
        self._drag = None
        if self._moved:
            return
        for block in range(12):
            if self.screen_block_rect(block).contains(event.position()):
                self.selected_block = block
                self._describe_selection()
                self.update()
                self.block_selected.emit(block)
                return

    def keyPressEvent(self, event):
        key = event.key()
        if key == Qt.Key.Key_Left:
            self.selected_block = max(0, self.selected_block - 1)
        elif key == Qt.Key.Key_Right:
            self.selected_block = min(11, self.selected_block + 1)
        elif key == Qt.Key.Key_Home:
            self.selected_block = 0
            self.fit()
        elif key == Qt.Key.Key_End:
            self.selected_block = 11
        elif key == Qt.Key.Key_F:
            self.focus_block(self.selected_block)
        elif key in (Qt.Key.Key_Plus, Qt.Key.Key_Equal):
            self.zoom_by(1.2)
        elif key == Qt.Key.Key_Minus:
            self.zoom_by(1 / 1.2)
        elif key in (Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space):
            self.block_selected.emit(self.selected_block)
        else:
            super().keyPressEvent(event)
            return
        self._describe_selection()
        self.update()
        event.accept()
