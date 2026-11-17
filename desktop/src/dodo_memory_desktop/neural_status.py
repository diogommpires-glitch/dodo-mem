"""Keep neural readiness visible while local catalogue browsing stays independent."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QWidget

from .ui import button, label
from .whole_library import CatalogueReads


class NeuralStatus(QWidget):
    def __init__(self, client):
        super().__init__()
        self.client = client
        self.reads = CatalogueReads(self)
        self.summary = label("Local neural service: checking…", "muted")
        self.summary.setTextFormat(Qt.TextFormat.PlainText)
        self.summary.setWordWrap(True)
        self.summary.setAccessibleName("Local neural memory status")
        self.retry = button("Check neural service")
        self.retry.clicked.connect(self.refresh)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.summary, 1)
        layout.addWidget(self.retry)

    def refresh(self):
        self.reads.request("neural-status", self.client.status, self._ready, self._unavailable)

    def _ready(self, status):
        state = status.get("state") or status.get("status")
        if not state:
            state = "ready" if status.get("ready") is True else "readiness not reported"
        fields = [f"Local neural memory: {state}"]
        for key, caption in (
            ("device", "Device"),
            ("model", "Model"),
            ("embedded", "Embedded"),
            ("pending", "Pending"),
            ("lag_seconds", "Index lag (s)"),
        ):
            if key in status and status[key] is not None:
                value = status[key]
                if key == "device" and isinstance(value, list):
                    value = ", ".join(value)
                elif key == "lag_seconds":
                    value = f"{value:.1f}"
                elif type(value) is int:
                    value = f"{value:,}"
                fields.append(f"{caption}: {value}")
        worker_alive = status.get("background_worker_alive")
        if worker_alive is not None:
            fields.append("Worker: online" if worker_alive else "Worker: stopped")
        if status.get("source_failures"):
            fields.append(f"Recorded source failures: {status['source_failures']:,}")
        if status.get("unavailable_providers"):
            fields.append(f"Unavailable source adapters: {status['unavailable_providers']:,}")
        reason = (status.get("reason") or "").strip()
        if (
            state in {"failed", "paused", "unavailable", "degraded", "blocked", "stopped"}
            or worker_alive is False
        ) and reason:
            fields.append("Reason: " + reason.splitlines()[0][:300])
        metrics = status.get("telemetry")
        self.summary.setToolTip("")
        if metrics is not None:
            fields.extend(
                [
                    f"Encoder busy: {metrics['inference_busy_percent']:.1f}% ({metrics['window_seconds']:g} s)",
                    f"Active: {metrics['active_inferences']}",
                    f"Completed: {metrics['completed_inferences']}",
                ]
            )
            if metrics["last_inference_ms"] is not None:
                fields.append(f"Last inference: {metrics['last_inference_ms']:.1f} ms")
            hardware = metrics["hardware_utilization_percent"]
            fields.append(
                "NPU hardware usage: unavailable"
                if hardware is None
                else f"NPU hardware usage: {hardware:.1f}%"
            )
            self.summary.setToolTip(
                "Encoder busy is measured inference time, not hardware utilisation. "
                + (metrics.get("hardware_utilization_reason") or "")
            )
        else:
            fields.append("Inference telemetry unavailable")
        if status.get("error"):
            fields.append("Some retrieval channels are unavailable")
        self.summary.setText(" · ".join(fields))

    def _unavailable(self):
        self.summary.setText(
            "Local neural service unavailable. Catalogue graph remains available; "
            "Topics and question paths need the local service."
        )
