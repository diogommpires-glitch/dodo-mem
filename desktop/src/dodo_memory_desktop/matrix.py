"""Prepare Matrix requests and approve their exact local execution boundary."""

import hashlib
import json
import re
from functools import partial
from uuid import uuid4

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QHeaderView,
    QLineEdit,
    QListWidget,
    QPlainTextEdit,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gateway import MEMORY_HOME, GatewayError
from .ui import ROLE, button, label, row_item


def checked_run(value):
    if not isinstance(value, dict):
        raise GatewayError("Matrix returned invalid run metadata.")
    request = value.get("request")
    if (
        not isinstance(value.get("run_id"), str)
        or not value["run_id"]
        or not isinstance(value.get("state"), str)
        or not isinstance(value.get("request_sha256"), str)
        or re.fullmatch(r"[0-9a-f]{64}", value["request_sha256"]) is None
        or not isinstance(request, dict)
        or not isinstance(request.get("question"), str)
        or not isinstance(request.get("mode"), str)
        or request.get("mode") not in {"plan", "work"}
        or not isinstance(request.get("providers"), list)
        or not request["providers"]
        or any(
            not isinstance(provider, str) or provider not in {"codex", "claude"}
            for provider in request["providers"]
        )
        or type(request.get("token_budget")) is not int
        or type(request.get("budget_seconds")) is not int
        or type(request.get("allow_uncertain_usage")) is not bool
        or (request.get("project") is not None and not isinstance(request["project"], str))
        or (value.get("result") is not None and not isinstance(value["result"], dict))
    ):
        raise GatewayError("Matrix returned invalid run metadata.")
    workers = (value.get("result") or {}).get("workers", [])
    if not isinstance(workers, list) or any(not isinstance(worker, dict) for worker in workers):
        raise GatewayError("Matrix returned invalid worker metadata.")
    return value


def request_text(request):
    return (
        f"Question\n{request['question']}\n\nMode: {request['mode']}\n"
        f"Project: {request.get('project') or 'None'}\n"
        f"Providers: {', '.join(request['providers'])}\n"
        f"Token budget: {request['token_budget']}\n"
        f"Time budget: {request['budget_seconds']} seconds\n"
        f"Allow uncertain usage: {request['allow_uncertain_usage']}"
    )


class Matrix(QWidget):
    def __init__(self, desktop):
        super().__init__()
        self.desktop = desktop
        self.connected = self.busy = self.active = self.owner_review = False
        self.current = self.approval_dialog = None
        home = str(getattr(desktop.gateway, "home", MEMORY_HOME)).casefold()
        scope = hashlib.sha256(home.encode("utf-8")).hexdigest()
        self.session_key = f"matrix/{scope}/session_id"
        saved = desktop.settings.value(self.session_key, "", type=str)
        self.session_id = saved if re.fullmatch(r"[0-9a-f]{32}", saved) else None
        self.runs = {}
        self.timer = QTimer(self, interval=2000)
        self.timer.timeout.connect(self._refresh)
        self.connect_matrix = button("Connect to agent matrix")
        self.connect_matrix.clicked.connect(self._connect)
        self.summary = label("Connect to prepare a request. Providers need local approval.")
        self.summary.setWordWrap(True)
        self.question = QPlainTextEdit(
            placeholderText="Task for the agents, up to 8,000 characters"
        )
        self.question.setAccessibleName("Matrix request")
        self.question.setMaximumHeight(115)
        self.question.textChanged.connect(self._actions)
        self.mode = QComboBox()
        self.mode.addItem("Plan", "plan")
        self.mode.addItem("Work in a Git project", "work")
        self.mode.currentIndexChanged.connect(self._actions)
        self.project = QLineEdit(placeholderText="Existing Git project for Work mode")
        self.project.setAccessibleName("Matrix Git project")
        self.project.textChanged.connect(self._actions)
        choose = button("Choose project")
        choose.clicked.connect(self._choose_project)
        project_row = QHBoxLayout()
        project_row.addWidget(self.project, 1)
        project_row.addWidget(choose)
        self.providers = [QCheckBox(name, checked=True) for name in ("Codex", "Claude")]
        provider_row = QHBoxLayout()
        for provider in self.providers:
            provider.toggled.connect(self._actions)
            provider_row.addWidget(provider)
        provider_row.addStretch()
        self.token_budget = QSpinBox(minimum=1, maximum=1000000, value=128000)
        self.seconds = QSpinBox(minimum=1, maximum=28800, value=28800)
        self.uncertain = QCheckBox("Allow providers with unreported usage or spending")
        self.prepare = button("Prepare request", role="primary")
        self.prepare.clicked.connect(self._prepare)
        form = QFormLayout()
        form.addRow("Mode", self.mode)
        form.addRow("Project", project_row)
        form.addRow("Providers", provider_row)
        budgets = QHBoxLayout()
        budgets.addWidget(label("Token budget"))
        budgets.addWidget(self.token_budget)
        budgets.addWidget(label("Time budget (seconds)"))
        budgets.addWidget(self.seconds)
        form.addRow(budgets)
        form.addRow(self.uncertain)
        form.addRow(self.prepare)
        self.run_list = QListWidget(accessibleName="Matrix runs")
        self.run_list.currentItemChanged.connect(self._selected)
        self.run_detail = QPlainTextEdit(readOnly=True)
        self.run_detail.setAccessibleName("Selected Matrix request")
        self.run_detail.setMaximumHeight(140)
        self.workers = QTableWidget(0, 5)
        self.workers.setAccessibleName("Reported Matrix workers")
        self.workers.setHorizontalHeaderLabels(("Provider", "Role", "Model", "Effort", "State"))
        self.workers.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.workers.verticalHeader().hide()
        self.workers.setMaximumHeight(135)
        self.output = QPlainTextEdit(readOnly=True)
        self.output.setAccessibleName("Matrix answer and errors")
        detail = QWidget()
        details = QVBoxLayout(detail)
        details.setContentsMargins(0, 0, 0, 0)
        details.addWidget(self.run_detail)
        details.addWidget(self.workers)
        details.addWidget(self.output, 1)
        split = QSplitter()
        split.addWidget(self.run_list)
        split.addWidget(detail)
        split.setSizes([230, 700])
        actions = QHBoxLayout()
        self.refresh = button("Refresh runs")
        self.refresh.clicked.connect(self._refresh)
        self.review_agents = button("Review agent requests")
        self.review_agents.setCheckable(True)
        self.review_agents.toggled.connect(self._review_agents)
        self.approve = button("Review and approve", role="primary")
        self.approve.clicked.connect(self._review)
        self.pause, self.resume, self.stop = button("Pause"), button("Resume"), button("Stop")
        for widget, action in ((self.pause, "pause"), (self.resume, "resume"), (self.stop, "stop")):
            widget.clicked.connect(partial(self._control, action))
        for widget in (
            self.refresh,
            self.review_agents,
            self.approve,
            self.pause,
            self.resume,
            self.stop,
        ):
            actions.addWidget(widget)
        actions.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 18, 24, 18)
        heading = QHBoxLayout()
        heading.addWidget(label("Agent matrix", "title"), 1)
        heading.addWidget(self.connect_matrix)
        layout.addLayout(heading)
        layout.addWidget(self.summary)
        layout.addWidget(self.question)
        layout.addLayout(form)
        layout.addLayout(actions)
        layout.addWidget(split, 1)
        layout.addWidget(label("Review and apply project changes in Hybrid.", "muted"))
        self._actions()

    def _actions(self):
        ready = (
            (self.connected or self.owner_review) and not self.busy and self.approval_dialog is None
        )
        self.connect_matrix.setEnabled(
            not self.busy and not self.connected and not self.owner_review
        )
        self.review_agents.setEnabled(not self.busy and self.approval_dialog is None)
        self.prepare.setEnabled(
            ready
            and not self.owner_review
            and 0 < len(self.question.toPlainText().strip()) <= 8000
            and any(provider.isChecked() for provider in self.providers)
            and (self.mode.currentData() != "work" or bool(self.project.text().strip()))
        )
        self.run_list.setEnabled(not self.busy)
        self.refresh.setEnabled(ready)
        state = self.current["state"] if self.current else None
        self.approve.setEnabled(ready and state == "awaiting_approval")
        self.pause.setEnabled(ready and state in {"queued", "running"})
        self.resume.setEnabled(ready and state == "paused")
        self.stop.setEnabled(
            ready
            and state in {"awaiting_approval", "queued", "running", "paused", "pause_requested"}
        )

    def _choose_project(self):
        path = QFileDialog.getExistingDirectory(self, "Choose an existing Git project")
        if path:
            self.project.setText(path)

    def _request(self, operation, complete):
        if self.busy:
            return
        for lane in ("matrix-list", "matrix-status"):
            self.desktop.pending.pop(lane, None)
        self.busy = True
        self._actions()

        def ready(value):
            self.busy = False
            complete(value)
            self._actions()

        def failed():
            self.busy = False
            self.summary.setText(self.desktop.status.text())
            self._actions()

        self.desktop.request("matrix-action", operation, ready, failed)

    def _connect(self):
        def connect():
            if self.session_id is not None:
                self._list_runs()
                return {"state": "ready", "session_id": self.session_id}
            value = self.desktop.gateway.call("connect_agent_matrix")
            if (
                not isinstance(value, dict)
                or value.get("state") != "ready"
                or not isinstance(value.get("session_id"), str)
                or re.fullmatch(r"[0-9a-f]{32}", value["session_id"]) is None
            ):
                raise GatewayError("Matrix is not ready for request preparation.")
            return value

        def ready(value):
            self.session_id = value["session_id"]
            self.desktop.settings.setValue(self.session_key, self.session_id)
            self.desktop.settings.sync()
            self.connected = True
            self.summary.setText("Connected · local approval required. No providers started.")
            self.set_active(self.active)

        self._request(connect, ready)

    def _run(self, tool, arguments, *, owner=False):
        call = self.desktop.gateway.matrix_owner if owner else self.desktop.gateway.call
        return checked_run(call(tool, arguments))

    def _prepare(self):
        if not self.prepare.isEnabled():
            return
        request = {
            "request_id": uuid4().hex,
            "session_id": self.session_id,
            "question": self.question.toPlainText().strip(),
            "mode": self.mode.currentData(),
            "project": self.project.text().strip() or None,
            "providers": [
                provider.text().lower() for provider in self.providers if provider.isChecked()
            ],
            "token_budget": self.token_budget.value(),
            "allow_uncertain_usage": self.uncertain.isChecked(),
            "budget_seconds": self.seconds.value(),
        }
        self._request(partial(self._run, "matrix_submit", request), self._prepared)

    def _prepared(self, run):
        self._show_run(run)
        self._listed({"runs": list(self.runs.values())})
        self.summary.setText("Request prepared · awaiting local approval. No providers started.")

    def _refresh(self):
        if not self.active or not (self.connected or self.owner_review) or self.busy:
            return

        if "matrix-list" not in self.desktop.pending:
            self.desktop.request(
                "matrix-list",
                partial(self._list_runs, self.owner_review),
                self._listed,
                self._read_failed,
            )
        if self.current and "matrix-status" not in self.desktop.pending:
            self._status(self.current["run_id"])

    def _list_runs(self, owner=False):
        if owner:
            value = self.desktop.gateway.matrix_owner("matrix_list", {"limit": 20})
        else:
            value = self.desktop.gateway.call(
                "matrix_list", {"session_id": self.session_id, "limit": 20}
            )
        if not isinstance(value, dict) or not isinstance(value.get("runs"), list):
            raise GatewayError("Matrix returned invalid run history.")
        return {"runs": [checked_run(run) for run in value["runs"]]}

    def _listed(self, value):
        selected = self.current["run_id"] if self.current else None
        self.runs = {run["run_id"]: run for run in value["runs"]}
        if selected in self.runs:
            # Status and explicit actions own the selected run; a list can arrive older.
            self.runs[selected] = self.current
        self.run_list.blockSignals(True)
        self.run_list.clear()
        for run in self.runs.values():
            row_item(
                self.run_list, f"{run['state']} · {run['request']['question'][:60]}", run["run_id"]
            )
            if run["run_id"] == selected:
                self.run_list.setCurrentRow(self.run_list.count() - 1)
        self.run_list.blockSignals(False)
        if selected not in self.runs and self.run_list.count():
            self.run_list.setCurrentRow(0)

    def _selected(self, item, _previous=None):
        self.desktop.pending.pop("matrix-status", None)
        if item is not None:
            identity = item.data(ROLE)
            self._show_run(self.runs[identity])
            if self.active and (self.connected or self.owner_review):
                self._status(identity)

    def _status(self, identity):
        owner = self.owner_review

        def ready(run):
            if self.current and self.current["run_id"] == identity and self.owner_review == owner:
                self._show_run(run)

        arguments = {"run_id": identity}
        if not owner:
            arguments["session_id"] = self.session_id
        self.desktop.request(
            "matrix-status",
            partial(self._run, "matrix_status", arguments, owner=owner),
            ready,
            self._read_failed,
        )

    def _show_run(self, run):
        self.current = run
        self.runs[run["run_id"]] = run
        scope = "Owner review · " if self.owner_review else ""
        self.summary.setText(f"{scope}Run {run['run_id']} · {run['state']}")
        self.run_detail.setPlainText(request_text(run["request"]))
        result = run.get("result") or {}
        workers = result.get("workers", [])
        self.workers.setRowCount(len(workers))
        for row, worker in enumerate(workers):
            for column, key in enumerate(("provider", "role", "model", "effort", "state")):
                item = QTableWidgetItem(str(worker.get(key) or "Not reported"))
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self.workers.setItem(row, column, item)
        answer = result.get("answer")
        error = run.get("error") or result.get("error")
        parts = [
            value if isinstance(value, str) else json.dumps(value, indent=2)
            for value in (answer, error)
            if value is not None
        ]
        self.output.setPlainText("\n\n".join(parts))
        self._actions()

    def _review(self):
        if not self.current or not self.approve.isEnabled():
            return
        run = self.current
        dialog = QDialog(self, windowTitle="Approve this Matrix request")
        dialog.setModal(True)
        dialog.resize(680, 520)
        dialog.request_text = QPlainTextEdit(request_text(run["request"]), readOnly=True)
        dialog.spending = QCheckBox("I accept uncertain provider usage and spending")
        uncertain = run["request"]["allow_uncertain_usage"]
        dialog.spending.setEnabled(uncertain)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        dialog.start = buttons.addButton(
            "Approve and start", QDialogButtonBox.ButtonRole.AcceptRole
        )
        dialog.start.setEnabled(not uncertain)
        dialog.spending.toggled.connect(dialog.start.setEnabled)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Review the exact request before providers start.", "section"))
        layout.addWidget(dialog.request_text, 1)
        budget_notice = label(
            "Budgets govern new dispatches. Provider spending has no guaranteed ceiling.", "muted"
        )
        budget_notice.setWordWrap(True)
        layout.addWidget(budget_notice)
        layout.addWidget(dialog.spending)
        layout.addWidget(buttons)
        self.approval_dialog = dialog
        self._actions()

        def finished(result):
            self.approval_dialog = None
            if result == QDialog.DialogCode.Accepted and (
                not uncertain or dialog.spending.isChecked()
            ):
                arguments = {
                    "run_id": run["run_id"],
                    "expected_request_sha256": run["request_sha256"],
                }

                def approve():
                    return checked_run(
                        self.desktop.gateway.matrix_owner("matrix_approve", arguments)
                    )

                self._request(approve, self._show_run)
            self._actions()
            dialog.deleteLater()

        dialog.finished.connect(finished)
        dialog.open()

    def _control(self, action):
        if not self.current or self.busy:
            return
        arguments = {
            "run_id": self.current["run_id"],
            "action": action,
        }
        if not self.owner_review:
            arguments["session_id"] = self.session_id
        self._request(
            partial(self._run, "matrix_control", arguments, owner=self.owner_review), self._show_run
        )

    def _review_agents(self, checked):
        for lane in ("matrix-list", "matrix-status"):
            self.desktop.pending.pop(lane, None)
        self.owner_review = checked
        self._clear_runs()
        self.summary.setText(
            "Owner review · checking agent requests." if checked else "Your Matrix session."
        )
        self._actions()
        self.set_active(self.active)

    def _read_failed(self):
        self.summary.setText(self.desktop.status.text())

    def unavailable(self):
        self.connected = self.busy = False
        self.owner_review = False
        self.review_agents.blockSignals(True)
        self.review_agents.setChecked(False)
        self.review_agents.blockSignals(False)
        self.timer.stop()
        if self.approval_dialog is not None:
            self.approval_dialog.reject()
        self._clear_runs()
        self.summary.setText("Matrix unavailable. Reconnect to inspect existing runs.")
        self._actions()

    def _clear_runs(self):
        self.current = None
        self.runs.clear()
        self.run_list.clear()
        self.run_detail.clear()
        self.workers.setRowCount(0)
        self.output.clear()

    def set_active(self, active):
        self.active = bool(active)
        if self.active and (self.connected or self.owner_review):
            self.timer.start()
            self._refresh()
        elif self.active and self.session_id is not None and not self.busy:
            self._connect()
        else:
            self.timer.stop()
            for lane in ("matrix-list", "matrix-status"):
                self.desktop.pending.pop(lane, None)

    def showEvent(self, event):
        self.set_active(True)
        super().showEvent(event)

    def hideEvent(self, event):
        self.set_active(False)
        super().hideEvent(event)
