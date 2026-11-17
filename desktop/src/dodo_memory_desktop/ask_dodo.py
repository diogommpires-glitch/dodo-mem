"""Ask Dodo against the private local library or approve a hosted evidence handoff."""

import sqlite3
from functools import partial
from pathlib import Path

from PySide6.QtCore import QObject, QRunnable, Qt, QThreadPool, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from .ask_evidence import (
    MAX_QUESTION,
    AskError,
    Context,
    EvidenceSession,
    Handoffs,
)
from .ask_models import ask_local, checked_answer, local_models, send_hosted
from .neural_ask import activated as neural_activated
from .neural_ask import inspect_neural
from .ui import button, label


def _location(locator):
    match locator["kind"]:
        case "lines":
            start, end = locator["start"], locator["end"]
            return f"Line {start}" if start == end else f"Lines {start}–{end}"
        case "page":
            return f"Page {locator['page']}"
        case "commit":
            return f"Commit {locator['commit']}"
        case kind:
            return f"{kind.title()} {locator['start']}"


def _connection(link):
    kind = link["kind"].replace("_", " ")
    return f"{link['from']} — {kind} — {link['to']}\n{link['evidence']}"


def _evidence_text(evidence):
    blocks = []
    for item in evidence["sources"]:
        paths = "\n\n".join(map(_connection, item["relationship_path"]))
        profile = item.get("profile")
        basis = f"{profile['section']} · {profile['basis']}\n" if profile else ""
        blocks.append(
            f"{item['cite']} {item['path']}\n{_location(item['locator'])}\n\n"
            + basis
            + item["excerpt"]
            + (f"\n\n{paths}" if paths else "")
        )
    if evidence["connections"]:
        blocks.append("Connections\n\n" + "\n\n".join(map(_connection, evidence["connections"])))
    coverage = evidence["coverage"]
    if "personal_sources" in coverage:
        count = coverage["personal_sources"]
        blocks.insert(0, f"Personal context · {count} verified source{'s' if count != 1 else ''}")
    if coverage["unavailable_evidence_reads"]:
        blocks.append("Some matching items no longer have verified excerpts.")
    if coverage["bounded_results"]:
        blocks.append("This search reached the evidence limit.")
    if coverage["index_scanning"]:
        blocks.append("Indexing is in progress, so more sources may become available.")
    return "\n\n─────\n\n".join(blocks)


class _Replies(QObject):
    completed = Signal(object, object, str)


class _Task(QRunnable):
    def __init__(self, operation, callback):
        super().__init__()
        self.operation = operation
        self.callback = callback
        self.replies = _Replies()

    def run(self):
        value = None
        error = ""
        try:
            value = self.operation()
        except AskError as problem:
            error = str(problem)
        except (OSError, sqlite3.Error):
            error = "The local index could not be read. Try again after indexing resumes."
        self.replies.completed.emit(self.callback, value, error)


class AskDodo(QWidget):
    def __init__(self, catalogue, *, send=send_hosted, local=ask_local, profile=None):
        super().__init__()
        self.catalogue = catalogue
        self.context = Context()
        self.context_name = None
        self.skill = None
        self.handoffs = Handoffs(catalogue, send)
        self.local = local
        self.profile = profile
        self._busy = False
        self._visible_evidence = None
        self._refreshing = False
        self._refresh_again = False

        self.question = QLineEdit(placeholderText="Ask about your files, projects, or decisions")
        self.question.setAccessibleName("Question for Ask Dodo")
        self.question.setMaxLength(MAX_QUESTION)
        self.question.returnPressed.connect(self.ask)
        self.status = label("", "muted")
        self.status.setWordWrap(True)
        self.model = QComboBox()
        self.model.setAccessibleName("Local model")
        self.model.addItem("Automatic · running local model")
        try:
            for model in local_models():
                self.model.addItem(model.name, model)
        except AskError as error:
            self.status.setText(str(error))
        self.ask_button = button("Ask local model", role="primary")
        self.memory = QCheckBox("Use local memory")
        self.memory.setChecked(True)
        self.memory.setAccessibleName("Use local memory for this request")
        self.memory.toggled.connect(self._clear_result)
        self.ask_button.clicked.connect(self.ask)
        self.inspect_button = button("Find evidence")
        self.inspect_button.clicked.connect(self.inspect)
        self.provider = QComboBox()
        self.provider.setAccessibleName("Hosted provider")
        self.provider.addItems(["Codex", "Claude"])
        self.hosted_button = button("Preview hosted request")
        self.hosted_button.clicked.connect(self.preview)
        self.scope = label("Search the whole library", "muted")
        self.scope.setTextFormat(Qt.TextFormat.PlainText)
        self.scope.setWordWrap(True)
        self.answer = QPlainTextEdit(readOnly=True)
        self.answer.setAccessibleName("Ask Dodo answer")
        self.evidence = QPlainTextEdit(readOnly=True)
        self.evidence.setAccessibleName("Verified evidence and relationship paths")
        self.question.textChanged.connect(self._clear_result)
        self.model.currentIndexChanged.connect(self._clear_result)

        primary = QHBoxLayout()
        primary.addWidget(self.model)
        primary.addWidget(self.memory)
        primary.addWidget(self.ask_button)
        primary.addWidget(self.inspect_button)
        primary.addStretch()
        hosted = QHBoxLayout()
        hosted.addWidget(self.provider)
        hosted.addWidget(self.hosted_button)
        hosted.addStretch()
        outputs = QSplitter()
        for heading, output in (("Answer", self.answer), ("Verified evidence", self.evidence)):
            pane = QWidget()
            box = QVBoxLayout(pane)
            box.setContentsMargins(0, 0, 0, 0)
            box.addWidget(label(heading, "section"))
            box.addWidget(output, 1)
            outputs.addWidget(pane)
        outputs.setSizes([550, 380])

        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 18, 22, 18)
        layout.addWidget(label("Ask Dodo", "title"))
        layout.addWidget(self.scope)
        layout.addWidget(self.question)
        layout.addLayout(primary)
        layout.addLayout(hosted)
        layout.addWidget(outputs, 1)
        layout.addWidget(self.status)

    def select_model(self, alias):
        for index in range(1, self.model.count()):
            if getattr(self.model.itemData(index), "alias", None) == alias:
                self.model.setCurrentIndex(index)
                return
        self.model.addItem(f"Unavailable · {alias}", alias)
        self.model.setCurrentIndex(self.model.count() - 1)
        self.status.setText(f"Local AI no longer lists {alias}. Choose another local model.")

    def set_context(self, node_id=None, workspace_path=None, *, node_name=None):
        context = Context(node_id, Path(workspace_path) if workspace_path else None)
        if context != self.context:
            self._clear_result()
        self.context = context
        self.context_name = node_name
        self._describe_scope()

    def set_skill(self, title: str, markdown: str):
        self.skill = (title[:160], markdown[:8000]) if title.strip() and markdown.strip() else None
        self._describe_scope()

    def invalidate_evidence(self):
        if self._visible_evidence is None:
            return
        self._clear_result()
        if not self._busy:
            self.status.setText("The library changed. Run the question again for current evidence.")

    def refresh_visible(self):
        snapshot = self._visible_evidence
        if snapshot is None:
            self._refresh_again |= self._busy
            return
        if self._refreshing:
            self._refresh_again = True
            return
        self._refreshing = True
        task = _Task(partial(snapshot.verify, self.catalogue), snapshot)
        task.replies.completed.connect(self._checked_visible)
        QThreadPool.globalInstance().start(task)

    def _checked_visible(self, snapshot, _result, error):
        self._refreshing = False
        if error and self._visible_evidence is snapshot:
            self.invalidate_evidence()
        if self._refresh_again:
            self._refresh_again = False
            self.refresh_visible()

    def _describe_scope(self):
        if self.context.workspace:
            name = self.context.workspace.name or str(self.context.workspace)
            text = f"Start with {name} and follow related evidence"
        elif self.context.node_id:
            name = self.context_name or "the selected item"
            text = f"Start with {name} and follow related evidence"
        else:
            text = "Search the whole library"
        if self.skill:
            text += f" · Local guidance from {self.skill[0]}"
        self.scope.setText(text)

    def _set_busy(self, busy):
        self._busy = busy
        for control in (
            self.question,
            self.model,
            self.memory,
            self.ask_button,
            self.inspect_button,
            self.provider,
            self.hosted_button,
        ):
            control.setEnabled(not busy)

    def _start(self, operation, callback, status):
        if self._busy:
            return
        self._set_busy(True)
        self.status.setText(status)
        request = (
            self.question.text(),
            self.context,
            self.skill,
            self.model.currentData(),
            self.memory.isChecked(),
        )
        task = _Task(operation, (request, callback))
        task.replies.completed.connect(self._finished)
        QThreadPool.globalInstance().start(task)

    def _finished(self, request_and_callback, result, error):
        self._set_busy(False)
        request, callback = request_and_callback
        if request != (
            self.question.text(),
            self.context,
            self.skill,
            self.model.currentData(),
            self.memory.isChecked(),
        ):
            self.status.setText("The question or context changed. Run the question again.")
            return
        if error:
            self.status.setText(error)
            return
        callback(result)
        if self._refresh_again and not self._refreshing:
            self._refresh_again = False
            self.refresh_visible()

    def _clear_result(self):
        self._visible_evidence = None
        self.answer.clear()
        self.evidence.clear()
        self.status.clear()

    def ask(self):
        model = self.model.currentData()
        if isinstance(model, str):
            self.status.setText(f"Local AI no longer lists {model}. Choose another local model.")
            return
        self._clear_result()
        self._start(
            partial(
                self.local,
                self.catalogue,
                self.question.text(),
                self.context,
                self.skill,
                include_snapshot=True,
                profile=self.profile,
                model=model,
                memory_enabled=self.memory.isChecked(),
            ),
            self._show_local,
            "Searching your local library and checking the model…",
        )

    def inspect(self):
        self._clear_result()
        question, context = self.question.text(), self.context

        def find():
            if neural_activated():
                node_id = context.node_id
                if not node_id and context.workspace:
                    selected = self.catalogue.resolve_path(context.workspace)
                    node_id = selected["id"] if selected else None
                return inspect_neural(question, node_id=node_id)
            session = EvidenceSession(
                self.catalogue, question, context, personal=True, profile=self.profile
            )
            session.fill()
            snapshot = session.snapshot()
            snapshot.verify(self.catalogue)
            return snapshot

        self._start(find, self._show_evidence, "Checking local evidence…")

    def _show_evidence(self, snapshot):
        self._visible_evidence = snapshot
        self.evidence.setPlainText(_evidence_text(snapshot.description))
        count = len(snapshot.sources)
        self.status.setText(f"{count} verified excerpt{'s' if count != 1 else ''}")

    def _show_local(self, result):
        answer, evidence, alias = result
        self.answer.setPlainText(answer)
        self._show_evidence(evidence)
        if alias:
            self.status.setText(f"Answered locally by {alias}")

    def preview(self):
        self._clear_result()
        self._start(
            partial(
                self.handoffs.prepare,
                self.provider.currentText(),
                self.question.text(),
                self.context,
            ),
            self._review,
            "Preparing the exact request for your review…",
        )

    def _review(self, packet):
        if not self.isVisible():
            self.handoffs.submit(packet, approved=False)
            return
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Review one request to {packet.provider}")
        dialog.resize(760, 570)
        layout = QVBoxLayout(dialog)
        explanation = label(
            f"Approving sends the question, excerpts, and paths below to {packet.provider} once.",
            "muted",
        )
        explanation.setWordWrap(True)
        layout.addWidget(explanation)
        preview = QPlainTextEdit(readOnly=True)
        preview.setAccessibleName("Exact evidence packet to be sent")
        preview.setPlainText(packet.preview)
        layout.addWidget(preview)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        cancel = actions.button(QDialogButtonBox.StandardButton.Cancel)
        cancel.setDefault(True)
        approve = actions.addButton(
            f"Send once to {packet.provider}", QDialogButtonBox.ButtonRole.AcceptRole
        )
        approve.setAutoDefault(False)
        actions.accepted.connect(dialog.accept)
        actions.rejected.connect(dialog.reject)
        layout.addWidget(actions)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.handoffs.submit(packet, approved=False)
            self.status.setText("Hosted request cancelled.")
            return
        self._start(
            partial(self.handoffs.submit, packet, approved=True),
            partial(self._show_hosted, packet),
            f"Sending the approved evidence to {packet.provider}…",
        )

    def _show_hosted(self, packet, answer):
        self._show_evidence(packet.snapshot)
        self.answer.setPlainText(checked_answer(answer, len(packet.snapshot.sources)))
        self.status.setText(f"{packet.provider} answered the approved request.")
