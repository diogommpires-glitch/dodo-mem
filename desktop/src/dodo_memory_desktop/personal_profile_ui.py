"""Edit local personal sources and review the exact profile intended for hosted use."""

from functools import partial
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QHeaderView,
    QPlainTextEdit,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .personal_context import BASES, SECTIONS, ProfileSource, shared_draft
from .ui import ROLE, button, label

PROFILE_KEY = "personalization/approved-profile"
PROFILE_NOTE = f"{PROFILE_KEY}.md"
WITHDRAWN = "<!-- DODO PROFILE WITHDRAWN -->"
PROFILE_REQUEST = "personal-profile"


class PersonalProfileEditor(QDialog):
    def __init__(self, desktop):
        super().__init__(desktop, windowTitle="Personal profile")
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(870, 570)
        self.desktop = desktop
        self.profile = desktop.personal_profile
        self.revision = None
        self.has_approved = False
        self.busy = False
        self.tabs = QTabWidget()
        layout = QVBoxLayout(self)
        layout.addWidget(self.tabs)
        self._source_tab()
        self._hosted_tab()
        self.tabs.currentChanged.connect(self._tab_changed)

    def _source_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = label(
            "Check the notes and folders Ask Dodo should use when you say "
            "“contextualize with mem”. These sources stay in your local profile."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.sources = QTableWidget(0, 3)
        self.sources.setAccessibleName("Local personal profile sources")
        self.sources.setHorizontalHeaderLabels(("Use source", "Section", "Basis"))
        header = self.sources.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.sources.verticalHeader().hide()
        layout.addWidget(self.sources, 1)
        actions = QHBoxLayout()
        self.add_file = button("Add notes")
        self.add_file.clicked.connect(self._choose_files)
        self.add_folder = button("Add folder")
        self.add_folder.clicked.connect(self._choose_folder)
        remove = button("Remove selected source")
        remove.clicked.connect(self._remove_source)
        self.save_sources = button("Save local profile", role="primary")
        self.save_sources.clicked.connect(self._save_sources)
        for widget in (self.add_file, self.add_folder, remove, self.save_sources):
            actions.addWidget(widget)
        actions.addStretch()
        layout.addLayout(actions)
        self.source_status = label("")
        self.source_status.setWordWrap(True)
        layout.addWidget(self.source_status)
        try:
            for source in self.profile.sources():
                self._append_source(source)
        except (OSError, ValueError) as error:
            self.source_status.setText(f"The local profile could not be read. {error}")
            self.save_sources.setEnabled(False)
        self.tabs.addTab(page, "Local sources")

    def _append_source(self, source):
        row = self.sources.rowCount()
        self.sources.insertRow(row)
        item = QTableWidgetItem(str(source.path))
        item.setFlags(
            (item.flags() | Qt.ItemFlag.ItemIsUserCheckable) & ~Qt.ItemFlag.ItemIsEditable
        )
        item.setCheckState(Qt.CheckState.Checked if source.enabled else Qt.CheckState.Unchecked)
        item.setData(ROLE, str(source.path))
        item.setToolTip(str(source.path))
        self.sources.setItem(row, 0, item)
        for column, values, current, title in (
            (1, SECTIONS, source.section, "Profile section"),
            (2, BASES, source.basis, "Preference basis"),
        ):
            options = QComboBox()
            options.setAccessibleName(title)
            options.addItems(values)
            options.setCurrentText(current)
            self.sources.setCellWidget(row, column, options)

    def _choose_files(self):
        paths, _ = QFileDialog.getOpenFileNames(self, "Choose local profile notes")
        self._add_sources(paths)

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Choose a local profile folder")
        if path:
            self._add_sources((path,))

    def _add_sources(self, paths):
        existing = {
            Path(self.sources.item(row, 0).data(ROLE)) for row in range(self.sources.rowCount())
        }
        for path in dict.fromkeys(map(Path, paths)):
            if path not in existing:
                self._append_source(ProfileSource(path, "Standing preferences", "observed", True))
        if paths:
            self.sources.selectRow(self.sources.rowCount() - 1)

    def _remove_source(self):
        if self.sources.currentRow() >= 0:
            self.sources.removeRow(self.sources.currentRow())

    def _save_sources(self):
        sources = []
        for row in range(self.sources.rowCount()):
            item = self.sources.item(row, 0)
            sources.append(
                ProfileSource(
                    Path(item.data(ROLE)),
                    self.sources.cellWidget(row, 1).currentText(),
                    self.sources.cellWidget(row, 2).currentText(),
                    item.checkState() == Qt.CheckState.Checked,
                )
            )
        try:
            self.profile.save(sources)
        except (OSError, ValueError) as error:
            self.source_status.setText(f"The local profile could not be saved. {error}")
        else:
            self.source_status.setText("Local profile saved.")

    def _hosted_tab(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        intro = label(
            "Review the exact draft below. Submit it to Review Inbox and approve it there "
            "before connected assistants can read it."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        self.published = label("Approved profile has not been checked.")
        self.published.setWordWrap(True)
        layout.addWidget(self.published)
        self.draft = QPlainTextEdit()
        self.draft.setAccessibleName("Exact hosted personalization draft")
        self.draft.setPlainText(shared_draft())
        self.draft.textChanged.connect(self._update_actions)
        layout.addWidget(self.draft, 1)
        actions = QHBoxLayout()
        self.reload = button("Check approved revision")
        self.reload.clicked.connect(self._load_approved)
        self.copy = button("Copy approved context")
        self.copy.clicked.connect(self._copy_approved)
        self.withdraw = button("Submit withdrawal")
        self.withdraw.clicked.connect(partial(self._submit, WITHDRAWN))
        self.submit = button("Submit draft for review", role="primary")
        self.submit.clicked.connect(lambda: self._submit(self.draft.toPlainText()))
        for widget in (self.reload, self.copy, self.withdraw, self.submit):
            actions.addWidget(widget)
        layout.addLayout(actions)
        self.hosted_status = label("")
        self.hosted_status.setWordWrap(True)
        layout.addWidget(self.hosted_status)
        self.tabs.addTab(page, "Hosted profile")
        self._update_actions()

    def _tab_changed(self, index):
        if index == 1 and self.revision is None and not self.busy:
            self._load_approved()

    def _update_actions(self):
        self.reload.setEnabled(not self.busy)
        self.copy.setEnabled(not self.busy and self.has_approved)
        self.withdraw.setEnabled(not self.busy and self.has_approved)
        text = self.draft.toPlainText()
        self.submit.setEnabled(
            not self.busy
            and self.revision is not None
            and bool(text.strip())
            and len(text) <= 100000
        )
        self.draft.setReadOnly(self.busy)

    def unavailable(self):
        self.revision = None
        self.has_approved = False
        self.busy = False
        self.published.setText("Approved profile unavailable.")
        self.hosted_status.setText(self.desktop.status.text())
        self._update_actions()

    def _request(self, tool, arguments, completed):
        if self.busy:
            return
        self.busy = True
        self.hosted_status.clear()
        self._update_actions()

        def finished(value):
            self.busy = False
            completed(value)
            self._update_actions()

        def failed():
            self.busy = False
            self.hosted_status.setText(self.desktop.status.text())
            self._update_actions()

        self.desktop.request(
            PROFILE_REQUEST,
            partial(self.desktop.gateway.call, tool, arguments),
            finished,
            failed,
        )

    def _load_approved(self):
        def ready(result):
            self.has_approved = result["status"] == "found" and result["text"] != WITHDRAWN
            self.revision = result["revision"] if result["status"] == "found" else 0
            if self.has_approved:
                self.published.setText(f"Approved revision {self.revision}.")
                if not self.draft.document().isModified():
                    self.draft.setPlainText(result["text"])
            elif self.revision:
                self.published.setText(f"Hosted profile withdrawn at revision {self.revision}.")
            else:
                self.published.setText("No hosted profile has been approved yet.")

        self._request("read_working_knowledge", {"entry_key": PROFILE_KEY}, ready)

    def _copy_approved(self):
        def copied(result):
            QApplication.clipboard().setText(result["text"])
            self.hosted_status.setText("Approved context copied.")

        self._request("get_note", {"path": PROFILE_NOTE}, copied)

    def _submit(self, text):
        if self.revision is None:
            return

        def submitted(_):
            self.hosted_status.setText("Submitted to Review Inbox.")
            self.desktop.refresh_inbox()

        self._request(
            "propose_working_knowledge",
            {
                "entry_key": PROFILE_KEY,
                "text": text,
                "expected_revision": self.revision,
                "scope": "shared",
                "assertion_type": "user_statement",
                "evidence": [],
            },
            submitted,
        )
