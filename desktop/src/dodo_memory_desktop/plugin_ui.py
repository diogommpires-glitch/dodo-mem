"""Install, approve and use local plugin packs."""

from functools import partial

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QListWidget,
    QMessageBox,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .plugins import EXAMPLES, EXECUTABLE, PACK_SUFFIX, PERMISSIONS, PluginError, PluginVersion
from .ui import ROLE, button, label, row_item

ENTRY_TYPES = {
    "theme": "Theme",
    "saved_view": "Saved view",
    "note_template": "Note template",
    "canvas_template": "Canvas template",
    "command": "Command",
    "panel": "Panel",
}


class Plugins(QWidget):
    changed = Signal()
    invocation_changed = Signal()
    run_requested = Signal(str, str, str, str)
    apply_requested = Signal(object)
    note_requested = Signal(str)

    def __init__(self, store):
        super().__init__()
        self.store = store
        self.selection = None
        self.setObjectName("plugins")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 32, 28, 24)
        layout.setSpacing(13)
        heading = QHBoxLayout()
        heading.addWidget(label("Plugins", "title"), 1)
        examples = button("Install bundled add-ins")
        examples.clicked.connect(self._install_examples)
        heading.addWidget(examples)
        install = button("Install pack", role="primary")
        install.clicked.connect(self._install)
        heading.addWidget(install)
        layout.addLayout(heading)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.packs = QListWidget()
        self.packs.setAccessibleName("Installed plugin versions")
        self.packs.setMinimumWidth(166)
        self.packs.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.packs.currentItemChanged.connect(self._selected)
        splitter.addWidget(self.packs)
        detail = QWidget()
        body = QVBoxLayout(detail)
        body.setContentsMargins(12, 2, 0, 0)
        body.setSpacing(10)
        self.name = label("No plugin selected", "section")
        body.addWidget(self.name)
        self.permissions = label("", "muted")
        self.permissions.setWordWrap(True)
        body.addWidget(self.permissions)
        controls = QHBoxLayout()
        self.enable = button("Approve and enable")
        self.enable.clicked.connect(self._enable)
        self.disable = button("Disable")
        self.disable.clicked.connect(self._disable)
        self.rollback = button("Restore previous")
        self.rollback.clicked.connect(self._rollback)
        for control in (self.enable, self.disable, self.rollback):
            controls.addWidget(control)
        controls.addStretch()
        body.addLayout(controls)
        self.actions = QListWidget()
        self.actions.setAccessibleName("Plugin entries")
        self.actions.setMaximumHeight(240)
        self.actions.currentItemChanged.connect(self._entry_selected)
        self.actions.itemActivated.connect(self._invoke)
        body.addWidget(self.actions)
        self.invoke = button("Open", role="primary")
        self.invoke.clicked.connect(self._invoke)
        invoke_row = QHBoxLayout()
        invoke_row.addWidget(self.invoke)
        invoke_row.addStretch()
        body.addLayout(invoke_row)
        self.result = label("", "section")
        self.result.setWordWrap(True)
        self.result.hide()
        body.addWidget(self.result)
        self.rows = QTreeWidget()
        self.rows.setAccessibleName("Plugin panel output")
        self.rows.setHeaderLabels(("Note", "Relationship"))
        self.rows.setRootIsDecorated(False)
        self.rows.setUniformRowHeights(True)
        self.rows.itemActivated.connect(self._open_note)
        self.rows.hide()
        body.addWidget(self.rows, 1)
        body.addStretch()
        splitter.addWidget(detail)
        splitter.setSizes((220, 780))
        layout.addWidget(splitter, 1)
        self.status = label("", "muted")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.refresh()

    def refresh(self, key=None):
        if key is None and self.selection:
            key = self.selection.id, self.selection.version
        try:
            versions = self.store.versions()
        except PluginError as error:
            versions = []
            self.status.setText(str(error))
        self.packs.clear()
        self._versions = versions
        latest = {plugin.id: plugin.version for plugin in versions}
        multiple = {plugin.id for plugin in versions if plugin.version != latest[plugin.id]}
        selected = 0
        for index, plugin in enumerate(versions):
            version_label = ""
            if plugin.id in multiple:
                version_label = " · latest" if plugin.version == latest[plugin.id] else " · earlier"
            suffix = " · enabled" if plugin.enabled else " · disabled"
            row_item(self.packs, f"{plugin.name}\n{plugin.version}{version_label}{suffix}", index)
            if (plugin.id, plugin.version) == key:
                selected = index
        if versions:
            self.packs.setCurrentRow(selected)
        else:
            self._selected(None)

    def _selected(self, item, _previous=None):
        self.selection = self._versions[item.data(ROLE)] if item is not None else None
        plugin = self.selection
        self.actions.clear()
        self.clear_result()
        self.enable.setEnabled(bool(plugin and not plugin.enabled))
        self.disable.setEnabled(bool(plugin and plugin.enabled))
        previous = None
        if plugin:
            try:
                previous = self.store.previous(plugin.id)
            except PluginError as error:
                self.status.setText(str(error))
        self.rollback.setEnabled(previous is not None)
        self.rollback.setToolTip(f"Restore version {previous.version}" if previous else "")
        self.name.setText(plugin.name if plugin else "No plugin selected")
        self.permissions.setText(
            "\n".join(PERMISSIONS[permission] for permission in plugin.permissions)
            if plugin and plugin.permissions
            else "No additional permissions"
            if plugin
            else ""
        )
        if plugin:
            for index, entry in enumerate(plugin.entries):
                row_item(self.actions, f"{entry['title']} · {ENTRY_TYPES[entry['kind']]}", index)
            self.actions.setCurrentRow(0)
        self._entry_selected()

    def _entry_selected(self, _item=None, _previous=None):
        self.clear_result()
        self.invocation_changed.emit()
        entry = self._current_entry()
        self.invoke.setEnabled(bool(entry and self.selection.enabled))
        self.invoke.setText("Run" if entry and entry["kind"] in EXECUTABLE else "Open")

    def _current_entry(self):
        item = self.actions.currentItem()
        return self.selection.entries[item.data(ROLE)] if item and self.selection else None

    def _invoke(self, _item=None):
        entry = self._current_entry()
        if entry is None or not self.selection.enabled:
            return
        self.clear_result()
        if entry["kind"] in EXECUTABLE:
            plugin = self.selection
            if plugin.generation is not None:
                self.run_requested.emit(plugin.id, entry["id"], plugin.sha256, plugin.generation)
        else:
            self.apply_requested.emit(entry)

    def _install(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Install plugin pack", "", "Dodo plugin packs (*.dodo-pack.zip)"
        )
        if not path:
            return
        self.install_path(path)

    def install_path(self, path):
        try:
            plugin = self.store.install(path)
        except PluginError as error:
            self.status.setText(str(error))
            return
        self.status.setText("Pack installed. Review its permissions before enabling it.")
        self.refresh((plugin.id, plugin.version))
        self.changed.emit()

    def _install_examples(self):
        packs = sorted(EXAMPLES.glob(f"*{PACK_SUFFIX}"))
        if not packs:
            self.status.setText("Bundled example packs are unavailable.")
            return
        try:
            for pack in packs:
                self.store.install(pack)
        except PluginError as error:
            self.status.setText(str(error))
        else:
            self.status.setText(
                "Examples installed disabled. Enable an earlier version, then the latest, to try rollback."
            )
        self.refresh()
        self.changed.emit()

    def _enable(self):
        plugin = self.selection
        if plugin is None:
            return
        box = QMessageBox(self)
        box.setWindowTitle("Approve plugin")
        box.setTextFormat(Qt.TextFormat.PlainText)
        box.setText(f"Enable {plugin.name} {plugin.version}?")
        permissions = "\n".join(PERMISSIONS[item] for item in plugin.permissions)
        box.setInformativeText(permissions or "This pack requests no additional permissions.")
        approve = box.addButton("Approve and enable", QMessageBox.ButtonRole.AcceptRole)
        cancel = box.addButton(QMessageBox.StandardButton.Cancel)
        box.setDefaultButton(cancel)
        box.exec()
        if box.clickedButton() is approve:
            enable = partial(self.store.enable, expected_sha256=plugin.sha256)
            self._change(enable, plugin.id, plugin.version, plugin.permissions)

    def _disable(self):
        if self.selection:
            self._change(self.store.disable, self.selection.id)

    def _rollback(self):
        if self.selection:
            self._change(self.store.rollback, self.selection.id)

    def _change(self, operation, *arguments):
        try:
            updated = operation(*arguments)
        except PluginError as error:
            self.status.setText(str(error))
            return
        self.status.clear()
        self.refresh((updated.id, updated.version) if isinstance(updated, PluginVersion) else None)
        self.changed.emit()

    def clear_result(self):
        self.result.clear()
        self.result.hide()
        self.rows.clear()
        self.rows.hide()

    def show_result(self, result):
        self.clear_result()
        if result is None:
            return
        self.result.setText(result["text"] if result["kind"] == "message" else result["title"])
        self.result.show()
        if result["kind"] == "panel":
            for row in result["rows"]:
                item = QTreeWidgetItem([row["label"], row["value"]])
                item.setData(0, ROLE, row.get("path"))
                self.rows.addTopLevelItem(item)
            self.rows.resizeColumnToContents(0)
            self.rows.show()

    def _open_note(self, item, _column):
        path = item.data(0, ROLE)
        if path:
            self.note_requested.emit(path)
