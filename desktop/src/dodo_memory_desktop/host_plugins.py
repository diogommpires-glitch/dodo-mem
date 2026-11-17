"""Discover Codex and Claude add-ins without importing their executables."""

import json
import sqlite3
import stat
import time
from pathlib import Path

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLineEdit,
    QPlainTextEdit,
    QSplitter,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .gateway import GatewayError
from .project_library import local_path
from .ui import ROLE, button, label
from .whole_index import safe_path

MAX_MANIFEST = 96 * 1024
MAX_SKILL = 64 * 1024
FEATURED = ("Codex", "created-by-me-remote", "dodo-kit")
PACK_PAGE = 100


def _bounded(path, root, limit):
    with local_path(path, root).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("Add-in file exceeds the catalogue size limit.")
    return raw.decode("utf-8-sig")


def _directories(path):
    try:
        if not stat.S_ISDIR(safe_path(path).st_mode):
            return []
        return sorted(item for item in path.iterdir() if _is_directory(item))
    except (OSError, ValueError):
        return []


def _is_directory(path):
    try:
        return stat.S_ISDIR(safe_path(path).st_mode)
    except (OSError, ValueError):
        return False


def _manifests(cache, manifest_dir):
    for market in _directories(cache):
        for package in _directories(market):
            for version in _directories(package):
                yield version / manifest_dir / "plugin.json"


def discover(hosts=None):
    hosts = hosts if hosts is not None else ()
    entries = []
    for host, cache, manifest_dir in hosts:
        for manifest in _manifests(cache, manifest_dir):
            try:
                data = json.loads(_bounded(manifest, cache, MAX_MANIFEST))
                if not isinstance(data, dict):
                    continue
                market, package, version, *_ = manifest.relative_to(cache).parts
                interface = data.get("interface")
                display = interface.get("displayName") if isinstance(interface, dict) else None
                name = (
                    display
                    if isinstance(display, str) and display.strip()
                    else str(data.get("name") or package)
                )
                featured = (host, market, package) == FEATURED
                base = manifest.parent.parent
                skills = [
                    directory / "SKILL.md"
                    for directory in _directories(base / "skills")
                    if _is_regular(directory / "SKILL.md")
                ]
                entries.append(
                    {
                        "host": host,
                        "market": market,
                        "name": name[:160],
                        "version": version[:80],
                        "description": str(data.get("description") or "")[:2500],
                        "featured": featured,
                        "cache": cache,
                        "skills": skills,
                    }
                )
            except (OSError, ValueError, RuntimeError):
                continue
    return sorted(
        entries,
        key=lambda row: (not row["featured"], row["name"].casefold(), row["host"], row["version"]),
    )


def _is_regular(path):
    try:
        return stat.S_ISREG(safe_path(path).st_mode)
    except (OSError, ValueError):
        return False


class HostPlugins(QWidget):
    skill_requested = Signal(str, str)
    pack_requested = Signal(str)

    def __init__(self, catalogue=None, request=None):
        super().__init__()
        self.catalogue = catalogue
        self.request = request
        self.entries = None
        self.host_refresh_due = 0.0
        self.packs = []
        self.offset = 0
        self.selected = None
        self.pack = None
        self.search = QLineEdit(placeholderText="Find an add-in or skill")
        self.search.setAccessibleName("Find a host add-in")
        self.search.textChanged.connect(self._populate)
        self.tree = QTreeWidget(accessibleName="Codex and Claude add-in catalogue")
        self.tree.setHeaderLabels(("Add-in or skill", "Host"))
        self.tree.setColumnWidth(0, 320)
        self.tree.currentItemChanged.connect(self._selected)
        self.status = label("Discovering local add-ins…", "muted")
        self.details = QPlainTextEdit(readOnly=True, accessibleName="Add-in details")
        self.use = button("Use skill with local Ask Dodo", role="primary")
        self.use.setEnabled(False)
        self.use.clicked.connect(self._use_skill)
        self.install = button("Install discovered pack")
        self.install.setEnabled(False)
        self.install.clicked.connect(self._install_pack)
        detail = QWidget()
        panel = QVBoxLayout(detail)
        panel.addWidget(self.details, 1)
        panel.addWidget(self.use)
        panel.addWidget(self.install)
        split = QSplitter()
        split.addWidget(self.tree)
        split.addWidget(detail)
        split.setSizes([410, 650])
        layout = QVBoxLayout(self)
        layout.addWidget(label("Host add-ins", "title"))
        layout.addWidget(self.search)
        layout.addWidget(self.status)
        layout.addWidget(split, 1)
        paging = QHBoxLayout()
        self.previous = button("Previous packs")
        self.previous.clicked.connect(lambda: self._page(-PACK_PAGE))
        self.next = button("More packs")
        self.next.clicked.connect(lambda: self._page(PACK_PAGE))
        for control in (self.previous, self.next):
            control.setEnabled(False)
            paging.addWidget(control)
        paging.addStretch()
        layout.addLayout(paging)

    def refresh(self):
        self._show_hosts(discover())
        self.refresh_packages()

    def _show_hosts(self, rows):
        self.host_refresh_due = time.monotonic() + 60
        if self.entries != rows:
            self.entries = rows
            self._populate()

    def _page(self, delta):
        self.offset = max(0, self.offset + delta)
        self.refresh_packages()

    def refresh_packages(self):
        if self.catalogue is None or self.request is None:
            return
        if time.monotonic() >= self.host_refresh_due:
            self.host_refresh_due = time.monotonic() + 60
            self.request("whole-host-addins", discover, self._show_hosts)
        offset = self.offset

        def read():
            try:
                return self.catalogue.addin_packages(PACK_PAGE + 1, offset)
            except (OSError, sqlite3.Error, ValueError) as error:
                raise GatewayError("Discovered packs could not be read.") from error

        def show(rows):
            if offset != self.offset:
                return
            self.previous.setEnabled(offset > 0)
            self.next.setEnabled(len(rows) > PACK_PAGE)
            if rows[:PACK_PAGE] != self.packs:
                self.packs = rows[:PACK_PAGE]
                self._populate()

        self.request("whole-discovered-addins", read, show, lambda: show([]))

    def _populate(self):
        if self.entries is None:
            return
        current = self.tree.currentItem()
        selected = current.data(0, ROLE) if current else None
        restored = None
        self.tree.clear()
        needle = self.search.text().strip().casefold()
        for row in self.entries:
            skills = row["skills"]
            if needle not in f"{row['name']} {row['host']} {row['market']}".casefold():
                skills = [path for path in skills if needle in path.parent.name.casefold()]
                if not skills:
                    continue
            caption = f"{row['name']} · {row['version']}"
            parent = QTreeWidgetItem(self.tree, [caption, row["host"]])
            parent.setData(0, ROLE, (row, None))
            if (row, None) == selected:
                restored = parent
            for path in skills:
                skill = QTreeWidgetItem(parent, [path.parent.name, row["host"]])
                skill.setData(0, ROLE, (row, path))
                if (row, path) == selected:
                    restored = skill
            parent.setExpanded(row["featured"] or bool(needle and skills))
        packs = [row for row in self.packs if needle in row["path"].casefold()]
        if packs:
            group = QTreeWidgetItem(self.tree, ["Discovered Dodo packs", "Whole library"])
            group.setFlags(group.flags() & ~group.flags().ItemIsSelectable)
            group.setExpanded(True)
            for pack in packs:
                path = Path(pack["path"])
                item = QTreeWidgetItem(group, [pack["name"], "Local pack"])
                item.setData(0, ROLE, (None, path))
                if (None, path) == selected:
                    restored = item
        self.status.setText(
            f"{len(self.entries)} cached add-in versions · Skill guidance stays local"
        )
        if self.tree.topLevelItemCount():
            self.tree.setCurrentItem(restored or self.tree.topLevelItem(0))

    def _selected(self, item, _previous):
        self.selected = None
        self.pack = None
        self.use.setEnabled(False)
        self.install.setEnabled(False)
        if item is None or item.data(0, ROLE) is None:
            self.details.clear()
            return
        row, skill = item.data(0, ROLE)
        if row is None:
            self.pack = skill
            self.details.setPlainText(
                f"{skill}\n\nInstall to inspect permissions. Enabling is a separate action."
            )
            self.install.setEnabled(True)
            return
        if skill is None:
            self.details.setPlainText(
                f"{row['name']}\n{row['host']} · {row['market']} · {row['version']}\n\n"
                f"{row['description']}\n\n{len(row['skills'])} skill files."
            )
            return
        try:
            body = _bounded(skill, row["cache"], MAX_SKILL)
        except (OSError, ValueError, RuntimeError):
            self.details.setPlainText("The skill file is unavailable or exceeds its read limit.")
            return
        self.details.setPlainText(body)
        self.selected = (skill.parent.name, body)
        self.use.setEnabled(True)

    def _use_skill(self):
        if self.selected:
            self.skill_requested.emit(*self.selected)

    def _install_pack(self):
        if self.pack and _is_regular(self.pack):
            self.pack_requested.emit(str(self.pack))
        else:
            self.details.setPlainText("The pack is no longer available at its indexed location.")
            self.install.setEnabled(False)
