"""The Library's local project folders and authenticated shared-note links."""

import json
import re
import stat
from html import escape
from pathlib import Path

from PySide6.QtCore import QStandardPaths, Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QHBoxLayout, QTreeWidget, QTreeWidgetItem, QVBoxLayout, QWidget

from .configuration import runtime_path
from .ui import ROLE, button, label

ARCHIVE = runtime_path("desktop") / "project-archive"
CATALOGUE_LIMIT = 4 * 1024 * 1024


def local_path(path: Path, *bases: Path) -> Path:
    if (
        not path.is_absolute()
        or path.drive.startswith("\\")
        or ".." in path.parts
        or not any(path != base and path.is_relative_to(base) for base in bases)
    ):
        raise ValueError("Project location is outside local folders")
    # Check ancestors before touching descendants so a junction cannot trigger network access.
    for part in (*reversed(path.parents), path):
        info = part.lstat()
        if stat.S_ISLNK(info.st_mode) or (
            getattr(info, "st_file_attributes", 0) & stat.FILE_ATTRIBUTE_REPARSE_POINT
        ):
            raise ValueError("Redirected project location")
    return path


def load_projects(archive: Path) -> list[dict]:
    catalogue = local_path(archive / "project-catalogue.json", archive)
    with catalogue.open("rb") as stream:
        raw = stream.read(CATALOGUE_LIMIT + 1)
    if len(raw) > CATALOGUE_LIMIT:
        raise ValueError("Project catalogue is too large")
    data = json.loads(raw)
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("Unsupported project catalogue")
    projects = data.get("projects")
    if not isinstance(projects, list) or len(projects) > 1000:
        raise ValueError("Invalid project list")
    return projects


def shared_note(entry: object) -> dict | None:
    if not isinstance(entry, dict):
        return None
    path, digest = entry.get("path"), entry.get("sha256")
    if not isinstance(path, str) or not isinstance(digest, str):
        return None
    parts = path.split("/")
    if (
        len(path) > 1024
        or not path.casefold().endswith(".md")
        or any(part in {"", ".", ".."} for part in parts)
        or parts[0].casefold() in {"private", "archive", "inbox"}
        or any(c in path for c in "\\:\0")
        or not re.fullmatch("[0-9a-f]{64}", digest)
    ):
        return None
    return {"path": path, "sha256": digest}


class ProjectLibrary(QWidget):
    note_requested = Signal(object)
    refs_requested = Signal()

    def __init__(self, archive: Path = ARCHIVE, desktop: Path | None = None):
        super().__init__()
        self.archive = archive.absolute()
        self._shared: list[tuple[QTreeWidgetItem, list[tuple[QTreeWidgetItem, dict]]]] = []
        self.desktop = desktop or Path(
            QStandardPaths.writableLocation(QStandardPaths.StandardLocation.DesktopLocation)
        )
        self.tree = QTreeWidget(
            objectName="projectTree", accessibleName="Project folders and notes"
        )
        self.tree.setHeaderHidden(True)
        self.tree.setUniformRowHeights(True)
        self.tree.setIndentation(12)
        self.tree.setExpandsOnDoubleClick(False)
        self.tree.itemActivated.connect(self._open_item)
        self.counts = label("", "muted")
        self.status = label("", "muted")
        self.status.setWordWrap(True)
        maps = button("Open all mind maps", role="primary")
        maps.setObjectName("openArchiveIndex")
        maps.clicked.connect(self.open_archive_index)
        reload = button("Reload projects")
        reload.clicked.connect(self.reload)
        actions = QHBoxLayout()
        actions.addWidget(maps)
        actions.addWidget(reload)
        actions.addStretch()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.addWidget(label("Project folders", "title"))
        layout.addWidget(self.counts)
        layout.addLayout(actions)
        layout.addWidget(self.tree, 1)
        layout.addWidget(self.status)

    def reload(self):
        self._shared.clear()
        self.tree.clear()
        self.counts.clear()
        self.status.clear()
        try:
            projects = load_projects(self.archive)
        except (OSError, ValueError, RuntimeError):
            self.status.setText("Project catalogue unavailable. Reload after updating the archive.")
            return
        desktop = QTreeWidgetItem(self.tree, ["Desktop folders"])
        other = QTreeWidgetItem(self.tree, ["Other projects"])
        folders: dict[tuple[str, ...], QTreeWidgetItem] = {}
        desktop_roots: set[Path] = set()
        project_count = 0
        for project in projects:
            if not isinstance(project, dict):
                continue
            title = project.get("title")
            if not isinstance(title, str) or not title.strip():
                continue
            project_count += 1
            roots = project.get("roots", [])
            roots = (
                [root for root in roots if isinstance(root, dict)]
                if isinstance(roots, list)
                else []
            )
            parents = set()
            for root in roots:
                value = root.get("path")
                if root.get("kind") != "desktop" or not isinstance(value, str):
                    continue
                path = Path(value)
                if (
                    path != self.desktop
                    and path.is_relative_to(self.desktop)
                    and ".." not in path.parts
                ):
                    parents.add(path.relative_to(self.desktop).parts[:-1])
                    desktop_roots.add(path)
            if not parents:
                self._add_project(other, project, title, roots)
                continue
            for parts in sorted(parents):
                parent = desktop
                for index, folder in enumerate(parts):
                    prefix = parts[: index + 1]
                    if prefix not in folders:
                        folders[prefix] = QTreeWidgetItem(parent, [folder])
                    parent = folders[prefix]
                self._add_project(parent, project, title, roots)
        for section in (desktop, other):
            section.setHidden(not section.childCount())
            section.setExpanded(True)
        self.tree.sortItems(0, Qt.SortOrder.AscendingOrder)
        if desktop.isHidden() and other.isHidden():
            self.status.setText("No projects in the catalogue.")
        else:
            projects_label = "project" if project_count == 1 else "projects"
            folders_label = "folder" if len(desktop_roots) == 1 else "folders"
            self.counts.setText(
                f"{project_count} {projects_label} · {len(desktop_roots)} Desktop {folders_label}"
            )
        self.refs_requested.emit()

    def open_archive_index(self) -> None:
        self._open_location("map", "index.html")

    def _add_project(self, parent, project, title, roots):
        item = QTreeWidgetItem(parent, [title])
        item.setToolTip(0, f"<span>{escape(title)}</span>")
        identity = project.get("id")
        if isinstance(identity, str) and re.fullmatch("[a-z0-9-]{1,150}", identity):
            relative = f"projects/{identity}/mind-map.html"
            if project.get("map") == relative:
                row = QTreeWidgetItem(item, ["Open mind map"])
                row.setData(0, ROLE, ("map", relative))
                row = QTreeWidgetItem(item, ["Open archive folder"])
                row.setData(0, ROLE, ("archive", f"projects/{identity}"))
        for root in roots:
            path = root.get("path")
            if not isinstance(path, str):
                continue
            caption = root.get("label")
            caption = caption if isinstance(caption, str) and caption else "Open project folder"
            row = QTreeWidgetItem(item, [caption])
            row.setToolTip(0, f"<span>{escape(path)}</span>")
            row.setData(0, ROLE, ("folder", path))
        notes = project.get("sharedNotes")
        if not isinstance(notes, list):
            return
        refs = [ref for entry in notes if (ref := shared_note(entry))]
        if not refs:
            return
        group = QTreeWidgetItem(item, ["Shared notes"])
        group.setHidden(True)
        rows = []
        for ref in refs:
            row = QTreeWidgetItem(group)
            row.setData(0, ROLE, ("note", ref))
            row.setHidden(True)
            rows.append((row, ref))
        self._shared.append((group, rows))

    def note_paths(self) -> list[str]:
        return list(dict.fromkeys(ref["path"] for _, rows in self._shared for _, ref in rows))

    def set_note_refs(self, available: dict):
        for group, rows in self._shared:
            visible = False
            for row, ref in rows:
                current = available.get(ref["path"], {})
                matches = current.get("sha256") == ref["sha256"]
                title = current.get("title") or ref["path"]
                row.setText(0, title if matches else "")
                row.setHidden(not matches)
                visible |= matches
            group.setHidden(not visible)

    def _open_item(self, item, _column):
        if item.isHidden():
            return
        action = item.data(0, ROLE)
        if not action:
            item.setExpanded(not item.isExpanded())
            return
        kind, target = action
        if kind == "note":
            self.note_requested.emit(target)
            return
        self._open_location(kind, target)

    def _open_location(self, kind: str, target: str):
        try:
            if kind in {"map", "archive"}:
                path = local_path(self.archive / target, self.archive)
            else:
                path = local_path(Path(target), Path.home(), self.desktop)
            if kind == "map" and not path.is_file():
                raise ValueError("Invalid project map")
            if kind != "map" and not path.is_dir():
                raise ValueError("Invalid project folder")
        except (OSError, ValueError, RuntimeError):
            self.status.setText("That project location is unavailable.")
            return
        if QDesktopServices.openUrl(QUrl.fromLocalFile(str(path))):
            self.status.clear()
        else:
            self.status.setText("Windows could not open that project location.")
