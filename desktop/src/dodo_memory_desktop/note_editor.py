"""Write ordinary shared Markdown with gateway-backed link completion."""

import re
from pathlib import PurePosixPath
from urllib.parse import quote, unquote

from PySide6.QtCore import QEvent, Qt, QTimer, Signal
from PySide6.QtGui import QKeySequence, QShortcut, QTextCursor
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QListWidget,
    QPlainTextEdit,
    QVBoxLayout,
)

from .ui import ROLE, label, row_item

OPEN_LINK = re.compile(r"(?<!\\)\[\[([^\[\]\r\n|]{0,256})$")
MAX_NOTE_BYTES = 256 * 1024


def note_title(target: str) -> str:
    name = PurePosixPath(unquote(target.split("#", 1)[0].strip())).name
    if name.casefold().endswith(".md"):
        name = name[:-3]
    return name


def new_note_path(title: str) -> str:
    title = re.sub(r'[\x00-\x1f<>:"/\\|?*\[\]#]', " ", title)
    title = " ".join(title.split()).strip(" .")[:120].rstrip(" .")
    if re.fullmatch(r"(CON|PRN|AUX|NUL|COM[1-9¹²³]|LPT[1-9¹²³])(?:\..*)?", title, re.IGNORECASE):
        title = f"Note {title}"
    if not title:
        raise ValueError("Enter a title for the note.")
    return f"notes/{title}.md"


class NoteEditor(QDialog):
    suggest_requested = Signal(object)
    save_requested = Signal(object)

    def __init__(self, note: dict, parent):
        super().__init__(
            parent, windowTitle="New shared note" if not note["sha256"] else "Edit note"
        )
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose)
        self.resize(720, 560)
        self.path = note["path"]
        self.sha256 = note["sha256"]
        self.markdown = QPlainTextEdit()
        self.markdown.setAccessibleName("Note Markdown")
        self.markdown.setPlainText(note["text"])
        self.markdown.installEventFilter(self)
        self.suggestions = QListWidget()
        self.suggestions.setMaximumHeight(135)
        self.suggestions.setAccessibleName("Matching notes and headings")
        self.suggestions.itemActivated.connect(self._insert)
        self.suggestions.installEventFilter(self)
        self.suggestions.hide()
        self.initial = self.markdown.toPlainText()
        self.status = label("Shared note. Type [[ to link a note or heading.", "muted")
        self.status.setWordWrap(True)
        actions = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel)
        self.save = actions.addButton("Save note", QDialogButtonBox.ButtonRole.AcceptRole)
        self.save.setProperty("role", "primary")
        self.save.clicked.connect(self._save)
        actions.rejected.connect(self.reject)
        root = QVBoxLayout(self)
        root.addWidget(label(PurePosixPath(self.path).stem, "section"))
        root.addWidget(self.markdown, 1)
        root.addWidget(self.suggestions)
        root.addWidget(self.status)
        root.addWidget(actions)
        self.timer = QTimer(self, interval=140, singleShot=True)
        self.timer.timeout.connect(self._suggest)
        self.markdown.textChanged.connect(self._changed)
        self.markdown.cursorPositionChanged.connect(self._changed)
        QShortcut(QKeySequence("Ctrl+S"), self, activated=self._save)
        self.busy = False
        self.match = None
        self.markdown.setFocus()
        self._changed()

    def _anchor(self):
        cursor = self.markdown.textCursor()
        cursor.clearSelection()
        position = cursor.position()
        cursor.setPosition(cursor.block().position(), QTextCursor.MoveMode.KeepAnchor)
        match = OPEN_LINK.search(cursor.selectedText())
        return (position, match[1]) if match else None

    def _changed(self):
        anchor = self._anchor()
        if anchor != self.match:
            self.match = anchor
            self.suggestions.hide()
            self.timer.stop()
            if anchor is not None:
                self.timer.start()
        text = self.markdown.toPlainText()
        fits = len(text.encode()) <= MAX_NOTE_BYTES
        if not fits:
            self.status.setText("This note exceeds 256 KiB.")
        elif self.status.text() == "This note exceeds 256 KiB.":
            self.status.setText("Shared note. Type [[ to link a note or heading.")
        self.save.setEnabled(
            not self.busy
            and fits
            and bool(text.strip())
            and (self.sha256 is None or text != self.initial)
        )

    def _suggest(self):
        if self.match is not None and not self.busy:
            self.suggest_requested.emit(self.match)

    def show_suggestions(self, anchor, result):
        if self.busy or not self.isVisible() or self._anchor() != anchor:
            return
        self.suggestions.clear()
        for row in result["items"]:
            caption = f"{row['title']}  ›  {row['heading']}" if row.get("heading") else row["title"]
            row_item(self.suggestions, f"{caption}  ·  {row['path']}", row)
        self.suggestions.setVisible(bool(self.suggestions.count()))
        self.suggestions.setCurrentRow(0)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Type.KeyPress and self.suggestions.isVisible():
            if event.key() == Qt.Key.Key_Down and watched is self.markdown:
                self.suggestions.setFocus()
                return True
            if event.key() == Qt.Key.Key_Escape:
                self.suggestions.hide()
                self.markdown.setFocus()
                return True
        return super().eventFilter(watched, event)

    def _insert(self, item):
        row = item.data(ROLE)
        anchor = self._anchor()
        if not row or anchor is None:
            return
        cursor = self.markdown.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.Right, QTextCursor.MoveMode.KeepAnchor, 2)
        end = cursor.position() if cursor.selectedText() == "]]" else anchor[0]
        units = len(anchor[1].encode("utf-16-le")) // 2
        cursor.setPosition(end)
        cursor.setPosition(anchor[0] - units, QTextCursor.MoveMode.KeepAnchor)
        heading = (
            f"#{quote(row['heading'].lstrip('# ').strip(), safe=' ,')}"
            if row.get("heading")
            else ""
        )
        cursor.insertText(f"/{quote(row['path'], safe='/ ,')}{heading}]]")
        self.markdown.setTextCursor(cursor)
        self.markdown.setFocus()

    def reject(self):
        if not self.busy:
            super().reject()

    def _save(self):
        if not self.save.isEnabled():
            return
        self.busy = True
        self.timer.stop()
        self.suggestions.hide()
        self.markdown.setReadOnly(True)
        self.save.setEnabled(False)
        self.status.setText("Saving shared note…")
        self.save_requested.emit(
            {"path": self.path, "text": self.markdown.toPlainText(), "expected_sha256": self.sha256}
        )

    def save_failed(self, message):
        self.busy = False
        self.markdown.setReadOnly(False)
        self.status.setText(message)
        self._changed()
