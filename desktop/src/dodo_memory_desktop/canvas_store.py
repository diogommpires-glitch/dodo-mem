"""Validate, import and save local JSON Canvas documents."""

import hashlib
import json
import os
import re
import stat
from ntpath import isreserved
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit

from PySide6.QtCore import QBuffer, QByteArray, QIODevice, QLockFile, QSaveFile, Qt
from PySide6.QtGui import QImage, QImageReader

MAX_JSON_BYTES = 2 * 1024**2
MAX_IMAGE_BYTES = 12 * 1024**2
MAX_NODES = 250
MAX_EDGES = 500
MAX_IMAGES = 24
MAX_IMPORT_IMAGE_BYTES = 32 * 1024**2
WORLD = 1_000_000
IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
SIDES = {"top", "right", "bottom", "left"}
ENDS = {"none", "arrow"}


def relative_file(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512:
        raise ValueError("Invalid canvas file reference.")
    parts = value.split("/")
    if any(
        not part
        or part in {".", ".."}
        or part[-1] in {" ", "."}
        or any(ord(character) < 32 or character in '<>:"\\|?*' for character in part)
        or isreserved(part)
        for part in parts
    ):
        raise ValueError("Canvas file references must be safe relative paths.")
    return PurePosixPath(value)


def web_url(value: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) > 2048
        or "\\" in value
        or any(ord(ch) < 33 for ch in value)
    ):
        raise ValueError("Enter a valid HTTP or HTTPS address.")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username is not None:
        raise ValueError("Enter an HTTP or HTTPS address without credentials.")
    return value


def _text(record: dict, key: str, limit: int) -> str:
    value = record.get(key)
    if not isinstance(value, str) or len(value) > limit:
        raise ValueError(f"Invalid canvas {key}.")
    return value


def _bounded_list(value: object, count: int) -> list:
    if not isinstance(value, list) or len(value) > count:
        raise ValueError("Canvas exceeds its card or connection limit.")
    return value


def _colour(record: dict, result: dict) -> None:
    if "color" in record:
        value = _text(record, "color", 7)
        if not re.fullmatch(r"#[a-fA-F0-9]{6}|[1-6]", value):
            raise ValueError("Invalid canvas colour.")
        result["color"] = value


def _node(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Invalid canvas card.")  # noqa: TRY004 - Untrusted document data.
    result = {key: _text(raw, key, 96) for key in ("id", "type")}
    if not result["id"]:
        raise ValueError("Canvas cards need identifiers.")
    for key in ("x", "y", "width", "height"):
        value = raw.get(key)
        lower, upper = (-WORLD, WORLD) if key in {"x", "y"} else (1, 4096)
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError("Canvas card geometry exceeds the supported bounds.")
        result[key] = value
    kind = result["type"]
    if kind == "text":
        result["text"] = _text(raw, "text", 20_000)
    elif kind == "link":
        result["url"] = web_url(raw.get("url"))
    elif kind == "file":
        path = relative_file(raw.get("file"))
        if path.suffix.lower() not in IMAGE_TYPES | {".md"}:
            raise ValueError("Canvas file cards support Markdown references and local images.")
        result["file"] = path.as_posix()
        if "subpath" in raw:
            subpath = _text(raw, "subpath", 400)
            if not subpath.startswith("#") or path.suffix.lower() != ".md":
                raise ValueError("Canvas subpaths must identify a Markdown heading.")
            result["subpath"] = subpath
    elif kind == "group":
        if raw.get("background"):
            raise ValueError("Canvas group backgrounds are not supported.")
        if "label" in raw:
            result["label"] = _text(raw, "label", 240)
    else:
        raise ValueError("Unsupported canvas card type.")
    _colour(raw, result)
    return result


def _edge(raw: object, nodes: set[str]) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("Invalid canvas connection.")  # noqa: TRY004 - Untrusted document data.
    result = {key: _text(raw, key, 96) for key in ("id", "fromNode", "toNode")}
    if not result["id"] or not {result["fromNode"], result["toNode"]} <= nodes:
        raise ValueError("Canvas connection refers to a missing card or identifier.")
    for key, allowed in (
        ("fromSide", SIDES),
        ("toSide", SIDES),
        ("fromEnd", ENDS),
        ("toEnd", ENDS),
    ):
        if key in raw:
            value = _text(raw, key, 8)
            if value not in allowed:
                raise ValueError("Invalid canvas connection endpoint.")
            result[key] = value
    if "label" in raw:
        result["label"] = _text(raw, "label", 240)
    _colour(raw, result)
    return result


def validate_board(raw: object) -> dict:
    if not isinstance(raw, dict):
        raise ValueError("A canvas must be a JSON object.")  # noqa: TRY004 - Untrusted document data.
    nodes = [_node(node) for node in _bounded_list(raw.get("nodes", []), MAX_NODES)]
    node_ids = {node["id"] for node in nodes}
    edges = [_edge(edge, node_ids) for edge in _bounded_list(raw.get("edges", []), MAX_EDGES)]
    if len(node_ids) != len(nodes) or len({edge["id"] for edge in edges}) != len(edges):
        raise ValueError("Canvas card and connection identifiers must be unique.")
    if node_ids & {edge["id"] for edge in edges}:
        raise ValueError("Canvas cards and connections cannot share an identifier.")
    images = {
        node["file"]
        for node in nodes
        if node["type"] == "file" and not node["file"].lower().endswith(".md")
    }
    if len(images) > MAX_IMAGES:
        raise ValueError(f"A canvas holds at most {MAX_IMAGES} distinct images.")
    title = raw.get("title", "Untitled canvas")
    if not isinstance(title, str) or not title.strip() or len(title) > 120:
        raise ValueError("Canvas titles must contain between 1 and 120 characters.")
    return {"title": title.strip(), "nodes": nodes, "edges": edges}


def read_bounded(path: Path, limit: int) -> bytes:
    path = path.absolute()
    if path.resolve(strict=True) != path or path.is_symlink():
        raise ValueError(f"{path.name} is a redirected canvas file.")
    with path.open("rb") as stream:
        opened = os.fstat(stream.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{path.name} is not a regular file.")
        content = stream.read(limit + 1)
        after = os.fstat(stream.fileno())
    if (
        path.resolve(strict=True) != path
        or not os.path.samestat(after, path.stat())
        or (opened.st_size, opened.st_mtime_ns) != (after.st_size, after.st_mtime_ns)
    ):
        raise ValueError(f"{path.name} changed while it was being read.")
    if len(content) > limit:
        raise ValueError(f"{path.name} exceeds the canvas size limit.")
    return content


def _unique_keys(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Canvas JSON contains duplicate keys.")
        result[key] = value
    return result


def read_board(path: Path) -> tuple[dict, str]:
    data = read_bounded(path, MAX_JSON_BYTES)
    try:
        raw = json.loads(data, object_pairs_hook=_unique_keys)
    except (UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Canvas is not valid JSON.") from error
    return validate_board(raw), hashlib.sha256(data).hexdigest()


def atomic_write(path: Path, data: bytes) -> None:
    output = QSaveFile(str(path))
    output.setDirectWriteFallback(False)
    if not output.open(QIODevice.OpenModeFlag.WriteOnly):
        raise OSError(output.errorString())
    if output.write(data) != len(data):
        output.cancelWriting()
        raise OSError(output.errorString())
    if not output.commit():
        raise OSError(output.errorString())


def save_board(path: Path, board: dict, expected: str | None, *, portable: bool = False) -> str:
    clean = validate_board(board)
    if portable:
        del clean["title"]
    data = (json.dumps(clean, ensure_ascii=False, indent=2) + "\n").encode()
    if len(data) > MAX_JSON_BYTES:
        raise ValueError("Canvas exceeds the size limit.")
    lock = QLockFile(str(path) + ".lock")
    if not lock.tryLock(0):
        raise OSError("The canvas is in use by another window.")
    try:
        if path.is_symlink():
            raise ValueError("Canvas files cannot be filesystem links.")
        current = None
        if path.exists():
            current = hashlib.sha256(read_bounded(path, MAX_JSON_BYTES)).hexdigest()
        if current != expected:
            raise ValueError("The canvas changed on disk. Reopen it before saving.")
        atomic_write(path, data)
    finally:
        lock.unlock()
    return hashlib.sha256(data).hexdigest()


def contained_file(root: Path, value: str) -> Path:
    parts = relative_file(value).parts
    root = root.resolve(strict=True)
    target = root
    for part in parts:
        target = target / part
        attrs = getattr(target.lstat(), "st_file_attributes", 0)
        if target.is_symlink() or attrs & stat.FILE_ATTRIBUTE_REPARSE_POINT:
            raise ValueError("Canvas images cannot traverse filesystem links.")
    if not target.is_file() or not target.resolve(strict=True).is_relative_to(root):
        raise ValueError("Canvas image is outside its import folder.")
    return target


def image_bytes(path: Path) -> bytes:
    content = read_bounded(path, MAX_IMAGE_BYTES)
    if _read_image(content).isNull():
        raise ValueError(f"{path.name} is not a supported, bounded image.")
    return content


def _read_image(content: bytes) -> QImage:
    data = QByteArray(content)
    stream = QBuffer(data)
    stream.open(QIODevice.OpenModeFlag.ReadOnly)
    reader = QImageReader(stream)
    if bytes(reader.format()).lower() not in {b"png", b"jpeg", b"webp", b"bmp"}:
        return QImage()
    size = reader.size()
    if not size.isValid() or size.width() * size.height() > 12_000_000:
        return QImage()
    if max(size.width(), size.height()) > 768:
        size.scale(768, 768, Qt.AspectRatioMode.KeepAspectRatio)
        reader.setScaledSize(size)
    reader.setAutoTransform(True)
    return reader.read()


def load_image(path: Path) -> QImage:
    return _read_image(read_bounded(path, MAX_IMAGE_BYTES))


def store_image(root: Path, content: bytes, suffix: str) -> str:
    filename = hashlib.sha256(content).hexdigest() + suffix.lower()
    target = root / "media" / filename
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.parent.resolve(strict=True) != target.parent or target.is_symlink():
        raise ValueError("Canvas media directory is redirected.")
    if not target.exists():
        atomic_write(target, content)
    elif read_bounded(target, MAX_IMAGE_BYTES) != content:
        raise ValueError("Canvas media failed its integrity check.")
    return f"media/{filename}"
